"""Parse the raw arXiv PDFs into position-aware, multimodal markdown.

For every PDF in ``data/raw_dataset/pdf/raw_pdf`` this writes one JSON document
holding the page text as markdown, with ``![img-0.png](img-0.png)`` /
``![table-0](table-0)`` placeholders sitting at the exact spot in the reading
flow where the figure or table appeared. The assets themselves are extracted to
PNG files (images) or markdown (tables) and described in per-section metadata,
so a chunker can carry the position binding through to retrieval.

Layout is recovered with a recursive XY-cut over the page's blocks, which keeps
two-column arXiv papers in reading order instead of interleaving the columns.

    cd Ingestion && uv run data_process.py [--force] [--limit N] [--layout]
"""

import argparse
import collections
import json
import os
import re
import statistics
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf
from tqdm import tqdm

from constants import (LOCAL_RAW_PDF_DATA_DIR, LOCAL_PROCESSED_DIR, PROCESSED_DOCS_DIR_NAME,
                       PROCESSED_IMAGES_DIR_NAME, PROCESSED_LAYOUT_DIR_NAME, FAILED_PARSES_JSON_NAME,
                       PDF_EXTENTION, JSON_EXTENSION, IMAGE_EXTENSION, RENDER_DPI, MIN_FIGURE_SIZE,
                       MAX_FIGURE_TEXT_COVERAGE, FIGURE_LABEL_MAX_CHARS, MIN_PROSE_WORDS, FIGURE_LABEL_PADDING,
                       MIN_RASTER_SIZE, FIGURE_MERGE_GAP, PROSE_CUT_OVERLAP, MAX_FIGURE_GROWTH,
                       MIN_TABLE_FILL, MIN_TABLE_ROWS, MIN_TABLE_COLS, REFINE_TABLES, MIN_COLUMN_GAP, MIN_ROW_GAP, HEADING_MAX_CHARS,
                       HEADING_MAX_LINES, HEADING_SIZE_RATIO, HEADING_L1_RATIO, HEADING_L2_RATIO,
                       BOLD_FLAG, ITALIC_FLAG, MAX_HEADING_LEVEL, MIN_HEADING_CHARS,
                       RUN_IN_TERMINATORS, RUN_IN_MAX_WORDS, MIN_HEADING_ALPHA, MARGIN_BAND,
                       RUNNING_HEAD_MIN_PAGES, RUNNING_HEAD_MIN_RATIO, CAPTION_MAX_DISTANCE, CAPTION_PREFIXES, MIN_CHARS_PER_PAGE,
                       OCR_IMAGE_PAGE_COVERAGE, IMAGE_ID_TEMPLATE, TABLE_ID_TEMPLATE,
                       PLACEHOLDER_TEMPLATE)

# pymupdf writes advisory notes straight to stdout, which would shred the
# progress bar. Route them into the void.
pymupdf.set_messages(text=f"path:{os.devnull}")

# pymupdf.layout is an optional ONNX layout model. With it, find_tables() locates
# borderless tables and `refine` recovers the rows a line-only grid merges into
# one cell — the difference between a 2x4 and a 25x8 reading of the same table.
try:
    import pymupdf.layout  # noqa: F401  (import activates the analyzer)
    HAS_LAYOUT_MODEL = True
except ImportError:
    HAS_LAYOUT_MODEL = False

SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+\S")
WHITESPACE_RE = re.compile(r"[ \t]+")
DIGITS_RE = re.compile(r"\d+")


def _running_heads(elements, pages):
    """Text that repeats in the top/bottom margins — journal heads, page numbers.

    Returns the normalised lines to drop. Page numbers differ per page, so
    digits are masked before counting repeats.
    """
    heights = {p["page"]: p["height"] for p in pages}
    seen = collections.defaultdict(set)
    for element in elements:
        if element["kind"] != "text":
            continue
        height = heights.get(element["page"], 0)
        if not height:
            continue
        top, bottom = element["bbox"][1], element["bbox"][3]
        if top > height * MARGIN_BAND and bottom < height * (1 - MARGIN_BAND):
            continue
        seen[DIGITS_RE.sub("#", element["text"])].add(element["page"])

    threshold = max(RUNNING_HEAD_MIN_PAGES, len(pages) * RUNNING_HEAD_MIN_RATIO)
    return {text for text, page_set in seen.items() if len(page_set) >= threshold}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _pad(bbox, amount):
    return (bbox[0] - amount, bbox[1] - amount, bbox[2] + amount, bbox[3] + amount)


def _covered_by(inner, outer, threshold=0.6):
    """True when `threshold` of `inner`'s area falls inside `outer`."""
    inner_area = _area(inner)
    return inner_area > 0 and _intersection_area(inner, outer) / inner_area >= threshold


def _merged_gaps(intervals, min_gap):
    """Largest clear gap between the merged 1-D `intervals`, or None."""
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    best = None
    for (_, end), (start, _) in zip(merged, merged[1:]):
        gap = start - end
        if gap >= min_gap and (best is None or gap > best[0]):
            best = (gap, (end + start) / 2)
    return best[1] if best else None


def _reading_order(boxes):
    """Order box indices by recursive XY-cut: columns first, then rows."""

    def cut(indices):
        if len(indices) <= 1:
            return list(indices)

        # A gutter running the full height of this group means real columns.
        split = _merged_gaps([(boxes[i][0], boxes[i][2]) for i in indices], MIN_COLUMN_GAP)
        if split is not None:
            left = [i for i in indices if boxes[i][0] < split]
            right = [i for i in indices if boxes[i][0] >= split]
            if left and right:
                return cut(left) + cut(right)

        split = _merged_gaps([(boxes[i][1], boxes[i][3]) for i in indices], MIN_ROW_GAP)
        if split is not None:
            top = [i for i in indices if boxes[i][1] < split]
            bottom = [i for i in indices if boxes[i][1] >= split]
            if top and bottom:
                return cut(top) + cut(bottom)

        # Overlapping boxes that resist cutting: fall back to top-left first.
        return sorted(indices, key=lambda i: (boxes[i][1], boxes[i][0]))

    return cut(range(len(boxes)))


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def _line_text(line):
    return "".join(span["text"] for span in line["spans"]).strip()


def _join(previous, addition):
    """Append a line to running text, undoing end-of-line hyphenation."""
    if not previous:
        return addition
    if previous.endswith("-"):
        return previous[:-1] + addition
    return previous + " " + addition


def _block_text(block):
    text = ""
    for line in block["lines"]:
        line_text = _line_text(line)
        if line_text:
            text = _join(text, line_text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _block_style(block):
    """(dominant size, bold fraction) for a text block, weighted by characters."""
    sizes = collections.Counter()
    bold_chars = total_chars = 0
    for line in block["lines"]:
        for span in line["spans"]:
            n = len(span["text"])
            if not n:
                continue
            sizes[round(span["size"], 1)] += n
            total_chars += n
            if span["flags"] & BOLD_FLAG:
                bold_chars += n
    if not total_chars:
        return 0.0, 0.0
    return max(sizes.items(), key=lambda kv: kv[1])[0], bold_chars / total_chars


def _is_rotated(block):
    return any(abs(line["dir"][1]) > 0.01 for line in block["lines"])


def _emphatic(span, body_size):
    """A span that stands out from body text: bold, italic, or larger."""
    return (bool(span["flags"] & (BOLD_FLAG | ITALIC_FLAG))
            or span["size"] >= body_size * HEADING_SIZE_RATIO)


def _heading_level(text, size, body_size):
    """Numbering depth when the heading is numbered, else relative font size."""
    numbered = SECTION_NUMBER_RE.match(text)
    if numbered:
        depth = numbered.group(1).rstrip(".").count(".") + 1
        return min(depth, MAX_HEADING_LEVEL)
    ratio = size / body_size if body_size else 1.0
    if ratio >= HEADING_L1_RATIO:
        return 1
    if ratio >= HEADING_L2_RATIO:
        return 2
    return 3


def _looks_like_heading(text, size, body_size):
    """Headings are numbered, all-caps, visually larger, or a run-in label."""
    if not text or not MIN_HEADING_CHARS <= len(text) <= HEADING_MAX_CHARS:
        return False
    if sum(character.isalpha() for character in text) < MIN_HEADING_ALPHA:
        return False  # display maths picked up from an italic font
    if SECTION_NUMBER_RE.match(text) or text.isupper():
        return True
    if size >= body_size * HEADING_SIZE_RATIO:
        return True
    # Run-in label such as "Proof." or "Numerical simulation:-", set off in bold
    # or italic ahead of the paragraph it opens. Guard against inline emphasis
    # and against italic maths by demanding a terminator and a real word.
    return (text[0].isupper() and text.endswith(RUN_IN_TERMINATORS)
            and len(text.split()) <= RUN_IN_MAX_WORDS
            and any(sum(c.isalpha() for c in word) >= 3 for word in text.split()))


def _split_heading(block, body_size):
    """Split a block into (heading, level, body).

    Handles both a heading on its own line and the run-in style these papers
    use, where the heading opens the same block as the paragraph that follows.
    """
    lines = block["lines"]
    if not lines or body_size <= 0 or _is_rotated(block):
        return None, None, _block_text(block)

    heading_lines, heading_size, consumed_spans = [], 0.0, 0
    for index, line in enumerate(lines):
        spans = line["spans"]
        if not any(span["text"].strip() for span in spans):
            continue
        emphatic = _leading_run(spans, body_size)
        if not emphatic:
            break
        heading_size = max(heading_size, max(s["size"] for s in emphatic))
        heading_lines.append("".join(s["text"] for s in emphatic).strip())
        consumed_spans = len(emphatic)
        if len(emphatic) < len(spans) or index + 1 >= HEADING_MAX_LINES:
            break

    heading = WHITESPACE_RE.sub(" ", _reduce(heading_lines)).strip()
    if not _looks_like_heading(heading, heading_size, body_size):
        return None, None, _block_text(block)

    body = _remaining_text(lines, len(heading_lines), consumed_spans)
    return heading, _heading_level(heading, heading_size, body_size), body


def _leading_run(spans, body_size):
    """Leading emphatic spans; whitespace-only spans neither start nor break it."""
    run = []
    for span in spans:
        if not span["text"].strip():
            if run:
                run.append(span)
            continue
        if not _emphatic(span, body_size):
            break
        run.append(span)
    while run and not run[-1]["text"].strip():
        run.pop()
    return run


def _reduce(parts):
    text = ""
    for part in parts:
        if part:
            text = _join(text, part)
    return text


def _remaining_text(lines, heading_line_count, consumed_spans):
    """Everything after the heading: the tail of its last line, then the rest."""
    text = ""
    last_index = heading_line_count - 1
    if 0 <= last_index < len(lines):
        spans = lines[last_index]["spans"]
        tail = "".join(s["text"] for s in spans[consumed_spans:]).strip()
        text = _join(text, tail) if tail else text
    for line in lines[heading_line_count:]:
        line_text = _line_text(line)
        if line_text:
            text = _join(text, line_text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _is_real_table(table):
    """Reject the ruled artwork find_tables() mistakes for a mostly empty grid."""
    if table.row_count < MIN_TABLE_ROWS or table.col_count < MIN_TABLE_COLS:
        return False
    cells = [cell for row in table.extract() for cell in row]
    if not cells:
        return False
    filled = sum(1 for cell in cells if cell and str(cell).strip())
    return filled / len(cells) >= MIN_TABLE_FILL


def _absorb_labels(bbox, text_blocks):
    """Grow a figure box over the axis ticks and legends sitting on its edge.

    Without this the labels are left stranded in the prose and clipped out of
    the rendered PNG.
    """
    limit = _area(bbox) * MAX_FIGURE_GROWTH
    grown = bbox
    for _ in range(3):
        merged = grown
        for block in text_blocks:
            if _is_prose(block) or len(block["text"]) > FIGURE_LABEL_MAX_CHARS:
                continue
            if _intersection_area(block["bbox"], _pad(merged, FIGURE_LABEL_PADDING)) <= 0:
                continue
            candidate = _union([merged, block["bbox"]])
            if _area(candidate) <= limit and not _hits_prose(candidate, text_blocks):
                merged = candidate
        if merged == grown:
            break
        grown = merged
    return grown


def _union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _is_prose(block):
    """Body text or a caption — the boundary a figure box must never cross.

    Counted in words, not characters: a row of axis ticks is long but wordless,
    and treating it as prose would cut a plot in half at its own x-axis.
    """
    text = block["text"]
    if text.lower().startswith(CAPTION_PREFIXES):
        return True
    words = sum(1 for word in text.split() if sum(c.isalpha() for c in word) >= 2)
    return words >= MIN_PROSE_WORDS


def _hits_prose(bbox, text_blocks):
    return any(_is_prose(b) and _intersection_area(b["bbox"], bbox) > 0 for b in text_blocks)


def _split_at_prose(bbox, text_blocks, drawing_rects):
    """Cut a drawing cluster that swallowed a caption into the bands around it.

    cluster_drawings() happily spans two stacked figures and the caption
    between them; each band is then tightened back onto the drawings it holds.
    """
    width = bbox[2] - bbox[0]
    barriers = sorted((b["bbox"] for b in text_blocks
                       if _is_prose(b) and _intersection_area(b["bbox"], bbox) > 0
                       and (min(bbox[2], b["bbox"][2]) - max(bbox[0], b["bbox"][0]))
                       >= width * PROSE_CUT_OVERLAP),
                      key=lambda barrier: barrier[1])
    if not barriers:
        return [bbox]

    bands, top = [], bbox[1]
    for barrier in barriers:
        if barrier[1] > top:
            bands.append((bbox[0], top, bbox[2], min(barrier[1], bbox[3])))
        top = max(top, barrier[3])
    if top < bbox[3]:
        bands.append((bbox[0], top, bbox[2], bbox[3]))

    tightened = []
    for band in bands:
        inside = [_clip(rect, band) for rect in drawing_rects
                  if _intersection_area(rect, band) > 0]
        if not inside:
            continue
        tight = _union(inside)
        if tight[2] - tight[0] >= MIN_FIGURE_SIZE and tight[3] - tight[1] >= MIN_FIGURE_SIZE:
            tightened.append(tight)
    return tightened


def _clip(bbox, window):
    return (max(bbox[0], window[0]), max(bbox[1], window[1]),
            min(bbox[2], window[2]), min(bbox[3], window[3]))


def _trim_to_column(bbox, text_blocks):
    """Pull a figure box back off the neighbouring column.

    A stray drawing op can push a cluster a few points into the next column,
    and because the figure is rendered as a clip, that column's text would be
    baked into the image.
    """
    left, top, right, bottom = bbox
    centre = (left + right) / 2
    for block in text_blocks:
        if not _is_prose(block):
            continue
        block_box = block["bbox"]
        if min(bottom, block_box[3]) - max(top, block_box[1]) <= 0:
            continue                        # no vertical overlap: not alongside
        if block_box[0] >= centre:
            right = min(right, block_box[0])
        elif block_box[2] <= centre:
            left = max(left, block_box[2])
    if right - left < MIN_FIGURE_SIZE:
        return bbox                         # trimming would erase the figure
    return (left, top, right, bottom)


def _merge_figures(candidates, text_blocks):
    """Group panels of one figure — a 2x2 subplot grid is one image, not four."""
    groups = [[candidate] for candidate in candidates]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                first, second = _union(b for b, _ in groups[i]), _union(b for b, _ in groups[j])
                if _intersection_area(_pad(first, FIGURE_MERGE_GAP), second) <= 0:
                    continue
                if _hits_prose(_union([first, second]), text_blocks):
                    continue     # panels separated by a caption are separate figures
                groups[i] += groups[j]
                del groups[j]
                merged = True
                break
            if merged:
                break
    return groups


def _find_caption(bbox, text_blocks, kind):
    """Nearest 'Figure 3: ...' / 'Table 1: ...' block above or below `bbox`."""
    best = None
    for block in text_blocks:
        text = block["text"]
        if not text.lower().startswith(CAPTION_PREFIXES):
            continue
        # Must sit in the same column band.
        overlap = min(bbox[2], block["bbox"][2]) - max(bbox[0], block["bbox"][0])
        if overlap <= 0:
            continue
        if block["bbox"][1] >= bbox[3]:          # below the asset
            distance = block["bbox"][1] - bbox[3]
        elif block["bbox"][3] <= bbox[1]:        # above it
            distance = bbox[1] - block["bbox"][3]
        else:
            continue
        if distance > CAPTION_MAX_DISTANCE:
            continue
        # Figures caption below, tables above — break ties that way.
        rank = (distance, 0 if kind == "table" else 1)
        if best is None or rank < best[0]:
            best = (rank, text)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

class PDFParser:
    """Turns one PDF into position-aware markdown plus extracted assets."""

    def __init__(self, pdf_path, doc_id, image_dir, render_dpi=RENDER_DPI):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.image_dir = image_dir
        self.render_dpi = render_dpi
        self.image_count = 0
        self.table_count = 0

    # -- assets ------------------------------------------------------------- #

    def _next_image_id(self):
        image_id = IMAGE_ID_TEMPLATE.format(n=self.image_count)
        self.image_count += 1
        return image_id

    def _save_pixmap(self, pixmap, image_id):
        if pixmap.is_unicolor:
            return None            # a blank spacer bitmap or an empty render
        os.makedirs(self.image_dir, exist_ok=True)
        path = os.path.join(self.image_dir, image_id)
        if pixmap.colorspace and pixmap.colorspace.n > 3:      # CMYK et al
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        pixmap.save(path)
        return path, pixmap.width, pixmap.height

    def _render_region(self, page, bbox, image_id):
        pixmap = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=self.render_dpi)
        return self._save_pixmap(pixmap, image_id)

    def _extract_raster(self, doc, page, xref, bbox, image_id):
        """Embedded bitmap at its native resolution; falls back to rendering."""
        try:
            pixmap = pymupdf.Pixmap(doc, xref)
            smask = doc.extract_image(xref).get("smask")
            if smask:
                pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(doc, smask))
            if pixmap.width < 2 or pixmap.height < 2:
                raise ValueError("degenerate image")
            saved = self._save_pixmap(pixmap, image_id)
            if saved is not None:
                return saved, "embedded"
        except Exception:
            pass
        return self._render_region(page, bbox, image_id), "rendered"

    # -- page elements ------------------------------------------------------ #

    def _page_elements(self, doc, page, page_no):
        """Every text block, table and figure on a page, in reading order."""
        tables = self._page_tables(page)
        table_boxes = [t["bbox"] for t in tables]

        raw_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
        text_blocks = []
        for block in raw_blocks:
            text = _block_text(block)
            if not text:
                continue
            size, _ = _block_style(block)
            text_blocks.append({"bbox": tuple(block["bbox"]), "text": text, "size": size,
                                "block": block})

        figures = self._page_figures(doc, page, page_no, table_boxes, text_blocks)
        asset_boxes = table_boxes + [f["bbox"] for f in figures]

        elements = list(figures) + list(tables)
        # Text that lives inside a table or figure is already carried by that asset.
        elements += [dict(b, kind="text") for b in text_blocks
                     if not any(_covered_by(b["bbox"], box) for box in asset_boxes)]

        order = _reading_order([e["bbox"] for e in elements])
        ordered = [elements[i] for i in order]
        for element in ordered:
            element["page"] = page_no
        return ordered, text_blocks

    def _page_tables(self, page):
        tables = []
        try:
            found = page.find_tables(refine=REFINE_TABLES and HAS_LAYOUT_MODEL)
        except Exception:
            return tables
        for table in found.tables:
            markdown = (table.to_markdown() or "").strip()
            if not markdown or not _is_real_table(table):
                continue
            tables.append({"kind": "table", "bbox": tuple(table.bbox), "markdown": markdown,
                           "rows": table.row_count, "cols": table.col_count})
        return tables

    def _page_figures(self, doc, page, page_no, table_boxes, text_blocks):
        """Embedded bitmaps plus clusters of vector drawing ops (LaTeX plots)."""
        candidates = self._figure_candidates(page, table_boxes, text_blocks)
        figures = []
        for group in _merge_figures(candidates, text_blocks):
            bbox = _trim_to_column(_absorb_labels(_union(box for box, _ in group), text_blocks),
                                   text_blocks)
            xref = group[0][1] if len(group) == 1 else None   # only a lone bitmap keeps its xref
            image_id = self._next_image_id()
            if xref is not None:
                extracted, mode = self._extract_raster(doc, page, xref, bbox, image_id)
            else:
                extracted, mode = self._render_region(page, bbox, image_id), "rendered"
            if extracted is None:
                self.image_count -= 1     # blank artwork; reuse the id
                continue
            path, width, height = extracted
            figures.append({"kind": "image", "bbox": bbox, "id": image_id, "path": path,
                            "width": width, "height": height, "extraction": mode,
                            "source": "raster" if xref is not None else "vector"})
        return figures

    def _figure_candidates(self, page, table_boxes, text_blocks):
        """(bbox, xref) pairs worth extracting; xref is None for vector artwork."""
        candidates, claimed, seen = [], list(table_boxes), set()

        for info in page.get_images(full=True):
            xref = info[0]
            for rect in page.get_image_rects(xref):
                bbox = tuple(rect)
                key = (xref, tuple(round(v) for v in bbox))
                if key in seen:
                    continue        # the same bitmap stamped twice at one spot
                seen.add(key)
                if rect.width < MIN_RASTER_SIZE or rect.height < MIN_RASTER_SIZE:
                    continue        # inline glyph or spacer, not a figure
                candidates.append((bbox, xref))
                claimed.append(bbox)

        try:
            drawings = page.get_drawings()
            clusters = page.cluster_drawings(drawings=drawings)
        except Exception:
            drawings, clusters = [], []
        drawing_rects = [tuple(drawing["rect"]) for drawing in drawings]

        for rect in clusters:
            if rect.width < MIN_FIGURE_SIZE or rect.height < MIN_FIGURE_SIZE:
                continue
            for bbox in _split_at_prose(tuple(rect), text_blocks, drawing_rects):
                if any(_intersection_area(bbox, box) / _area(bbox) > 0.5 for box in claimed):
                    continue        # already covered by a table or a bitmap
                text_area = sum(_intersection_area(b["bbox"], bbox) for b in text_blocks)
                if text_area / _area(bbox) > MAX_FIGURE_TEXT_COVERAGE:
                    continue        # a boxed paragraph, not a figure
                candidates.append((bbox, None))
                claimed.append(bbox)
        return candidates

    # -- document ----------------------------------------------------------- #

    def parse(self):
        with pymupdf.open(self.pdf_path) as doc:
            metadata = doc.metadata or {}
            elements, pages = [], []
            for page_no, page in enumerate(doc, start=1):
                page_elements, text_blocks = self._page_elements(doc, page, page_no)
                for element in page_elements:
                    if element["kind"] in ("image", "table"):
                        element["caption"] = _find_caption(element["bbox"], text_blocks,
                                                           element["kind"])
                elements.extend(page_elements)
                pages.append(self._page_summary(page, page_no, page_elements))

            elements = self._strip_running_heads(elements, pages)
            body_size = self._body_size(elements)
            sections = self._build_sections(elements, body_size)
            return {
                "id": self.doc_id,
                "source_pdf": os.path.relpath(self.pdf_path),
                "title": (metadata.get("title") or "").strip() or self._first_heading(sections),
                "authors": (metadata.get("author") or "").strip(),
                "page_count": len(pages),
                "body_font_size": body_size,
                "needs_ocr": any(p["needs_ocr"] for p in pages),
                "counts": {"sections": len(sections), "images": self.image_count,
                           "tables": self.table_count},
                "pages": pages,
                "sections": sections,
            }

    @staticmethod
    def _page_summary(page, page_no, page_elements):
        chars = sum(len(e["text"]) for e in page_elements if e["kind"] == "text")
        page_area = _area(tuple(page.rect))
        image_area = sum(_area(e["bbox"]) for e in page_elements if e["kind"] == "image")
        return {
            "page": page_no,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "chars": chars,
            "needs_ocr": chars < MIN_CHARS_PER_PAGE and page_area > 0
            and image_area / page_area >= OCR_IMAGE_PAGE_COVERAGE,
        }

    @staticmethod
    def _strip_running_heads(elements, pages):
        heads = _running_heads(elements, pages)
        return [e for e in elements
                if e["kind"] != "text" or DIGITS_RE.sub("#", e["text"]) not in heads]

    @staticmethod
    def _body_size(elements):
        sizes = [e["size"] for e in elements if e["kind"] == "text" and e["size"]]
        return statistics.median(sizes) if sizes else 0.0

    @staticmethod
    def _first_heading(sections):
        for section in sections:
            if section["heading"]:
                return section["heading"]
        return ""

    def _build_sections(self, elements, body_size):
        """Split the ordered elements at headings; emit markdown per section."""
        sections = []
        current = self._new_section(0, None, None)

        for element in elements:
            if element["kind"] == "text":
                heading, level, body = _split_heading(element["block"], body_size)
                if heading:
                    if current["text"] or current["heading"]:
                        sections.append(current)
                    current = self._new_section(len(sections), heading, level)
                    self._append(current, "#" * level + " " + heading)
                current["pages"].add(element["page"])
                if body:
                    self._append(current, body)
                continue

            asset_id = element.get("id") or TABLE_ID_TEMPLATE.format(n=self.table_count)
            current["pages"].add(element["page"])
            offset = self._append(current, PLACEHOLDER_TEMPLATE.format(id=asset_id))
            record = {"page": element["page"], "bbox": [round(v, 2) for v in element["bbox"]],
                      "caption": element.get("caption"), "char_offset": offset,
                      "order": len(current["elements"])}
            if element["kind"] == "image":
                record.update(path=os.path.relpath(element["path"], os.path.dirname(self.image_dir)),
                              width=element["width"], height=element["height"],
                              source=element["source"], extraction=element["extraction"])
                current["images"][asset_id] = record
            else:
                self.table_count += 1
                record.update(markdown=element["markdown"], rows=element["rows"],
                              cols=element["cols"])
                current["tables"][asset_id] = record
            current["elements"].append({"id": asset_id, "kind": element["kind"],
                                        "page": element["page"], "char_offset": offset})

        if current["text"] or current["heading"]:
            sections.append(current)
        return [self._finalise(section) for section in sections]

    @staticmethod
    def _new_section(section_id, heading, level):
        return {"section_id": section_id, "heading": heading, "level": level, "text": "",
                "pages": set(), "images": {}, "tables": {}, "elements": []}

    @staticmethod
    def _append(section, chunk):
        """Append a markdown chunk; returns its character offset in the section."""
        if section["text"]:
            section["text"] += "\n\n"
        offset = len(section["text"])
        section["text"] += chunk
        return offset

    @staticmethod
    def _finalise(section):
        section["pages"] = sorted(section["pages"])
        section["chars"] = len(section["text"])
        return section


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

def _process_one(args):
    """Worker entry point — must be module level so it can be pickled."""
    pdf_path, doc_id, docs_dir, images_dir, layout_dir = args
    try:
        parser = PDFParser(pdf_path, doc_id, os.path.join(images_dir, doc_id))
        document = parser.parse()
        out_path = os.path.join(docs_dir, doc_id + JSON_EXTENSION)
        tmp_path = out_path + ".part"
        if layout_dir:
            document["layout_file"] = _write_layout(document, layout_dir, doc_id)
        with open(tmp_path, "w") as f:
            json.dump(document, f, ensure_ascii=False, indent=1)
        os.replace(tmp_path, out_path)
        return doc_id, document["counts"], None
    except Exception as e:
        return doc_id, None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"


def _write_layout(document, layout_dir, doc_id):
    """Per-section element positions, split out so the doc JSON stays small."""
    os.makedirs(layout_dir, exist_ok=True)
    path = os.path.join(layout_dir, doc_id + JSON_EXTENSION)
    layout = {"id": doc_id, "pages": document["pages"], "sections": [
        {"section_id": s["section_id"], "heading": s["heading"], "pages": s["pages"],
         "elements": s["elements"]} for s in document["sections"]]}
    with open(path, "w") as f:
        json.dump(layout, f, ensure_ascii=False, indent=1)
    return os.path.relpath(path, os.path.dirname(layout_dir))


class DataProcessor:
    """Parses every raw PDF into ``data/processed/pdf/``, resumably."""

    def __init__(self, pdf_dir, out_dir):
        self.pdf_dir = pdf_dir
        self.out_dir = out_dir
        self.docs_dir = os.path.join(out_dir, PROCESSED_DOCS_DIR_NAME)
        self.images_dir = os.path.join(out_dir, PROCESSED_IMAGES_DIR_NAME)
        self.layout_dir = os.path.join(out_dir, PROCESSED_LAYOUT_DIR_NAME)
        self.failures_path = os.path.join(out_dir, FAILED_PARSES_JSON_NAME)

    def _pending(self, force, limit, only):
        names = sorted(n for n in os.listdir(self.pdf_dir) if n.endswith(PDF_EXTENTION))
        if only:
            wanted = {o if o.endswith(PDF_EXTENTION) else o + PDF_EXTENTION for o in only}
            names = [n for n in names if n in wanted]

        pending, done = [], 0
        for name in names:
            doc_id = name[: -len(PDF_EXTENTION)]
            if not force and os.path.exists(os.path.join(self.docs_dir, doc_id + JSON_EXTENSION)):
                done += 1
                continue
            pending.append((os.path.join(self.pdf_dir, name), doc_id))
            if limit and len(pending) >= limit:
                break
        return pending, len(names), done

    def _save_failures(self, failures):
        if not failures:
            try:
                os.remove(self.failures_path)
            except FileNotFoundError:
                pass
            return
        with open(self.failures_path, "w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)

    def run(self, workers=None, force=False, limit=None, only=None, layout=False):
        for directory in (self.docs_dir, self.images_dir):
            os.makedirs(directory, exist_ok=True)
        layout_dir = self.layout_dir if layout else None

        pending, total, done = self._pending(force, limit, only)
        print(f"{total} PDFs | {done} already parsed | {len(pending)} to parse")
        if not pending:
            return

        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        jobs = [(path, doc_id, self.docs_dir, self.images_dir, layout_dir)
                for path, doc_id in pending]
        failures, totals = {}, collections.Counter()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
                doc_id, counts, error = future.result()
                if error:
                    failures[doc_id] = error
                    tqdm.write(f"Failed {doc_id}: {error.splitlines()[0]}")
                else:
                    totals.update(counts)
                    totals["docs"] += 1

        self._save_failures(failures)
        print(f"Parsed {totals['docs']} docs | {totals['sections']} sections | "
              f"{totals['images']} images | {totals['tables']} tables | {len(failures)} failed")
        if failures:
            print(f"Failures logged to {self.failures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse raw PDFs into position-aware markdown with image and table references")
    parser.add_argument("--force", action="store_true", help="re-parse PDFs already on disk")
    parser.add_argument("--limit", type=int, help="stop after N unparsed PDFs (smoke tests)")
    parser.add_argument("--only", nargs="+", metavar="PAPER_ID", help="parse just these ids")
    parser.add_argument("--workers", type=int, help="parallel processes (default: cores - 1)")
    parser.add_argument("--layout", action="store_true",
                        help=f"also write element positions to {PROCESSED_LAYOUT_DIR_NAME}/")
    args = parser.parse_args()

    DataProcessor(LOCAL_RAW_PDF_DATA_DIR, LOCAL_PROCESSED_DIR).run(
        workers=args.workers, force=args.force, limit=args.limit, only=args.only,
        layout=args.layout)
