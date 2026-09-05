import os

LOCAL_RAW_DATA_DIR = "./data"
LOCAL_RAW_DATASET_DIR = os.path.join(LOCAL_RAW_DATA_DIR, "dataset")
HUGGINGFACE_DATASET_NAME = "vectara/open_ragbench"
LOCAL_RAW_PDF_DATA_DIR = os.path.join(LOCAL_RAW_DATA_DIR, "raw_dataset/pdf/raw_pdf")
PDF_URLS_JSON_NAME = "pdf_urls.json"
PDF_EXTENTION=".pdf"
FAILED_DOWNLOADS_JSON_NAME = "failed_downloads.json"
PART_EXTENSION = ".part"
PDF_MAGIC_BYTES = b"%PDF"
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_RETRY_BACKOFF_SECONDS = 60

# --- PDF parsing (data_process.py) -------------------------------------------
LOCAL_PROCESSED_DIR = os.path.join(LOCAL_RAW_DATA_DIR, "processed/pdf")
PROCESSED_DOCS_DIR_NAME = "docs"
PROCESSED_IMAGES_DIR_NAME = "images"
PROCESSED_LAYOUT_DIR_NAME = "layout"
FAILED_PARSES_JSON_NAME = "failed_parses.json"
JSON_EXTENSION = ".json"

# Rendering / asset extraction
RENDER_DPI = 200                    # dpi used when rasterising vector figures
MIN_FIGURE_SIZE = 60.0              # pt; smaller vector clusters are rules/glyphs, not figures
MAX_FIGURE_TEXT_COVERAGE = 0.70     # a "figure" whose area is mostly text is a text box
FIGURE_LABEL_MAX_CHARS = 80         # axis ticks and legends absorbed into the figure box
MIN_PROSE_WORDS = 5                 # word-like tokens that make a block prose, not a label
MAX_FIGURE_GROWTH = 1.5             # cap on how far absorbing labels may grow a figure
FIGURE_LABEL_PADDING = 12.0         # pt around a figure searched for its labels
MIN_RASTER_SIZE = 24.0              # pt; smaller bitmaps are inline glyphs or spacers
FIGURE_MERGE_GAP = 10.0             # pt; panels this close belong to one figure
PROSE_CUT_OVERLAP = 0.5             # share of a cluster's width prose must span to cut it
MIN_TABLE_FILL = 0.35               # share of cells with content; below this it is ruled artwork
MIN_TABLE_ROWS = 2
REFINE_TABLES = True                # split rows the line grid merged (needs pymupdf.layout)
MIN_TABLE_COLS = 2
IMAGE_EXTENSION = ".png"

# Reading order (recursive XY-cut)
MIN_COLUMN_GAP = 10.0               # pt of clear vertical whitespace that marks a column gutter
MIN_ROW_GAP = 1.0                   # pt of clear horizontal whitespace that separates blocks

# Heading detection
HEADING_MAX_CHARS = 200
HEADING_MAX_LINES = 3
HEADING_SIZE_RATIO = 1.10           # span size relative to body text that marks a heading
HEADING_L1_RATIO = 1.40
HEADING_L2_RATIO = 1.18
BOLD_FLAG = 1 << 4                  # pymupdf span flag bits
ITALIC_FLAG = 1 << 1
MAX_HEADING_LEVEL = 4
MIN_HEADING_CHARS = 3
RUN_IN_TERMINATORS = (".", ":", "-", "\u2014")   # "Proof.", "Model-", "Key idea:"
RUN_IN_MAX_WORDS = 8
MIN_HEADING_ALPHA = 4               # letters required, so display maths is not a heading

# Running headers / footers: repeated margin text (journal name, page number)
MARGIN_BAND = 0.08                  # fraction of page height treated as margin
RUNNING_HEAD_MIN_PAGES = 3
RUNNING_HEAD_MIN_RATIO = 0.30       # share of pages a line must repeat on to be dropped

# Captions
CAPTION_MAX_DISTANCE = 80.0         # pt between an asset and its caption block
CAPTION_PREFIXES = ("figure", "fig.", "fig ", "table", "tab.", "algorithm", "listing")

# Scanned-page detection
MIN_CHARS_PER_PAGE = 50             # below this, with a full-page image, the page needs OCR
OCR_IMAGE_PAGE_COVERAGE = 0.60

# Placeholders embedded in the section markdown, e.g. ![img-0.png](img-0.png)
IMAGE_ID_TEMPLATE = "img-{n}" + IMAGE_EXTENSION
TABLE_ID_TEMPLATE = "table-{n}"
PLACEHOLDER_TEMPLATE = "![{id}]({id})"

# --- Corpus mirror on the Hugging Face Hub (fetch_data.py) -------------------
# The 6.8 GB of PDFs, extracted figures and parse output lives here rather than
# in git. Everything except `descriptions/` is reproducible from ingestion.py
# and data_process.py — the descriptions cost money, so they are the reason the
# mirror exists at all.
HF_CORPUS_REPO = "sandeep-halder/research-query-corpus"
HF_TOKEN_ENV_VARS = ("HF_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Named subsets, so a laptop can pull the 508 KB it needs instead of 6.8 GB.
CORPUS_PARTS = {
    "descriptions": ["processed/pdf/descriptions/**"],
    "docs":         ["processed/pdf/docs/**", "processed/pdf/layout/**"],
    "processed":    ["processed/**"],
    "dataset":      ["dataset/**"],
    "raw":          ["raw_dataset/**"],
    "all":          None,
}
