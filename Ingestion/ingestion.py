import argparse
import asyncio
import json
import os
import httpx
from huggingface_hub import snapshot_download
from constants import (LOCAL_RAW_DATASET_DIR, HUGGINGFACE_DATASET_NAME, LOCAL_RAW_PDF_DATA_DIR,
                       PDF_URLS_JSON_NAME, PDF_EXTENTION, FAILED_DOWNLOADS_JSON_NAME, PART_EXTENSION,
                       PDF_MAGIC_BYTES, RETRYABLE_STATUS_CODES, MAX_RETRY_BACKOFF_SECONDS)
from tqdm.asyncio import tqdm_asyncio


class DATAIngestor:
    def __init__(self, dataset_path, pdf_path):
        self.dataset_path = dataset_path
        self.pdf_path = pdf_path
        
        self.failures_path = os.path.join(os.path.dirname(self.pdf_path) or ".", FAILED_DOWNLOADS_JSON_NAME)

    async def download_dataset(self):
        
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=HUGGINGFACE_DATASET_NAME,
            repo_type="dataset",
            local_dir=self.dataset_path
        )
        return path

    def _pdf_file_path(self, paper_id):
        file_name = paper_id if paper_id.endswith(PDF_EXTENTION) else paper_id + PDF_EXTENTION
        return os.path.join(self.pdf_path, file_name)

    @staticmethod
    def _is_complete_pdf(file_path):
        try:
            if os.path.getsize(file_path) == 0:
                return False
            with open(file_path, "rb") as f:
                return f.read(len(PDF_MAGIC_BYTES)) == PDF_MAGIC_BYTES
        except OSError:
            return False

    def load_failures(self):
        try:
            with open(self.failures_path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_failures(self, failures):
        if not failures:
            try:
                os.remove(self.failures_path)
            except FileNotFoundError:
                pass
            return
        os.makedirs(os.path.dirname(self.failures_path) or ".", exist_ok=True)
        with open(self.failures_path, "w") as f:
            json.dump(failures, f, indent=2, sort_keys=True)

    @staticmethod
    def _record_failure(failures, paper_id, pdf_url, reason, attempts, detail):
        failures[paper_id] = {"url": pdf_url, "reason": reason, "attempts": attempts, "detail": detail}
        tqdm_asyncio.write(f"Failed {paper_id} ({reason}): {detail}")

    @staticmethod
    async def _sleep_before_retry(attempt, response=None):
        backoff = 2 ** (attempt - 1)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                backoff = max(backoff, float(retry_after))
            except ValueError:
                pass
        await asyncio.sleep(min(backoff, MAX_RETRY_BACKOFF_SECONDS))

    def _load_pdf_urls(self):
        pdf_urls = {}
        for root, _, files in os.walk(self.dataset_path):
            if PDF_URLS_JSON_NAME in files:
                json_path = os.path.join(root, PDF_URLS_JSON_NAME)
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        pdf_urls.update(data)
        return pdf_urls

    async def _download_single_pdf(self, client: httpx.AsyncClient, paper_id: str, pdf_url: str,
                                   semaphore: asyncio.Semaphore, failures: dict, retries: int = 3):
        file_path = self._pdf_file_path(paper_id)
        part_path = file_path + PART_EXTENSION
        async with semaphore:
            try:
                for attempt in range(1, retries + 1):
                    try:
                        response = await client.get(pdf_url, follow_redirects=True)
                        if response.status_code == 200:
                            if not response.content.startswith(PDF_MAGIC_BYTES):
                                self._record_failure(failures, paper_id, pdf_url, "not_a_pdf", attempt,
                                                     "response body is not a PDF")
                                return
                            with open(part_path, "wb") as f:
                                f.write(response.content)
                            os.replace(part_path, file_path)
                            failures.pop(paper_id, None)
                            return
                        if response.status_code in RETRYABLE_STATUS_CODES and attempt < retries:
                            await self._sleep_before_retry(attempt, response)
                            continue
                        self._record_failure(failures, paper_id, pdf_url, "http_status", attempt,
                                             f"status code {response.status_code}")
                        return
                    except Exception as e:
                        if attempt < retries:
                            await self._sleep_before_retry(attempt)
                            continue
                        self._record_failure(failures, paper_id, pdf_url, type(e).__name__, attempt, str(e))
                        return
            finally:
                try:
                    os.remove(part_path)
                except FileNotFoundError:
                    pass

    async def download_pdf(self, max_concurrent: int = 10, retry_failed: bool = False, force: bool = False):
        pdf_urls = self._load_pdf_urls()
        if not pdf_urls:
            print(f"No {PDF_URLS_JSON_NAME} found under {self.dataset_path}")
            return

        failures = self.load_failures()
        if retry_failed:
            if not failures:
                print(f"No recorded failures in {self.failures_path}, nothing to retry")
                return
            pdf_urls = {pid: url for pid, url in pdf_urls.items() if pid in failures}

        os.makedirs(self.pdf_path, exist_ok=True)

        pending = {}
        skipped = 0
        for paper_id, pdf_url in pdf_urls.items():
            if not force and self._is_complete_pdf(self._pdf_file_path(paper_id)):
                skipped += 1
                failures.pop(paper_id, None)
                continue
            pending[paper_id] = pdf_url

        print(f"{len(pdf_urls)} PDFs known | {skipped} already on disk | {len(pending)} to download")
        if not pending:
            self._save_failures(failures)
            return

        semaphore = asyncio.Semaphore(max_concurrent)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=max_concurrent)
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            tasks = [
                self._download_single_pdf(client, paper_id, pdf_url, semaphore, failures)
                for paper_id, pdf_url in pending.items()
            ]
            await tqdm_asyncio.gather(*tasks, desc="Downloading PDFs")

        self._save_failures(failures)
        failed_now = [paper_id for paper_id in pending if paper_id in failures]
        print(f"Downloaded {len(pending) - len(failed_now)} | failed {len(failed_now)}")
        if failed_now:
            print(f"Failures logged to {self.failures_path} — re-run with --retry-failed")

    async def run(self, retry_failed: bool = False, force: bool = False):
        await self.download_dataset()
        await self.download_pdf(retry_failed=retry_failed, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the Open RAG Benchmark dataset and its source PDFs")
    parser.add_argument("--retry-failed", action="store_true",
                        help=f"only re-attempt ids recorded in {FAILED_DOWNLOADS_JSON_NAME}")
    parser.add_argument("--force", action="store_true",
                        help="re-download every PDF, ignoring files already on disk")
    args = parser.parse_args()

    ingestor = DATAIngestor(LOCAL_RAW_DATASET_DIR, LOCAL_RAW_PDF_DATA_DIR)
    asyncio.run(ingestor.run(retry_failed=args.retry_failed, force=args.force))
