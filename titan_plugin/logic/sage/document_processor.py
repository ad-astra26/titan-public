"""
logic/sage/document_processor.py

The Document Intelligence Engine for Titan V2.0 Step 5 (The Stealth-Sage).

Downloads documents (.pdf, .docx, .pptx, .xlsx) to a temporary "Safe-Room" directory,
parses them using the Unstructured library, distills the content via a local Ollama
phi3:mini model, and immediately deletes the temp file. Enforces sequential processing
via asyncio.Queue to prevent CPU spike on the VPS (load average guard).
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# Maximum characters extracted from Unstructured elements sent to Ollama
_MAX_CONTENT_CHARS = 8_000

# Supported document extensions
_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Dedicated phi3:mini prompt for document analysis (production-grade)
_DOCUMENT_PROMPT_TEMPLATE = (
    "You are the Titan's Document Analyzer. "
    "Below are raw elements extracted from a document. "
    "Extract only the key technical facts and data and format your response "
    "as a single JSON object with these exact keys: "
    '"topic" (string: one-sentence description of the document subject), '
    '"key_points" (list of strings: up to 7 most important facts), '
    '"source_type" (string: always set to "Document"). '
    "Respond ONLY with the JSON object, no markdown fencing, no extra text.\n\n"
    "Document content:\n{content}"
)


class DocumentProcessor:
    """
    Sequential, VPS-safe document parsing engine.

    Uses an asyncio.Queue to ensure only one document is processed at a time,
    preventing CPU overload on constrained VPS hardware. Documents are downloaded
    to a temporary directory and deleted immediately after content extraction.

    Integrates:
        - Unstructured (auto-detect PDF/DOCX/PPTX/XLSX)
        - Ollama phi3:mini (local LLM summarization)
    """

    def __init__(
        self,
        safe_room: str = "/tmp/titan_sage_docs",
        max_load_avg: float = 2.0,
        proxy_url: Optional[str] = None,
    ) -> None:
        """
        Initializes the DocumentProcessor.

        Args:
            safe_room (str): Directory path for temporary document downloads.
            max_load_avg (float): Maximum 1-minute load average before skipping processing.
            proxy_url (Optional[str]): Webshare rotating proxy URL for document downloads.
        """
        # Ollama Cloud client — wired by TitanPlugin.__init__ if configured
        self._ollama_cloud = None
        self._safe_room = Path(safe_room)
        self._max_load_avg = max_load_avg
        self._proxy_url = proxy_url.strip() if proxy_url else None
        self._queue: asyncio.Queue = asyncio.Queue()

        # Ensure the safe room directory exists
        self._safe_room.mkdir(parents=True, exist_ok=True)

        # Lazy-check unstructured availability at init time
        self._unstructured_available = self._check_unstructured()

    @staticmethod
    def _check_unstructured() -> bool:
        """Checks whether the unstructured library is importable."""
        try:
            from unstructured.partition.auto import partition  # noqa: F401
            return True
        except ImportError:
            log.warning(
                "[DocumentProcessor] 'unstructured' library not installed. "
                "Document deep-dives will be skipped. "
                "Run: pip install \"unstructured[pdf,docx,pptx,xlsx]\""
            )
            return False

    @staticmethod
    def is_document_url(url: str) -> bool:
        """
        Returns True if the URL points to a supported document type.

        Supported extensions: .pdf, .docx, .pptx, .xlsx

        Args:
            url (str): The URL to inspect.

        Returns:
            bool: True if the URL path ends with a supported document extension.
        """
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in _DOC_EXTENSIONS)

    def _is_load_safe(self) -> bool:
        """
        Checks the VPS 1-minute load average against the configured threshold.
        Returns False if the system is under high load (processing should be deferred).
        """
        try:
            load1, _, _ = os.getloadavg()
            if load1 > self._max_load_avg:
                log.warning(
                    f"[DocumentProcessor] System load {load1:.2f} > "
                    f"{self._max_load_avg}. Skipping document processing."
                )
                return False
        except AttributeError:
            # os.getloadavg() not available on Windows — allow processing
            pass
        return True

    async def _download_document(self, url: str) -> Optional[Path]:
        """
        Downloads a document from the given URL into the safe_room directory.

        Handles both http(s):// URLs and local file:// paths.
        Uses the configured Webshare proxy for remote downloads.

        Args:
            url (str): The URL or file:// path of the document to download.

        Returns:
            Optional[Path]: Path to the downloaded temp file, or None on failure.
        """
        parsed = urlparse(url)

        # Handle local file:// paths (e.g., for test cases)
        if parsed.scheme == "file":
            local_path = Path(parsed.path)
            if local_path.exists():
                return local_path
            log.error(f"[DocumentProcessor] Local file not found: {local_path}")
            return None

        # Determine extension for temp file naming
        ext = Path(parsed.path).suffix.lower() or ".tmp"
        proxy = self._proxy_url if self._proxy_url else None

        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

            # Write to a unique temp file in the safe room
            tmp_file = self._safe_room / f"sage_doc_{id(response)}{ext}"
            tmp_file.write_bytes(response.content)
            log.info(f"[DocumentProcessor] Downloaded document to {tmp_file} ({len(response.content)} bytes).")
            return tmp_file

        except httpx.HTTPStatusError as e:
            log.warning(f"[DocumentProcessor] HTTP error downloading {url}: {e.response.status_code}")
        except Exception as e:
            log.warning(f"[DocumentProcessor] Failed to download document from {url}: {e}")
        return None

    async def _parse_with_unstructured(self, file_path: Path) -> str:
        """
        Parses a document file using Unstructured's auto-detection and returns
        a concatenated string of element texts up to _MAX_CONTENT_CHARS.

        Args:
            file_path (Path): Path to the document file.

        Returns:
            str: Extracted text content from the document.
        """
        try:
            # Run blocking unstructured I/O in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()

            def _partition():
                from unstructured.partition.auto import partition
                elements = partition(filename=str(file_path))
                return "\n".join(str(el) for el in elements)

            raw_text = await loop.run_in_executor(None, _partition)
            return raw_text[:_MAX_CONTENT_CHARS]

        except Exception as e:
            log.error(f"[DocumentProcessor] Unstructured parsing failed for {file_path}: {e}")
            return ""

    async def _distill_with_ollama(self, content: str) -> dict:
        """
        Sends the extracted document content to Ollama Cloud for structured
        JSON extraction. Returns the parsed JSON dict, or {} on failure.

        Args:
            content (str): Raw document text to distill.

        Returns:
            dict: Parsed JSON with keys 'topic', 'key_points', 'source_type'.
        """
        if not content.strip():
            return {}

        if not self._ollama_cloud:
            log.warning("[DocumentProcessor] No Ollama Cloud client — skipping distillation.")
            return {}

        prompt = _DOCUMENT_PROMPT_TEMPLATE.format(content=content)

        try:
            from titan_plugin.utils.ollama_cloud import get_model_for_task
            model = get_model_for_task("document_distill")
            raw_text = await self._ollama_cloud.complete(
                prompt=prompt,
                model=model,
                temperature=0.1,
                max_tokens=500,
                timeout=60.0,
            )

            if not raw_text:
                return {"source_type": "Document", "topic": "Unknown", "key_points": []}

            # Strip markdown fencing if the model added it despite instructions
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").lstrip("json").strip()

            result = json.loads(raw_text)
            # Always enforce the source_type tag
            result["source_type"] = "Document"
            return result

        except json.JSONDecodeError as e:
            log.warning(f"[DocumentProcessor] Ollama Cloud returned non-JSON response: {e}")
            return {"source_type": "Document", "topic": "Unknown", "key_points": []}
        except Exception as e:
            log.warning(f"[DocumentProcessor] Ollama Cloud distillation failed: {e}")
            return {}

    async def process(self, doc_url: str) -> dict:
        """
        Full document processing pipeline: download → parse → distill → clean up.

        Enforces VPS load guard and sequential queue to prevent CPU spikes.
        Guarantees temp file deletion even on failure (uses finally block).

        Args:
            doc_url (str): URL or file:// path of the document to process.

        Returns:
            dict: Result with keys 'topic' (str), 'key_points' (list), 'source_type' ("Document"),
                  plus a 'summary' field with a formatted human-readable summary string.
                  Returns {} if processing is skipped, unavailable, or fails.
        """
        if not self._unstructured_available:
            return {}

        if not self._is_load_safe():
            return {}

        # Serialize via queue to prevent concurrent document downloads on VPS
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((doc_url, future))

        async def _worker():
            item_url, item_future = await self._queue.get()
            tmp_path: Optional[Path] = None
            try:
                tmp_path = await self._download_document(item_url)
                if tmp_path is None:
                    item_future.set_result({})
                    return

                raw_content = await self._parse_with_unstructured(tmp_path)
                if not raw_content.strip():
                    log.warning(f"[DocumentProcessor] No extractable content from {item_url}.")
                    item_future.set_result({})
                    return

                result = await self._distill_with_ollama(raw_content)

                if result:
                    # Build a human-readable summary string for the research pipeline
                    topic = result.get("topic", "Unknown topic")
                    points = result.get("key_points", [])
                    summary_lines = [f"Document Topic: {topic}"]
                    for pt in points:
                        summary_lines.append(f"  • {pt}")
                    result["summary"] = "\n".join(summary_lines)
                    log.info(f"[DocumentProcessor] Successfully processed document: {item_url}")

                item_future.set_result(result)

            except Exception as e:
                log.error(f"[DocumentProcessor] Pipeline error for {item_url}: {e}")
                if not item_future.done():
                    item_future.set_result({})
            finally:
                # Always clean up the temp file — never leave documents on disk
                if tmp_path and tmp_path.exists():
                    # Only delete files WE downloaded (not local file:// test inputs)
                    if tmp_path.parent == self._safe_room:
                        try:
                            tmp_path.unlink()
                            log.debug(f"[DocumentProcessor] Cleaned up temp file: {tmp_path}")
                        except OSError as e:
                            log.warning(f"[DocumentProcessor] Could not delete temp file {tmp_path}: {e}")
                self._queue.task_done()

        asyncio.create_task(_worker())
        return await future
