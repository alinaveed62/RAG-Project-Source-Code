"""JSONL export for RAG-ready chunks."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from keats_scraper.models.chunk import Chunk
from keats_scraper.models.document import Document
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class JSONLExporter:
    """Exports documents and chunks to JSONL format."""

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_documents(
        self,
        documents: list[Document],
        filename: str = "documents.jsonl",
    ) -> Path:
        """
        Export documents to JSONL file.

        The output is the pre-chunking snapshot of the scraped corpus. It is
        consumed by the validate CLI subcommand
        (processors.ContentValidator) and by
        analyses/coverage_report.py for discovery-vs-processed
        reconciliation. It is not on the RAG retrieval path.

        Args:
            documents: List of documents to export
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for doc in documents:
                line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        logger.info(f"Exported {len(documents)} documents to {filepath}")
        return filepath

    def export_chunks(
        self,
        chunks: list[Chunk],
        filename: str = "handbook_chunks.jsonl",
    ) -> Path:
        """
        Export chunks to JSONL file.

        Full-metadata backup of chunks (includes heading_path,
        chunk_index, content_hash, etc. via ChunkMetadata).
        Consumed by analyses/coverage_report.py and
        notebooks/01_data_exploration.ipynb. The RAG pipeline itself
        reads the slimmer chunks_for_embedding.jsonl produced by
        export_embedding_format, not this file.

        Args:
            chunks: List of chunks to export
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        logger.info(f"Exported {len(chunks)} chunks to {filepath}")
        return filepath

    def export_embedding_format(
        self,
        chunks: list[Chunk],
        filename: str = "chunks_for_embedding.jsonl",
    ) -> Path:
        """
        Export chunks in format optimized for embedding.

        Args:
            chunks: List of chunks
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(chunk.to_embedding_format(), ensure_ascii=False)
                f.write(line + "\n")

        logger.info(f"Exported {len(chunks)} chunks for embedding to {filepath}")
        return filepath

    def create_index(
        self,
        chunks: list[Chunk],
        filename: str = "chunk_index.json",
    ) -> Path:
        """
        Create a quick lookup index for chunks.

        Provides section -> chunk-id and document-id -> chunk-id lookups so
        notebooks/01_data_exploration.ipynb can navigate the corpus
        without re-reading the full chunk payload. Not on the RAG
        retrieval path.

        Args:
            chunks: List of chunks
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        index = {
            "created_at": datetime.now(UTC).isoformat(),
            "total_chunks": len(chunks),
            "chunks_by_section": {},
            "chunks_by_document": {},
        }

        for chunk in chunks:
            # Index by section
            section = chunk.metadata.section or "Unknown"
            if section not in index["chunks_by_section"]:
                index["chunks_by_section"][section] = []
            index["chunks_by_section"][section].append(chunk.id)

            # Index by document
            doc_id = chunk.metadata.document_id
            if doc_id not in index["chunks_by_document"]:
                index["chunks_by_document"][doc_id] = {
                    "title": chunk.metadata.document_title,
                    "chunks": [],
                }
            index["chunks_by_document"][doc_id]["chunks"].append(chunk.id)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        logger.info(f"Created chunk index at {filepath}")
        return filepath

    @staticmethod
    def load_chunks(filepath: Path) -> Iterator[Chunk]:
        """
        Load chunks from JSONL file.

        Args:
            filepath: Path to JSONL file

        Yields:
            Chunk objects
        """
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield Chunk(**data)

    @staticmethod
    def load_documents(filepath: Path) -> Iterator[Document]:
        """
        Load documents from JSONL file.

        Args:
            filepath: Path to JSONL file

        Yields:
            Document objects
        """
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield Document(**data)
