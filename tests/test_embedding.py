import hashlib
import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config.config import ROOT, Settings
from ingest.embedding.build_index import build_index
from ingest.embedding.documents import load_preprocessed_chunks
from ingest.embedding.storage import file_hash, load_bundle
from service.retrieval import paper_index
from service.retrieval.paper_index import PaperIndex, get_paper_index


class EncoderFixture:
    name = "BAAI/bge-m3"
    revision = "a" * 40
    dimension = 12
    max_tokens = 8192

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=8):
        self.calls.append(list(texts))
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.frombuffer(digest[:self.dimension], dtype=np.uint8).astype(np.float32) + 1
            values.append(vector / np.linalg.norm(vector))
        return np.stack(values)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.chunks_path = self.root / "chunks.jsonl"
        self.manifest_path = self.root / "manifest.json"
        documents = []
        chunks = []
        for doc_id, technology, role in (("mla", "MLA", "primary"), ("cxl", "CXL-PNM", "primary"),
                                         ("baseline", "COMMON", "serving_baseline")):
            documents.append({"doc_id": doc_id, "file_name": f"{doc_id}.pdf", "technology": technology,
                              "role": role, "title": f"{doc_id} title", "url": f"https://example.org/{doc_id}",
                              "published_at": "2025-01-01", "allowed": True, "page_count": 3,
                              "sha256": "a" * 64})
            chunks.append({"chunk_id": f"{doc_id}_chunk_0001", "doc_id": doc_id,
                           "document_title": f"{doc_id} title", "technology": technology, "role": role,
                           "section": "Results", "section_level": 2, "section_path": ["Results"],
                           "pdf_pages": [1], "text": f"{doc_id} exact preprocessed chunk", "token_count": 8,
                           "needs_review": False, "review_reason": None})
        review = dict(chunks[0], chunk_id="mla_chunk_0002", text="mla review chunk", needs_review=True,
                      review_reason="표 추출 확인 필요")
        chunks.append(review)
        self.manifest_path.write_text(json.dumps(documents), encoding="utf-8")
        self.chunks_path.write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8")
        self.cfg = Settings(_env_file=None, preprocessed_chunks=self.chunks_path,
                            document_manifest=self.manifest_path, faiss_index_dir=self.root / "faiss",
                            embedding_dimension=None)
        self.encoder = EncoderFixture()

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **overrides):
        return build_index(self.cfg.model_copy(update=overrides), encoder=self.encoder)

    def test_build_preserves_preprocessed_chunks_and_excludes_review_by_default(self):
        manifest, reused = self.build()
        self.assertFalse(reused)
        self.assertEqual((manifest["source_chunk_count"], manifest["indexed_chunk_count"]), (4, 3))
        self.assertEqual(manifest["excluded_review_chunk_ids"], ["mla_chunk_0002"])
        # 전달받은 JSONL text가 한 번만 임베딩되며 제목 추가나 재청킹이 일어나지 않는다.
        self.assertEqual(self.encoder.calls, [["mla exact preprocessed chunk", "cxl exact preprocessed chunk",
                                               "baseline exact preprocessed chunk"]])
        _, stored_chunks, _ = load_bundle(self.cfg.faiss_index_dir)
        review = next(chunk for chunk in stored_chunks if chunk["id"] == "mla_chunk_0002")
        self.assertEqual(review["page"], 1)
        self.assertEqual(review["index_group"], "sw")
        self.assertEqual(review["review_reason"], "표 추출 확인 필요")

        retriever = PaperIndex(self.cfg)
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=EncoderFixture()):
            results = retriever.search("preprocessed query", "sw", 5)
        self.assertNotIn("mla_chunk_0002", {row["id"] for row in results})
        self.assertTrue(all(row["index_group"] in {"sw", "common"} for row in results))
        self.assertTrue(all({"id", "title", "url", "page", "pdf_pages", "section_path", "role", "score"} <= row.keys()
                            for row in results))

    def test_include_review_chunks_is_explicit_opt_in(self):
        manifest, _ = self.build(include_review_chunks=True)
        self.assertEqual(manifest["indexed_chunk_count"], 4)
        self.assertEqual(manifest["excluded_review_chunk_ids"], [])

    def test_bundle_can_be_shared_without_source_jsonl(self):
        self.build()
        target = self.root / "team" / "faiss"
        shutil.copytree(self.cfg.faiss_index_dir, target)
        self.chunks_path.unlink()
        self.manifest_path.unlink()
        cfg = self.cfg.model_copy(update={"faiss_index_dir": target})
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=EncoderFixture()):
            self.assertTrue(PaperIndex(cfg).search("query", "hw"))

    def test_reuse_detects_source_change(self):
        self.build()
        calls = len(self.encoder.calls)
        _, reused = self.build()
        self.assertTrue(reused)
        self.assertEqual(len(self.encoder.calls), calls)
        data = self.chunks_path.read_text(encoding="utf-8").replace("exact preprocessed chunk", "updated chunk", 1)
        self.chunks_path.write_text(data, encoding="utf-8")
        _, reused = self.build()
        self.assertFalse(reused)

    def test_invalid_review_metadata_and_mapping_are_rejected(self):
        original = self.chunks_path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in original.splitlines()]
        records[-1]["review_reason"] = None
        self.chunks_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "review_reason"):
            load_preprocessed_chunks(self.chunks_path, self.manifest_path)
        self.chunks_path.write_text(original, encoding="utf-8")
        self.build()
        manifest_path = self.cfg.faiss_index_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["indexes"]["sw"]["chunk_ids"] = ["unexpected_chunk"]
        manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "행과 청크"):
            load_bundle(self.cfg.faiss_index_dir)

    def test_corrupt_bundle_is_rejected(self):
        self.build()
        chunks_path = self.cfg.faiss_index_dir / "chunks.json"
        chunks_path.write_text(chunks_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "chunks.json"):
            PaperIndex(self.cfg)

    def test_model_dimension_mismatch_is_rejected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "인덱스 차원"):
            PaperIndex(self.cfg.model_copy(update={"embedding_dimension": 1024}))

    def test_invalid_vectors_do_not_replace_existing_bundle(self):
        self.build()
        before = file_hash(self.cfg.faiss_index_dir / "manifest.json")
        self.encoder.encode = lambda texts, batch_size=8: np.zeros((len(texts), 12), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "정규화"):
            self.build(include_review_chunks=True)
        self.assertEqual(file_hash(self.cfg.faiss_index_dir / "manifest.json"), before)

    def test_settings_paths_are_project_relative(self):
        cfg = Settings(_env_file=None, preprocessed_chunks="database/chunks.jsonl", document_manifest="database/manifest.json")
        self.assertEqual(cfg.preprocessed_chunks, ROOT / "database/chunks.jsonl")
        self.assertEqual(cfg.document_manifest, ROOT / "database/manifest.json")

    def test_nodes_share_one_index_even_on_parallel_first_use(self):
        paper_index._cached_paper_index.cache_clear()
        shared = object()

        def create():
            time.sleep(0.02)
            return shared

        try:
            with patch("service.retrieval.paper_index.PaperIndex", side_effect=create) as factory:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    indexes = list(pool.map(lambda _: get_paper_index(), range(4)))
                self.assertTrue(all(index is shared for index in indexes))
                factory.assert_called_once_with()
        finally:
            paper_index._cached_paper_index.cache_clear()


if __name__ == "__main__":
    unittest.main()
