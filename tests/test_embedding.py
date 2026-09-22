import hashlib
import json
import re
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
from ingest.embedding.chunking import chunk_document
from ingest.embedding.documents import Document, Page, load_documents, read_pages
from ingest.embedding.storage import file_hash, load_bundle
from service.retrieval import paper_index
from service.retrieval.paper_index import PaperIndex, get_paper_index


class TokenizerFixture:
    name_or_path = "test-characters"
    init_kwargs = {"_commit_hash": "tokenizer-fixture"}

    def encode(self, text, add_special_tokens=True):
        return list(text) + (["<s>", "</s>"] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def get_vocab(self):
        return {"test-characters": 0}


class EncoderFixture:
    name = "BAAI/bge-m3"
    revision = "a" * 40
    dimension = 12
    max_tokens = 8192

    def __init__(self):
        self.tokenizer = TokenizerFixture()
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
        self.input = self.root / "processed"
        self.input.mkdir()
        records = []
        for group in ("sw", "hw", "common"):
            filename = f"{group}.json"
            (self.input / filename).write_text(json.dumps({"pages": [
                {"page": 1, "text": "# Results\n" + f"{group} KV cache mechanism and measurements. " * 5},
                {"page": 3, "text": f"{group} costs limitations and validation."},
            ]}), encoding="utf-8")
            records.append({"doc_id": f"paper_{group}", "path": filename, "title": f"{group} paper",
                            "technology": group, "url": f"https://example.org/{group}",
                            "original_pages": 3, "allowed": True})
        self.manifest = self.input / "documents.json"
        self.manifest.write_text(json.dumps(records), encoding="utf-8")
        self.cfg = Settings(_env_file=None, processed_manifest=self.manifest,
                            faiss_index_dir=self.root / "faiss", chunk_size=100,
                            chunk_tokenizer_models=[], embedding_dimension=None)
        self.encoder = EncoderFixture()

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **overrides):
        config = self.cfg.model_copy(update=overrides)
        return build_index(config, encoder=self.encoder, comparison_tokenizers=[])

    def test_build_reload_and_numpy_parity_with_common(self):
        manifest, reused = self.build()
        self.assertFalse(reused)
        self.assertEqual(manifest["embedding"]["dimension"], 12)
        query = "KV cache limitations"
        retriever = PaperIndex(self.cfg)
        query_encoder = EncoderFixture()
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=query_encoder) as factory:
            results = retriever.search(query, "sw", 5)
            retriever.search(query, "hw", 5)
            factory.assert_called_once_with("BAAI/bge-m3", "a" * 40, self.cfg.embedding_device)
        self.assertTrue(all(call == [query] for call in query_encoder.calls))
        allowed = [chunk for chunk in retriever.chunks if chunk["technology"] in {"sw", "common"}]
        vectors = EncoderFixture().encode([chunk["embedding_text"] for chunk in allowed])
        query_vector = EncoderFixture().encode([query])[0]
        expected = sorted(zip(allowed, vectors @ query_vector), key=lambda item: (-float(item[1]), item[0]["id"]))[:5]
        self.assertEqual([row["id"] for row in results], [row["id"] for row, _ in expected])
        np.testing.assert_allclose([row["score"] for row in results], [score for _, score in expected], atol=1e-6)
        self.assertTrue(all(row["technology"] != "hw" for row in results))
        self.assertEqual(len(results), 5)
        self.assertTrue(all(row["pdf_pages"] == [row["page"]] for row in results))
        self.assertTrue(all({"id", "title", "url", "page", "text", "score"} <= row.keys() for row in results))
        # 호출자가 결과를 수정해도 내부 메타데이터는 유지한다.
        results[0]["pdf_pages"].append(999)
        self.assertNotIn(999, retriever.search(query, "sw")[0]["pdf_pages"])

    def test_team_copy_needs_no_processed_documents(self):
        self.build()
        target = self.root / "team" / "faiss"
        shutil.copytree(self.cfg.faiss_index_dir, target)
        self.manifest.unlink()
        cfg = self.cfg.model_copy(update={"faiss_index_dir": target})
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=EncoderFixture()):
            index = PaperIndex(cfg)
            self.assertTrue(index.search("query", "sw"))

    def test_reuse_and_input_or_settings_change(self):
        self.build()
        calls = len(self.encoder.calls)
        _, reused = self.build()
        self.assertTrue(reused)
        self.assertEqual(len(self.encoder.calls), calls)
        _, reused = self.build(chunk_size=90)
        self.assertFalse(reused)
        data = json.loads((self.input / "sw.json").read_text())
        data["pages"][0]["text"] += " New finding."
        (self.input / "sw.json").write_text(json.dumps(data))
        _, reused = self.build(chunk_size=90)
        self.assertFalse(reused)

    def test_corrupt_bundle_rejected(self):
        self.build()
        path = self.cfg.faiss_index_dir / "chunks.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "chunks.json"):
            PaperIndex(self.cfg)

    def test_mapping_and_index_integrity(self):
        self.build()
        path = self.cfg.faiss_index_dir / "manifest.json"
        data = json.loads(path.read_text())
        data["indexes"]["sw"]["chunk_ids"].reverse()
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "행과 청크"):
            load_bundle(self.cfg.faiss_index_dir)
        data["indexes"]["sw"]["chunk_ids"].reverse()
        path.write_text(json.dumps(data))
        sw_index = self.cfg.faiss_index_dir / "sw.faiss"
        sw_index.write_bytes(sw_index.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "FAISS 파일"):
            load_bundle(self.cfg.faiss_index_dir)

    def test_dimension_and_revision_mismatch(self):
        with self.assertRaisesRegex(ValueError, "출력 차원"):
            self.build(embedding_dimension=1024)
        self.build()
        with self.assertRaisesRegex(ValueError, "인덱스 차원"):
            PaperIndex(self.cfg.model_copy(update={"embedding_dimension": 1024}))
        with self.assertRaisesRegex(ValueError, "revision"):
            PaperIndex(self.cfg.model_copy(update={"embedding_revision": "b" * 40}))
        wrong = EncoderFixture()
        wrong.revision = "b" * 40
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "질의 모델"):
                PaperIndex(self.cfg).search("query", "sw")

    def test_lazy_model_and_parallel_query_loading(self):
        self.build()
        with patch("service.retrieval.paper_index.DenseEncoder", return_value=EncoderFixture()) as factory:
            retriever = PaperIndex(self.cfg)
            factory.assert_not_called()
            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(lambda side: retriever.search("query", side, 999), ("sw", "hw")))
            factory.assert_called_once()
            self.assertTrue(all(row for row in rows))
            with self.assertRaises(ValueError):
                retriever.search("", "sw")
            with self.assertRaises(ValueError):
                retriever.search("query", "bad")
            with self.assertRaises(ValueError):
                retriever.search("query", "sw", 0)

    def test_invalid_vectors_do_not_replace_existing_bundle(self):
        self.build()
        before = file_hash(self.cfg.faiss_index_dir / "manifest.json")
        self.encoder.encode = lambda texts, batch_size=8: np.zeros((len(texts), 12), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "정규화"):
            self.build(chunk_size=90)
        self.assertEqual(file_hash(self.cfg.faiss_index_dir / "manifest.json"), before)
        load_bundle(self.cfg.faiss_index_dir)

    def test_nodes_share_one_index_even_on_parallel_first_use(self):
        paper_index._cached_paper_index.cache_clear()
        shared = object()

        def create():
            time.sleep(0.02)  # 생성 중 다른 노드의 최초 호출이 들어오는 상황
            return shared

        try:
            with patch("service.retrieval.paper_index.PaperIndex", side_effect=create) as factory:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    indexes = list(pool.map(lambda _: get_paper_index(), range(4)))
                self.assertTrue(all(index is shared for index in indexes))
                self.assertIs(get_paper_index(), shared)
                factory.assert_called_once_with()
        finally:
            paper_index._cached_paper_index.cache_clear()

    def test_input_page_markers_and_json(self):
        path = self.input / "sample.md"
        path.write_text("<!-- PAGE: 2 -->\n# Heading\nParagraph.\n\n<!-- PAGE: 7 -->\nNext.")
        pages = read_pages(path)
        self.assertEqual([page.page for page in pages], [2, 7])
        path.write_text("본문은 있으나 페이지가 없습니다.")
        with self.assertRaisesRegex(ValueError, "페이지 마커"):
            read_pages(path)
        path.write_text("<!-- PAGE: 2 -->\nFirst\n<!-- PAGE: 2 -->\nSecond")
        with self.assertRaisesRegex(ValueError, "중복 없이"):
            read_pages(path)

    def test_document_scope_validation_before_model_loading(self):
        records = json.loads(self.manifest.read_text())
        records[0]["allowed"] = False
        self.manifest.write_text(json.dumps(records))
        with patch("ingest.embedding.build_index.DenseEncoder") as factory:
            with self.assertRaisesRegex(ValueError, "허용 여부"):
                build_index(self.cfg)
            factory.assert_not_called()
        records[0]["allowed"] = True
        records[0]["original_pages"] = 201
        self.manifest.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "200쪽"):
            load_documents(self.manifest)

    def test_chunk_boundaries_token_limit_and_overlap(self):
        doc = Document(doc_id="test", path="test.md", title="Title", technology="sw", original_pages=2, allowed=True)
        source = "First paragraph with words.\n\nSecond paragraph is quite long. " * 6
        pages = [Page(page=1, text="# Heading\n" + source), Page(page=2, text="마지막 페이지 문단입니다.")]
        tokenizer = TokenizerFixture()
        chunks = chunk_document(doc, pages, [tokenizer], size=80, overlap=0, include_section=True)
        self.assertTrue(all(len(tokenizer.encode(c["embedding_text"])) <= 80 for c in chunks))
        combined = " ".join(c["text"] for c in chunks if c["page"] == 1)
        self.assertEqual(re.sub(r"\s+", "", combined), re.sub(r"\s+", "", source))
        self.assertEqual(chunks[-1]["pdf_pages"], [2])
        overlap = chunk_document(doc, pages[:1], [tokenizer], size=80, overlap=5, include_section=True)
        self.assertTrue(all(len(tokenizer.encode(c["embedding_text"])) <= 80 for c in overlap))
        self.assertTrue(overlap[1]["text"].startswith(overlap[0]["text"][-5:].lstrip()))

    def test_comparison_tokenizer_limit_and_heading_overhead(self):
        class LargerTokenizer(TokenizerFixture):
            def encode(self, text, add_special_tokens=True):
                return super().encode(text, add_special_tokens) * 2

        doc = Document(doc_id="test", path="test.txt", title="Title", technology="hw", original_pages=1, allowed=True)
        tokens = [TokenizerFixture(), LargerTokenizer()]
        chunks = chunk_document(doc, [Page(page=1, text="paragraph " * 50)], tokens,
                                size=80, overlap=0, include_section=True)
        self.assertTrue(all(len(t.encode(c["embedding_text"])) <= 80 for c in chunks for t in tokens))
        with self.assertRaisesRegex(ValueError, "제목과 소절"):
            chunk_document(doc, [Page(page=1, text="text")], tokens, size=5, overlap=0, include_section=True)

    def test_settings_paths_are_project_relative(self):
        cfg = Settings(_env_file=None, faiss_index_dir="artifacts/faiss", processed_manifest="data/processed/documents.json")
        self.assertEqual(cfg.faiss_index_dir, ROOT / "artifacts/faiss")
        self.assertEqual(cfg.processed_manifest, ROOT / "data/processed/documents.json")

if __name__ == "__main__":
    unittest.main()
