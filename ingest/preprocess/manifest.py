import hashlib
import json
from pathlib import Path

import pymupdf


# 프로젝트 경로
ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs"
DATABASE_DIR = ROOT_DIR / "database"
MANIFEST_PATH = DATABASE_DIR / "manifest.json"


# RAG에 사용하는 문서 목록
DOCUMENTS = [
    {
        "doc_id": "sw_deepseek_v2",
        "file_name": "sw_deepseek_v2.pdf",
        "technology": "MLA",
        "role": "primary",
        "title": (
            "DeepSeek-V2: A Strong, Economical, and Efficient "
            "Mixture-of-Experts Language Model"
        ),
        "allowed": True,
    },
    {
        "doc_id": "sw_mha2mla",
        "file_name": "2025.acl-long.1597.pdf",
        "technology": "MLA",
        "role": "adoption",
        "title": (
            "Towards Economical Inference: Enabling DeepSeek’s "
            "Multi-Head Latent Attention in Any Transformer-based LLMs"
        ),
        "allowed": True,
    },
    {
        "doc_id": "hw_pim_cxl",
        "file_name": "hw_pim_cxl.pdf",
        "technology": "CXL-PNM",
        "role": "primary",
        "title": (
            "Scalable Processing-Near-Memory for 1M-Token LLM Inference: "
            "CXL-Enabled KV-Cache Management Beyond GPU Limits"
        ),
        "allowed": True,
    },
    {
        "doc_id": "hw_pond",
        "file_name": "2023_Pond_asplos23_official_asplos_version.pdf",
        "technology": "CXL-PNM",
        "role": "domain",
        "title": "Pond: CXL-Based Memory Pooling Systems for Cloud Platforms",
        "allowed": True,
    },
    {
        "doc_id": "common_pagedattention",
        "file_name": "2309.06180v1.pdf",
        "technology": "COMMON",
        "role": "serving_baseline",
        "title": (
            "Efficient Memory Management for Large Language "
            "Model Serving with PagedAttention"
        ),
        "allowed": True,
    },
    {
        "doc_id": "common_splitwise",
        "file_name": "2311.18677v2.pdf",
        "technology": "COMMON",
        "role": "operation_baseline",
        "title": (
            "Splitwise: Efficient Generative LLM Inference "
            "Using Phase Splitting"
        ),
        "allowed": True,
    },
]

def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_page_count(file_path: Path) -> int:
    with pymupdf.open(file_path) as pdf:
        return len(pdf)


def build_manifest():
    manifest = []

    for document in DOCUMENTS:
        file_path = DOCS_DIR / document["file_name"]

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF 파일을 찾을 수 없습니다: {file_path}"
            )

        item = document.copy()

        item["page_count"] = get_page_count(file_path)
        item["sha256"] = calculate_sha256(file_path)

        manifest.append(item)

    return manifest


def save_manifest(manifest):
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    manifest = build_manifest()
    save_manifest(manifest)

    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"Documents: {len(manifest)}")

    for doc in manifest:
        print(
            f"- {doc['doc_id']}: "
            f"{doc['page_count']} pages"
        )