# 임베딩 모델 선정 설계

## SUMMARY

- 두 KV-cache 기술 논문을 대상으로 BGE-M3, E5-instruct, Qwen3-Embedding-0.6B의 dense 검색 성능과 실행 비용을 비교했다.
- 모델 선택에는 test를 사용하지 않고 `dev Recall@5 → dev AllEvidence@5 → dev MRR@10 → 질의 p50 지연` 순서를 적용했다.
- BGE-M3와 Qwen은 dev Recall@5 75.0%, AllEvidence@5 72.2%로 같았으나, BGE-M3가 MRR@10과 질의 지연에서 앞서 현재 모델로 잠정 선정됐다.
- Qwen은 dev Recall@10에서 88.9%로 BGE-M3의 86.1%보다 높았다. 향후 더 넓은 후보 회수가 중요하거나 긴 청크를 사용하는 성능 개선 단계에서는 동일 조건으로 재평가한 뒤 Qwen 교체를 고려한다.
- 이 결론은 한국어 질의와 영어 논문 2개를 사용한 소규모 지정 근거 회수 실험에 한정되며, 생성 답변 정확도나 보편적인 모델 우위를 뜻하지 않는다.

## 1. 선정 목적

기술 조사 에이전트와 도메인 평가 에이전트가 SW·HW 논문에서 질문에 필요한 근거를 검색할 수 있도록 오픈소스 임베딩 모델을 선정한다. 공개 리더보드 순위만 사용하지 않고, 실제 과제 문서와 질의 조건에서 검색 품질과 실행 비용을 비교한다.

## 2. 적용 환경

| 구분 | 내용 |
|---|---|
| SW 문서 | `sw_deepseek_v2.pdf` — DeepSeek-V2의 MLA 기반 KV cache 축소 기술 |
| HW 문서 | `hw_pim_cxl.pdf` — CXL-PNM 기반 KV cache·attention 처리 기술 |
| 원본 규모 | PDF 2개, 총 65쪽 |
| 검색 corpus | 목차·참고문헌 등을 제외한 35쪽, 123개 청크 |
| 문서 언어 | 영어 |
| 질의 언어 | 한국어 |
| 적용 에이전트 | 기술 조사 에이전트, 도메인 평가 에이전트 |

두 논문은 기술 용어, 약어, 실험 조건과 표·그림 설명을 포함한다. 이에 따라 한국어 질의와 영어 문서 간 교차 언어 검색 품질, 복수 근거 회수 능력, 로컬 실행 비용을 주요 선정 기준으로 삼았다.

## 3. 비교 모델

공개 가중치와 공개 라이선스를 제공하며 다국어 검색이 가능한 다음 세 모델을 비교했다.

| 모델 | 파라미터 | 임베딩 차원 | 최대 입력 길이 | 주요 특징 |
|---|---:|---:|---:|---|
| `BAAI/bge-m3` | 568M | 1,024 | 8,192 tokens | 다국어, Dense·Sparse·Multi-vector 지원 |
| `intfloat/multilingual-e5-large-instruct` | 560M | 1,024 | 514 tokens | 다국어 instruction 기반 retrieval |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1,024 | 32,768 tokens | 긴 입력과 다국어 retrieval 지원 |

이번 비교에서는 모델 고유의 부가 기능 차이가 결과를 왜곡하지 않도록 세 모델 모두 normalized dense 검색만 사용했다.

## 4. 평가 데이터 설계

- 한국어 질문 40개를 구성했다.
- 답변 가능한 질문은 dev 18개와 test 18개로 분리했다.
- 답변 불가 진단 질문 4개는 recall과 MRR 계산에서 제외했다.
- 원문에서 직접 확인한 지정 근거 41개를 정답으로 사용했다.
- dev/test는 질문과 evidence ID를 분리했지만 동일한 두 논문을 사용했다.
- 평가 질문과 청크는 첫 모델 검색 전에 동결했다.
- test 결과는 모델 선택 기준에 사용하지 않고 선택 이후의 분리 평가로만 사용했다.

## 5. 공정한 비교 조건

| 항목 | 통제 조건 |
|---|---|
| 청크 | 세 모델에 동일한 123개 청크 사용 |
| 청크 길이 | 모든 후보 tokenizer에서 제목 포함 400 tokens 이내 |
| Overlap | 0 |
| 임베딩 | 1,024차원, float32, L2 normalization |
| 검색 | Cosine similarity 기반 dense 전수 검색 |
| 장치 | Apple M5 Pro, MPS |
| 배치 | 문서 인덱싱 batch 8, 질의 측정 batch 1 |
| 제외 기능 | BM25, reranker, 질문 번역, LLM judge, parent 확장 |

BGE-M3는 질문 지시문을 사용하지 않았고 E5와 Qwen은 동일한 영어 task instruction을 질문에만 적용했다. 문서에는 논문명과 실제 소절명만 추가했으며 정답이나 평가 질문은 넣지 않았다.

## 6. 선정 지표

모델 선택 순서는 다음과 같이 정했다.

1. dev Recall@5
2. dev AllEvidence@5
3. dev MRR@10
4. 앞의 값이 모두 같으면 질의 p50 지연

Recall@5는 질문별 지정 근거 중 상위 5개에서 회수한 비율의 평균이다. AllEvidence@5는 질문에 필요한 지정 근거를 상위 5개에서 모두 찾은 비율이며, MRR@10은 첫 지정 근거가 나타난 순위를 평가한다.

세부 동률 해소 순서는 첫 BGE 실행 후 보고서 작성 과정에서 구체화했으므로 사전 등록된 규칙은 아니다. 다만 평가셋과 청크는 모든 모델 실행 전에 동결하여 모델별 입력 조건을 동일하게 유지했다.

## 7. 실험 결과

### 7.1 모델 선택용 dev 결과

| 모델 | Recall@1 | Recall@5 | Recall@10 | AllEvidence@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **bge-m3** | **55.6%** | **75.0%** | 86.1% | **72.2%** | **0.651** |
| e5-instruct | 27.8% | 58.3% | 69.4% | 55.6% | 0.414 |
| qwen3-0.6b | 44.4% | **75.0%** | **88.9%** | **72.2%** | 0.601 |

### 7.2 분리한 test 결과

| 모델 | Recall@1 | Recall@5 | Recall@10 | AllEvidence@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **bge-m3** | **47.2%** | **83.3%** | **94.4%** | **77.8%** | **0.681** |
| e5-instruct | 11.1% | 47.2% | 66.7% | 44.4% | 0.263 |
| qwen3-0.6b | 33.3% | 77.8% | 83.3% | **77.8%** | 0.492 |

### 7.3 실행 성능

| 모델 | 123청크 인덱싱 | 질의 p50 | 질의 p95 | 프로세스 RSS peak | MPS driver 관측 최대 |
|---|---:|---:|---:|---:|---:|
| **bge-m3** | 4.67s | **17.2ms** | **20.0ms** | **0.78GiB** | 2.99GiB |
| e5-instruct | **4.41s** | 19.6ms | 21.7ms | 2.92GiB | 3.02GiB |
| qwen3-0.6b | 9.35s | 28.0ms | 38.5ms | 2.92GiB | 3.67GiB |

실행시간은 모델 로딩·다운로드·워밍업과 PDF 파싱을 제외한 값이다. 인덱싱은 1회, 질의는 40문항을 각 3회 실행해 측정했으며 전용 벤치마크 환경은 아니다.

## 8. 최종 선정

본 설계의 임베딩 모델은 **`BAAI/bge-m3`로 잠정 선정**한다.

BGE-M3와 Qwen3-Embedding-0.6B는 dev Recall@5 75.0%, AllEvidence@5 72.2%로 같았다. 다음 선택 기준인 MRR@10에서 BGE-M3가 0.651로 Qwen의 0.601보다 높았다. 질의 p50 지연도 BGE-M3가 17.2ms로 Qwen의 28.0ms보다 짧았으며, 관측된 프로세스 RSS peak도 가장 낮았다. 분리한 test에서도 BGE-M3가 Recall@5 83.3%, MRR@10 0.681로 가장 높은 결과를 보였다.

따라서 현재 과제의 한국어 질의·영어 기술문서 dense 검색 조건에서는 BGE-M3가 검색 품질과 실행 비용의 균형이 가장 적절하다고 판단했다. 이는 지정 근거 회수 파일럿에 따른 프로젝트 내부 선택이며, 생성 답변의 정확도나 임베딩 모델의 보편적 우위를 의미하지 않는다.

다만 Qwen3-Embedding-0.6B는 dev Recall@10에서 88.9%로 BGE-M3의 86.1%보다 높았다. 또한 최대 입력 길이가 더 길지만, 이번 실험은 모든 모델에 동일한 400토큰 이하 청크를 사용했으므로 장문맥 지원의 이점은 검증하지 않았다. 향후 top-10 후보 회수율을 높여야 하거나 긴 청크를 직접 검색해야 하는 경우에는 Qwen을 동일한 운영 데이터로 다시 평가하고 교체 여부를 결정한다.

## 9. 적용 전략

초기 RAG 파이프라인에는 다음 구성을 적용한다.

```text
Embedding model: BAAI/bge-m3
Retrieval mode: dense
Embedding dimension: 1024
Normalization: L2
Similarity: cosine
Chunk size: 모델별 tokenizer 기준 제목 포함 400 tokens 이하
Chunk overlap: 0
```

복합 질문에서는 검색 결과가 한 논문에 편중되는 현상이 확인됐다. 이 문제는 임베딩 모델을 추가하는 대신 질문을 SW·HW 하위 질문으로 분해하고 문서별 검색을 수행하는 방식으로 보완한다. BGE-M3의 sparse·multi-vector 기능이나 reranker는 dense 기준선이 실제 파이프라인에서 부족할 때만 추가 검토한다.

Qwen 교체는 다음 조건 중 하나가 실제 운영 평가에서 확인될 때 검토한다.

- 후속 생성 단계가 `top_k=10` 후보를 활용하며 Recall@10 개선이 답변 품질 향상으로 이어지는 경우
- 400토큰보다 긴 청크가 필요해 BGE-M3 대비 Qwen의 장문맥 지원 이점을 다시 비교할 필요가 있는 경우
- 동일 장치·동일 데이터 재평가에서 검색 품질 향상이 추가 지연과 메모리 비용보다 큰 경우

## 10. 한계

- 정답은 원문에서 지정한 quote이며 다른 페이지의 동등한 근거가 누락됐을 수 있다.
- 수식 복원, 표·그래프 전체 해석, 생성 답변 정확도와 인용 타당성은 평가하지 않았다.
- 두 논문의 한국어 질문만 사용했으므로 다른 문서와 도메인에 일반화할 수 없다.
- 동일한 400토큰 청크를 사용해 장문맥 지원 모델의 최대 입력 능력을 비교하지 않았다.
- 답변 불가 질문에서 dense 검색은 항상 결과를 반환하므로 별도의 abstention 기준이 필요하다.
- 소규모 파일럿이므로 모델 간 차이의 통계적 유의성을 주장하지 않는다.

## 참고자료

- BAAI. *BAAI/bge-m3 Model Card*. Hugging Face, https://huggingface.co/BAAI/bge-m3
- intfloat. *intfloat/multilingual-e5-large-instruct Model Card*. Hugging Face, https://huggingface.co/intfloat/multilingual-e5-large-instruct
- Qwen Team. *Qwen/Qwen3-Embedding-0.6B Model Card*. Hugging Face, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
