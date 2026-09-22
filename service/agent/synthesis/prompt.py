SYNTHESIS_SYSTEM_PROMPT = (
    "기술, 시장, 이해관계자, 도메인 평가를 종합하고 품질을 점검하세요. "
    "입력에 없는 사실이나 근거를 추가하지 말고 한국어로 작성하세요. "
    "평가 간 모순, 근거 부족, 과도한 일반화는 quality_feedback에 작성하세요. "
    "문제가 없으면 status를 complete로, 재작업이 필요하면 partial 또는 error로 설정하세요."
)
## 프롬프트만 과제에 맞게 추가하고