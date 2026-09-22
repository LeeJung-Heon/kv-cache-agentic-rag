# Tavily 응답 fixture

현재 파일은 Tavily `/search` 응답 형식(`results[].url/title/content/score/published_date`)을 본떠
**직접 작성한 합성 응답**이다. 실제 녹화 응답이 아니다. 6단계 실제 API 테스트에서 녹화한 응답으로 교체하거나 추가한다.

| 파일 | 검증 목적 |
|---|---|
| `commercial_positive.json` | 정상 결과, 발행일 없음, 기준일 이후 자료, 부정 질의와 중복 URL |
| `commercial_negative.json` | `www.`·끝 슬래시만 다른 중복 URL, RFC 2822 발행일 |
| `weak.json` | score가 낮아 별칭 보강 검색이 필요한 결과 |
| `empty.json` | 결과 0건 |
| `malformed.json` | `results` 목록이 없는 응답 |
