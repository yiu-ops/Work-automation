"""
교무지원과 업무 자동화 - LLM 파싱 모듈

processor.py 가 생성한 JSON 결과를 읽어
Gemini API 에 전달한 뒤 구조화된 업무 데이터로 변환한다.

최종 산출물: data/output.json
"""

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================
# 경로 설정
# =============================================
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "data" / "output"
FINAL_FILE  = BASE_DIR / "data" / "output.json"

# =============================================
# Gemini API 초기화
# =============================================
# 사용할 모델 (무료 티어 지원)
MODEL_ID = "gemini-2.5-flash-lite"

# Gemini API 호출 설정 (JSON Mode 강제)
_GENERATE_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.1,
)


def _init_client() -> genai.Client:
    """환경 변수에서 API 키를 로드하고 Gemini 클라이언트를 초기화한다."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            ".env 파일에 GEMINI_API_KEY 가 설정되지 않았습니다.\n"
            ".env.example 을 복사해 .env 를 만들고 키를 입력하세요."
        )
    client = genai.Client(api_key=api_key)
    logger.info("Gemini 클라이언트 초기화 완료 (%s)", MODEL_ID)
    return client


# =============================================
# 프롬프트 템플릿
# =============================================
SYSTEM_PROMPT = """\
당신은 대학 교무지원과의 행정 문서 분석 전문가입니다.
아래 문서 텍스트를 읽고, 다음 JSON 스키마에 정확히 맞춰 정보를 추출하세요.

### 출력 JSON 스키마
{
  "task_name":      "업무명 (문서 제목 기반, 없으면 빈 문자열)",
  "description":    "업무 개요 (2~4문장으로 요약)",
  "precautions":    ["주의사항 1", "주의사항 2", ...],
  "timeline":       [
    {"date": "날짜 또는 기간 (예: 2025-03-01)", "action": "해야 할 일"}
  ],
  "related_depts":  ["협조 부서 1", "협조 부서 2", ...],
  "deliverables":   ["산출물 1", "산출물 2", ...]
}

### 지침
- 반드시 위 스키마만 출력하고, 설명·마크다운 코드블록·주석은 절대 포함하지 마세요.
- 문서에 해당 항목이 없으면 빈 배열([]) 또는 빈 문자열("")을 사용하세요.
- 날짜가 명시되지 않은 경우 timeline 은 빈 배열로 반환하세요.
- 한국어로 작성하세요.

### 문서 텍스트
"""


# =============================================
# 단일 문서 파싱
# =============================================
def parse_document(client: genai.Client, text: str, filename: str) -> dict:
    """텍스트를 Gemini 에 보내 구조화된 딕셔너리로 반환한다.

    API 오류 발생 시 에러 정보를 담은 딕셔너리를 반환하고 프로그램을 계속한다.
    """
    if not text or text.strip() in ("(텍스트 없음)", ""):
        logger.warning("텍스트 없음, 건너뜀: %s", filename)
        return {
            "task_name": filename,
            "description": "",
            "precautions": [],
            "timeline": [],
            "related_depts": [],
            "deliverables": [],
            "_parse_status": "skipped_empty",
        }

    # 텍스트가 너무 길면 앞 6000자만 전달 (Gemini 무료 티어 입력 제한 고려)
    MAX_CHARS = 6000
    trimmed = text[:MAX_CHARS]
    if len(text) > MAX_CHARS:
        trimmed += "\n... (이하 생략)"

    prompt = SYSTEM_PROMPT + trimmed

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=_GENERATE_CONFIG,
        )
        raw_json = response.text.strip()

        # 혹시 마크다운 코드블록이 붙는 경우 제거
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
            raw_json = raw_json.strip()

        parsed = json.loads(raw_json)
        parsed["_parse_status"] = "ok"
        logger.info("파싱 완료: %s", filename)
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("JSON 파싱 실패 (%s): %s", filename, exc)
        logger.debug("원본 응답:\n%s", getattr(response, 'text', 'N/A'))
        return {
            "task_name": filename,
            "_parse_status": "error_json",
            "_error": str(exc),
            "_raw_response": getattr(response, 'text', ""),
        }
    except Exception as exc:
        logger.error("API 오류 (%s): %s", filename, exc)
        return {
            "task_name": filename,
            "_parse_status": "error_api",
            "_error": str(exc),
        }


# =============================================
# 배치 처리 — processor.py 결과 JSON 읽기
# =============================================
def load_extracted_records(output_dir: Path) -> list[dict]:
    """output_dir 에서 가장 최근 result_*.json 을 읽어 반환한다."""
    result_files = sorted(output_dir.glob("result_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"{output_dir} 에서 result_*.json 파일을 찾을 수 없습니다.\n"
            "먼저 processor.py 를 실행하세요."
        )
    latest = result_files[-1]
    logger.info("입력 파일: %s", latest)
    return json.loads(latest.read_text(encoding="utf-8"))


# =============================================
# 결과 저장
# =============================================
def save_output(results: list[dict], output_path: Path) -> None:
    """파싱된 결과 목록을 output.json 으로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("최종 결과 저장 완료 → %s", output_path)


# =============================================
# 진입점
# =============================================
if __name__ == "__main__":
    # 1. Gemini 클라이언트 초기화
    client = _init_client()

    # 2. processor.py 가 생성한 레코드 로드
    records = load_extracted_records(OUTPUT_DIR)
    logger.info("총 %d개 레코드 로드", len(records))

    parsed_results: list[dict] = []
    ok_count = skipped_count = error_count = 0

    for i, record in enumerate(records, 1):
        filename = record.get("filename", f"record_{i}")
        source   = record.get("source_zip") or "직접 파일"
        text     = record.get("text", "")
        ext      = record.get("extension", "")

        logger.info("[%d/%d] %s ← %s", i, len(records), filename, source)

        result = parse_document(client, text, filename)

        # 원본 메타데이터를 함께 보존
        result["_meta"] = {
            "source_zip":    record.get("source_zip"),
            "folder_in_zip": record.get("folder_in_zip"),
            "filename":      filename,
            "extension":     ext,
        }
        parsed_results.append(result)

        # 파싱 상태 집계
        status = result.get("_parse_status", "")
        if status == "ok":
            ok_count += 1
        elif "skipped" in status:
            skipped_count += 1
        else:
            error_count += 1

        # API 레이트 리밋 방지 (무료 티어: 15 req/min)
        if i < len(records):
            time.sleep(2)

    # 3. 최종 저장
    save_output(parsed_results, FINAL_FILE)

    # 4. 요약
    print("\n===== LLM 파싱 완료 =====")
    print(f"  성공    : {ok_count}개")
    print(f"  건너뜀  : {skipped_count}개 (텍스트 없음)")
    print(f"  오류    : {error_count}개")
    print(f"  결과 파일: {FINAL_FILE}")
