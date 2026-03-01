"""
supabase_uploader.py
────────────────────
rag_analyzer.py 에서 추출한 업무 인사이트 딕셔너리를 받아
Supabase gyomu_tasks 테이블에 upsert 합니다.

- task_name 이 이미 존재하면 → UPDATE (덮어쓰기)
- task_name 이 없으면        → INSERT (새 레코드)
- 에러 발생 시 프로그램 미종료 + error.log 기록
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import create_client, Client

# ── 환경 변수 로드 ──────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        ".env 파일에 SUPABASE_URL 과 SUPABASE_KEY 가 설정되어 있지 않습니다."
    )

# ── Supabase 클라이언트 초기화 ───────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 에러 로그 설정 ────────────────────────────────────────────────
_error_log = Path(__file__).parent / "error.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _to_json_safe(value: Any) -> Any:
    """list / dict 는 그대로, 문자열 JSON 은 파싱해서 반환."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def upload_to_supabase(task_data: dict[str, Any]) -> bool:
    """
    task_data 딕셔너리를 gyomu_tasks 테이블에 upsert 합니다.

    Parameters
    ----------
    task_data : dict
        task_name        (str)  – 필수
        target_date      (str | None)  – YYYY-MM-DD 형식
        core_regulations (list[str])
        action_triggers  (list | dict)
        lessons_learned  (str | None)

    Returns
    -------
    bool : 성공 여부
    """
    task_name: str = task_data.get("task_name", "").strip()
    if not task_name:
        logger.warning("task_name 이 비어 있어 업로드를 건너뜁니다.")
        return False

    record: dict[str, Any] = {
        "task_name":        task_name,
        "target_date":      task_data.get("target_date"),
        "core_regulations": _to_json_safe(task_data.get("core_regulations", [])),
        "action_triggers":  _to_json_safe(task_data.get("action_triggers", [])),
        "lessons_learned":  task_data.get("lessons_learned"),
    }

    try:
        (
            supabase
            .table("gyomu_tasks")
            .upsert(record, on_conflict="task_name")
            .execute()
        )
        print(f"✅ [{task_name}] 데이터가 Supabase에 성공적으로 적재/업데이트 되었습니다.")
        return True

    except Exception:  # noqa: BLE001
        err_msg = traceback.format_exc()
        logger.error("업로드 실패: %s\n%s", task_name, err_msg)
        with _error_log.open("a", encoding="utf-8") as f:
            f.write(f"[{task_name}]\n{err_msg}\n{'─'*60}\n")
        return False


# ── 직접 실행 시 테스트 업로드 ───────────────────────────────────
if __name__ == "__main__":
    dummy: dict[str, Any] = {
        "task_name": "테스트_업무_supabase_uploader",
        "target_date": str(date.today()),
        "core_regulations": [
            "교원인사규정 제15조 (겸직허가)",
            "위임전결규정 제3조 (전결 범위)",
        ],
        "action_triggers": [
            {
                "trigger": "겸직허가 신청서 수령",
                "action":  "허가 여부 검토 후 총장 결재",
            },
            {
                "trigger": "겸직현황 실태조사 공문 발송",
                "action":  "각 단과대학 회신 취합 및 결과 보고",
            },
        ],
        "lessons_learned": (
            "겸직허가 기준 명확화가 필요하며, "
            "매 학기 초 일괄 점검 체계를 구축하면 업무 효율이 향상됩니다."
        ),
    }

    print("── 테스트 업로드 시작 ──")
    success = upload_to_supabase(dummy)
    print("── 테스트 업로드 완료 ──" if success else "── 테스트 업로드 실패 ──")
