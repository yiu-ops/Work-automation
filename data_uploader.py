"""
교무 행정 업무 자동화 - 업무 인사이트 데이터 업로더

rag_analyzer.py 가 생성한 JSON 파일을 읽어,
아래 두 가지 방법 중 하나로 대시보드(Vercel)에 데이터를 전달합니다.

  방법 1 (Supabase)  : .env의 SUPABASE_URL + SUPABASE_KEY 를 사용하여
                       gyomu_tasks 테이블에 upsert
  방법 2 (REST API)  : .env의 DASHBOARD_API_URL 엔드포인트로 JSON POST

환경 변수 UPLOAD_METHOD 로 전환
  UPLOAD_METHOD=supabase   → 방법 1
  UPLOAD_METHOD=api        → 방법 2  (기본값)

────────────────────────────────────────────────────────────────
[방법 1] Supabase 테이블 스키마 (SQL)
────────────────────────────────────────────────────────────────
  CREATE TABLE gyomu_tasks (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    task_name        TEXT        UNIQUE NOT NULL,        -- upsert 기준 컬럼
    target_date      TEXT,
    core_regulations JSONB,      -- 배열 저장
    action_triggers  JSONB,      -- 배열 저장
    lessons_learned  TEXT,
    source_file      TEXT,       -- 원본 JSON 파일명 (추적용)
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
  );

  -- updated_at 자동 갱신 트리거 (선택)
  CREATE OR REPLACE FUNCTION set_updated_at()
  RETURNS TRIGGER LANGUAGE plpgsql AS $$
  BEGIN NEW.updated_at = now(); RETURN NEW; END; $$;

  CREATE TRIGGER trg_gyomu_tasks_updated_at
  BEFORE UPDATE ON gyomu_tasks
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

────────────────────────────────────────────────────────────────
[방법 2] Next.js / Vercel API 수신부 기본 형태
  (파일 위치: gyomu-dashboard/app/api/tasks/route.ts)
────────────────────────────────────────────────────────────────
  import { NextResponse } from "next/server";

  // POST /api/tasks  ← 단건 upsert
  export async function POST(request: Request) {
    const body = await request.json();
    // body: { task_name, target_date, core_regulations[], action_triggers[], lessons_learned }
    // 여기에 DB upsert 로직(예: Prisma, Supabase client) 추가
    return NextResponse.json({ success: true });
  }

  // POST /api/tasks/bulk  ← 다건 upsert
  export async function POST_bulk(request: Request) {
    const { tasks } = await request.json();
    // tasks: TaskRecord[]
    return NextResponse.json({ inserted: tasks.length });
  }
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ── 로깅 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ── 경로 / 환경 변수 ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = DATA_DIR / "analysis_result"
FAILED_LOG = BASE_DIR / "failed_uploads.log"

load_dotenv(BASE_DIR / ".env")

UPLOAD_METHOD      = os.getenv("UPLOAD_METHOD", "api").lower()  # "supabase" | "api"
SUPABASE_URL       = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY", "")
DASHBOARD_API_URL  = os.getenv(
    "DASHBOARD_API_URL",
    "https://gyomu-dashboard.vercel.app/api/insights",
)
API_SECRET         = os.getenv("DASHBOARD_API_SECRET", "")  # Bearer 토큰 (선택)


# ══════════════════════════════════════════════════════════════════
#  공통 유틸리티
# ══════════════════════════════════════════════════════════════════

def _log_failure(record: dict[str, Any], reason: str) -> None:
    """전송 실패한 레코드를 failed_uploads.log 에 append."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "record": record,
    }
    with FAILED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.warning("  → 실패 기록 저장: %s", FAILED_LOG.name)


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    """
    core_regulations / action_triggers / reference_documents 가
    문자열로 들어온 경우 리스트로 변환.
    lessons_learned 가 리스트인 경우 줄바꿈 문자열로 병합.
    document_count 가 없으면 0 으로 기본값 설정.
    """
    for key in ("core_regulations", "action_triggers", "reference_documents"):
        val = record.get(key)
        if isinstance(val, str):
            record[key] = [v.strip() for v in val.split(",") if v.strip()]
        elif val is None:
            record[key] = []

    ll = record.get("lessons_learned")
    if isinstance(ll, list):
        record["lessons_learned"] = "\n".join(ll)

    # 신규 텍스트 필드 기본값
    for key in ("compliance_check", "recurrence_pattern", "semester"):
        if record.get(key) is None:
            record[key] = ""

    # document_count 기본값
    if record.get("document_count") is None:
        record["document_count"] = 0

    return record


# ══════════════════════════════════════════════════════════════════
#  방법 1 : Supabase Uploader
# ══════════════════════════════════════════════════════════════════

class SupabaseUploader:
    """
    supabase-py 를 사용해 gyomu_tasks 테이블에 upsert 합니다.
    중복 방지 기준: task_name (UNIQUE 제약)
    """

    def __init__(self) -> None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise EnvironmentError(
                ".env 에 SUPABASE_URL 과 SUPABASE_KEY 를 설정해 주세요."
            )
        try:
            from supabase import create_client  # type: ignore
        except ImportError:
            raise ImportError(
                "supabase 패키지가 없습니다. 'pip install supabase' 를 실행하세요."
            )
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("[Supabase] 클라이언트 초기화 완료")

    def upload(self, record: dict[str, Any], source_file: str = "") -> bool:
        """
        단건 upsert.
        on_conflict="task_name" → 동일 task_name 이면 UPDATE, 없으면 INSERT.
        """
        payload = _normalize(record.copy())
        payload["source_file"] = source_file

        try:
            resp = (
                self.client.table("gyomu_tasks")
                .upsert(payload, on_conflict="task_name")
                .execute()
            )
            # supabase-py v2 는 resp.data 가 삽입/수정된 행 리스트
            if resp.data:
                task = resp.data[0].get("task_name", "")
                logger.info("  ✓ upsert 성공: %s", task)
                return True
            else:
                logger.warning("  ⚠ 응답 데이터 없음: %s", resp)
                _log_failure(payload, "supabase: empty response data")
                return False
        except Exception as exc:
            logger.error("  ✗ Supabase 오류: %s", exc)
            _log_failure(payload, f"supabase: {exc}")
            return False

    def upload_bulk(self, records: list[dict], source_file: str = "") -> tuple[int, int]:
        """다건 처리. (성공 수, 실패 수) 반환."""
        ok, fail = 0, 0
        for rec in records:
            (ok if self.upload(rec, source_file) else fail).__class__  # noqa
            if self.upload(rec, source_file):
                ok += 1
            else:
                fail += 1
        return ok, fail


# ══════════════════════════════════════════════════════════════════
#  방법 2 : REST API Uploader
# ══════════════════════════════════════════════════════════════════

class ApiUploader:
    """
    requests 를 사용해 DASHBOARD_API_URL 에 JSON POST 합니다.

    중복 방지 전략:
      1) GET {url}?task_name=<name> 로 존재 여부 확인
         - 200 + 본문에 task_name 포함 → PUT {url}/{id} 로 UPDATE
         - 404 또는 빈 목록 → POST 로 INSERT
      2) 서버가 upsert 엔드포인트를 제공한다면 POST 만으로 처리 가능
         (UPSERT_MODE=true 환경 변수로 전환)
    """

    TIMEOUT = 15  # seconds

    def __init__(self) -> None:
        self.base_url = DASHBOARD_API_URL.rstrip("/")
        self.upsert_mode = os.getenv("UPSERT_MODE", "true").lower() == "true"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        if API_SECRET:
            self.headers["Authorization"] = f"Bearer {API_SECRET}"
        logger.info("[API] 엔드포인트: %s (upsert_mode=%s)", self.base_url, self.upsert_mode)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────

    def _existing_id(self, task_name: str) -> str | None:
        """
        GET /api/tasks?task_name=<name> 로 기존 레코드 ID 조회.
        없으면 None 반환.
        """
        try:
            resp = requests.get(
                self.base_url,
                params={"task_name": task_name},
                headers=self.headers,
                timeout=self.TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                # [{"id": "...", "task_name": "..."}, ...] 또는 {"id": "...", ...}
                if isinstance(data, list) and data:
                    return data[0].get("id")
                if isinstance(data, dict) and data.get("task_name") == task_name:
                    return data.get("id")
            return None
        except Exception as exc:
            logger.debug("  기존 레코드 조회 실패 (무시): %s", exc)
            return None

    def _post(self, payload: dict) -> bool:
        try:
            resp = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("  ✓ POST 성공: %s (HTTP %s)", payload.get("task_name"), resp.status_code)
            return True
        except requests.HTTPError as exc:
            logger.error("  ✗ HTTP 오류: %s | 응답: %s", exc, exc.response.text[:200])
            _log_failure(payload, f"api POST HTTPError: {exc}")
        except requests.RequestException as exc:
            logger.error("  ✗ 네트워크 오류: %s", exc)
            _log_failure(payload, f"api POST RequestException: {exc}")
        return False

    def _put(self, record_id: str, payload: dict) -> bool:
        try:
            resp = requests.put(
                f"{self.base_url}/{record_id}",
                json=payload,
                headers=self.headers,
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("  ✓ PUT 성공: %s (HTTP %s)", payload.get("task_name"), resp.status_code)
            return True
        except requests.HTTPError as exc:
            logger.error("  ✗ HTTP 오류 (PUT): %s", exc)
            _log_failure(payload, f"api PUT HTTPError: {exc}")
        except requests.RequestException as exc:
            logger.error("  ✗ 네트워크 오류 (PUT): %s", exc)
            _log_failure(payload, f"api PUT RequestException: {exc}")
        return False

    # ── 공개 인터페이스 ───────────────────────────────────────────

    def upload(self, record: dict[str, Any], source_file: str = "") -> bool:
        payload = _normalize(record.copy())
        payload["source_file"] = source_file
        task_name = payload.get("task_name", "")

        if self.upsert_mode:
            # 서버가 upsert 를 처리하므로 POST 만 전송
            return self._post(payload)

        # upsert_mode=False : GET → POST or PUT
        existing_id = self._existing_id(task_name)
        if existing_id:
            logger.info("  → 기존 레코드 발견 (%s), 업데이트합니다.", task_name)
            return self._put(existing_id, payload)
        return self._post(payload)

    def upload_bulk(self, records: list[dict], source_file: str = "") -> tuple[int, int]:
        """
        /api/tasks/bulk 엔드포인트가 있다면 일괄 전송, 없으면 단건 반복.
        """
        bulk_url = self.base_url.rstrip("/tasks").rstrip("/") + "/tasks/bulk"
        payloads = [_normalize(r.copy()) | {"source_file": source_file} for r in records]

        try:
            resp = requests.post(
                bulk_url,
                json={"tasks": payloads},
                headers=self.headers,
                timeout=self.TIMEOUT,
            )
            if resp.status_code in (200, 201):
                logger.info("  ✓ 일괄 전송 성공: %d건", len(payloads))
                return len(payloads), 0
            logger.warning("  ⚠ bulk 엔드포인트 미지원 (HTTP %s), 단건 전환", resp.status_code)
        except requests.RequestException:
            logger.warning("  ⚠ bulk 엔드포인트 접속 실패, 단건 전환")

        # 단건 폴백
        ok, fail = 0, 0
        for rec, src_payload in zip(records, payloads):
            if self.upload(rec, source_file):
                ok += 1
            else:
                fail += 1
        return ok, fail


# ══════════════════════════════════════════════════════════════════
#  JSON 파일 로더
# ══════════════════════════════════════════════════════════════════

def load_result_files(result_dir: Path) -> list[tuple[str, list[dict]]]:
    """
    analysis_result/ 내 JSON 파일을 읽어
    [(파일명, [레코드, ...]), ...] 형태로 반환.

    지원 포맷:
      - 단일 객체   : { "task_name": "...", ... }
      - 객체 배열   : [{ ... }, { ... }]
      - tasks 래퍼  : { "tasks": [...] }
    """
    if not result_dir.exists():
        logger.error("결과 디렉터리가 없습니다: %s", result_dir)
        return []

    results: list[tuple[str, list[dict]]] = []
    json_files = sorted(result_dir.glob("*.json"))

    if not json_files:
        logger.warning("업로드할 JSON 파일이 없습니다: %s", result_dir)
        return []

    for jf in json_files:
        try:
            with jf.open(encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("파일 읽기 실패 [%s]: %s", jf.name, exc)
            _log_failure({"file": jf.name}, f"JSON parse error: {exc}")
            continue

        if isinstance(raw, dict):
            records = raw.get("tasks", [raw])
        elif isinstance(raw, list):
            records = raw
        else:
            logger.warning("알 수 없는 JSON 구조 [%s], 건너뜁니다.", jf.name)
            continue

        # task_name 없는 레코드 필터
        valid = [r for r in records if isinstance(r, dict) and r.get("task_name")]
        skipped = len(records) - len(valid)
        if skipped:
            logger.warning("  [%s] task_name 없는 레코드 %d건 제외", jf.name, skipped)

        if valid:
            results.append((jf.name, valid))

    return results


# ══════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 55)
    logger.info("업무 인사이트 데이터 업로더 시작")
    logger.info("업로드 방법: %s", UPLOAD_METHOD.upper())
    logger.info("=" * 55)

    # ── 업로더 초기화 ─────────────────────────────────────────────
    if UPLOAD_METHOD == "supabase":
        uploader: SupabaseUploader | ApiUploader = SupabaseUploader()
    else:
        uploader = ApiUploader()

    # ── JSON 파일 로드 ────────────────────────────────────────────
    file_records = load_result_files(RESULT_DIR)
    if not file_records:
        logger.info("처리할 파일이 없습니다. 종료합니다.")
        return

    total_ok = total_fail = 0
    for filename, records in file_records:
        logger.info("\n── 파일 처리 중: %s (%d건) ──", filename, len(records))
        ok, fail = uploader.upload_bulk(records, source_file=filename)
        total_ok += ok
        total_fail += fail

    # ── 결과 요약 ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("업로드 완료  ✓ 성공: %d건  ✗ 실패: %d건", total_ok, total_fail)
    if total_fail:
        logger.warning("실패 내역은 '%s' 를 확인하세요.", FAILED_LOG.name)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
