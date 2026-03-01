"""
main.py — 교무지원과 업무 자동화 파이프라인 원클릭 실행

  [Step 1] data/raw/*.zip  → 압축 해제 + 텍스트 추출  (processor.py)
  [Step 2] 추출 텍스트     → RAG 분석 → JSON          (rag_analyzer.py)
  [Step 3] JSON 인사이트   → Supabase upsert           (supabase_uploader.py)
  [Step 4] data/temp/ 임시 파일 정리                    (cleanup)

실행:
    python main.py
    python main.py --skip-extract   # extraction 건너뛰고 RAG 분석부터
    python main.py --rebuild-db     # 벡터 DB 강제 재생성
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── 환경 변수 로드 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ── 로깅 설정 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 경로 상수 ────────────────────────────────────────────────────
RAW_DIR       = BASE_DIR / "data" / "raw"
TEMP_DIR      = BASE_DIR / "data" / "temp"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
RESULT_DIR    = BASE_DIR / "data" / "analysis_result"
FAILED_LOG    = BASE_DIR / "failed_files.log"


# ══════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════

def _banner(msg: str, width: int = 60) -> None:
    """구분선으로 감싼 배너 출력."""
    line = "─" * width
    logger.info(line)
    logger.info("  %s", msg)
    logger.info(line)


def _log_failure(filename: str, reason: str) -> None:
    """실패 파일명과 사유를 failed_files.log 에 기록."""
    with FAILED_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{filename}\t{reason}\n")


# ══════════════════════════════════════════════════════════════════
# Step 1 : ZIP 압축 해제 + 텍스트 추출 → data/extracted/*.txt
# ══════════════════════════════════════════════════════════════════

def step_extract() -> list[Path]:
    """
    data/raw/ 의 ZIP 파일을 data/temp/ 에 풀고,
    각 파일의 텍스트를 추출하여 data/extracted/ 에 .txt 로 저장한다.

    Returns
    -------
    list[Path] : 저장된 .txt 파일 경로 목록
    """
    _banner("STEP 1 / 3  ▶  ZIP 압축 해제 & 텍스트 추출")

    from processor import extract_zips, process_files  # 지연 임포트

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # ZIP 이 없으면 경고 후 기존 extracted 파일 그대로 사용
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning("data/raw/ 에 ZIP 파일이 없습니다. 기존 extracted 파일로 진행합니다.")
        return sorted(EXTRACTED_DIR.glob("*.txt"))

    entries = extract_zips(RAW_DIR, TEMP_DIR)
    records = process_files(entries)

    saved_txts: list[Path] = []
    ok_count = skipped_count = error_count = 0

    for rec in records:
        if rec["status"] == "skipped":
            skipped_count += 1
            continue

        text: str = rec.get("text", "")
        if not text or text.startswith("[오류]"):
            logger.warning("  텍스트 추출 실패: %s", rec["filename"])
            _log_failure(rec["filename"], text or "빈 텍스트")
            error_count += 1
            continue

        # 원본 ZIP 명을 prefix 로 사용해 중복 파일명 충돌 방지
        stem = Path(rec["filename"]).stem
        prefix = Path(rec["source_zip"]).stem if rec["source_zip"] else "direct"
        txt_name = f"[{prefix}]_{stem}.txt"
        txt_path = EXTRACTED_DIR / txt_name

        try:
            txt_path.write_text(text, encoding="utf-8")
            saved_txts.append(txt_path)
            ok_count += 1
            logger.info("  ✓ 추출 저장: %s", txt_name)
        except Exception:
            logger.error("  파일 저장 실패: %s\n%s", txt_name, traceback.format_exc())
            _log_failure(rec["filename"], "txt 저장 실패")
            error_count += 1

    logger.info(
        "Step 1 완료 — 저장: %d건 / 건너뜀: %d건 / 오류: %d건",
        ok_count, skipped_count, error_count,
    )
    return saved_txts


# ══════════════════════════════════════════════════════════════════
# Step 2 : RAG 분석 → data/analysis_result/*.json
# ══════════════════════════════════════════════════════════════════

def step_analyze(txt_files: list[Path], rebuild_db: bool = False) -> list[dict]:
    """
    추출된 .txt 파일을 RAGSystem 으로 분석해 인사이트 딕셔너리 목록을 반환한다.

    Returns
    -------
    list[dict] : AnalysisResult.model_dump() 목록
    """
    _banner("STEP 2 / 3  ▶  RAG 분석 (규정 매핑 & 인사이트 추출)")

    if not txt_files:
        logger.warning("분석할 .txt 파일이 없습니다. STEP 2 를 건너뜁니다.")
        return []

    from rag_analyzer import RAGSystem  # 지연 임포트 (무거운 초기화)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rag = RAGSystem()
    rag.build_vector_db(force_rebuild=rebuild_db)

    results: list[dict] = []
    success_count = fail_count = 0

    logger.info("분석 대상: %d개 문서", len(txt_files))

    for txt_path in txt_files:
        try:
            result = rag.analyze_document(txt_path)
            if result is None:
                raise ValueError("analyze_document 가 None 을 반환했습니다.")

            data = result.model_dump()

            # JSON 파일로도 저장 (캐싱 목적)
            json_path = RESULT_DIR / f"{txt_path.stem}_analysis.json"
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            results.append(data)
            success_count += 1
            logger.info("  ✓ 분석 완료: %s → %s", txt_path.name, data.get("task_name", ""))

        except Exception:
            logger.error("  분석 실패: %s\n%s", txt_path.name, traceback.format_exc())
            _log_failure(txt_path.name, "RAG 분석 실패")
            fail_count += 1
            # ← 실패해도 다음 문서로 계속 진행

    logger.info(
        "Step 2 완료 — 성공: %d건 / 실패: %d건", success_count, fail_count
    )
    return results


# ══════════════════════════════════════════════════════════════════
# Step 3 : Supabase upsert
# ══════════════════════════════════════════════════════════════════

def step_upload(analysis_results: list[dict]) -> None:
    """분석 결과 딕셔너리 목록을 Supabase 에 upsert 한다."""
    _banner("STEP 3 / 3  ▶  Supabase 업로드")

    if not analysis_results:
        logger.warning("업로드할 분석 결과가 없습니다. STEP 3 를 건너뜁니다.")
        return

    from supabase_uploader import upload_to_supabase  # 지연 임포트

    success_count = fail_count = 0

    for data in analysis_results:
        task_name = data.get("task_name", "(이름 없음)")
        try:
            ok = upload_to_supabase(data)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                _log_failure(task_name, "upload_to_supabase 반환값 False")
        except Exception:
            logger.error("  업로드 예외: %s\n%s", task_name, traceback.format_exc())
            _log_failure(task_name, "업로드 예외")
            fail_count += 1
            # ← 실패해도 다음 레코드 계속 처리

    logger.info(
        "Step 3 완료 — 성공: %d건 / 실패: %d건", success_count, fail_count
    )


# ══════════════════════════════════════════════════════════════════
# 정리 : data/temp/ 임시 파일 삭제
# ══════════════════════════════════════════════════════════════════

def cleanup() -> None:
    """data/temp/ 안의 모든 파일·폴더를 삭제한다."""
    _banner("CLEANUP  ▶  임시 파일 정리")

    if not TEMP_DIR.exists():
        logger.info("  data/temp/ 폴더가 없습니다. (건너뜀)")
        return

    removed_files = removed_dirs = 0
    for item in TEMP_DIR.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                removed_files += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed_dirs += 1
        except Exception:
            logger.warning("  삭제 실패: %s\n%s", item, traceback.format_exc())

    logger.info(
        "정리 완료 — 파일 %d개, 폴더 %d개 삭제", removed_files, removed_dirs
    )


# ══════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="교무지원과 업무 자동화 파이프라인 원클릭 실행"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Step 1(ZIP 압축 해제·추출)을 건너뛰고 기존 extracted 파일 사용",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="ChromaDB 벡터 DB를 강제로 재생성",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Step 3(Supabase 업로드)를 건너뜀 (분석만 수행)",
    )
    args = parser.parse_args()

    start_time = time.time()
    _banner("교무지원과 업무 자동화 파이프라인 시작", width=60)

    # STEP 1 ─ 추출
    if args.skip_extract:
        logger.info("--skip-extract 옵션 → 기존 data/extracted/ 파일 사용")
        txt_files = sorted(EXTRACTED_DIR.glob("*.txt"))
        logger.info("  기존 txt 파일: %d개", len(txt_files))
    else:
        txt_files = step_extract()

    # STEP 2 ─ RAG 분석
    analysis_results = step_analyze(txt_files, rebuild_db=args.rebuild_db)

    # STEP 3 ─ Supabase 업로드
    if args.skip_upload:
        logger.info("--skip-upload 옵션 → 업로드 건너뜀")
    else:
        step_upload(analysis_results)

    # 정리
    cleanup()

    elapsed = time.time() - start_time
    _banner(f"파이프라인 완료  (소요 시간: {elapsed:.1f}초)", width=60)

    if FAILED_LOG.exists() and FAILED_LOG.stat().st_size > 0:
        logger.warning("일부 실패 항목이 있습니다. %s 를 확인하세요.", FAILED_LOG)
    else:
        logger.info("모든 처리가 정상적으로 완료되었습니다. ✅")


if __name__ == "__main__":
    main()
