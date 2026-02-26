"""
교무지원과 업무 자동화 - 파일 처리 모듈

ZIP 파일을 압축 해제하고 확장자별로 텍스트를 추출한 뒤 JSON 으로 저장한다.

설계 원칙:
- ZIP 내부 폴더 구조(DOC/ATTDOC/ATT 여부)는 처리 로직에 영향을 주지 않는다.
- 폴더명은 메타데이터로만 기록하여 나중에 참고할 수 있도록 한다.
- raw/ 에는 공문 ZIP 외에 참고 파일(.pdf, .xlsx 등)을 직접 올릴 수도 있다.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

# =============================================
# 경로 설정
# =============================================
BASE_DIR    = Path(__file__).parent
RAW_DIR     = BASE_DIR / "data" / "raw"
TEMP_DIR    = BASE_DIR / "data" / "temp"
OUTPUT_DIR  = BASE_DIR / "data" / "output"

# 처리할 확장자 목록
SUPPORTED_EXTENSIONS = {".xlsx", ".hwp", ".hwpx", ".pdf", ".docx"}


# =============================================
# 한글 파일명 복원 헬퍼
# =============================================
def _decode_zip_name(info: zipfile.ZipInfo) -> str:
    """CP949(Windows)로 저장된 ZIP 항목명을 올바른 문자열로 반환한다."""
    if info.flag_bits & 0x800:          # UTF-8 플래그가 켜진 경우
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return info.filename            # 변환 실패 시 원본 유지


# =============================================
# 압축 해제 → 파일 메타데이터 목록 반환
# =============================================
def extract_zips(raw_dir: Path, temp_dir: Path) -> list[dict]:
    """raw_dir 안의 ZIP 파일을 temp_dir 에 풀고 파일별 메타데이터 목록을 반환한다.

    반환 형식 (파일 1개당):
    {
        "source_zip": "공문제목.zip",   # 원본 ZIP 파일명 (직접 올린 파일이면 None)
        "folder_in_zip": "DOC",        # ZIP 내부 폴더명 (없으면 빈 문자열)
        "filepath": Path(...),         # 추출된 실제 경로
        "filename": "파일명.hwp",
        "extension": ".hwp",
    }
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_entries: list[dict] = []

    # ── ZIP 파일 처리 ──────────────────────────────────────
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        print(f"[INFO] {raw_dir} 에서 ZIP 파일을 찾을 수 없습니다.")
    else:
        for zip_path in sorted(zip_files):
            print(f"[INFO] 압축 해제 중: {zip_path.name}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    correct_name = _decode_zip_name(info)
                    dest_path = temp_dir / correct_name

                    if info.is_dir():
                        dest_path.mkdir(parents=True, exist_ok=True)
                        continue

                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())

                    # ZIP 내부 폴더명 추출 (예: "DOC", "ATT", "ATTDOC" 또는 그룹웨어 폴더명)
                    parts = Path(correct_name).parts
                    folder_in_zip = parts[0] if len(parts) > 1 else ""

                    file_entries.append({
                        "source_zip":    zip_path.name,
                        "folder_in_zip": folder_in_zip,
                        "filepath":      dest_path,
                        "filename":      dest_path.name,
                        "extension":     dest_path.suffix.lower(),
                    })

    # ── raw/ 에 직접 올린 파일 처리 (ZIP 아닌 것) ─────────
    for direct_file in sorted(raw_dir.iterdir()):
        if direct_file.suffix.lower() == ".zip" or direct_file.is_dir():
            continue
        print(f"[INFO] 직접 파일 감지: {direct_file.name}")
        file_entries.append({
            "source_zip":    None,          # ZIP 에서 온 게 아님
            "folder_in_zip": "",
            "filepath":      direct_file,
            "filename":      direct_file.name,
            "extension":     direct_file.suffix.lower(),
        })

    print(f"[INFO] 총 {len(file_entries)}개 파일 수집 완료")
    return file_entries


# =============================================
# 확장자별 텍스트 추출 함수 (추후 구현 예정)
# =============================================
def extract_xlsx(filepath: Path) -> str:
    """엑셀(.xlsx) 파일에서 텍스트 추출 — TODO: openpyxl"""
    return f"[xlsx] {filepath.name} 에서 텍스트 추출 성공"

def extract_hwp(filepath: Path) -> str:
    """한글(.hwp) 파일에서 텍스트 추출 — TODO: pyhwp"""
    return f"[hwp] {filepath.name} 에서 텍스트 추출 성공"

def extract_hwpx(filepath: Path) -> str:
    """한글(.hwpx) 파일에서 텍스트 추출 — TODO: hwpx 파싱"""
    return f"[hwpx] {filepath.name} 에서 텍스트 추출 성공"

def extract_pdf(filepath: Path) -> str:
    """PDF 파일에서 텍스트 추출 — TODO: pdfplumber / pymupdf"""
    return f"[pdf] {filepath.name} 에서 텍스트 추출 성공"

def extract_docx(filepath: Path) -> str:
    """워드(.docx) 파일에서 텍스트 추출 — TODO: python-docx"""
    return f"[docx] {filepath.name} 에서 텍스트 추출 성공"

EXTRACTOR_MAP = {
    ".xlsx": extract_xlsx,
    ".hwp":  extract_hwp,
    ".hwpx": extract_hwpx,
    ".pdf":  extract_pdf,
    ".docx": extract_docx,
}


# =============================================
# 파일 처리 → 결과 레코드 반환
# =============================================
def process_files(entries: list[dict]) -> list[dict]:
    """각 파일 항목에 텍스트 추출 결과를 추가해 레코드 목록으로 반환한다.

    반환 레코드 구조:
    {
        "source_zip":    "공문제목.zip" | null,
        "folder_in_zip": "DOC" | "ATT" | "그룹웨어폴더명" | "",
        "filename":      "파일명.hwp",
        "extension":     ".hwp",
        "status":        "ok" | "skipped",
        "text":          "추출된 텍스트 또는 스킵 사유",
    }
    """
    records: list[dict] = []

    for entry in entries:
        filepath = entry["filepath"]
        if not filepath.is_file():
            continue

        ext = entry["extension"]
        base_record = {
            "source_zip":    entry["source_zip"],
            "folder_in_zip": entry["folder_in_zip"],
            "filename":      entry["filename"],
            "extension":     ext,
        }

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[SKIP] 지원하지 않는 확장자: {entry['filename']}")
            records.append({**base_record, "status": "skipped",
                             "text": f"지원하지 않는 확장자: {ext}"})
            continue

        text = EXTRACTOR_MAP[ext](filepath)
        print(f"[OK]   {text}")
        records.append({**base_record, "status": "ok", "text": text})

    return records


# =============================================
# JSON 저장
# =============================================
def save_results_json(records: list[dict], output_dir: Path) -> Path:
    """처리 결과를 JSON 파일로 저장하고 저장 경로를 반환한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"result_{timestamp}.json"

    # Path 객체는 JSON 직렬화 불가 → 문자열로 변환
    serializable = [
        {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
        for r in records
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 결과 저장 완료 → {output_path}")
    return output_path


# =============================================
# 진입점
# =============================================
if __name__ == "__main__":
    # 1. ZIP 압축 해제 + 직접 파일 수집
    entries = extract_zips(RAW_DIR, TEMP_DIR)

    # 2. 파일별 텍스트 추출
    records = process_files(entries)

    # 3. JSON 저장
    output_path = save_results_json(records, OUTPUT_DIR)

    # 4. 요약 출력
    ok_count      = sum(1 for r in records if r["status"] == "ok")
    skipped_count = sum(1 for r in records if r["status"] == "skipped")
    print(f"\n===== 처리 완료 =====")
    print(f"  처리 성공: {ok_count}개")
    print(f"  건너뜀  : {skipped_count}개")
    print(f"  결과 파일: {output_path}")
