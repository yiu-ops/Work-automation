"""
교무 행정 업무 자동화 - RAG(Retrieval-Augmented Generation) 분석기 v2

[핵심 설계 원칙]
  · 분석 단위: 개별 문서 X  →  같은 [업무명]으로 묶인 문서 '그룹' O
  · 동일 업무의 여러 문서(주 기안문 + 하위 회신 + 첨부 규정)를 한꺼번에
    LLM에 제공하여 "종합 인사이트"를 생성합니다.
  · 필수 법령/상위부서 지침까지 명시적으로 매핑합니다.

[기능]
1. 규정집 벡터화 (Build Vector DB)
   data/reference/ 내 PDF/TXT/MD 파일 → ChromaDB (청크 저장)
2. 그룹 분석 (analyze_task_group)
   [업무명] prefix 가 같은 파일 묶음 → 단일 LLM 호출 → 종합 JSON
3. 결과 저장: data/analysis_result/<업무명>_group_analysis.json

[출력 JSON 신규 필드]
  reference_documents : 교육부 가이드라인·법령·규정집 등 필수 참고 문서 목록
  compliance_check    : 규정 준수 상태 평가 (현황 진단)
  recurrence_pattern  : 반복 주기 및 권장 착수 시점
  document_count      : 그룹 내 처리된 문서 수
  semester            : 대상 학기/시즌 (예: 2025-1학기)

[실행 방법]
  python rag_analyzer.py               # 전체 그룹 분석
  python rag_analyzer.py --rebuild-db  # 벡터 DB 재생성 후 분석
"""

import os
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
OUTPUT_DIR = DATA_DIR / "analysis_result"

# 환경 변수 로드
load_dotenv()

# =============================================
# Pydantic 모델 정의 (출력 스키마 v3 — SOP 생성기)
# =============================================
class AnalysisResult(BaseModel):
    """
    업무 그룹 종합 분석 결과 v3 (능동적 업무 실행 매뉴얼).
    2024 교무지원과 행정편람의 31개 업무 분류를 절대 기준으로 삼아
    실무자가 '즉시 실행 가능한' SOP를 도출합니다.
    """
    task_name: str = Field(
        description="업무명. 행정편람 31개 분류명과 일치하도록 정규화. 예: '전임교원 겸직현황 실태조사'"
    )
    semester: str = Field(
        description="대상 학기. 문서 날짜 기준 추출. 예: '2025-1학기', '2025-2학기'"
    )
    reference_documents: List[str] = Field(
        description=(
            "반드시 참조해야 할 상위 법령·교육부 지침·본교 규정집 조항 목록. "
            "'교육공무원법 제19조(겸직 금지)', "
            "'사교육 관련 대학교원 겸직 가이드라인(교육부 2023) 제3조 제1항' 처럼 "
            "구체적 조항까지 명시. 규정집 검색 결과 + 기안문 인용 법령 모두 포함."
        )
    )
    core_regulations: List[str] = Field(
        description=(
            "본교 내부 규정 중 직접 적용되는 조항 목록. "
            "'용인대학교 교원인사규정 제25조(겸직허가)' 형식. "
            "reference_documents와 중복 없이 내부 규정만."
        )
    )
    target_date: str = Field(
        description="다음 사이클 업무 완료 목표일. YYYY-MM-DD. 행정편람의 처리 시기 기준으로 산정."
    )
    recurrence_pattern: str = Field(
        description=(
            "행정편람에 명시된 처리 시기를 바탕으로 한 반복 주기와 권장 착수 시점. "
            "예: '매 학기 말(8월/2월) 마감, D-60부터 각 학과 자료 수집 공문 발송 권장'."
        )
    )
    action_triggers: List[str] = Field(
        description=(
            "다음 사이클을 위한 D-Day 기준 사전 작업 리스트. "
            "형식: 'D-60: [구체적 행동]'. 최소 4개 이상, 행정편람 처리 절차 기반."
        )
    )
    compliance_check: str = Field(
        description=(
            "이번 사이클 규정 준수 현황 진단. "
            "기한 준수 여부, 누락 학과·부서, 서식 오류 등을 서술형으로 기술."
        )
    )
    lessons_learned: str = Field(
        description=(
            "확인된 문제점·개선 필요 사항. 최소 2개 이상. "
            "회신 지연·누락·형식 오류 등 실무 패턴을 구체적으로 기술."
        )
    )
    document_count: int = Field(
        description="분석에 사용된 문서 수 (기안문 + 회신문 + 첨부 등 합계)"
    )

    # ── v3 신규 필드 ──────────────────────────────────────────────
    standard_timeline: str = Field(
        description=(
            "【v3 신규】행정편람에 명시된 이 업무의 처리 시기를 상대적 기준으로 서술. "
            "절대 날짜(2025-05-10)가 아닌 상대 표현 사용. "
            "예: '매년 5월 둘째 주 마감 / 1학기 개강 30일 전 착수 / 매 학기 말 정기 시행'. "
            "행정편람에서 해당 업무 시기를 찾지 못하면 문서 패턴으로 추론."
        )
    )
    compliance_checklists: List[str] = Field(
        description=(
            "【v3 신규】기안 전 실무자가 반드시 확인해야 할 사항을 "
            "'~했는가?' 형태의 체크리스트 배열로 생성. "
            "상위법·행정편람·본교 규정 기반. 최소 5개 이상. "
            "예: '교육부 사교육 관련 가이드라인 최신 버전(2023년판)을 확인했는가?', "
            "    '겸직 허가 신청서 서식이 최신 버전인지 확인했는가?', "
            "    '전임교원 전원에게 신청 안내 공문을 개강 30일 전에 발송했는가?'"
        )
    )
    early_warning: str = Field(
        description=(
            "【v3 신규】과거 문서(회신문·반려 이력 등)에서 추출한 "
            "가장 중요한 주의사항을 단문(2~4줄) 긴급 주의보로 정제. "
            "업무 착수 시 관리자 화면에 가장 먼저 표시됨. "
            "형식: '🚨 [핵심 위험] 설명'. "
            "예: '🚨 매년 A학과·B학과에서 제출 지연 발생 — 착수 시 해당 학과 우선 개별 독촉 필요'"
        )
    )
    auto_draft_context: str = Field(
        description=(
            "【v3 신규】이번 학기 기안문 초안 뼈대. "
            "과거 문서 내용을 바탕으로 실무자가 그대로 복사해서 쓸 수 있는 수준으로 작성. "
            "포함 요소: 제목/수신/경유/발신/제목/목적/근거 법령·규정/주요 일정/제출 서류/담당부서. "
            "마크다운 서식(## 소제목, - 목록) 사용 가능. "
            "예:\n## 제목\n전임교원 겸직현황 실태조사 결과 보고\n\n## 목적\n..."
        )
    )

# =============================================
# 모델 폴백 목록 (쿼터 초과 시 순서대로 시도)
# ─────────────────────────────────────────────
# 형식:
#   "gemini-xxx"         → Google Gemini API
#   "groq:model-name"    → Groq API (langchain-groq)
# =============================================
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "groq:llama-3.3-70b-versatile",   # 무료 14,400회/일, 빠른 응답
    "groq:llama-3.1-8b-instant",      # 더 가벼운 Groq 백업
]

def _pick_model_from_args() -> str:
    """
    CLI: --model gemini-2.0-flash  또는  env: GEMINI_MODEL=gemini-2.0-flash
    지정이 없으면 _FALLBACK_MODELS[0] 사용.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv("GEMINI_MODEL", _FALLBACK_MODELS[0])


# =============================================
# RAG 시스템 클래스 v2
# =============================================
class RAGSystem:
    def __init__(self, model: Optional[str] = None):
        # .env의 GEMINI_API_KEY 또는 GOOGLE_API_KEY 모두 지원
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

        self._model_name = model or _pick_model_from_args()
        logger.info(f"[LLM] 사용 모델: {self._model_name}")

        # 임베딩 모델 (항상 Gemini 사용)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.api_key
        )

        # LLM (temperature=0: 재현성 확보)
        self.llm = self._make_llm(self._model_name)

        self.vector_store = None

    def _make_llm(self, model_spec: str) -> BaseChatModel:
        """model_spec이 'groq:...' 이면 ChatGroq, 아니면 ChatGoogleGenerativeAI 반환."""
        if model_spec.startswith("groq:"):
            if not _GROQ_AVAILABLE:
                raise ImportError("langchain-groq 패키지가 설치되지 않았습니다. pip install langchain-groq")
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY 환경 변수가 설정되지 않았습니다.")
            groq_model = model_spec[len("groq:"):]
            logger.info(f"  → Groq API 사용: {groq_model}")
            return ChatGroq(
                model=groq_model,
                temperature=0,
                groq_api_key=self.groq_api_key,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model_spec,
                temperature=0,
                google_api_key=self.api_key,
            )

    # ──────────────────────────────────────────────────────────
    # 1단계: 규정집 벡터화
    # ──────────────────────────────────────────────────────────
    def build_vector_db(self, force_rebuild: bool = False):
        """data/reference/ 폴더의 PDF/TXT/MD 파일을 청크로 나누어 ChromaDB에 저장."""
        if CHROMA_DB_DIR.exists() and not force_rebuild:
            logger.info(f"기존 벡터 DB를 로드합니다: {CHROMA_DB_DIR}")
            self.vector_store = Chroma(
                persist_directory=str(CHROMA_DB_DIR),
                embedding_function=self.embeddings
            )
            return

        if force_rebuild and CHROMA_DB_DIR.exists():
            logger.warning(f"기존 벡터 DB를 삭제하고 재생성합니다: {CHROMA_DB_DIR}")
            shutil.rmtree(CHROMA_DB_DIR)

        logger.info("규정집 문서를 로드하고 청크로 분할합니다...")
        documents = []

        for pdf_file in REFERENCE_DIR.glob("**/*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()
                documents.extend(pages)
                logger.info(f"  PDF 로드: {pdf_file.name} ({len(pages)}페이지)")
            except Exception as e:
                logger.warning(f"  PDF 로드 실패 [{pdf_file.name}]: {e}")

        for txt_file in REFERENCE_DIR.glob("**/*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"  TXT 로드 실패 [{txt_file.name}]: {e}")

        for md_file in REFERENCE_DIR.glob("**/*.md"):
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"  MD 로드 실패 [{md_file.name}]: {e}")

        if not documents:
            logger.warning("data/reference/ 폴더에 처리할 문서가 없습니다.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"총 {len(documents)}개 문서를 {len(splits)}개 청크로 분할했습니다.")

        logger.info("벡터 DB를 생성 중입니다...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        logger.info("벡터 DB 구축 완료.")

    # ──────────────────────────────────────────────────────────
    # 2단계: 업무 그룹 종합 분석 (핵심 메서드)
    # ──────────────────────────────────────────────────────────
    def analyze_task_group(
        self,
        task_name: str,
        doc_paths: List[Path]
    ) -> Optional[AnalysisResult]:
        """
        동일 [업무명]으로 묶인 문서 목록을 한꺼번에 분석하여
        종합 인사이트 AnalysisResult를 반환합니다.

        - 주 기안문 + 회신문 + 첨부 규정 등을 모두 포함해 패턴 분석
        - 벡터 DB에서 업무명 기반으로 관련 규정 청크를 검색
        """
        if not self.vector_store:
            self.build_vector_db()

        if not self.vector_store:
            logger.error("벡터 DB가 준비되지 않아 분석을 수행할 수 없습니다.")
            return None

        # ── 모든 문서 내용 취합 ──────────────────────────────
        all_contents: list[str] = []
        for dp in doc_paths:
            try:
                text = dp.read_text(encoding="utf-8").strip()
                if text:
                    all_contents.append(f"=== 문서: {dp.name} ===\n{text}")
            except Exception as e:
                logger.warning(f"  문서 읽기 실패 ({dp.name}): {e}")

        if not all_contents:
            logger.error(f"[{task_name}] 읽어올 수 있는 문서가 없습니다.")
            return None

        combined_text = "\n\n".join(all_contents)
        logger.info(f"[{task_name}] 총 {len(all_contents)}개 문서 취합 완료 "
                    f"({len(combined_text):,}자)")

        # ── 벡터 DB에서 관련 규정 검색 (k=8) ───────────────
        # 업무명 + 첫 번째 문서 앞부분을 쿼리로 사용
        query_text = task_name + "\n" + all_contents[0][:1500]
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 8}
        )
        relevant_docs = retriever.invoke(query_text)
        context_text = "\n\n".join(
            [f"[규정 참고 {i+1}] (출처: {d.metadata.get('source','?')})\n{d.page_content}"
             for i, d in enumerate(relevant_docs)]
        )
        logger.info(f"[{task_name}] 관련 규정 청크 {len(relevant_docs)}개 검색됨")

        # ── LLM 프롬프트 구성 ────────────────────────────────
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)

        system_prompt = (
            "당신은 대학 교무행정 전문가이자 SOP(표준업무절차) 설계 전문가입니다.\n"
            "주어진 자료를 분석하여 실무자가 '즉시 실행 가능한 업무 매뉴얼(SOP)'을 생성해야 합니다.\n\n"

            "【절대 기준: 2024 교무지원과 행정편람】\n"
            "규정 참고 자료에 '2024 교무지원과 행정편람'이 포함되어 있습니다.\n"
            "이 편람에는 교무지원과의 31개 업무 분류와 각 업무의 처리 시기가 명시되어 있습니다.\n"
            "모든 분석은 이 편람을 절대 기준으로 삼아야 합니다.\n\n"

            "【v3 신규 필드 작성 지침】\n"

            "1. standard_timeline (처리 기준 시점):\n"
            "   - 행정편람에서 이 업무의 '처리 시기' 항목을 찾아 '상대적 표현'으로 기술하세요.\n"
            "   - 절대 날짜(2025-05-10) 사용 금지. 반드시 상대 표현 사용.\n"
            "   - 예: '매년 5월 둘째 주 마감', '1학기 개강 30일 전 착수', '매 학기 말(6월/12월) 시행'\n"
            "   - 편람에서 해당 시기를 찾지 못하면 제공된 문서의 날짜 패턴으로 추론하세요.\n\n"

            "2. compliance_checklists (기안 전 체크리스트):\n"
            "   - 상위법·행정편람·본교 규정을 근거로 체크리스트를 작성하세요.\n"
            "   - 반드시 '~했는가?' 또는 '~확인했는가?' 질문 형태로 작성.\n"
            "   - 최소 5개 이상. 각 항목은 근거 법령/규정을 포함할 것.\n"
            "   - 예: '교육부 겸직 가이드라인(2023) 최신 버전을 확인했는가?'\n"
            "       '전임교원 전원에게 개강 30일 전 신청 안내 공문을 발송했는가?'\n\n"

            "3. early_warning (긴급 주의보):\n"
            "   - 과거 문서(회신 지연·반려·누락 등)와 compliance_check에서 도출된\n"
            "     가장 중요한 위험 요소를 단문 2~4줄로 압축하세요.\n"
            "   - 형식: '🚨 [핵심 위험] 구체 설명'\n"
            "   - 예: '🚨 A학과·B학과 매번 제출 지연 — 착수 즉시 개별 독촉 필요'\n"
            "   - '위험 없음' 또는 공문(空文)으로 작성하지 마세요.\n\n"

            "4. auto_draft_context (기안문 초안 뼈대):\n"
            "   - 과거 문서 내용을 바탕으로 이번 학기 기안에 바로 사용할 수 있는 뼈대를 작성하세요.\n"
            "   - 포함 필수 요소: 제목 / 목적 / 근거(법령·규정) / 주요 일정 / 제출 서류 / 담당부서.\n"
            "   - 마크다운 형식(## 소제목, - 목록) 사용.\n"
            "   - 실제 학과명·날짜·수치는 '[학과명]', '[날짜]' 같은 플레이스홀더로 표기.\n\n"

            "【기존 필드 작성 지침】\n"
            "5. reference_documents: 교육부 지침·법령의 구체적 조항까지 명시.\n"
            "6. core_regulations: 본교 내부 규정 조항만 별도 기재.\n"
            "7. compliance_check: 기한·형식·누락 부서 등 전체 패턴 진단.\n"
            "8. lessons_learned: 최소 2개 이상 구체적 문제점 기술. '없음' 회피 금지.\n"
            "9. recurrence_pattern: 행정편람 처리 시기 기반 반복 주기와 착수 시점.\n"
            "10. document_count: 분석 문서 수 정확히 기입.\n\n"

            "반드시 아래 JSON 포맷으로만 응답하세요.\n\n"
            "{format_instructions}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
             "### 규정 참고 자료 (벡터 DB 검색 결과):\n{context}\n\n"
             "### 기안문서 묶음 ({doc_count}개 문서):\n{task_docs}\n\n"
             "위 내용을 바탕으로 종합 분석해 주세요. "
             "문서 수={doc_count}, 업무명={task_name}")
        ])

        invoke_kwargs = {
            "context": context_text,
            "task_docs": combined_text[:12000],  # 루프 안에서 크기 조정됨
            "doc_count": len(all_contents),
            "task_name": task_name,
            "format_instructions": parser.get_format_instructions(),
        }

        # ── 모델 폴백 시도 ───────────────────────────────────
        # 쿼터 초과(429) → 다음 모델
        # 페이로드 초과(413) → 텍스트 절반 축소 후 동일 모델 재시도(최대 3회)
        # 파싱 실패(null) → 다음 모델
        current_idx = _FALLBACK_MODELS.index(self._model_name) \
            if self._model_name in _FALLBACK_MODELS else 0
        candidates = _FALLBACK_MODELS[current_idx:] + _FALLBACK_MODELS[:current_idx]

        text_limit = 12000  # 초기 텍스트 한도

        for attempt_model in candidates:
            if attempt_model != self._model_name:
                logger.warning(f"[{task_name}] 모델 전환 시도: {attempt_model}")

            size_retries = 0
            current_limit = text_limit

            while size_retries <= 3:
                invoke_kwargs["task_docs"] = combined_text[:current_limit]
                try:
                    llm = self._make_llm(attempt_model)
                    result = (prompt | llm | parser).invoke(invoke_kwargs)
                    if attempt_model != self._model_name:
                        logger.info(f"[{task_name}] {attempt_model} 로 성공 (텍스트 {current_limit}자)")
                    # document_count 보정
                    if result.document_count == 0:
                        object.__setattr__(result, "document_count", len(all_contents))
                    return result

                except Exception as e:
                    err = str(e)

                    # 413 Payload Too Large → 텍스트 축소 후 재시도
                    is_size_err = (
                        "413" in err
                        or "payload too large" in err.lower()
                        or "request too large" in err.lower()
                        or "context_length_exceeded" in err.lower()
                    )
                    if is_size_err:
                        current_limit = current_limit // 2
                        size_retries += 1
                        logger.warning(
                            f"[{task_name}] {attempt_model} 페이로드 초과 → "
                            f"텍스트 {current_limit}자로 축소 재시도 ({size_retries}/3)"
                        )
                        continue

                    # 쿼터/속도 제한 → 다음 모델
                    is_quota_err = (
                        "429" in err
                        or "RESOURCE_EXHAUSTED" in err
                        or "quota" in err.lower()
                        or "rate_limit" in err.lower()
                        or "rate limit" in err.lower()
                    )
                    if is_quota_err:
                        logger.warning(
                            f"[{task_name}] {attempt_model} 쿼터/한도 초과 → 다음 모델 시도"
                        )
                        break  # while 탈출 → 다음 모델

                    # JSON 파싱 실패 → 다음 모델
                    is_parse_err = (
                        "Failed to parse" in err
                        or "completion null" in err
                        or "validation error" in err.lower()
                    )
                    if is_parse_err:
                        logger.warning(
                            f"[{task_name}] {attempt_model} JSON 파싱 실패 → 다음 모델 시도"
                        )
                        break  # while 탈출 → 다음 모델

                    # 그 외 오류 (키 오류, 네트워크 등) → 즉시 실패
                    logger.error(f"[{task_name}] LLM 분석 오류 ({attempt_model}): {e}")
                    return None

        logger.error(f"[{task_name}] 모든 모델 실패. API 키 및 쿼터를 확인하세요.")
        return None
        return None

    # ──────────────────────────────────────────────────────────
    # (하위 호환) 단일 문서 분석 — 기존 코드 호환용
    # ──────────────────────────────────────────────────────────
    def analyze_document(self, doc_path: Path) -> Optional[AnalysisResult]:
        """단일 문서를 1개짜리 그룹으로 처리 (하위 호환)."""
        task_name = doc_path.stem  # 파일명을 업무명으로 사용
        return self.analyze_task_group(task_name, [doc_path])

# =============================================
# 파일명에서 [업무명] 추출 유틸
# =============================================
_GROUP_RE = re.compile(r"^\[(.+?)\]")

def extract_task_name(filename: str) -> Optional[str]:
    """
    '[전임교원 겸직현황 실태조사 결과 보고의 건]_xxx.txt'
    → '전임교원 겸직현황 실태조사 결과 보고의 건'
    매칭 실패 시 None 반환.
    """
    m = _GROUP_RE.match(filename)
    return m.group(1) if m else None


# =============================================
# 메인 실행 블록
# =============================================
if __name__ == "__main__":
    force_rebuild = "--rebuild-db" in sys.argv
    # --model 인자로 시작 모델 지정 가능
    # 예: python rag_analyzer.py --model gemini-2.0-flash
    start_model = _pick_model_from_args()

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    rag = RAGSystem(model=start_model)
    rag.build_vector_db(force_rebuild=force_rebuild)

    if not EXTRACTED_DIR.exists():
        logger.warning(f"'{EXTRACTED_DIR}' 폴더가 없습니다.")
        sys.exit(1)

    # ── 파일을 [업무명]으로 그룹핑 ──────────────────────────
    # 패턴 있는 파일: [업무명]_개별문서.txt → groups["업무명"] 에 추가
    # 패턴 없는 파일: 파일명 자체를 업무명으로 취급
    groups: dict[str, list[Path]] = defaultdict(list)
    for txt_file in sorted(EXTRACTED_DIR.glob("*.txt")):
        task_key = extract_task_name(txt_file.name) or txt_file.stem
        groups[task_key].append(txt_file)

    if not groups:
        logger.warning("분석할 .txt 파일이 없습니다.")
        sys.exit(0)

    logger.info(f"업무 그룹 {len(groups)}개 발견:")
    for name, files in groups.items():
        logger.info(f"  [{name}]  문서 {len(files)}개")

    # ── 그룹별 종합 분석 실행 ────────────────────────────────
    ok_count, fail_count = 0, 0
    for task_name, doc_paths in groups.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"분석 시작: {task_name} ({len(doc_paths)}개 문서)")

        result = rag.analyze_task_group(task_name, doc_paths)
        if result:
            # 안전한 파일명 생성 (특수문자 제거)
            safe_name = re.sub(r'[\\/:*?"<>|]', "_", task_name)
            output_path = OUTPUT_DIR / f"{safe_name}_group_analysis.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 저장 완료: {output_path.name}")
            ok_count += 1
        else:
            logger.error(f"✗ 분석 실패: {task_name}")
            fail_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"완료: 성공 {ok_count}건 / 실패 {fail_count}건")
