"""
교무 행정 업무 자동화 - RAG(Retrieval-Augmented Generation) 분석기

이 모듈은 학교 규정집(Reference)을 벡터화하여 Knowledge Base를 구축하고,
기안 문서(Task Document)를 분석하여 관련 규정을 매핑하고 인사이트를 도출합니다.

[기능]
1. 규정집 벡터화 (Build Vector DB): data/reference/ 내 텍스트 파일을 청크로 나누어 ChromaDB에 저장
2. 문서 분석 (Analyze Document): data/extracted/ 내 기안 문서를 입력받아 관련 규정을 검색하고 LLM으로 분석
3. 결과 저장: 분석 결과를 JSON 파일로 저장

[실행 방법]
1. 환경 변수 설정 (.env 파일에 GOOGLE_API_KEY 설정)
2. 규정집 파일 준비 (data/reference/*.txt)
3. 기안 문서 준비 (data/extracted/*.txt)
4. 실행: python rag_analyzer.py
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Processor 모듈 임포트 (ZIP 및 다양한 문서 포맷 처리용)
try:
    import processor
except ImportError:
    # processor.py가 같은 디렉토리에 없을 경우를 대비해 경로 추가
    import sys
    sys.path.append(str(Path(__file__).parent))
    import processor

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

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
# Pydantic 모델 정의 (출력 스키마)
# =============================================
class AnalysisResult(BaseModel):
    task_name: str = Field(description="업무명 (기안 문서의 제목 또는 핵심 주제)")
    core_regulations: List[str] = Field(description="매핑된 필수 관련 문서 및 규정 조항 (예: 사교육 관련 대학교원 겸직 가이드라인 제3조)")
    target_date: str = Field(description="업무 완료 목표일 (YYYY-MM-DD 형식 또는 '2024-1학기 말' 등 특정 시기)")
    action_triggers: List[str] = Field(description="다가올 기간을 예측하여 미리 해야 할 사전 작업 리스트 (예: D-14: 교육부 공문 발송)")
    lessons_learned: List[str] = Field(description="문서에서 파악된 지난 학기 부족했던 점이나 주의사항, 개선점")

# =============================================
# ZIP 및 문서 전처리 함수
# =============================================
def preprocess_extracted_data():
    """
    data/extracted/ 폴더 내의 ZIP 파일을 찾아 압축을 풀고,
    processor.py를 이용해 텍스트를 추출하여 .txt 파일로 변환합니다.
    """
    if not EXTRACTED_DIR.exists():
        return

    # ZIP 파일이 있는지 확인
    zip_files = list(EXTRACTED_DIR.glob("*.zip"))
    if not zip_files:
        return

    logger.info(f"ZIP 파일 {len(zip_files)}개를 발견하여 전처리를 시작합니다...")

    # 임시 폴더 설정
    temp_dir = EXTRACTED_DIR / "temp_extracted"
    
    # 1. 압축 해제 및 파일 목록 수집 (processor.py 활용)
    # 주의: extract_zips는 폴더 내 모든 파일을 스캔하므로, 반환값에서 ZIP 출처인 것만 필터링해야 함
    try:
        entries = processor.extract_zips(EXTRACTED_DIR, temp_dir)
    except Exception as e:
        logger.error(f"ZIP 파일 처리 중 오류 발생: {e}")
        return

    # 2. ZIP에서 나온 파일만 필터링
    zip_entries = [e for e in entries if e.get("source_zip") is not None]
    
    if not zip_entries:
        logger.info("처리할 압축 파일 내용이 없습니다.")
        return

    # 3. 텍스트 추출
    logger.info(f"문서 변환 시작 ({len(zip_entries)}개 파일)...")
    records = processor.process_files(zip_entries)

    # 4. 추출된 텍스트를 .txt 파일로 저장
    converted_count = 0
    for record in records:
        if record.get("status") == "ok" and record.get("text"):
            # 파일명 생성: [ZIP파일명]_[원본파일명].txt
            zip_name = Path(record["source_zip"]).stem
            orig_name = Path(record["filename"]).stem
            # 특수문자 제거 등은 processor가 해주지 않으므로 간단히 처리
            safe_name = f"[{zip_name}]_{orig_name}.txt"
            
            output_path = EXTRACTED_DIR / safe_name
            
            # 이미 존재하면 덮어쓸지 여부? 여기선 덮어씀
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(record["text"])
                converted_count += 1
            except Exception as e:
                logger.error(f"파일 저장 실패 ({output_path}): {e}")

    logger.info(f"전처리 완료: {converted_count}개의 텍스트 파일이 생성되었습니다.")
    
    # (선택) 임시 폴더 정리
    # shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================
# RAG 시스템 클래스
# =============================================
class RAGSystem:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 임베딩 모델 설정 (text-embedding-004 추천, 없으면 embedding-001 사용)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=self.api_key
        )
        
        # LLM 모델 설정 (Gemini 1.5 Flash 사용)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=self.api_key
        )
        
        self.vector_store = None

    def build_vector_db(self, force_rebuild: bool = False):
        """
        1단계: 규정집 벡터화 (Knowledge Base 구축)
        data/reference/ 폴더의 텍스트 파일들을 읽어 ChromaDB에 저장
        """
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
        
        # 문서 로드 (.txt, .md 파일 지원)
        loader = DirectoryLoader(str(REFERENCE_DIR), glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()
        
        if not documents:
            # .md 파일도 시도
            loader_md = DirectoryLoader(str(REFERENCE_DIR), glob="**/*.md", loader_cls=TextLoader, show_progress=True)
            documents.extend(loader_md.load())
            
        # .pdf 파일 추가
        loader_pdf = DirectoryLoader(str(REFERENCE_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        pdf_docs = loader_pdf.load()
        if pdf_docs:
            documents.extend(pdf_docs)
            
        if not documents:
            logger.warning("data/reference/ 폴더에 처리할 문서가 없습니다.")
            return

        # 청크 분할 (CharacterTextSplitter 사용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"총 {len(documents)}개 문서를 {len(splits)}개 청크로 분할했습니다.")

        # 벡터 DB 생성 및 저장
        logger.info("벡터 DB를 생성 중입니다 (시간이 소요될 수 있습니다)...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        logger.info("벡터 DB 구축 완료.")

    def analyze_document(self, doc_path: Path) -> Optional[AnalysisResult]:
        """
        2단계 & 3단계: 업무 문서 분석 및 매핑 -> JSON 결과 반환
        """
        if not self.vector_store:
            self.build_vector_db()
            
        if not self.vector_store:
             logger.error("벡터 DB가 준비되지 않아 분석을 수행할 수 없습니다.")
             return None

        # 기안 문서 읽기
        try:
            if doc_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(doc_path))
                pages = loader.load()
                doc_content = "\n".join([page.page_content for page in pages])
            else:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc_content = f.read()
        except Exception as e:
            logger.error(f"문서 읽기 실패 ({doc_path}): {e}")
            return None

        logger.info(f"문서 분석 시작: {doc_path.name}")

        # 1. 관련 규정 검색 (Retrieve)
        # 유사도 기반 상위 3개 청크 검색
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(doc_content[:2000]) # 문서 앞부분 2000자 기반 검색 (너무 길면 자름)
        
        context_text = "\n\n".join([f"규정 {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # 2. LLM 프롬프트 구성
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 대학 교무 행정 전문가입니다. 주어진 '기안 문서'와 '관련 규정'을 분석하여 행정 처리에 필요한 핵심 정보를 추출해야 합니다.\n"
                       "반드시 아래 JSON 포맷으로 응답해주세요.\n\n"
                       "{format_instructions}"),
            ("human", "### 관련 규정 (Reference Context):\n{context}\n\n"
                      "### 기안 문서 (Task Document):\n{task_doc}\n\n"
                      "위 내용을 바탕으로 문서를 분석해 주세요.")
        ])

        # 3. 체인 실행
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "context": context_text,
                "task_doc": doc_content,
                "format_instructions": parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.error(f"LLM 분석 중 오류 발생: {e}")
            return None

# =============================================
# 메인 실행 블록
# =============================================
if __name__ == "__main__":
    # 출력 디렉토리 생성
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    rag = RAGSystem()
    
    # 1. 벡터 DB 구축 (최초 1회 또는 업데이트 시 실행)
    # rag.build_vector_db(force_rebuild=True) 
    rag.build_vector_db()

    # 2. ZIP 파일 등 전처리 (텍스트 추출)
    preprocess_extracted_data()

    # 3. 추출된 기안 문서 처리
    processed_count = 0
    if not EXTRACTED_DIR.exists():
         logger.warning(f"'{EXTRACTED_DIR}' 폴더가 없습니다. 텍스트 파일을 넣어주세요.")
    else:
        # .txt 및 .pdf 파일 모두 검색
        doc_files = list(EXTRACTED_DIR.glob("*.txt")) + list(EXTRACTED_DIR.glob("*.pdf"))
        logger.info(f"분석 대상 문서: {len(doc_files)}개")

        for doc_file in doc_files:
            result = rag.analyze_document(doc_file)
            if result:
                # 결과 저장
                output_path = OUTPUT_DIR / f"{doc_file.stem}_analysis.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
                logger.info(f"분석 완료 및 저장: {output_path}")
                processed_count += 1
            else:
                logger.warning(f"분석 실패: {doc_file.name}")

    logger.info(f"총 {processed_count}개 문서 분석 완료.")
