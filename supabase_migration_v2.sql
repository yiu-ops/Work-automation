-- ============================================================
-- gyomu_tasks 테이블 스키마 마이그레이션 v2
-- 실행 위치: Supabase Dashboard > SQL Editor
-- ============================================================

-- 1. 신규 컬럼 추가
ALTER TABLE gyomu_tasks
  ADD COLUMN IF NOT EXISTS reference_documents JSONB    DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS compliance_check    TEXT     DEFAULT '',
  ADD COLUMN IF NOT EXISTS recurrence_pattern  TEXT     DEFAULT '',
  ADD COLUMN IF NOT EXISTS document_count      INTEGER  DEFAULT 0,
  ADD COLUMN IF NOT EXISTS semester            TEXT     DEFAULT '';

-- 2. 기존 데이터의 기본값 채우기 (신규 컬럼 NULL→빈값)
UPDATE gyomu_tasks
SET
  reference_documents = COALESCE(reference_documents, '[]'::jsonb),
  compliance_check    = COALESCE(compliance_check,    ''),
  recurrence_pattern  = COALESCE(recurrence_pattern,  ''),
  document_count      = COALESCE(document_count,      0),
  semester            = COALESCE(semester,            '')
WHERE
  reference_documents IS NULL
  OR compliance_check IS NULL
  OR recurrence_pattern IS NULL
  OR document_count IS NULL
  OR semester IS NULL;

-- 3. 확인 쿼리
SELECT
  column_name,
  data_type,
  column_default
FROM information_schema.columns
WHERE table_name = 'gyomu_tasks'
ORDER BY ordinal_position;
