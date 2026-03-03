-- ============================================================
-- gyomu_tasks v3 마이그레이션 (SOP 생성기 신규 필드)
-- Supabase Dashboard → SQL Editor 에서 실행하세요.
-- ============================================================

ALTER TABLE gyomu_tasks
  ADD COLUMN IF NOT EXISTS standard_timeline      TEXT     DEFAULT '',
  ADD COLUMN IF NOT EXISTS compliance_checklists  JSONB    DEFAULT '[]',
  ADD COLUMN IF NOT EXISTS early_warning          TEXT     DEFAULT '',
  ADD COLUMN IF NOT EXISTS auto_draft_context     TEXT     DEFAULT '';

-- 실행 확인
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'gyomu_tasks'
ORDER BY ordinal_position;
