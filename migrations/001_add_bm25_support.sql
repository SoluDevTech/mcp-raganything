-- Migration: Add BM25 support via pg_textsearch
-- Version: 001
-- Date: 2026-04-07
-- Jira: BRIC-7

-- This migration adds BM25 full-text search capability using the pg_textsearch extension
-- It creates tsvector columns and BM25 indexes for chunk content

-- ========================================
-- Prerequisites: Install pg_textsearch extension
-- ========================================

-- Install pg_textsearch extension (requires superuser privileges)
-- Run this in PostgreSQL:
-- CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- If pg_textsearch is not available, you can install it from:
-- https://github.com/timescale/pg_textsearch

-- ========================================
-- Step 1: Add tsvector column to chunks table
-- ========================================

-- Add tsvector column for BM25 indexing
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- ========================================
-- Step 2: Create GIN index for tsvector operations
-- ========================================

-- Create GIN index for tsvector column (used for @@ and @@@ operators)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv 
ON chunks USING GIN(content_tsv);

-- ========================================
-- Step 3: Create BM25 index using pg_textsearch
-- ========================================

-- Create BM25 index using pg_textsearch <@> operator
-- Note: This requires pg_textsearch extension to be installed
-- If extension is not available, this will be skipped
DO $$ 
BEGIN
    -- Check if pg_textsearch extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') THEN
        -- Create BM25 index for content column
        -- Uses 'english' text search configuration
        -- The <@> operator will be available for BM25 ranking
        CREATE INDEX IF NOT EXISTS idx_chunks_bm25 
        ON chunks USING bm25(content) 
        WITH (text_config='english');
        
        RAISE NOTICE 'BM25 index created successfully';
    ELSE
        RAISE NOTICE 'pg_textsearch extension not found. BM25 index creation skipped.';
        RAISE NOTICE 'Install pg_textsearch and run: CREATE EXTENSION pg_textsearch;';
    END IF;
END $$;

-- ========================================
-- Step 4: Create auto-update trigger for tsvector
-- ========================================

-- Create function to auto-update tsvector on INSERT/UPDATE
CREATE OR REPLACE FUNCTION update_chunks_tsv()
RETURNS TRIGGER AS $$
BEGIN
    -- Update tsvector column with English text configuration
    -- This uses PostgreSQL's built-in to_tsvector function
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for auto-indexing
DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON chunks;
CREATE TRIGGER trg_chunks_content_tsv
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();

-- ========================================
-- Step 5: Backfill existing documents
-- ========================================

-- Backfill tsvector for existing documents (if any)
UPDATE chunks 
SET content_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE content_tsv IS NULL;

-- ========================================
-- Step 6: Verify migration
-- ========================================

-- Verify indexes were created
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'chunks' 
  AND (indexname LIKE '%tsv%' OR indexname LIKE '%bm25%')
ORDER BY indexname;

-- Verify trigger exists
SELECT 
    trigger_name,
    event_manipulation,
    action_timing,
    action_statement
FROM information_schema.triggers
WHERE event_object_table = 'chunks'
  AND trigger_name = 'trg_chunks_content_tsv';

-- ========================================
-- Step 7 (Optional): Performance stats
-- ========================================

-- Check BM25 index usage (run after some queries)
-- SELECT * FROM pg_stat_user_indexes WHERE indexrelid::regclass::text LIKE '%bm25%';

-- ========================================
-- Rollback instructions (if needed)
-- ========================================

/*
-- To rollback this migration:
DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON chunks;
DROP FUNCTION IF EXISTS update_chunks_tsv();
DROP INDEX IF EXISTS idx_chunks_bm25;
DROP INDEX IF EXISTS idx_chunks_content_tsv;
ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv;
*/

-- ========================================
-- Usage examples
-- ========================================

/*
-- Example 1: Basic BM25 search using pg_textsearch
SELECT 
    chunk_id,
    content,
    file_path,
    content <@> websearch_to_tsquery('english', 'PostgreSQL database') AS score
FROM chunks
WHERE content_tsv @@ websearch_to_tsquery('english', 'PostgreSQL database')
ORDER BY score
LIMIT 10;

-- Example 2: BM25 search with working_dir filter
SELECT 
    chunk_id,
    content,
    file_path,
    content <@> websearch_to_tsquery('english', 'search terms') AS score
FROM chunks
WHERE working_dir = 'your-project'
  AND content_tsv @@ websearch_to_tsquery('english', 'search terms')
ORDER BY score
LIMIT 10;

-- Example 3: Traditional full-text search (without BM25 ranking)
SELECT 
    chunk_id,
    content,
    file_path,
    ts_rank_cd(content_tsv, websearch_to_tsquery('english', 'search terms')) AS score
FROM chunks
WHERE content_tsv @@ websearch_to_tsquery('english', 'search terms')
ORDER BY score DESC
LIMIT 10;

-- Note: The <@> operator returns negative scores (lower is better)
-- Convert to positive: ABS(content <@> query)
*/