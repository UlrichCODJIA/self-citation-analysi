-- ===================================================================
-- PERFORMANCE TUNING SETTINGS - Optimized for 4 cores, 16GB RAM
-- ===================================================================

-- Bulk import optimizations
SET synchronous_commit = OFF;                    -- Faster writes, acceptable for bulk import

-- Memory allocation
SET maintenance_work_mem = '3GB';                -- Large for faster index builds (sequential)
SET work_mem = '256MB';                          -- Per-operation (multiple can run in parallel)
SET temp_buffers = '256MB';                      -- Reasonable for temp tables
SET effective_cache_size = '12GB';               -- Query planner hint (75% of RAM)
SET temp_file_limit = '100GB';                   -- Large temp files allowed

-- Parallelism
SET max_parallel_maintenance_workers = 3;        -- 4 cores - 1 for system
SET max_parallel_workers = 4;                    -- Total parallel workers
SET max_parallel_workers_per_gather = 3;         -- Aggressive for bulk (normally 2)

-- I/O optimization (these CAN be set via SET)
SET effective_io_concurrency = 200;              -- Assumes SSD (can set via SET)
SET random_page_cost = 1.1;                      -- SSD optimization (can set via SET)

-- Query optimization
SET hash_mem_multiplier = 2.0;                   -- More memory for hash operations


\echo 'Disabling autovacuum on imported tables...'
-- ALTER TABLE openalex.topics SET (autovacuum_enabled = false);
-- ALTER TABLE openalex.works SET (autovacuum_enabled = false);
ALTER TABLE openalex.works_authorships SET (autovacuum_enabled = false);
-- ALTER TABLE openalex.works_topics SET (autovacuum_enabled = false);
-- ALTER TABLE openalex.works_referenced_works SET (autovacuum_enabled = false);


\echo 'Dropping indexes on imported tables to speed up data import...'
-- DO $$
-- DECLARE
--     idx RECORD;
-- BEGIN
--     FOR idx IN
--         SELECT indexname
--         FROM pg_indexes
--         WHERE tablename = 'topics'
--           AND schemaname = 'openalex'
--           AND indexname NOT LIKE '%_pkey'
--     LOOP
--         RAISE NOTICE 'Dropping index % for faster import...', idx.indexname;
--         EXECUTE format('DROP INDEX IF EXISTS openalex.%I', idx.indexname);
--     END LOOP;
-- END $$;


-- DO $$
-- DECLARE
--     idx RECORD;
-- BEGIN
--     FOR idx IN
--         SELECT indexname
--         FROM pg_indexes
--         WHERE tablename = 'works'
--           AND schemaname = 'openalex'
--           AND indexname NOT LIKE '%_pkey'
--     LOOP
--         RAISE NOTICE 'Dropping index % for faster import...', idx.indexname;
--         EXECUTE format('DROP INDEX IF EXISTS openalex.%I', idx.indexname);
--     END LOOP;
-- END $$;


DO $$
DECLARE
    idx RECORD;
BEGIN
    FOR idx IN
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = 'works_authorships'
          AND schemaname = 'openalex'
          AND indexname NOT LIKE '%_pkey'
    LOOP
        RAISE NOTICE 'Dropping index % for faster import...', idx.indexname;
        EXECUTE format('DROP INDEX IF EXISTS openalex.%I', idx.indexname);
    END LOOP;
END $$;


-- DO $$
-- DECLARE
--     idx RECORD;
-- BEGIN
--     FOR idx IN
--         SELECT indexname
--         FROM pg_indexes
--         WHERE tablename = 'works_topics'
--           AND schemaname = 'openalex'
--           AND indexname NOT LIKE '%_pkey'
--     LOOP
--         RAISE NOTICE 'Dropping index % for faster import...', idx.indexname;
--         EXECUTE format('DROP INDEX IF EXISTS openalex.%I', idx.indexname);
--     END LOOP;
-- END $$;


-- DO $$
-- DECLARE
--     idx RECORD;
-- BEGIN
--     FOR idx IN
--         SELECT indexname
--         FROM pg_indexes
--         WHERE tablename = 'works_referenced_works'
--           AND schemaname = 'openalex'
--           AND indexname NOT LIKE '%_pkey'
--     LOOP
--         RAISE NOTICE 'Dropping index % for faster import...', idx.indexname;
--         EXECUTE format('DROP INDEX IF EXISTS openalex.%I', idx.indexname);
--     END LOOP;
-- END $$;


\echo 'Converting imported tables to UNLOGGED for faster data import...'
-- ALTER TABLE openalex.topics SET UNLOGGED;
-- ALTER TABLE openalex.works SET UNLOGGED;
ALTER TABLE openalex.works_authorships SET UNLOGGED;
-- ALTER TABLE openalex.works_topics SET UNLOGGED;
-- ALTER TABLE openalex.works_referenced_works SET UNLOGGED;


\echo 'Starting data import...'
--authors

-- \copy openalex.authors (id, orcid, display_name, display_name_alternatives, works_count, cited_by_count, last_known_institution, works_api_url, updated_date) from program 'gzip -d -c csv-files/authors.csv.gz | grep -v "^$" | awk -F"," "NF==9" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.authors_ids (author_id, openalex, orcid, scopus, twitter, wikipedia, mag) from program 'gzip -d -c csv-files/authors_ids.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.authors_counts_by_year (author_id, year, works_count, cited_by_count, oa_works_count) from program 'gzip -d -c csv-files/authors_counts_by_year.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header

-- topics

-- \copy openalex.topics (id, display_name, subfield_id, subfield_display_name, field_id, field_display_name, domain_id, domain_display_name, description, keywords, works_api_url, wikipedia_id, works_count, cited_by_count, updated_date) from program 'pigz -d -c csv-files/topics.csv.gz | python3 -c "import sys, csv; reader = csv.reader(sys.stdin); writer = csv.writer(sys.stdout); writer.writerow(next(reader)[:15]); [writer.writerow(row[:15]) for row in reader]"' csv header
-- \echo 'Re-enabling autovacuum on imported table...'
-- ALTER TABLE openalex.topics SET (autovacuum_enabled = true);
-- \echo 'Converting imported table back to LOGGED...'
-- ALTER TABLE openalex.topics SET LOGGED;
-- \echo 'Performing VACUUM ANALYZE on imported table...'
-- VACUUM ANALYZE openalex.topics;

--concepts

-- \copy openalex.concepts (id, wikidata, display_name, level, description, works_count, cited_by_count, image_url, image_thumbnail_url, works_api_url, updated_date) from program 'gzip -d -c csv-files/concepts.csv.gz | grep -v "^$" | awk -F"," "NF==11" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.concepts_ancestors (concept_id, ancestor_id) from program 'gzip -d -c csv-files/concepts_ancestors.csv.gz | grep -v "^$" | awk -F"," "NF==2" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.concepts_counts_by_year (concept_id, year, works_count, cited_by_count, oa_works_count) from program 'gzip -d -c csv-files/concepts_counts_by_year.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.concepts_ids (concept_id, openalex, wikidata, wikipedia, umls_aui, umls_cui, mag) from program 'gzip -d -c csv-files/concepts_ids.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.concepts_related_concepts (concept_id, related_concept_id, score) from program 'gzip -d -c csv-files/concepts_related_concepts.csv.gz | grep -v "^$" | awk -F"," "NF==3" | iconv -f UTF-8 -t UTF-8 -c' csv header

--institutions

-- \copy openalex.institutions (id, ror, display_name, country_code, type, homepage_url, image_url, image_thumbnail_url, display_name_acronyms, display_name_alternatives, works_count, cited_by_count, works_api_url, updated_date) from program 'gzip -d -c csv-files/institutions.csv.gz | grep -v "^$" | awk -F"," "NF==14" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.institutions_ids (institution_id, openalex, ror, grid, wikipedia, wikidata, mag) from program 'gzip -d -c csv-files/institutions_ids.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.institutions_geo (institution_id, city, geonames_city_id, region, country_code, country, latitude, longitude) from program 'gzip -d -c csv-files/institutions_geo.csv.gz | grep -v "^$" | awk -F"," "NF==8" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.institutions_associated_institutions (institution_id, associated_institution_id, relationship) from program 'gzip -d -c csv-files/institutions_associated_institutions.csv.gz | grep -v "^$" | awk -F"," "NF==3" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.institutions_counts_by_year (institution_id, year, works_count, cited_by_count, oa_works_count) from program 'gzip -d -c csv-files/institutions_counts_by_year.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header

--publishers

-- \copy openalex.publishers (id, display_name, alternate_titles, country_codes, hierarchy_level, parent_publisher, works_count, cited_by_count, sources_api_url, updated_date) from program 'gzip -d -c csv-files/publishers.csv.gz | grep -v "^$" | awk -F"," "NF==10" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.publishers_ids (publisher_id, openalex, ror, wikidata) from program 'gzip -d -c csv-files/publishers_ids.csv.gz | grep -v "^$" | awk -F"," "NF==4" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.publishers_counts_by_year (publisher_id, year, works_count, cited_by_count, oa_works_count) from program 'gzip -d -c csv-files/publishers_counts_by_year.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header

--sources

-- \copy openalex.sources (id, issn_l, issn, display_name, publisher, works_count, cited_by_count, is_oa, is_in_doaj, homepage_url, works_api_url, updated_date) from program 'gzip -d -c csv-files/sources.csv.gz | grep -v "^$" | awk -F"," "NF==12" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.sources_ids (source_id, openalex, issn_l, issn, mag, wikidata, fatcat) from program 'gzip -d -c csv-files/sources_ids.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.sources_counts_by_year (source_id, year, works_count, cited_by_count, oa_works_count) from program 'gzip -d -c csv-files/sources_counts_by_year.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header

--works

-- \copy openalex.works (id, doi, title, display_name, publication_year, publication_date, type, cited_by_count, is_retracted, is_paratext, cited_by_api_url, abstract_inverted_index, language) from program 'gzip -d -c csv-files/works.csv.gz | grep -v "^$" | awk -F"," "NF==13" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works (id, doi, title, display_name, publication_year, publication_date, type, cited_by_count, is_retracted, is_paratext, cited_by_api_url, language) from program 'pigz -d -c csv-files/works.csv.gz | grep -v "^$" | python3 -c "import sys, csv; reader = csv.reader(sys.stdin); writer = csv.writer(sys.stdout); writer.writerow(next(reader)[:12]); [writer.writerow(row[:12]) for row in reader]"' csv header
-- \echo 'Re-enabling autovacuum on imported table...'
-- ALTER TABLE openalex.works SET (autovacuum_enabled = true);
-- \echo 'Converting imported table back to LOGGED...'
-- ALTER TABLE openalex.works SET LOGGED;
-- \echo 'Performing VACUUM ANALYZE on imported table...'
-- VACUUM ANALYZE openalex.works;
-- \copy openalex.works_primary_locations (work_id, source_id, landing_page_url, pdf_url, is_oa, version, license) from program 'gzip -d -c csv-files/works_primary_locations.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_locations (work_id, source_id, landing_page_url, pdf_url, is_oa, version, license) from program 'gzip -d -c csv-files/works_locations.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_best_oa_locations (work_id, source_id, landing_page_url, pdf_url, is_oa, version, license) from program 'gzip -d -c csv-files/works_best_oa_locations.csv.gz | grep -v "^$" | awk -F"," "NF==7" | iconv -f UTF-8 -t UTF-8 -c' csv header
\copy openalex.works_authorships (work_id, author_position, author_id, institution_id) from program 'pigz -d -c csv-files/works_authorships.csv.gz | python3 -c "import sys, csv; r = csv.reader(sys.stdin); next(r); w = csv.writer(sys.stdout); [w.writerow(x[:4]) for x in r]"' csv header
\echo 'Re-enabling autovacuum on imported table...'
ALTER TABLE openalex.works_authorships SET (autovacuum_enabled = true);
\echo 'Converting imported table back to LOGGED...'
ALTER TABLE openalex.works_authorships SET LOGGED;
\echo 'Performing VACUUM ANALYZE on imported table...'
VACUUM ANALYZE openalex.works_authorships;
-- \copy openalex.works_biblio (work_id, volume, issue, first_page, last_page) from program 'gzip -d -c csv-files/works_biblio.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_topics (work_id, topic_id, score) from program 'pigz -d -c csv-files/works_topics.csv.gz | grep -v "^$" | awk -F"," "NF==3" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_topics (work_id, topic_id, score) from program 'pigz -d -c csv-files/works_topics.csv.gz | grep -v "^$" | awk -F"," "NF==3"' csv header
-- \echo 'Re-enabling autovacuum on imported table...'
-- ALTER TABLE openalex.works_topics SET (autovacuum_enabled = true);
-- \echo 'Converting imported table back to LOGGED...'
-- ALTER TABLE openalex.works_topics SET LOGGED;
-- \echo 'Performing VACUUM ANALYZE on imported table...'
-- VACUUM ANALYZE openalex.works_topics;
-- \copy openalex.works_concepts (work_id, concept_id, score) from program 'gzip -d -c csv-files/works_concepts.csv.gz | grep -v "^$" | awk -F"," "NF==3" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_ids (work_id, openalex, doi, mag, pmid, pmcid) from program 'gzip -d -c csv-files/works_ids.csv.gz | grep -v "^$" | awk -F"," "NF==6" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_mesh (work_id, descriptor_ui, descriptor_name, qualifier_ui, qualifier_name, is_major_topic) from program 'gzip -d -c csv-files/works_mesh.csv.gz | grep -v "^$" | awk -F"," "NF==6" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_open_access (work_id, is_oa, oa_status, oa_url, any_repository_has_fulltext) from program 'gzip -d -c csv-files/works_open_access.csv.gz | grep -v "^$" | awk -F"," "NF==5" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_referenced_works (work_id, referenced_work_id) from program 'pigz -d -c csv-files/works_referenced_works.csv.gz | grep -v "^$" | awk -F"," "NF==2" | iconv -f UTF-8 -t UTF-8 -c' csv header
-- \copy openalex.works_referenced_works (work_id, referenced_work_id) from program 'pigz -d -c csv-files/works_referenced_works.csv.gz | grep -v "^$" | awk -F"," "NF==2"' csv header
-- \echo 'Re-enabling autovacuum on imported table...'
-- ALTER TABLE openalex.works_referenced_works SET (autovacuum_enabled = true);
-- \echo 'Converting imported table back to LOGGED...'
-- ALTER TABLE openalex.works_referenced_works SET LOGGED;
-- \echo 'Performing VACUUM ANALYZE on imported table...'
-- VACUUM ANALYZE openalex.works_referenced_works;
-- \copy openalex.works_related_works (work_id, related_work_id) from program 'gzip -d -c csv-files/works_related_works.csv.gz | grep -v "^$" | awk -F"," "NF==2" | iconv -f UTF-8 -t UTF-8 -c' csv header


-- ===================================================================
-- CRITICAL INDEXES FOR SELF-CITATION ANALYSIS PERFORMANCE
-- ===================================================================
\echo 'Creating indexes for optimized query performance...'
-- CREATE INDEX IF NOT EXISTS works_id_idx ON openalex.works (id);
CREATE INDEX IF NOT EXISTS idx_self_citation_detection ON openalex.works_authorships (author_id, work_id) WHERE author_id IS NOT NULL INCLUDE (author_position, institution_id);
-- CREATE INDEX IF NOT EXISTS works_referenced_works_reverse_idx ON openalex.works_referenced_works (referenced_work_id, work_id);
-- CREATE INDEX IF NOT EXISTS works_referenced_works_both_idx ON openalex.works_referenced_works (work_id, referenced_work_id);
-- CREATE INDEX IF NOT EXISTS works_referenced_works_work_id_idx ON openalex.works_referenced_works (work_id);
-- CREATE INDEX IF NOT EXISTS topics_field_display_name_idx ON openalex.topics (field_display_name) WHERE field_display_name IS NOT NULL;
-- CREATE INDEX IF NOT EXISTS works_topics_optimized_idx ON openalex.works_topics (topic_id, work_id);
-- CREATE INDEX IF NOT EXISTS works_topics_composite_idx ON openalex.works_topics (work_id, topic_id);
-- CREATE INDEX IF NOT EXISTS works_publication_year_idx ON openalex.works (publication_year) WHERE publication_year IS NOT NULL;
-- CREATE INDEX IF NOT EXISTS works_topics_work_id_idx ON openalex.works_topics (work_id);
-- CREATE INDEX IF NOT EXISTS works_cited_by_count_idx ON openalex.works (cited_by_count) WHERE cited_by_count IS NOT NULL;
CREATE INDEX IF NOT EXISTS works_authorships_work_id_idx ON openalex.works_authorships USING btree (work_id);
-- CREATE INDEX IF NOT EXISTS works_topics_author_discipline_idx ON openalex.works_topics (topic_id) INCLUDE (work_id);
-- CREATE INDEX IF NOT EXISTS topics_id_idx ON openalex.topics(id);
CREATE INDEX IF NOT EXISTS works_authorships_author_id_idx ON openalex.works_authorships(author_id) WHERE author_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS works_authorships_author_work_idx ON openalex.works_authorships(author_id, work_id) WHERE author_id IS NOT NULL;
-- CREATE INDEX IF NOT EXISTS topics_id_idx ON openalex.topics(id);
-- CREATE INDEX IF NOT EXISTS works_year_citations_idx ON openalex.works(publication_year, id)  INCLUDE (cited_by_count) WHERE publication_year IS NOT NULL AND publication_year >= 1970;


-- Show index creation progress
-- SELECT
--     psi.schemaname AS schema_name,
--     psi.relname AS table_name,
--     psi.indexrelname AS index_name,
--     pg_size_pretty(pg_relation_size(psi.indexrelid)) AS index_size,
--     psi.idx_scan AS number_of_scans,
--     psi.idx_tup_read AS tuples_read,
--     psi.idx_tup_fetch AS tuples_fetched,
--     CASE
--         WHEN pi.indisprimary THEN 'PRIMARY KEY'
--         WHEN pi.indisunique THEN 'UNIQUE'
--         ELSE 'INDEX'
--     END AS index_type,
--     pi.indisvalid AS is_valid,
--     pi.indisready AS is_ready,
--     pg_get_indexdef(psi.indexrelid) AS index_definition,
--     array_to_string(array_agg(pa.attname ORDER BY array_position(pi.indkey::integer[], pa.attnum)), ', ') AS indexed_columns
-- FROM pg_stat_user_indexes psi
-- JOIN pg_index pi ON psi.indexrelid = pi.indexrelid
-- LEFT JOIN pg_attribute pa ON pa.attrelid = psi.relid
--     AND pa.attnum = ANY(pi.indkey)
-- WHERE psi.schemaname NOT IN ('pg_catalog', 'information_schema')
-- GROUP BY
--     psi.schemaname,
--     psi.relname,
--     psi.indexrelname,
--     psi.indexrelid,
--     psi.idx_scan,
--     psi.idx_tup_read,
--     psi.idx_tup_fetch,
--     -- pi.indisprimary,
--     -- pi.indisunique,
--     -- pi.indisvalid,
--     -- pi.indisready,
--     pi.indkey
-- ORDER BY psi.schemaname, psi.relname, psi.indexrelname;


\echo 'Updating table statistics for query optimizer...'
-- ANALYZE openalex.topics;
-- ANALYZE openalex.works;
ANALYZE openalex.works_authorships;
-- ANALYZE openalex.works_topics;
-- ANALYZE openalex.works_referenced_works;


-- Show final disk space
SELECT pg_size_pretty(pg_database_size('openalex_db')) as total_database_size;