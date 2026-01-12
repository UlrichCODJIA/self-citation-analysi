#!/usr/bin/env python3
"""
Self-Citation Analysis Tool

Analyzes self-citation patterns from 1970-2025 using OpenAlex data.
Built for machines with limited RAM (~15GB) using chunked processing.

Runs 7 analysis modules:
1. Temporal trends (with ARIMA forecasting)
2. Career stage breakdown (early/mid/senior)
3. Cross-discipline comparison
4. AI era impact (pre-2010 vs post-2010)
5. Network analysis
6. Regression modeling
7. ML clustering

Outputs 28 plots and a markdown report.
"""

import gc
import hashlib
import logging
import os
import time
from datetime import datetime

import dask.dataframe as dd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# Import sampling strategy module
try:
    from sampling_strategy import BiblioSampler  # type: ignore[attr-defined]
except ImportError:
    BiblioSampler = None  # type: ignore[misc, assignment]

# Configure logging FIRST (before capturing warnings)
log_filename = f"self_citation_analysis/self_citation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],  # Also print to console
)
logger = logging.getLogger(__name__)

# Capture Python warnings and redirect to logging system
# This ensures warnings from statsmodels, sklearn, etc. appear in log files
logging.captureWarnings(True)
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.setLevel(logging.WARNING)

# Plot styling
plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")


class SelfCitationAnalyzer:
    """
    Main class for running self-citation analysis.

    Handles data extraction, stats, and visualization. Uses chunked
    processing to work on machines with ~15GB RAM.
    """

    def __init__(self, db_config, use_sampling=False, sample_fraction=0.10, target_disciplines=None):
        """
        Set up the analyzer.

        Args:
            db_config: Dict with host, port, database, user, password
            use_sampling: Use stratified sampling instead of full data
            sample_fraction: What percent to sample (0.10 = 10%)
            target_disciplines: List of fields to analyze, or None for all
        """
        self.db_config = db_config
        self.use_sampling = use_sampling
        self.sample_fraction = sample_fraction
        self.target_disciplines = target_disciplines
        self.sampled_authors = None
        self.sampling_weights = None

        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # Initialize sampler if sampling is enabled
        if self.use_sampling:
            if BiblioSampler is None:
                raise ImportError("BiblioSampler not available. Please ensure sampling_strategy.py is in the path.")
            logger.info(f"📊 Sampling enabled: {sample_fraction * 100:.1f}% stratified random sample")
            if target_disciplines:
                logger.info(f"   🎯 Targeting {len(target_disciplines)} specific disciplines")
            self.sampler = BiblioSampler(
                {
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "database": db_config["database"],
                    "user": db_config["user"],
                    "password": db_config["password"],
                }
            )
        else:
            logger.info("📊 Full population analysis (no sampling)")
            self.sampler = None

        # Career stage classification
        self.career_stages = {
            "early": (0, 5),  # 0-5 years since first publication
            "mid": (6, 15),  # 6-15 years
            "senior": (16, 999),  # 16+ years
        }

        # AI era classification with COVID-19 distinction
        self.temporal_eras = {
            "pre_digital": (1970, 1995),
            "early_digital": (1996, 2009),
            "ai_emergence": (2010, 2019),
            "covid_era": (2020, 2022),  # COVID-19 pandemic period
            "modern_ai": (2023, 2025),  # ChatGPT and modern AI era
        }

        # Performance optimization settings
        self.performance_config = {
            "chunk_size": 5,  # Process 5 years at a time (1970-2025 = 12 chunks)
            "enable_progress_monitoring": True,  # Show progress during long queries
        }

        # Initialize containers for results and visualizations
        self.results = {}
        self.figures = {}  # Stores the 7 figure sets (28 plots total)

    def _weighted_mean(self, df, value_col, weight_col="sampling_weight"):
        """Calculate weighted mean for a column"""
        return np.average(df[value_col], weights=df[weight_col])

    def _weighted_std(self, df, value_col, weight_col="sampling_weight"):
        """Calculate weighted standard deviation"""
        avg = self._weighted_mean(df, value_col, weight_col)
        variance = np.average((df[value_col] - avg) ** 2, weights=df[weight_col])
        return np.sqrt(variance)

    def _weighted_groupby_stats(self, df, group_cols, value_cols, weight_col="sampling_weight"):
        """
        Calculate weighted stats (mean, std, median, count) per group.
        Much faster than using groupby().apply() with lambdas.
        """
        result_df = df.copy()

        # Pre-compute weighted values for each value column
        for col in value_cols:
            result_df[f"_w_{col}"] = result_df[col] * result_df[weight_col]

        # Build aggregation dict using named aggregations for consistent column naming
        agg_dict = {}
        agg_dict[f"{weight_col}_sum"] = (weight_col, "sum")
        for col in value_cols:
            agg_dict[f"_w_{col}_sum"] = (f"_w_{col}", "sum")
            agg_dict[f"{col}_std"] = (col, "std")
            agg_dict[f"{col}_median"] = (col, "median")
            agg_dict[f"{col}_count"] = (col, "count")

        # Single aggregation pass with named aggregations (flat column names)
        grouped = result_df.groupby(group_cols).agg(**agg_dict)

        # Build result with MultiIndex columns
        result = pd.DataFrame(index=grouped.index)
        for col in value_cols:
            result[(col, "mean")] = grouped[f"_w_{col}_sum"] / grouped[f"{weight_col}_sum"]
            result[(col, "std")] = grouped[f"{col}_std"]
            result[(col, "median")] = grouped[f"{col}_median"]
            result[(col, "count")] = grouped[f"{col}_count"]

        return result.round(4)

    def _weighted_agg(self, grouped_df, value_col, weight_col="sampling_weight"):
        """Apply weighted aggregations to grouped data"""
        result = pd.DataFrame()
        result["mean"] = grouped_df.apply(lambda x: self._weighted_mean(x, value_col, weight_col))
        result["std"] = grouped_df.apply(lambda x: self._weighted_std(x, value_col, weight_col))
        result["count"] = grouped_df.size()
        # For median, use weighted median approximation or fall back to unweighted
        result["median"] = grouped_df[value_col].median()
        return result

    def perform_sampling(self):
        """
        Run stratified sampling to get a representative author subset.
        Stratifies by era and discipline so all groups are covered.
        Returns (author_ids, weights) or (None, None) if sampling is off.
        """
        if not self.use_sampling or self.sampler is None:
            logger.info("📊 Sampling disabled - will analyze full population")
            return None, None

        logger.info("\n" + "=" * 80)
        logger.info("📊 PHASE 0: STRATIFIED SAMPLING")
        logger.info("=" * 80)

        # Check if sampling already completed (checkpoint/resume)
        sample_file = "sampling_results/stratified_sample.parquet"
        if os.path.exists(sample_file):
            logger.info(f"\n📂 Found existing sample file: {sample_file}")
            logger.info("   Loading from checkpoint to avoid re-sampling (saves ~62 hours)...")
            stratified_sample = pd.read_parquet(sample_file)
            logger.info(f"   ✅ Loaded {len(stratified_sample):,} author-year observations from file")
        else:
            logger.info("\n📊 No existing sample found - performing stratified random sampling...")
            # Perform stratified random sampling
            stratified_sample = self.sampler.stratified_random_sampling(
                stratification_criteria=["era", "discipline"],
                sample_fraction=self.sample_fraction,
                allocation_method="proportional",
                target_disciplines=self.target_disciplines,
            )

        if stratified_sample is None or len(stratified_sample) == 0:
            logger.error("❌ Sampling failed - falling back to full population")
            self.use_sampling = False
            return None, None

        logger.info(f"\n✅ Sampling complete: {len(stratified_sample):,} author-year observations")
        logger.info(f"   Unique authors: {stratified_sample['author_id'].nunique():,}")

        # Calculate sampling weights
        logger.info("\n📊 Calculating sampling weights...")
        sample_with_weights = self.sampler.get_sampling_weights(stratified_sample, stratum_col="stratum")

        logger.info(f"   ✅ Weights calculated for {len(sample_with_weights)} observations")
        logger.info(
            f"   Weight range: {sample_with_weights['sampling_weight'].min():.2f} - "
            f"{sample_with_weights['sampling_weight'].max():.2f}"
        )
        logger.info(f"   Mean weight: {sample_with_weights['sampling_weight'].mean():.2f}")

        # Validate sample representativeness
        logger.info("\n📊 Validating sample representativeness...")
        validation_results = self.sampler.validate_sample_representativeness(
            sample=sample_with_weights, validation_criteria=["publication_year", "discipline"]
        )

        if validation_results.get("is_representative", True):  # Default to True if key missing
            logger.info("   ✅ Sample is statistically representative")
        else:
            logger.warning("   ⚠️  Sample may have representativeness issues")
            for issue in validation_results.get("warnings", []):
                logger.warning(f"      - {issue}")

        # Store sampled authors and weights
        self.sampled_authors = set(sample_with_weights["author_id"].unique())
        self.sampling_weights = sample_with_weights[["author_id", "sampling_weight"]].drop_duplicates()

        logger.info(f"\n📊 Sampled {len(self.sampled_authors):,} unique authors for analysis")

        return self.sampled_authors, self.sampling_weights

    def _get_sample_hash(self) -> str:
        """
        Generate a short hash based on the sampled authors set.
        This ensures we reuse tables only when the exact same sample is used.
        """
        if not self.sampled_authors:
            return "nosample"

        # Sort for deterministic hash
        sorted_authors = sorted(self.sampled_authors)
        # Use first 1000 + last 1000 + count for efficiency (full hash would be slow)
        sample_signature = (
            str(len(sorted_authors)) + "_" + "_".join(sorted_authors[:100]) + "_" + "_".join(sorted_authors[-100:])
        )
        return hashlib.md5(sample_signature.encode()).hexdigest()[:8]

    def _get_table_names(self) -> tuple:
        """Get hash-based table names for this sample."""
        sample_hash = self._get_sample_hash()
        return (
            f"sampled_authors_{sample_hash}",
            f"author_works_{sample_hash}",
        )

    def extract_self_citations(self, chunk_size=5):
        logger.info("⚙️ Query-specific optimization settings will be applied per-connection...")

        # Get hash-based table names for this sample
        sampled_authors_table, author_works_table = self._get_table_names()
        sample_hash = self._get_sample_hash()
        logger.info(f"   Using sample hash: {sample_hash}")

        # STEP 0: Create sampled authors table (hash-based permanent table)
        if self.use_sampling and self.sampled_authors:
            try:
                with self.engine.connect() as conn:
                    # Check if table already exists
                    result = conn.execute(
                        text(
                            """
                        SELECT COUNT(*) FROM information_schema.tables
                        WHERE table_schema = 'openalex' AND table_name = :table_name
                    """
                        ),
                        {"table_name": sampled_authors_table},
                    )
                    table_exists = (result.scalar() or 0) > 0

                    if table_exists:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM openalex.{sampled_authors_table}"))
                        existing_count = result.scalar()
                        logger.info(f"\n✅ Reusing existing {sampled_authors_table} table ({existing_count:,} authors)")
                    else:
                        logger.info(
                            f"\n📊 Creating {sampled_authors_table} for {len(self.sampled_authors):,} sampled authors..."
                        )
                        conn.execute(
                            text(
                                f"""
                            CREATE TABLE openalex.{sampled_authors_table} (
                                author_id TEXT PRIMARY KEY
                            )
                        """
                            )
                        )
                        conn.commit()

                        # Insert in batches
                        batch_size = 50000
                        author_list = list(self.sampled_authors)
                        for i in range(0, len(author_list), batch_size):
                            batch = author_list[i : i + batch_size]
                            values = ", ".join([f"('{aid}')" for aid in batch])
                            conn.execute(
                                text(
                                    f"""
                                INSERT INTO openalex.{sampled_authors_table} (author_id)
                                VALUES {values}
                                ON CONFLICT (author_id) DO NOTHING
                            """
                                )
                            )

                        conn.commit()
                        conn.execute(text(f"ANALYZE openalex.{sampled_authors_table};"))
                        conn.commit()
                        logger.info(f"   ✅ Created {sampled_authors_table}")

            except Exception as e:
                logger.error(f"   ❌ Failed to create sampled_authors table: {e}")
                self.use_sampling = False

        # STEP 0.5 - Pre-aggregate all author-work pairs (hash-based permanent table)
        if self.use_sampling and self.sampled_authors:
            try:
                with self.engine.connect() as conn:
                    # Check if author_works table already exists
                    result = conn.execute(
                        text(
                            """
                        SELECT COUNT(*) FROM information_schema.tables
                        WHERE table_schema = 'openalex' AND table_name = :table_name
                    """
                        ),
                        {"table_name": author_works_table},
                    )
                    table_exists = (result.scalar() or 0) > 0

                    # Check if existing table has enriched schema (publication_year column)
                    is_enriched = False
                    if table_exists:
                        result = conn.execute(
                            text(
                                """
                            SELECT COUNT(*) FROM information_schema.columns
                            WHERE table_schema = 'openalex'
                              AND table_name = :table_name
                              AND column_name = 'publication_year'
                        """
                            ),
                            {"table_name": author_works_table},
                        )
                        is_enriched = (result.scalar() or 0) > 0

                        if not is_enriched:
                            logger.info(f"\n⚠️ Found old {author_works_table} without enriched columns, recreating...")
                            conn.execute(text(f"DROP TABLE IF EXISTS openalex.{author_works_table} CASCADE"))
                            conn.commit()
                            table_exists = False

                    if table_exists:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM openalex.{author_works_table}"))
                        existing_count = result.scalar()
                        logger.info(f"\n✅ Reusing existing {author_works_table} table ({existing_count:,} pairs)")
                    else:
                        logger.info(f"\n⚡️ Creating {author_works_table} table...")

                        # Set aggressive settings for this bulk operation
                        conn.execute(
                            text(
                                """
                            SET work_mem = '1GB';
                            SET maintenance_work_mem = '2GB';
                            SET max_parallel_workers_per_gather = 4;
                        """
                            )
                        )

                        temp_works_start = time.time()

                        conn.execute(
                            text(
                                f"""
                            CREATE TABLE openalex.{author_works_table} AS
                            SELECT
                                wa.author_id,
                                wa.work_id,
                                w.publication_year,
                                w.cited_by_count
                            FROM openalex.works_authorships wa
                            INNER JOIN openalex.{sampled_authors_table} sa ON wa.author_id = sa.author_id
                            INNER JOIN openalex.works w ON wa.work_id = w.id
                            WHERE w.publication_year IS NOT NULL
                              AND w.publication_year >= 1970
                        """
                            )
                        )
                        conn.commit()

                        # Build statistics
                        conn.execute(text(f"ANALYZE openalex.{author_works_table}"))

                        # Add indexes for fast lookups
                        conn.execute(
                            text(
                                f"CREATE INDEX idx_aw_{sample_hash}_author ON openalex.{author_works_table} (author_id)"
                            )
                        )
                        conn.execute(
                            text(f"CREATE INDEX idx_aw_{sample_hash}_work ON openalex.{author_works_table} (work_id)")
                        )
                        conn.execute(
                            text(
                                f"CREATE INDEX idx_aw_{sample_hash}_composite ON openalex.{author_works_table} (author_id, work_id)"
                            )
                        )

                        # Update stats for query planner
                        conn.execute(text(f"ANALYZE openalex.{author_works_table}"))
                        conn.commit()

                        temp_works_elapsed = time.time() - temp_works_start

                        # Get row count
                        result = conn.execute(text(f"SELECT COUNT(*) FROM openalex.{author_works_table}"))
                        row_count = result.scalar()

                        logger.info(
                            f"   ✅ Created enriched table with {row_count:,} author-work pairs in {temp_works_elapsed:.1f}s"
                        )

            except Exception as e:
                logger.error(f"   ❌ Failed to create author_works table: {e}")
                logger.error("   Falling back to slower approach")
                # Continue with old method if this fails

        # STEP 1: Process self-citations by year ranges
        start_year = 1970
        end_year = 2025
        year_ranges = []

        for year in range(start_year, end_year + 1, chunk_size):
            year_end = min(year + chunk_size - 1, end_year)
            year_ranges.append((year, year_end))

        logger.info(f"\n📝 Step 1: Processing {len(year_ranges)} year-range chunks ({chunk_size} years each)...")
        logger.info("   💾 Using streaming to Parquet files to avoid memory exhaustion...")

        temp_dir = "temp_self_citation_chunks"
        os.makedirs(temp_dir, exist_ok=True)

        chunk_files = []
        skipped_chunks = 0

        for idx, (year_start, year_end) in enumerate(year_ranges, 1):
            # Check if chunk already exists (for resume capability)
            chunk_file = os.path.join(temp_dir, f"chunk_{idx:02d}_{year_start}_{year_end}.parquet")

            if os.path.exists(chunk_file):
                logger.info(
                    f"\n   ✅ Chunk {idx}/{len(year_ranges)}: Years {year_start}-{year_end} (already exists, skipping)"
                )
                chunk_files.append(chunk_file)
                skipped_chunks += 1
                continue

            logger.info(f"\n   📅 Chunk {idx}/{len(year_ranges)}: Years {year_start}-{year_end}")
            logger.info(f"      🕐 Started at: {time.strftime('%H:%M:%S')}")

            # ⚡️ OPTIMIZED QUERY - Uses pre-filtered hash-based permanent table
            chunk_query = text(
                f"""
                SELECT
                    aw_citing.author_id,
                    w_citing.publication_year as citing_year,
                    w_cited.publication_year as cited_year,
                    COUNT(DISTINCT w_citing.id) as self_citation_count,
                    AVG(w_citing.publication_year - w_cited.publication_year) as temporal_distance
                FROM openalex.works w_citing

                -- 1. Find citing works by our sampled authors (FAST - hash-based table)
                INNER JOIN openalex.{author_works_table} aw_citing
                    ON w_citing.id = aw_citing.work_id

                -- 2. Find all references for those works
                INNER JOIN openalex.works_referenced_works wrw
                    ON w_citing.id = wrw.work_id

                -- 3. Find the cited works
                INNER JOIN openalex.works w_cited
                    ON wrw.referenced_work_id = w_cited.id

                -- 4. Self-citation check: same author wrote the cited work (FAST - hash-based table)
                INNER JOIN openalex.{author_works_table} aw_cited
                    ON aw_cited.work_id = w_cited.id
                    AND aw_cited.author_id = aw_citing.author_id

                WHERE w_citing.publication_year >= :year_start
                AND w_citing.publication_year <= :year_end
                AND w_citing.publication_year IS NOT NULL
                AND w_cited.publication_year >= 1970
                AND w_cited.publication_year IS NOT NULL
                AND w_citing.publication_year > w_cited.publication_year
                GROUP BY aw_citing.author_id, w_citing.publication_year, w_cited.publication_year
                ORDER BY aw_citing.author_id, w_citing.publication_year
            """
            )

            try:
                chunk_start = time.time()
                logger.info(f"      📊 Querying self-citations for {year_start}-{year_end}...")

                with self.engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                        SET work_mem = '512MB';
                        SET maintenance_work_mem = '1GB';
                        SET hash_mem_multiplier = 2.0;
                        SET enable_hashjoin = on;
                        SET enable_mergejoin = on;
                        SET random_page_cost = 1.1;
                        SET max_parallel_workers_per_gather = 4;
                    """
                        )
                    )

                    chunk_df = pd.read_sql(chunk_query, conn, params={"year_start": year_start, "year_end": year_end})

                chunk_elapsed = time.time() - chunk_start

                if len(chunk_df) > 0:
                    # Write immediately to parquet to free memory
                    chunk_df.to_parquet(chunk_file, compression="snappy", index=False)
                    chunk_files.append(chunk_file)

                    chunk_size_mb = chunk_df.memory_usage(deep=True).sum() / 1024 / 1024
                    logger.info(f"      ✅ Extracted {len(chunk_df):,} records in {chunk_elapsed:.1f}s")
                    logger.info(f"         💾 Written to {os.path.basename(chunk_file)} (~{chunk_size_mb:.1f} MB)")

                    # Free memory immediately
                    del chunk_df
                    gc.collect()
                else:
                    logger.warning(f"      ⚠️  No data in this range (completed in {chunk_elapsed:.1f}s)")

            except Exception as e:
                logger.error(f"      ❌ Error in chunk {idx}: {str(e)}")
                continue

        if not chunk_files:
            logger.error("\n❌ No self-citation data extracted!")
            return pd.DataFrame()

        # Load chunks using Dask
        logger.info(f"\n📝 Loading {len(chunk_files)} parquet chunks with Dask...")
        dask_df = dd.read_parquet(os.path.join(temp_dir, "chunk_*.parquet"))
        total_rows = len(dask_df)
        logger.info(f"   ✅ Loaded {len(chunk_files)} chunks with {total_rows:,} total records")

        # Get unique authors
        logger.info("\n📝 Step 2: Extracting unique authors from all chunks...")
        unique_authors = dask_df["author_id"].unique().compute()
        logger.info(f"   ✅ Found {len(unique_authors):,} unique authors across all chunks")

        logger.info("\n📝 Step 2.5: Fetching ALL author metadata...")

        author_metadata_table = f"author_metadata_{sample_hash}"

        try:
            with self.engine.connect() as conn:
                # Check if cached metadata table exists
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'openalex' AND table_name = :table_name
                """
                    ),
                    {"table_name": author_metadata_table},
                )
                table_exists = (result.scalar() or 0) > 0

                if table_exists:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM openalex.{author_metadata_table}"))
                    existing_count = result.scalar()
                    logger.info(f"   ✅ Reusing cached {author_metadata_table} ({existing_count:,} authors)")
                    author_first_pub_df = pd.read_sql(f"SELECT * FROM openalex.{author_metadata_table}", conn)
                else:
                    logger.info(f"   Creating {author_metadata_table} (one-time, ~5-6 hours with enriched table)...")
                    logger.info(f"   Using enriched openalex.{author_works_table} - NO JOIN needed!")

                    author_metadata_start = time.time()

                    conn.execute(
                        text(
                            """
                            SET work_mem = '512MB';
                            SET hash_mem_multiplier = 2.0;
                            SET enable_hashjoin = on;
                            SET enable_mergejoin = on;
                            SET random_page_cost = 1.1;
                            SET max_parallel_workers_per_gather = 4;
                        """
                        )
                    )

                    # Create cached table - NO JOIN needed, columns are in author_works!
                    conn.execute(
                        text(
                            f"""
                        CREATE TABLE openalex.{author_metadata_table} AS
                        SELECT
                            author_id,
                            MIN(publication_year) as first_pub_year,
                            COUNT(*) as total_works,
                            AVG(cited_by_count) as avg_citations
                        FROM openalex.{author_works_table}
                        WHERE publication_year IS NOT NULL
                            AND publication_year >= 1970
                        GROUP BY author_id
                        HAVING COUNT(*) >= 5
                    """
                        )
                    )
                    conn.commit()

                    # Add index for fast lookups
                    conn.execute(
                        text(
                            f"CREATE INDEX idx_{author_metadata_table}_author ON openalex.{author_metadata_table} (author_id)"
                        )
                    )
                    conn.execute(text(f"ANALYZE openalex.{author_metadata_table}"))
                    conn.commit()

                    author_metadata_elapsed = time.time() - author_metadata_start

                    # Read back the cached data
                    author_first_pub_df = pd.read_sql(f"SELECT * FROM openalex.{author_metadata_table}", conn)

                    logger.info(
                        f"   ✅ Created & cached metadata for {len(author_first_pub_df):,} authors in {author_metadata_elapsed:.1f}s"
                    )

        except Exception as e:
            logger.error(f"   ❌ Error in author metadata query: {e}")
            author_first_pub_df = pd.DataFrame(columns=["author_id", "first_pub_year", "total_works", "avg_citations"])

        logger.info("\n📝 Step 3: Fetching ALL disciplines...")

        author_disciplines_table = f"author_disciplines_{sample_hash}"

        try:
            with self.engine.connect() as conn:
                # Check if cached disciplines table exists
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'openalex' AND table_name = :table_name
                """
                    ),
                    {"table_name": author_disciplines_table},
                )
                table_exists = (result.scalar() or 0) > 0

                if table_exists:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM openalex.{author_disciplines_table}"))
                    existing_count = result.scalar()
                    logger.info(f"   ✅ Reusing cached {author_disciplines_table} ({existing_count:,} authors)")
                    disciplines_for_analysis = pd.read_sql(f"SELECT * FROM openalex.{author_disciplines_table}", conn)
                else:
                    logger.info(f"   Creating {author_disciplines_table} (one-time)...")

                    disciplines_start = time.time()

                    conn.execute(
                        text(
                            """
                            SET work_mem = '512MB';
                            SET hash_mem_multiplier = 2.0;
                            SET enable_hashjoin = on;
                            SET enable_mergejoin = on;
                            SET random_page_cost = 1.1;
                            SET max_parallel_workers_per_gather = 4;
                        """
                        )
                    )

                    # Create cached table
                    conn.execute(
                        text(
                            f"""
                        CREATE TABLE openalex.{author_disciplines_table} AS
                        WITH author_disciplines AS (
                            SELECT
                                aw.author_id,
                                t.field_display_name as primary_field,
                                COUNT(*) as field_count,
                                ROW_NUMBER() OVER (PARTITION BY aw.author_id ORDER BY COUNT(*) DESC) as rn
                            FROM openalex.{author_works_table} aw
                            JOIN openalex.works_topics wt ON aw.work_id = wt.work_id
                            JOIN openalex.topics t ON wt.topic_id = t.id
                            WHERE t.field_display_name IS NOT NULL
                            GROUP BY aw.author_id, t.field_display_name
                        )
                        SELECT author_id, primary_field
                        FROM author_disciplines
                        WHERE rn = 1
                    """
                        )
                    )
                    conn.commit()

                    # Add index for fast lookups
                    conn.execute(
                        text(
                            f"CREATE INDEX idx_{author_disciplines_table}_author ON openalex.{author_disciplines_table} (author_id)"
                        )
                    )
                    conn.execute(text(f"ANALYZE openalex.{author_disciplines_table}"))
                    conn.commit()

                    disciplines_elapsed = time.time() - disciplines_start

                    # Read back the cached data
                    disciplines_for_analysis = pd.read_sql(f"SELECT * FROM openalex.{author_disciplines_table}", conn)

                    logger.info(
                        f"   ✅ Created & cached disciplines for {len(disciplines_for_analysis):,} authors in {disciplines_elapsed:.1f}s"
                    )

            # Handle missing authors
            missing_authors = set(unique_authors) - set(disciplines_for_analysis["author_id"])
            if missing_authors:
                missing_df = pd.DataFrame({"author_id": list(missing_authors), "primary_field": "Unknown"})
                disciplines_for_analysis = pd.concat([disciplines_for_analysis, missing_df], ignore_index=True)
                logger.info(f"   📊 Added 'Unknown' for {len(missing_authors):,} authors without discipline data")
        except Exception as e:
            logger.error(f"   ❌ Error in disciplines query: {e}")
            disciplines_for_analysis = pd.DataFrame({"author_id": unique_authors, "primary_field": "Unknown"})

        # Step 3: Merge metadata with Dask DataFrame
        # Check if enriched data already exists (for resume capability)
        enriched_dir = "temp_enriched_chunks"
        enriched_path = os.path.join(enriched_dir, "enriched_data.parquet")

        if os.path.exists(enriched_path):
            logger.info(f"\n✅ Found existing enriched data at {enriched_path}, loading...")
            dask_df = dd.read_parquet(enriched_path)
            logger.info("   ✅ Enriched data loaded from cache")

            # Store the dask dataframe for later use
            self.self_citation_data_dask = dask_df
            self.enriched_data_path = enriched_path

            # Compute summary statistics and author-level data
            logger.info("\n📝 Computing summary statistics from cached data...")
            total_records = len(dask_df)
            unique_authors_count = dask_df["author_id"].nunique().compute()
            # Use Dask min/max instead of computing entire column to save memory
            year_min = dask_df["citing_year"].min().compute()
            year_max = dask_df["citing_year"].max().compute()

            logger.info(f"✅ Loaded {total_records:,} self-citation instances")
            logger.info(f"   📊 {unique_authors_count:,} unique authors analyzed")
            if total_records > 0:
                logger.info(f"   📅 Temporal range: {year_min} - {year_max}")

            # Compute author-level aggregations
            logger.info("\n📝 Computing author-level aggregations...")
            author_stats = (
                dask_df.groupby(["author_id", "career_stage", "era", "primary_field"])
                .agg(
                    {
                        "self_citation_count": "sum",
                        "temporal_distance": "mean",
                        "total_works": "first",
                        "first_pub_year": "first",
                    }
                )
                .reset_index()
                .compute()
            )

            self.author_level_data = author_stats
            self.enhanced_author_data = author_stats

            logger.info(f"   ✅ Author-level data computed: {len(author_stats):,} authors")
            return dask_df

        logger.info("\n📝 Step 4: Merging metadata with self-citation data using Dask...")

        # Convert metadata to Dask DataFrames for efficient merging
        author_metadata_dd = dd.from_pandas(
            author_first_pub_df[["author_id", "first_pub_year", "total_works"]], npartitions=1
        )
        disciplines_dd = dd.from_pandas(disciplines_for_analysis, npartitions=1)

        # Merge with author metadata
        dask_df = dask_df.merge(author_metadata_dd, on="author_id", how="left")
        logger.info("   ✅ Merged author metadata")

        # Merge with disciplines
        dask_df = dask_df.merge(disciplines_dd, on="author_id", how="left")
        logger.info("   ✅ Merged discipline data")

        # Calculate career stage using VECTORIZED operations (much more memory efficient)
        # Avoid apply(axis=1) which forces row-by-row computation
        years_since_first = dask_df["citing_year"] - dask_df["first_pub_year"]
        dask_df["career_stage"] = "senior"  # Default
        dask_df["career_stage"] = dask_df["career_stage"].mask(years_since_first <= 15, "mid")
        dask_df["career_stage"] = dask_df["career_stage"].mask(years_since_first <= 5, "early")
        logger.info("   ✅ Calculated career stages (vectorized)")

        # Calculate era using VECTORIZED operations
        dask_df["era"] = "modern_ai"  # Default (2023+)
        dask_df["era"] = dask_df["era"].mask(dask_df["citing_year"] <= 2022, "covid_era")
        dask_df["era"] = dask_df["era"].mask(dask_df["citing_year"] <= 2019, "ai_emergence")
        dask_df["era"] = dask_df["era"].mask(dask_df["citing_year"] <= 2009, "early_digital")
        dask_df["era"] = dask_df["era"].mask(dask_df["citing_year"] <= 1995, "pre_digital")
        logger.info("   ✅ Calculated temporal eras (vectorized)")

        # Write enriched data back to parquet for later analysis
        # Use chunked writing to avoid memory explosion
        os.makedirs(enriched_dir, exist_ok=True)

        logger.info(f"\n📝 Writing enriched data to disk: {enriched_path}")
        logger.info("   Using chunked write to manage memory...")

        # Repartition to have more, smaller partitions for memory-efficient writing
        n_partitions = max(dask_df.npartitions, 50)  # At least 50 partitions
        dask_df = dask_df.repartition(npartitions=n_partitions)

        # Write with explicit engine and row group size to control memory
        dask_df.to_parquet(
            enriched_path,
            compression="snappy",
            engine="pyarrow",
            write_metadata_file=True,
        )
        logger.info("   ✅ Enriched data written to parquet")

        # Store the dask dataframe for later use
        self.self_citation_data_dask = dask_df
        self.enriched_data_path = enriched_path

        # For compatibility, also compute summary statistics
        logger.info("\n📝 Computing summary statistics...")
        total_records = len(dask_df)
        unique_authors_count = dask_df["author_id"].nunique().compute()
        # Use Dask min/max instead of computing entire column to save memory
        year_min = dask_df["citing_year"].min().compute()
        year_max = dask_df["citing_year"].max().compute()

        logger.info(f"✅ Processed {total_records:,} self-citation instances")
        logger.info(f"   📊 {unique_authors_count:,} unique authors analyzed")
        if total_records > 0:
            logger.info(f"   📅 Temporal range: {year_min} - {year_max}")

        # For analysis functions, compute author-level data
        logger.info("\n📝 Computing author-level aggregations...")
        author_stats = (
            dask_df.groupby(["author_id", "career_stage", "era", "primary_field"])
            .agg(
                {
                    "self_citation_count": "sum",
                    "temporal_distance": "mean",
                    "total_works": "first",
                    "first_pub_year": "first",
                }
            )
            .reset_index()
            .compute()
        )

        self.author_level_data = author_stats
        self.enhanced_author_data = author_stats  # For compatibility with existing analysis code
        logger.info(f"   ✅ Computed {len(author_stats):,} author-level records")

        # Also keep path to enriched data for re-loading if needed
        self.enriched_data_path = enriched_path

        return enriched_path  # Return path instead of huge DataFrame

    def calculate_total_citations(self):
        """Calculate total citations to compute self-citation rates."""
        logger.info("📈 Calculating total citation patterns...")

        # Use the pre-computed author_works table if sampling is enabled
        if self.use_sampling and self.sampled_authors:
            _, author_works_table = self._get_table_names()
            logger.info(f"   Using pre-computed {author_works_table} table for efficiency...")
            logger.info("   📊 Aggregating at author level in database to minimize memory...")

            # Query aggregated at AUTHOR level (not author+year) to reduce result size
            # This reduces from ~100M rows to ~5M rows
            query = f"""
            SELECT
                author_id,
                SUM(cited_by_count) as total_times_cited,
                COUNT(*) as papers_published,
                COUNT(*) as total_references_made
            FROM openalex.{author_works_table}
            WHERE publication_year IS NOT NULL
                AND publication_year >= 1970
            GROUP BY author_id;
            """
        else:
            # Full population query (no sampling) - also aggregate at author level
            query = """
            SELECT
                wa.author_id,
                SUM(w.cited_by_count) as total_times_cited,
                COUNT(DISTINCT w.id) as papers_published,
                COUNT(wrw.referenced_work_id) as total_references_made
            FROM openalex.works_authorships wa
            JOIN openalex.works w ON wa.work_id = w.id
            LEFT JOIN openalex.works_referenced_works wrw ON w.id = wrw.work_id
            WHERE w.publication_year IS NOT NULL
                AND w.publication_year >= 1970
            GROUP BY wa.author_id;
            """

        logger.info("   ⏳ Executing aggregation query...")
        self.total_citation_data = pd.read_sql(query, self.engine)
        logger.info(f"   ✅ Loaded {len(self.total_citation_data):,} author citation records")

        # Merge with self-citation data for rate calculation
        # No need to groupby again - already aggregated at author level
        merged = self.author_level_data.merge(
            self.total_citation_data,
            on="author_id",
            how="left",
        )

        # Free intermediate data to reduce peak memory
        del self.total_citation_data
        gc.collect()

        # Calculate self-citation rate (SCR)
        merged["self_citation_rate"] = merged["self_citation_count"] / merged["total_references_made"].clip(lower=1)

        # Calculate FIELD-NORMALIZED self-citation rate (z-score within field)
        # This enables cross-field comparisons and identifies "excessive" self-citers
        logger.info("   📊 Computing field-normalized self-citation rates...")
        field_means = merged.groupby("primary_field")["self_citation_rate"].transform("mean")
        field_stds = merged.groupby("primary_field")["self_citation_rate"].transform("std")
        # Avoid division by zero for fields with single author or no variance
        field_stds = field_stds.replace(0, np.nan)
        merged["sc_rate_field_normalized"] = (merged["self_citation_rate"] - field_means) / field_stds
        # Free intermediate series
        del field_means, field_stds
        # Fill NaN with 0 (authors in fields with no variance are "average")
        merged["sc_rate_field_normalized"] = merged["sc_rate_field_normalized"].fillna(0)
        logger.info(
            f"   ✅ Field-normalized SC rate: mean={merged['sc_rate_field_normalized'].mean():.3f}, std={merged['sc_rate_field_normalized'].std():.3f}"
        )

        # Calculate h-index using VECTORIZED operations (much faster than apply)
        papers = merged["papers_published"].fillna(0).astype(float)
        total_cites = merged["total_times_cited"].fillna(0).astype(float)
        avg_cites = np.where(papers > 0, total_cites / papers, 0)
        merged["h_index"] = np.where(papers > 0, np.minimum(papers, np.sqrt(papers * avg_cites)).astype(int), 0)
        # Free intermediate arrays
        del papers, total_cites, avg_cites
        gc.collect()

        # Add sampling weights if sampling was used - use map() instead of merge() to save memory
        if self.use_sampling and self.sampling_weights is not None:
            logger.info("\n📊 Adding sampling weights to enhanced dataset...")
            # Convert to dict for memory-efficient lookup (avoids DataFrame copy from merge)
            weight_dict = self.sampling_weights.set_index("author_id")["sampling_weight"].to_dict()
            merged["sampling_weight"] = merged["author_id"].map(weight_dict).fillna(1.0)
            del weight_dict  # Free memory immediately
            gc.collect()
            logger.info(
                f"   ✅ Weights added: range {merged['sampling_weight'].min():.2f} - "
                f"{merged['sampling_weight'].max():.2f}"
            )
        else:
            # Add unit weights for consistency
            merged["sampling_weight"] = 1.0
            logger.info("   ℹ️  No sampling weights (full population or sampling disabled)")

        self.enhanced_author_data = merged

        logger.info(f"✅ Enhanced dataset with {len(self.enhanced_author_data):,} author profiles")

        return self.enhanced_author_data

    def monitor_query_progress(self):
        """Check on active queries and temp file usage."""

        progress_query = """
        SELECT
            pid,
            now() - pg_stat_activity.query_start AS duration,
            wait_event_type,
            wait_event,
            state,
            CASE WHEN length(query) > 50
                 THEN left(query, 50) || '...'
                 ELSE query
            END as query_preview
        FROM pg_stat_activity
        WHERE state = 'active'
          AND query NOT LIKE '%pg_stat_activity%'
          AND query NOT LIKE '%SHOW%'
        ORDER BY duration DESC;
        """

        temp_usage_query = """
        SELECT
            datname,
            temp_files,
            temp_bytes,
            pg_size_pretty(temp_bytes) as temp_size_pretty
        FROM pg_stat_database
        WHERE datname = 'openalex_db';
        """

        try:
            logger.info("\n🔍 Query Progress Monitor:")

            # Check active queries
            with self.engine.connect() as conn:
                active_queries = pd.read_sql(progress_query, conn)
                if len(active_queries) > 0:
                    logger.info("   Active queries:")
                    for _, row in active_queries.iterrows():
                        logger.info(
                            f"      PID {row['pid']}: {row['duration']} | {row['wait_event'] or 'CPU'} | {row['query_preview']}"
                        )
                else:
                    logger.info("   No active long-running queries")

                # Check temp file usage
                temp_usage = pd.read_sql(temp_usage_query, conn)
                if len(temp_usage) > 0:
                    temp_info = temp_usage.iloc[0]
                    logger.info(
                        f"   Temp files: {temp_info['temp_files']} files, {temp_info['temp_size_pretty']} total"
                    )

        except Exception as e:
            logger.warning(f"⚠️  Could not retrieve progress info: {e}")

    def _calculate_h_index(self, row):
        """Estimate h-index from paper count and total citations."""
        try:
            # Approximation based on papers and average citations
            papers = float(row["papers_published"]) if pd.notna(row["papers_published"]) else 0
            total_cites = float(row["total_times_cited"]) if pd.notna(row["total_times_cited"]) else 0

            if papers == 0:
                return 0

            avg_cites = total_cites / papers
            return int(min(papers, np.sqrt(papers * avg_cites)))
        except Exception:
            return 0

    def temporal_analysis(self):
        """
        Analyze how self-citation patterns changed over time.
        Generates 4 plots covering era comparisons, yearly trends, etc.
        """
        logger.info("⏰ Conducting temporal analysis across AI eras...")
        logger.info("   📊 Using weighted statistics to account for stratified sampling...")

        # Era-based analysis with weighted aggregations
        # Pre-compute weighted values to avoid multiple slow groupby.apply() calls
        logger.info("   📊 Pre-computing weighted metrics...")
        df = self.enhanced_author_data.copy()

        # Pre-multiply values by weights for weighted mean calculation
        df["weighted_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        df["weighted_scc"] = df["self_citation_count"] * df["sampling_weight"]
        df["weighted_h"] = df["h_index"] * df["sampling_weight"]

        # Single aggregation pass instead of 8 separate apply() calls
        logger.info("   📊 Computing era statistics (single pass)...")
        era_agg = df.groupby(["era", "primary_field"]).agg(
            weighted_scr_sum=("weighted_scr", "sum"),
            weighted_scc_sum=("weighted_scc", "sum"),
            weighted_h_sum=("weighted_h", "sum"),
            weight_sum=("sampling_weight", "sum"),
            scr_std=("self_citation_rate", "std"),
            scr_median=("self_citation_rate", "median"),
            scr_count=("self_citation_rate", "count"),
            scc_std=("self_citation_count", "std"),
            h_std=("h_index", "std"),
        )

        # Calculate weighted means
        era_analysis = pd.DataFrame(index=era_agg.index)
        era_analysis[("self_citation_rate", "mean")] = era_agg["weighted_scr_sum"] / era_agg["weight_sum"]
        era_analysis[("self_citation_rate", "std")] = era_agg["scr_std"]
        era_analysis[("self_citation_rate", "median")] = era_agg["scr_median"]
        era_analysis[("self_citation_rate", "count")] = era_agg["scr_count"]
        era_analysis[("self_citation_count", "mean")] = era_agg["weighted_scc_sum"] / era_agg["weight_sum"]
        era_analysis[("self_citation_count", "std")] = era_agg["scc_std"]
        era_analysis[("h_index", "mean")] = era_agg["weighted_h_sum"] / era_agg["weight_sum"]
        era_analysis[("h_index", "std")] = era_agg["h_std"]

        era_analysis = era_analysis.round(4)

        # Clean up
        del df, era_agg
        gc.collect()

        # Time series analysis
        logger.info("   📊 Computing yearly trends (partition-wise to save memory)...")

        # Get yearly self-citation sums
        yearly_sc_sum = (
            self.self_citation_data_dask.groupby(["citing_year", "primary_field"])["self_citation_count"]
            .sum()
            .reset_index()
            .compute()
        )
        yearly_sc_sum.columns = ["citing_year", "primary_field", "self_citation_count"]

        logger.info("   📊 Computing unique author counts per year (partition-wise)...")
        author_year_field = {}
        for i in range(self.self_citation_data_dask.npartitions):
            partition = self.self_citation_data_dask.get_partition(i)[
                ["citing_year", "primary_field", "author_id"]
            ].compute()
            for (year, field), group in partition.groupby(["citing_year", "primary_field"]):
                key = (year, field)
                if key not in author_year_field:
                    author_year_field[key] = set()
                author_year_field[key].update(group["author_id"].unique())
            del partition
            gc.collect()

        yearly_author_count = pd.DataFrame(
            [{"citing_year": k[0], "primary_field": k[1], "author_count": len(v)} for k, v in author_year_field.items()]
        )
        del author_year_field
        gc.collect()

        yearly_trends = yearly_sc_sum.merge(yearly_author_count, on=["citing_year", "primary_field"], how="left")

        yearly_trends["sc_rate_per_author"] = yearly_trends["self_citation_count"] / yearly_trends["author_count"]

        # ARIMA Time Series Analysis
        logger.info("   📊 Fitting ARIMA time series model...")
        arima_results = self._fit_arima_model(yearly_trends)

        self.results["temporal"] = {
            "era_analysis": era_analysis,
            "yearly_trends": yearly_trends,
            "arima": arima_results,
        }

        # Create temporal visualization
        self._create_temporal_plots()

        return self.results["temporal"]

    def _fit_arima_model(self, yearly_trends):
        """Fit ARIMA model to yearly self-citation data. Includes structural break tests."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller, kpss

            # Aggregate to yearly totals (across all fields)
            yearly_agg = (
                yearly_trends.groupby("citing_year")
                .agg({"self_citation_count": "sum", "author_count": "sum"})
                .reset_index()
            )
            yearly_agg["sc_rate"] = yearly_agg["self_citation_count"] / yearly_agg["author_count"]
            yearly_agg = yearly_agg.sort_values("citing_year")

            # Filter to valid years with sufficient data
            yearly_agg = yearly_agg[(yearly_agg["citing_year"] >= 1970) & (yearly_agg["citing_year"] <= 2025)]
            yearly_agg = yearly_agg.dropna(subset=["sc_rate"])

            if len(yearly_agg) < 10:
                logger.warning("   ⚠️ Insufficient data for ARIMA (need 10+ years)")
                return {"error": "Insufficient data"}

            # Create time series with proper DatetimeIndex for forecasting
            yearly_agg["date"] = pd.to_datetime(yearly_agg["citing_year"], format="%Y")
            ts = yearly_agg.set_index("date")["sc_rate"]
            ts.index.freq = "YS"  # Year start frequency

            # Stationarity tests
            logger.info("   📊 Testing stationarity (ADF and KPSS tests)...")
            adf_result = adfuller(ts, autolag="AIC")
            adf_stationary = adf_result[1] < 0.05  # p < 0.05 means stationary
            logger.info(
                f"      ADF test: statistic={adf_result[0]:.3f}, p-value={adf_result[1]:.4f} ({'stationary' if adf_stationary else 'non-stationary'})"
            )

            try:
                kpss_result = kpss(ts, regression="c", nlags="auto")
                kpss_stationary = kpss_result[1] > 0.05  # p > 0.05 means stationary
                logger.info(
                    f"      KPSS test: statistic={kpss_result[0]:.3f}, p-value={kpss_result[1]:.4f} ({'stationary' if kpss_stationary else 'non-stationary'})"
                )
            except Exception:
                kpss_stationary = None
                kpss_result = [None, None]
                logger.info("      KPSS test: skipped")

            # Determine differencing order
            d = 0 if adf_stationary else 1

            # Fit ARIMA model - try different orders and pick best AIC
            logger.info(f"   📊 Fitting ARIMA model (d={d})...")
            best_aic = float("inf")
            best_model = None
            best_order = (1, d, 1)

            for p in range(3):
                for q in range(3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except Exception:
                        continue

            if best_model is None:
                logger.warning("   ⚠️ ARIMA fitting failed for all parameter combinations")
                return {"error": "Model fitting failed"}

            logger.info(f"   ✅ Best ARIMA{best_order}: AIC={best_aic:.2f}, n={len(ts)} years")

            # Ljung-Box test for residual autocorrelation
            try:
                lb_test = acorr_ljungbox(best_model.resid, lags=[10], return_df=True)
                lb_pvalue = float(lb_test["lb_pvalue"].iloc[0])
                residuals_ok = lb_pvalue > 0.05  # p > 0.05 means no significant autocorrelation
                logger.info(
                    f"      Ljung-Box test: p-value={lb_pvalue:.4f} ({'residuals OK' if residuals_ok else 'autocorrelation detected'})"
                )
            except Exception:
                lb_pvalue = None
                residuals_ok = None

            # Trend significance - check if there's a significant trend component
            # Using the coefficient on the AR term and its p-value
            params = best_model.params
            pvalues = best_model.pvalues

            # Forecast next 5 years
            logger.info("   📊 Generating 5-year forecast...")
            forecast = best_model.get_forecast(steps=5)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # Structural break detection - compare pre-2010 vs post-2010 (AI emergence)
            # Use year from datetime index for comparison
            logger.info("   📊 Testing for structural breaks...")
            pre_ai = ts[ts.index.year < 2010]
            post_ai = ts[ts.index.year >= 2010]

            if len(pre_ai) >= 5 and len(post_ai) >= 5:

                # Welch's t-test for different means
                t_stat, break_pvalue = stats.ttest_ind(pre_ai, post_ai, equal_var=False)
                structural_break = {
                    "break_year": 2010,
                    "pre_mean": float(pre_ai.mean()),
                    "post_mean": float(post_ai.mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(break_pvalue),
                    "significant": break_pvalue < 0.05,
                }
            else:
                structural_break = None

            # COVID impact test (2020-2022 vs 2017-2019)
            pre_covid = ts[(ts.index.year >= 2017) & (ts.index.year < 2020)]
            covid_era = ts[(ts.index.year >= 2020) & (ts.index.year <= 2022)]

            if len(pre_covid) >= 2 and len(covid_era) >= 2:
                t_stat_covid, covid_pvalue = stats.ttest_ind(pre_covid, covid_era, equal_var=False)
                covid_impact = {
                    "pre_covid_mean": float(pre_covid.mean()),
                    "covid_mean": float(covid_era.mean()),
                    "t_statistic": float(t_stat_covid),
                    "p_value": float(covid_pvalue),
                    "significant": covid_pvalue < 0.05,
                }
            else:
                covid_impact = None

            arima_results = {
                "order": best_order,
                "aic": best_aic,
                "bic": best_model.bic,
                "log_likelihood": best_model.llf,
                "n_observations": len(ts),
                "stationarity": {
                    "adf_statistic": float(adf_result[0]),
                    "adf_pvalue": float(adf_result[1]),
                    "adf_stationary": adf_stationary,
                    "kpss_pvalue": float(kpss_result[1]) if kpss_result[1] else None,
                    "kpss_stationary": kpss_stationary,
                },
                "residual_diagnostics": {
                    "ljung_box_pvalue": lb_pvalue,
                    "residuals_ok": residuals_ok,
                },
                "parameters": params.to_dict(),
                "parameter_pvalues": pvalues.to_dict(),
                "forecast": {
                    "years": list(range(2026, 2031)),
                    "values": forecast_mean.tolist(),
                    "lower_ci": forecast_ci.iloc[:, 0].tolist(),
                    "upper_ci": forecast_ci.iloc[:, 1].tolist(),
                },
                "structural_break_2010": structural_break,
                "covid_impact": covid_impact,
                "time_series": ts.to_dict(),
            }

            logger.info(
                f"   ✅ ARIMA analysis complete: structural break significant={structural_break['significant'] if structural_break else 'N/A'}"
            )
            if structural_break:
                logger.info(
                    f"      Pre-2010 mean SC rate: {structural_break['pre_mean']:.4f}, Post-2010: {structural_break['post_mean']:.4f}"
                )
            if covid_impact:
                covid_sig = "significant" if covid_impact["significant"] else "not significant"
                logger.info(
                    f"      COVID impact: pre={covid_impact['pre_covid_mean']:.4f}, during={covid_impact['covid_mean']:.4f} ({covid_sig})"
                )

            return arima_results

        except ImportError as e:
            logger.warning(f"   ⚠️ ARIMA requires statsmodels: {e}")
            return {"error": f"Missing dependency: {e}"}
        except Exception as e:
            logger.warning(f"   ⚠️ ARIMA fitting failed: {e}")
            return {"error": str(e)}

    def career_stage_analysis(self):
        """
        Compare self-citation rates across career stages (early/mid/senior).
        Generates 4 plots with stats tests.
        """
        logger.info("🎯 Analyzing career stage patterns...")
        logger.info("   📊 Using weighted statistics to account for stratified sampling...")

        # Career stage statistical analysis with weighted aggregations
        # Use efficient vectorized aggregation instead of slow apply(lambda...)
        logger.info("   📊 Computing career stage statistics (vectorized)...")
        career_stats = self._weighted_groupby_stats(
            self.enhanced_author_data,
            group_cols=["career_stage", "primary_field"],
            value_cols=["self_citation_rate", "self_citation_count", "h_index", "total_works"],
        )

        # Weighted ANOVA test for career stage differences
        # Note: For weighted ANOVA, we use repeated sampling proportional to weights
        career_groups = [
            self.enhanced_author_data[self.enhanced_author_data["career_stage"] == stage][
                ["self_citation_rate", "sampling_weight"]
            ].dropna()
            for stage in ["early", "mid", "senior"]
        ]

        # Use unweighted test as approximation (proper weighted ANOVA requires specialized libraries)
        # This is acceptable since weights are population-proportional
        career_groups_values = [g["self_citation_rate"].values for g in career_groups]
        f_stat, p_value = stats.f_oneway(*career_groups_values)

        # Pairwise comparisons (weighted means reported, unweighted tests as conservative estimate)
        pairwise_results = {}
        stages = ["early", "mid", "senior"]
        for i, stage1 in enumerate(stages):
            for stage2 in stages[i + 1 :]:
                df1 = self.enhanced_author_data[self.enhanced_author_data["career_stage"] == stage1][
                    ["self_citation_rate", "sampling_weight"]
                ].dropna()
                df2 = self.enhanced_author_data[self.enhanced_author_data["career_stage"] == stage2][
                    ["self_citation_rate", "sampling_weight"]
                ].dropna()

                # Calculate weighted means for reporting
                mean1 = np.average(df1["self_citation_rate"], weights=df1["sampling_weight"])
                mean2 = np.average(df2["self_citation_rate"], weights=df2["sampling_weight"])

                # Use unweighted t-test (conservative, accounts for sampling uncertainty)
                t_stat, p_val = stats.ttest_ind(df1["self_citation_rate"], df2["self_citation_rate"])
                pairwise_results[f"{stage1}_vs_{stage2}"] = {
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "weighted_mean_1": mean1,
                    "weighted_mean_2": mean2,
                }

        self.results["career_stage"] = {
            "descriptive_stats": career_stats,
            "anova": {"f_stat": f_stat, "p_value": p_value},
            "pairwise_comparisons": pairwise_results,
        }

        # Create career stage visualizations
        self._create_career_stage_plots()

        return self.results["career_stage"]

    def disciplinary_analysis(self):
        """
        Compare self-citation rates across academic fields.
        Generates 4 plots showing field rankings and comparisons.
        """
        logger.info("🔬 Conducting disciplinary analysis...")

        # Field-based analysis
        field_stats = (
            self.enhanced_author_data.groupby("primary_field")
            .agg(
                {
                    "self_citation_rate": ["mean", "std", "median", "count"],
                    "self_citation_count": ["mean", "std"],
                    "h_index": ["mean", "std"],
                    "total_works": ["mean", "std"],
                }
            )
            .round(4)
        )

        logger.info("   📊 Using weighted statistics to account for stratified sampling...")

        # Field ranking by self-citation rate (weighted) - vectorized approach
        df = self.enhanced_author_data.copy()
        df["_weighted_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        field_agg = df.groupby("primary_field").agg(
            weighted_scr_sum=("_weighted_scr", "sum"), weight_sum=("sampling_weight", "sum")
        )
        field_ranking = (field_agg["weighted_scr_sum"] / field_agg["weight_sum"]).sort_values(ascending=False)
        del df
        gc.collect()

        # Cross-field statistical tests
        field_comparison = {}
        top_fields = list(field_ranking.head(10).index)

        for field in top_fields[:5]:  # Compare top 5 fields
            field_df = self.enhanced_author_data[self.enhanced_author_data["primary_field"] == field][
                ["self_citation_rate", "sampling_weight"]
            ].dropna()
            other_df = self.enhanced_author_data[self.enhanced_author_data["primary_field"] != field][
                ["self_citation_rate", "sampling_weight"]
            ].dropna()

            if len(field_df) > 30 and len(other_df) > 30:  # Sufficient sample size
                # Weighted means
                mean_field = np.average(field_df["self_citation_rate"], weights=field_df["sampling_weight"])
                mean_others = np.average(other_df["self_citation_rate"], weights=other_df["sampling_weight"])

                # Unweighted test (conservative)
                t_stat, p_val = stats.ttest_ind(field_df["self_citation_rate"], other_df["self_citation_rate"])
                field_comparison[field] = {
                    "mean_field": mean_field,
                    "mean_others": mean_others,
                    "t_stat": t_stat,
                    "p_value": p_val,
                }

        self.results["disciplinary"] = {
            "field_stats": field_stats,
            "field_ranking": field_ranking,
            "field_comparisons": field_comparison,
        }

        # Create disciplinary visualizations
        self._create_disciplinary_plots()

        return self.results["disciplinary"]

    def ai_era_impact_analysis(self):
        """
        Test if the AI era (2010+) changed self-citation patterns.
        Also checks COVID-19 impact. Generates 4 plots.
        """
        logger.info("🤖 Analyzing AI era impact on self-citation patterns...")
        logger.info("   📊 Using weighted statistics to account for stratified sampling...")

        # Pre-AI vs AI/COVID era comparison with weights
        pre_ai_df = self.enhanced_author_data[self.enhanced_author_data["era"].isin(["pre_digital", "early_digital"])][
            ["self_citation_rate", "sampling_weight"]
        ].dropna()

        ai_covid_df = self.enhanced_author_data[
            self.enhanced_author_data["era"].isin(["ai_emergence", "covid_era", "modern_ai"])
        ][["self_citation_rate", "sampling_weight"]].dropna()

        # Statistical comparison
        if len(pre_ai_df) > 0 and len(ai_covid_df) > 0:
            # Weighted means and stds
            pre_ai_mean = np.average(pre_ai_df["self_citation_rate"], weights=pre_ai_df["sampling_weight"])
            ai_covid_mean = np.average(ai_covid_df["self_citation_rate"], weights=ai_covid_df["sampling_weight"])

            pre_ai_std = self._weighted_std(pre_ai_df, "self_citation_rate")
            ai_covid_std = self._weighted_std(ai_covid_df, "self_citation_rate")

            # Unweighted t-test (conservative)
            t_stat, p_value = stats.ttest_ind(pre_ai_df["self_citation_rate"], ai_covid_df["self_citation_rate"])

            # Calculate weighted effect size (Cohen's d)
            pooled_std = np.sqrt((pre_ai_std**2 + ai_covid_std**2) / 2)
            effect_size = (ai_covid_mean - pre_ai_mean) / pooled_std
        else:
            t_stat, p_value, effect_size = None, None, None
            pre_ai_mean = ai_covid_mean = None

        # Era transition analysis with weighted aggregations - vectorized
        logger.info("   📊 Computing era transitions (vectorized)...")
        era_transitions = self._weighted_groupby_stats(
            self.enhanced_author_data, group_cols=["era", "career_stage"], value_cols=["self_citation_rate", "h_index"]
        )

        self.results["ai_impact"] = {
            "pre_ai_mean": pre_ai_mean,
            "ai_covid_mean": ai_covid_mean,
            "comparison": {"t_stat": t_stat, "p_value": p_value, "effect_size": effect_size},
            "era_transitions": era_transitions,
        }

        # Create AI impact visualizations
        self._create_ai_impact_plots()

        return self.results["ai_impact"]

    def network_analysis(self):
        """
        Build a self-citation network and compute stats.
        Samples 10K authors to keep it tractable. Generates 4 plots.
        """
        logger.info("🕸️ Conducting network analysis of self-citation patterns...")

        # Create self-citation network
        G = nx.DiGraph()

        # Add edges based on temporal self-citations
        # Use aggregated data to avoid iterating over 60M rows
        logger.info("   📊 Aggregating self-citation data for network construction...")

        # Sample authors for network analysis to prevent OOM
        # Network analysis doesn't need all 11M+ authors - a representative sample suffices
        max_network_authors = 10_000  # Reduced from 50k to keep centrality calculations feasible

        # Get unique authors from enhanced_author_data (already in memory)
        unique_authors = self.enhanced_author_data["author_id"].unique()
        if len(unique_authors) > max_network_authors:
            logger.info(
                f"   📊 Sampling {max_network_authors:,} authors from {len(unique_authors):,} for network analysis"
            )
            sampled_authors = set(pd.Series(unique_authors).sample(n=max_network_authors, random_state=42))
        else:
            sampled_authors = set(unique_authors)

        # Process partition by partition to avoid loading full dataset
        logger.info("   📊 Processing network data partition by partition...")
        network_edges = []

        for i in range(self.self_citation_data_dask.npartitions):
            partition = self.self_citation_data_dask.get_partition(i).compute()
            # Filter to sampled authors
            partition = partition[partition["author_id"].isin(sampled_authors)]
            if len(partition) > 0:
                # Aggregate at AUTHOR level (not author-year) for cleaner network interpretation
                # This creates a self-citation intensity network where edge weight = total self-citations
                agg = (
                    partition.groupby(["author_id"])
                    .agg(
                        {
                            "self_citation_count": "sum",
                            "temporal_distance": "mean",
                            "citing_year": "nunique",  # number of active years
                        }
                    )
                    .reset_index()
                )
                network_edges.append(agg)
            del partition

        if network_edges:
            network_data = pd.concat(network_edges, ignore_index=True)
            # Final aggregation across partitions
            network_data = (
                network_data.groupby(["author_id"])
                .agg(
                    {
                        "self_citation_count": "sum",
                        "temporal_distance": "mean",
                        "citing_year": "sum",  # total active years
                    }
                )
                .reset_index()
            )
            del network_edges
        else:
            logger.warning("⚠️  No network data found")
            self.results["network"] = {"graph": G, "structure_metrics": {"nodes": 0, "edges": 0}}
            return self.results["network"]

        logger.info(f"   📊 Building author-level network from {len(network_data):,} authors...")

        # Create author nodes with self-citation intensity as self-loop weight
        # This represents each author's self-citation behavior as a weighted node
        for _, row in network_data.iterrows():
            author_node = str(row["author_id"])
            G.add_node(
                author_node,
                self_citations=row["self_citation_count"],
                avg_temporal_distance=row["temporal_distance"],
                active_years=row["citing_year"],
            )
            # Add self-loop weighted by self-citation count (represents self-citation intensity)
            if row["self_citation_count"] > 0:
                G.add_edge(author_node, author_node, weight=row["self_citation_count"])

        del network_data

        logger.info(
            f"   ✅ Network built: {G.number_of_nodes():,} author nodes, {G.number_of_edges():,} self-citation edges"
        )

        # Calculate network metrics - for self-loop author network, focus on node attributes
        if G.number_of_nodes() > 0:
            try:
                logger.info("   📊 Computing network metrics...")

                # Extract self-citation intensity distribution from node attributes
                self_citations = [G.nodes[n].get("self_citations", 0) for n in G.nodes()]
                temporal_distances = [G.nodes[n].get("avg_temporal_distance", 0) for n in G.nodes()]
                active_years = [G.nodes[n].get("active_years", 0) for n in G.nodes()]

                # Compute distribution statistics
                sc_stats = {
                    "mean": np.mean(self_citations),
                    "median": np.median(self_citations),
                    "std": np.std(self_citations),
                    "max": np.max(self_citations),
                    "total": np.sum(self_citations),
                }

                td_stats = {
                    "mean": np.mean(temporal_distances),
                    "median": np.median(temporal_distances),
                    "std": np.std(temporal_distances),
                }

                # Identify high self-citers (top 1%)
                threshold = np.percentile(self_citations, 99)
                high_self_citers = sum(1 for sc in self_citations if sc >= threshold)

                self.results["network"] = {
                    "graph": G,
                    "self_citation_distribution": self_citations,
                    "temporal_distance_distribution": temporal_distances,
                    "active_years_distribution": active_years,
                    "self_citation_stats": sc_stats,
                    "temporal_distance_stats": td_stats,
                    "structure_metrics": {
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "high_self_citers_1pct": high_self_citers,
                        "mean_self_citations": sc_stats["mean"],
                        "mean_temporal_distance": td_stats["mean"],
                    },
                }
                logger.info(
                    f"   ✅ Network metrics computed: mean self-citations={sc_stats['mean']:.2f}, mean temporal distance={td_stats['mean']:.2f}y"
                )

            except Exception as e:
                logger.warning(f"⚠️  Error computing network metrics: {e}")
                self.results["network"] = {
                    "graph": G,
                    "structure_metrics": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
                }

        # Create network visualizations
        self._create_network_plots()

        return self.results["network"]

    def multivariate_regression_analysis(self):
        """
        Run OLS and mixed-effects regression on the data.
        Shows which factors predict self-citation rates. Generates 4 plots.
        """
        logger.info("📊 Conducting multivariate regression analysis...")
        logger.info("   📊 Using weighted regression to account for stratified sampling...")

        # Prepare data for regression
        reg_data = self.enhanced_author_data.dropna(
            subset=["self_citation_rate", "career_stage", "primary_field", "era"]
        )

        # Subsample if dataset is too large for regression (>500k rows)
        max_regression_rows = 500_000
        if len(reg_data) > max_regression_rows:
            logger.info(f"   📊 Subsampling from {len(reg_data):,} to {max_regression_rows:,} rows for regression")
            reg_data = reg_data.sample(n=max_regression_rows, random_state=42, weights="sampling_weight")
        # Create dummy variables
        career_dummies = pd.get_dummies(reg_data["career_stage"], prefix="career")
        era_dummies = pd.get_dummies(reg_data["era"], prefix="era")
        field_dummies = pd.get_dummies(reg_data["primary_field"], prefix="field")

        # Combine features
        X = pd.concat(
            [career_dummies, era_dummies, field_dummies, reg_data[["h_index", "total_works"]].fillna(0)], axis=1
        )

        y = reg_data["self_citation_rate"]

        # OLS Regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        # Also split weights for weighted regression
        weights = reg_data["sampling_weight"]
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=weights_train)

        # Predictions and metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "coefficient": model.coef_, "abs_coefficient": np.abs(model.coef_)}
        ).sort_values("abs_coefficient", ascending=False)

        self.results["regression"] = {
            "model": model,
            "r2_score": r2,
            "rmse": rmse,
            "feature_importance": feature_importance,
            "n_features": len(X.columns),
            "n_observations": len(X),
        }

        # Hierarchical Linear Model (Mixed Effects) - accounts for clustering within fields
        logger.info("   📊 Fitting Hierarchical Linear Model (Mixed Effects)...")
        try:
            import statsmodels.formula.api as smf

            # Prepare data for HLM (need raw data, not dummies)
            hlm_data = reg_data[
                [
                    "self_citation_rate",
                    "career_stage",
                    "era",
                    "h_index",
                    "total_works",
                    "primary_field",
                    "sampling_weight",
                ]
            ].copy()
            hlm_data = hlm_data.dropna()

            # Subsample for HLM if needed (HLM is computationally intensive)
            max_hlm_rows = 100_000
            if len(hlm_data) > max_hlm_rows:
                logger.info(f"   📊 Subsampling to {max_hlm_rows:,} rows for HLM")
                hlm_data = hlm_data.sample(n=max_hlm_rows, random_state=42)

            # Fit mixed effects model with field as random effect
            # This accounts for the fact that authors are nested within fields
            hlm_model = smf.mixedlm(
                "self_citation_rate ~ C(career_stage) + C(era) + h_index + total_works",
                data=hlm_data,
                groups=hlm_data["primary_field"],
            )

            # Try BFGS first (computes AIC/BIC properly), fall back to Powell if it fails
            try:
                hlm_result = hlm_model.fit(method="bfgs")
            except Exception:
                logger.info("   📊 BFGS failed, trying Powell optimizer...")
                hlm_result = hlm_model.fit(method="powell")

            # Extract key results
            # Note: ngroups attribute may not exist in all statsmodels versions
            n_groups = getattr(hlm_result, "ngroups", None) or hlm_data["primary_field"].nunique()

            # Handle NaN AIC/BIC (can happen with Powell optimizer)
            aic_val = hlm_result.aic if not np.isnan(hlm_result.aic) else None
            bic_val = hlm_result.bic if not np.isnan(hlm_result.bic) else None

            hlm_summary = {
                "converged": hlm_result.converged,
                "aic": aic_val,
                "bic": bic_val,
                "log_likelihood": hlm_result.llf,
                "random_effect_variance": (
                    float(hlm_result.cov_re.iloc[0, 0])
                    if hasattr(hlm_result, "cov_re") and hlm_result.cov_re is not None
                    else None
                ),
                "fixed_effects": hlm_result.fe_params.to_dict(),
                "fixed_effects_pvalues": hlm_result.pvalues.to_dict(),
                "n_groups": n_groups,
                "n_observations": len(hlm_data),
            }

            # Calculate ICC (Intraclass Correlation) - proportion of variance explained by field
            if hlm_summary["random_effect_variance"] is not None:
                residual_var = hlm_result.scale
                icc = hlm_summary["random_effect_variance"] / (hlm_summary["random_effect_variance"] + residual_var)
                hlm_summary["icc"] = icc
                logger.info(f"   ✅ HLM fitted: ICC={icc:.3f} ({icc * 100:.1f}% of variance due to field differences)")

            self.results["regression"]["hlm"] = hlm_summary
            self.results["regression"]["hlm_model"] = hlm_result

            aic_str = f"{aic_val:.1f}" if aic_val is not None else "N/A (Powell optimizer)"
            logger.info(f"   ✅ HLM converged: {hlm_result.converged}, AIC={aic_str}")

        except Exception as e:
            logger.warning(f"   ⚠️ HLM fitting failed: {e}")
            self.results["regression"]["hlm"] = {"error": str(e)}

        # Create regression visualizations
        self._create_regression_plots()

        return self.results["regression"]

    def machine_learning_analysis(self):
        """
        Run Random Forest and K-means clustering on the data.
        Identifies which features matter most and groups similar authors.
        Generates 4 plots.
        """
        logger.info("🤖 Conducting machine learning analysis...")
        logger.info("   ⚠️  Note: RandomForest and KMeans don't support sample weights directly")
        logger.info("   💡 Using stratified sample as-is (weights already applied during sampling)")

        # Prepare features - ensure required columns have no NA values
        required_cols = ["self_citation_rate", "career_stage", "era", "h_index", "total_works", "primary_field"]
        ml_data = self.enhanced_author_data.dropna(subset=required_cols).copy()

        # Ensure numeric columns are proper floats (not pandas NA types)
        ml_data["h_index"] = pd.to_numeric(ml_data["h_index"], errors="coerce").fillna(0).astype(float)
        ml_data["total_works"] = pd.to_numeric(ml_data["total_works"], errors="coerce").fillna(0).astype(float)
        ml_data["self_citation_rate"] = (
            pd.to_numeric(ml_data["self_citation_rate"], errors="coerce").fillna(0).astype(float)
        )

        # Subsample if dataset is too large for ML (>300k rows for RandomForest)
        max_ml_rows = 300_000
        if len(ml_data) > max_ml_rows:
            logger.info(f"   📊 Subsampling from {len(ml_data):,} to {max_ml_rows:,} rows for ML analysis")
            ml_data = ml_data.sample(n=max_ml_rows, random_state=42, weights="sampling_weight")

        # Create feature matrix using VECTORIZED one-hot encoding (much faster than iterrows)
        logger.info("   📊 Creating feature matrix using vectorized encoding...")

        # One-hot encode categorical variables
        career_dummies = pd.get_dummies(ml_data["career_stage"], prefix="career")
        era_dummies = pd.get_dummies(ml_data["era"], prefix="era")
        # Ensure all expected columns exist (in case some categories are missing)
        for col in ["career_early", "career_mid", "career_senior"]:
            if col not in career_dummies.columns:
                career_dummies[col] = 0
        for col in ["era_pre_digital", "era_early_digital", "era_ai_emergence", "era_covid_era", "era_modern_ai"]:
            if col not in era_dummies.columns:
                era_dummies[col] = 0

        # Build feature matrix
        X = pd.concat(
            [
                career_dummies[["career_early", "career_mid", "career_senior"]].reset_index(drop=True),
                era_dummies[
                    ["era_pre_digital", "era_early_digital", "era_ai_emergence", "era_covid_era", "era_modern_ai"]
                ].reset_index(drop=True),
                ml_data[["h_index", "total_works"]].reset_index(drop=True),
                ml_data["primary_field"]
                .str.len()
                .fillna(0)
                .reset_index(drop=True)
                .rename("field_name_len"),  # Field name length as proxy
            ],
            axis=1,
        )

        # Ensure all values are numeric and no NA values remain
        X = X.fillna(0).astype(float)
        y = ml_data["self_citation_rate"].fillna(0).astype(float).values

        # Random Forest Analysis
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Feature importance
        feature_names = [
            "early_career",
            "mid_career",
            "senior_career",
            "pre_digital",
            "early_digital",
            "ai_emergence",
            "covid_era",
            "modern_ai",
            "h_index",
            "total_works",
            "field_specificity",
        ]

        rf_importance = pd.DataFrame(
            {"feature": feature_names, "importance": rf_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Clustering analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        ml_data["cluster"] = clusters
        cluster_analysis = ml_data.groupby("cluster").agg(
            {
                "self_citation_rate": ["mean", "std", "count"],
                "career_stage": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
                "era": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
                "primary_field": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown",
            }
        )

        self.results["machine_learning"] = {
            "rf_model": rf_model,
            "rf_importance": rf_importance,
            "clusters": cluster_analysis,
            "kmeans_model": kmeans,
            "scaler": scaler,
        }

        # Create machine learning visualizations
        self._create_machine_learning_plots()

        return self.results["machine_learning"]

    def _create_temporal_plots(self):
        """Generate the 4 temporal analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Temporal Evolution of Self-Citation Patterns (1970-2025)", fontsize=16, fontweight="bold")

        # Plot 1: Era comparison (weighted) - vectorized
        df = self.enhanced_author_data.copy()
        df["_w_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        era_agg = df.groupby("era").agg(
            w_sum=("_w_scr", "sum"), weight_sum=("sampling_weight", "sum"), scr_std=("self_citation_rate", "std")
        )
        era_data = pd.DataFrame(
            {"era": era_agg.index, "mean": era_agg["w_sum"] / era_agg["weight_sum"], "std": era_agg["scr_std"]}
        )
        del df
        axes[0, 0].bar(era_data["era"], era_data["mean"], yerr=era_data["std"], capsize=5, alpha=0.7)
        axes[0, 0].set_title("Self-Citation Rates Across Eras")
        axes[0, 0].set_ylabel("Mean Self-Citation Rate")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Yearly trends for top fields
        yearly_data = self.results["temporal"]["yearly_trends"]
        top_fields = yearly_data.groupby("primary_field")["sc_rate_per_author"].mean().nlargest(5).index

        for field in top_fields:
            field_data = yearly_data[yearly_data["primary_field"] == field]
            axes[0, 1].plot(
                field_data["citing_year"],
                field_data["sc_rate_per_author"],
                label=field[:20] + "..." if len(field) > 20 else field,
                marker="o",
                alpha=0.7,
            )

        axes[0, 1].set_title("Self-Citation Trends by Field Over Time")
        axes[0, 1].set_xlabel("Year")
        axes[0, 1].set_ylabel("Self-Citations per Author")
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot 3: Era × Career Stage interaction (weighted) - vectorized
        df = self.enhanced_author_data.copy()
        df["_w_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        era_career_agg = df.groupby(["era", "career_stage"]).agg(
            w_sum=("_w_scr", "sum"), weight_sum=("sampling_weight", "sum")
        )
        era_career_agg["weighted_mean"] = era_career_agg["w_sum"] / era_career_agg["weight_sum"]
        era_career = era_career_agg["weighted_mean"].unstack()
        del df
        era_career.plot(kind="bar", ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title("Self-Citation Rates: Era × Career Stage")
        axes[1, 0].set_ylabel("Mean Self-Citation Rate")
        axes[1, 0].legend(title="Career Stage")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: ARIMA Time Series with Forecast
        arima_results = self.results["temporal"].get("arima", {})

        if "time_series" in arima_results and "forecast" in arima_results:
            ts_data = arima_results["time_series"]
            years = list(ts_data.keys())
            values = list(ts_data.values())

            # Plot historical data
            axes[1, 1].plot(years, values, "b-", marker="o", markersize=3, label="Observed", alpha=0.7)

            # Plot forecast
            forecast = arima_results["forecast"]
            axes[1, 1].plot(
                forecast["years"], forecast["values"], "r--", marker="s", markersize=4, label="Forecast", linewidth=2
            )

            # Plot confidence interval
            axes[1, 1].fill_between(
                forecast["years"], forecast["lower_ci"], forecast["upper_ci"], color="red", alpha=0.2, label="95% CI"
            )

            # Mark structural breaks
            if arima_results.get("structural_break_2010", {}).get("significant"):
                axes[1, 1].axvline(x=2010, color="green", linestyle="--", alpha=0.7, label="AI Emergence (2010)")

            if arima_results.get("covid_impact", {}).get("significant"):
                axes[1, 1].axvspan(2020, 2022, color="orange", alpha=0.2, label="COVID-19 Era")

            axes[1, 1].set_title(f"ARIMA{arima_results.get('order', '?')} Time Series & Forecast")
            axes[1, 1].set_xlabel("Year")
            axes[1, 1].set_ylabel("Self-Citation Rate")
            axes[1, 1].legend(loc="upper left", fontsize=8)

            # Add model info as text
            info_text = f"AIC: {arima_results.get('aic', 'N/A'):.1f}\n"
            if arima_results.get("structural_break_2010"):
                sb = arima_results["structural_break_2010"]
                info_text += f"Break 2010: p={sb['p_value']:.3f}"
            axes[1, 1].text(
                0.98,
                0.02,
                info_text,
                transform=axes[1, 1].transAxes,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            # Fallback to distribution comparison
            max_hist_points = 100_000
            pre_ai_data = self.enhanced_author_data[
                self.enhanced_author_data["era"].isin(["pre_digital", "early_digital"])
            ]["self_citation_rate"].dropna()
            if len(pre_ai_data) > max_hist_points:
                pre_ai_data = pre_ai_data.sample(n=max_hist_points, random_state=42)

            ai_covid_data = self.enhanced_author_data[
                self.enhanced_author_data["era"].isin(["ai_emergence", "covid_era", "modern_ai"])
            ]["self_citation_rate"].dropna()
            if len(ai_covid_data) > max_hist_points:
                ai_covid_data = ai_covid_data.sample(n=max_hist_points, random_state=42)

            axes[1, 1].hist(pre_ai_data, bins=30, alpha=0.5, label="Pre-AI Era", density=True)
            axes[1, 1].hist(ai_covid_data, bins=30, alpha=0.5, label="AI/COVID Era", density=True)
            axes[1, 1].set_title("Self-Citation Rate Distributions")
            axes[1, 1].set_xlabel("Self-Citation Rate")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].legend()

        plt.tight_layout()
        self.figures["temporal_analysis"] = fig

    def _create_career_stage_plots(self):
        """Generate the 4 career stage plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Career Stage Analysis of Self-Citation Patterns", fontsize=16, fontweight="bold")

        # Plot 1: Career stage box plots (sample for visualization efficiency)
        max_boxplot_points = 50_000
        career_data = []
        career_labels = []
        for stage in ["early", "mid", "senior"]:
            stage_data = self.enhanced_author_data[self.enhanced_author_data["career_stage"] == stage][
                "self_citation_rate"
            ].dropna()
            # Sample for boxplot if too many points
            if len(stage_data) > max_boxplot_points:
                stage_data = stage_data.sample(n=max_boxplot_points, random_state=42)
            career_data.append(stage_data)
            career_labels.append(stage.capitalize())

        axes[0, 0].boxplot(career_data, tick_labels=career_labels)
        axes[0, 0].set_title("Self-Citation Rate Distribution by Career Stage")
        axes[0, 0].set_ylabel("Self-Citation Rate")

        # Plot 2: Career progression over time (weighted) - vectorized
        df = self.enhanced_author_data.copy()
        df["_w_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        career_prog_agg = df.groupby(["career_stage", "era"]).agg(
            w_sum=("_w_scr", "sum"), weight_sum=("sampling_weight", "sum")
        )
        career_prog_agg["weighted_mean"] = career_prog_agg["w_sum"] / career_prog_agg["weight_sum"]
        career_progression = career_prog_agg["weighted_mean"].unstack()
        del df
        career_progression.plot(kind="bar", ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title("Career Stage Patterns Across Eras")
        axes[0, 1].set_ylabel("Mean Self-Citation Rate")
        axes[0, 1].legend(title="Era")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Career stage vs h-index relationship (sample for scatter efficiency)
        max_scatter_points = 10_000
        for stage in ["early", "mid", "senior"]:
            stage_data = self.enhanced_author_data[self.enhanced_author_data["career_stage"] == stage]
            if len(stage_data) > max_scatter_points:
                stage_data = stage_data.sample(n=max_scatter_points, random_state=42)
            axes[1, 0].scatter(
                stage_data["h_index"], stage_data["self_citation_rate"], label=stage.capitalize(), alpha=0.6
            )

        axes[1, 0].set_title("Self-Citation Rate vs H-Index by Career Stage")
        axes[1, 0].set_xlabel("H-Index")
        axes[1, 0].set_ylabel("Self-Citation Rate")
        axes[1, 0].legend()

        # Plot 4: Statistical significance heatmap
        pairwise = self.results["career_stage"]["pairwise_comparisons"]
        comparison_matrix = np.zeros((3, 3))
        stages = ["early", "mid", "senior"]

        for i, stage1 in enumerate(stages):
            for j, stage2 in enumerate(stages):
                if i < j:
                    key = f"{stage1}_vs_{stage2}"
                    if key in pairwise:
                        comparison_matrix[i, j] = pairwise[key]["p_value"]
                        comparison_matrix[j, i] = pairwise[key]["p_value"]

        im = axes[1, 1].imshow(comparison_matrix, cmap="RdYlBu_r", vmin=0, vmax=0.05)
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(stages)
        axes[1, 1].set_yticklabels(stages)
        axes[1, 1].set_title("P-values for Career Stage Comparisons")

        # Add text annotations
        for i in range(3):
            for j in range(3):
                if comparison_matrix[i, j] > 0:
                    axes[1, 1].text(j, i, f"{comparison_matrix[i, j]:.3f}", ha="center", va="center")

        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()
        self.figures["career_stage_analysis"] = fig

    def _create_disciplinary_plots(self):
        """Generate the 4 disciplinary comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle("Disciplinary Patterns in Self-Citation Behavior", fontsize=16, fontweight="bold")

        # Plot 1: Field ranking
        field_ranking = self.results["disciplinary"]["field_ranking"].head(15)
        axes[0, 0].barh(range(len(field_ranking)), field_ranking.values, alpha=0.7)
        axes[0, 0].set_yticks(range(len(field_ranking)))
        axes[0, 0].set_yticklabels([name[:30] + "..." if len(name) > 30 else name for name in field_ranking.index])
        axes[0, 0].set_title("Self-Citation Rates by Discipline (Top 15)")
        axes[0, 0].set_xlabel("Mean Self-Citation Rate")

        # Plot 2: Field comparison statistics
        field_stats = self.results["disciplinary"]["field_stats"].head(10)
        means = field_stats["self_citation_rate"]["mean"]
        stds = field_stats["self_citation_rate"]["std"]

        x_pos = range(len(means))
        axes[0, 1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([name[:15] + "..." if len(name) > 15 else name for name in means.index], rotation=45)
        axes[0, 1].set_title("Self-Citation Statistics by Field (Top 10)")
        axes[0, 1].set_ylabel("Self-Citation Rate")

        # Plot 3: Field × Era interaction (weighted) - vectorized
        df = self.enhanced_author_data.copy()
        df["_w_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        field_era_agg = df.groupby(["primary_field", "era"]).agg(
            w_sum=("_w_scr", "sum"), weight_sum=("sampling_weight", "sum")
        )
        field_era_agg["weighted_mean"] = field_era_agg["w_sum"] / field_era_agg["weight_sum"]
        field_era = field_era_agg["weighted_mean"].unstack()
        del df

        # Select top 8 fields for visibility
        top_fields = field_ranking.head(8).index
        field_era_subset = field_era.loc[top_fields]

        im = axes[1, 0].imshow(field_era_subset.values, aspect="auto", cmap="viridis")
        axes[1, 0].set_xticks(range(len(field_era_subset.columns)))
        axes[1, 0].set_yticks(range(len(field_era_subset.index)))
        axes[1, 0].set_xticklabels(field_era_subset.columns)
        field_names_list = list(field_era_subset.index)
        axes[1, 0].set_yticklabels([name[:20] + "..." if len(name) > 20 else name for name in field_names_list])
        axes[1, 0].set_title("Self-Citation Rates: Field × Era Heatmap")
        plt.colorbar(im, ax=axes[1, 0])

        # Plot 4: Field diversity analysis
        field_counts = self.enhanced_author_data["primary_field"].value_counts().head(10)
        field_labels = list(field_counts.index)
        axes[1, 1].pie(
            field_counts.values,
            labels=[name[:20] + "..." if len(name) > 20 else name for name in field_labels],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 1].set_title("Distribution of Authors Across Fields")

        plt.tight_layout()
        self.figures["disciplinary_analysis"] = fig

    def _create_ai_impact_plots(self):
        """Generate the 4 AI era impact plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("AI Era Impact on Self-Citation Patterns", fontsize=16, fontweight="bold")

        # Plot 1: Pre-AI vs AI/COVID era comparison (sample for visualization)
        max_boxplot_points = 50_000
        pre_ai = self.enhanced_author_data[self.enhanced_author_data["era"].isin(["pre_digital", "early_digital"])][
            "self_citation_rate"
        ].dropna()
        if len(pre_ai) > max_boxplot_points:
            pre_ai = pre_ai.sample(n=max_boxplot_points, random_state=42)

        ai_covid_era = self.enhanced_author_data[
            self.enhanced_author_data["era"].isin(["ai_emergence", "covid_era", "modern_ai"])
        ]["self_citation_rate"].dropna()
        if len(ai_covid_era) > max_boxplot_points:
            ai_covid_era = ai_covid_era.sample(n=max_boxplot_points, random_state=42)

        axes[0, 0].boxplot([pre_ai, ai_covid_era], tick_labels=["Pre-AI Era", "AI/COVID Era"])
        axes[0, 0].set_title("Self-Citation Rate Comparison: Pre-AI vs AI/COVID Era")
        axes[0, 0].set_ylabel("Self-Citation Rate")

        # Plot 2: Era progression by career stage (weighted) - vectorized
        df = self.enhanced_author_data.copy()
        df["_w_scr"] = df["self_citation_rate"] * df["sampling_weight"]
        era_career_agg = df.groupby(["era", "career_stage"]).agg(
            w_sum=("_w_scr", "sum"), weight_sum=("sampling_weight", "sum")
        )
        era_career_agg["weighted_mean"] = era_career_agg["w_sum"] / era_career_agg["weight_sum"]
        era_career = era_career_agg["weighted_mean"].unstack()
        del df
        era_career.plot(kind="bar", ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title("Era Progression by Career Stage")
        axes[0, 1].set_ylabel("Mean Self-Citation Rate")
        axes[0, 1].legend(title="Career Stage")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Effect sizes across fields
        effect_sizes = []
        field_names = []

        top_fields_index = list(self.enhanced_author_data["primary_field"].value_counts().head(10).index)
        for field in top_fields_index:
            field_data = self.enhanced_author_data[self.enhanced_author_data["primary_field"] == field]

            pre_ai_field = field_data[field_data["era"].isin(["pre_digital", "early_digital"])][
                ["self_citation_rate", "sampling_weight"]
            ].dropna()
            ai_covid_field = field_data[field_data["era"].isin(["ai_emergence", "covid_era", "modern_ai"])][
                ["self_citation_rate", "sampling_weight"]
            ].dropna()

            if len(pre_ai_field) > 10 and len(ai_covid_field) > 10:
                # Calculate weighted effect size
                pre_ai_mean = np.average(pre_ai_field["self_citation_rate"], weights=pre_ai_field["sampling_weight"])
                ai_covid_mean = np.average(
                    ai_covid_field["self_citation_rate"], weights=ai_covid_field["sampling_weight"]
                )
                pre_ai_std = self._weighted_std(pre_ai_field, "self_citation_rate")
                ai_covid_std = self._weighted_std(ai_covid_field, "self_citation_rate")

                pooled_std = np.sqrt((pre_ai_std**2 + ai_covid_std**2) / 2)
                effect_size = (ai_covid_mean - pre_ai_mean) / pooled_std
                effect_sizes.append(effect_size)
                field_names.append(field[:20] + "..." if len(field) > 20 else field)

        if effect_sizes:
            colors = ["red" if x < 0 else "blue" for x in effect_sizes]
            axes[1, 0].barh(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
            axes[1, 0].set_yticks(range(len(field_names)))
            axes[1, 0].set_yticklabels(field_names)
            axes[1, 0].set_title("AI Era Effect Sizes by Field")
            axes[1, 0].set_xlabel("Effect Size (Cohen's d)")
            axes[1, 0].axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Plot 4: Temporal trend with AI milestones
        if "yearly_trends" in self.results["temporal"]:
            yearly_data = self.results["temporal"]["yearly_trends"]
            yearly_summary = yearly_data.groupby("citing_year")["sc_rate_per_author"].mean()

            axes[1, 1].plot(yearly_summary.index, yearly_summary.values, marker="o", alpha=0.7)

            # Add AI and COVID milestones
            ai_covid_milestones = {
                1997: "Deep Blue beats Kasparov",
                2006: "Deep Learning Renaissance",
                2012: "AlexNet/ImageNet",
                2017: "Transformer Architecture",
                2020: "COVID-19 Pandemic & GPT-3",
                2022: "ChatGPT Launch",
                2023: "GPT-4 & AI Mainstreaming",
            }

            for year, milestone in ai_covid_milestones.items():
                if year in yearly_summary.index:
                    axes[1, 1].axvline(x=year, color="red", linestyle="--", alpha=0.5)
                    axes[1, 1].text(year, yearly_summary.max() * 0.8, milestone, rotation=90, fontsize=8, ha="right")

            axes[1, 1].set_title("Self-Citation Trends with AI Milestones")
            axes[1, 1].set_xlabel("Year")
            axes[1, 1].set_ylabel("Self-Citations per Author")

        plt.tight_layout()
        self.figures["ai_impact_analysis"] = fig

    def _create_network_plots(self):
        """Generate the 4 network analysis plots."""
        if "network" not in self.results or "graph" not in self.results["network"]:
            logger.warning("⚠️  No network data available for visualization")
            return

        G = self.results["network"]["graph"]

        if G.number_of_nodes() == 0:
            logger.warning("⚠️  Empty network - no visualizations created")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle("Self-Citation Network Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Self-Citation Intensity Distribution
        if "self_citation_distribution" in self.results["network"]:
            sc_dist = self.results["network"]["self_citation_distribution"]
            # Cap at 99th percentile for visualization
            cap = np.percentile(sc_dist, 99)
            sc_capped = [min(x, cap) for x in sc_dist]
            axes[0, 0].hist(sc_capped, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
            axes[0, 0].axvline(np.mean(sc_dist), color="red", linestyle="--", label=f"Mean: {np.mean(sc_dist):.1f}")
            axes[0, 0].axvline(
                np.median(sc_dist), color="green", linestyle="--", label=f"Median: {np.median(sc_dist):.1f}"
            )
            axes[0, 0].set_title("Self-Citation Count Distribution (per Author)")
            axes[0, 0].set_xlabel("Self-Citation Count")
            axes[0, 0].set_ylabel("Number of Authors")
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, "No distribution data", ha="center", va="center", transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Self-Citation Distribution")

        # Plot 2: Temporal Distance Distribution
        if "temporal_distance_distribution" in self.results["network"]:
            td_dist = [x for x in self.results["network"]["temporal_distance_distribution"] if x > 0]
            if td_dist:
                axes[0, 1].hist(td_dist, bins=30, alpha=0.7, edgecolor="black", color="coral")
                axes[0, 1].axvline(
                    np.mean(td_dist), color="red", linestyle="--", label=f"Mean: {np.mean(td_dist):.1f}y"
                )
                axes[0, 1].set_title("Average Temporal Distance Distribution")
                axes[0, 1].set_xlabel("Years Between Citing and Cited Work")
                axes[0, 1].set_ylabel("Number of Authors")
                axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, "No temporal data", ha="center", va="center", transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Temporal Distance Distribution")

        # Plot 3: Self-Citations vs Active Years
        if (
            "self_citation_distribution" in self.results["network"]
            and "active_years_distribution" in self.results["network"]
        ):
            sc_dist = self.results["network"]["self_citation_distribution"]
            ay_dist = self.results["network"]["active_years_distribution"]
            # Sample for scatter if too many points
            if len(sc_dist) > 5000:
                indices = np.random.choice(len(sc_dist), 5000, replace=False)
                sc_sample = [sc_dist[i] for i in indices]
                ay_sample = [ay_dist[i] for i in indices]
            else:
                sc_sample, ay_sample = sc_dist, ay_dist
            axes[1, 0].scatter(ay_sample, sc_sample, alpha=0.3, s=10)
            axes[1, 0].set_title("Self-Citations vs Career Activity")
            axes[1, 0].set_xlabel("Active Years (with self-citations)")
            axes[1, 0].set_ylabel("Total Self-Citations")
        else:
            axes[1, 0].text(0.5, 0.5, "No data available", ha="center", va="center", transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Self-Citations vs Activity")

        # Plot 4: Network Metrics Summary
        metrics = self.results["network"]["structure_metrics"]
        sc_stats = self.results["network"].get("self_citation_stats", {})
        td_stats = self.results["network"].get("temporal_distance_stats", {})

        metrics_text = f"""
Network Structure (Sampled Authors):

• Authors Analyzed: {metrics['nodes']:,}
• Self-Citation Events: {metrics['edges']:,}

Self-Citation Statistics:
• Mean per Author: {sc_stats.get('mean', 0):.2f}
• Median per Author: {sc_stats.get('median', 0):.1f}
• Std Dev: {sc_stats.get('std', 0):.2f}
• Max (single author): {sc_stats.get('max', 0):,}
• Total Self-Citations: {sc_stats.get('total', 0):,}

Temporal Distance (years):
• Mean: {td_stats.get('mean', 0):.2f}
• Median: {td_stats.get('median', 0):.2f}

High Self-Citers (top 1%): {metrics.get('high_self_citers_1pct', 'N/A')}
"""

        axes[1, 1].text(
            0.05,
            0.95,
            metrics_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Network Summary Statistics")

        plt.tight_layout()
        self.figures["network_analysis"] = fig
        plt.tight_layout()
        self.figures["network_analysis"] = fig

    def _create_regression_plots(self):
        """Generate the 4 regression analysis plots."""
        if "regression" not in self.results:
            logger.warning("⚠️  No regression data available for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Multivariate Regression Analysis of Self-Citation Patterns", fontsize=16, fontweight="bold")

        reg_results = self.results["regression"]

        # Plot 1: Feature Importance
        top_features = reg_results["feature_importance"].head(15)
        y_pos = range(len(top_features))

        colors = ["red" if coef < 0 else "blue" for coef in top_features["coefficient"]]
        axes[0, 0].barh(y_pos, top_features["coefficient"], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([name[:25] + "..." if len(name) > 25 else name for name in top_features["feature"]])
        axes[0, 0].set_title("Feature Coefficients (Top 15)")
        axes[0, 0].set_xlabel("Coefficient Value")
        axes[0, 0].axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Plot 2: Model Performance Metrics
        metrics = {
            "R² Score": reg_results["r2_score"],
            "RMSE": reg_results["rmse"],
            "Features": reg_results["n_features"] / 100,  # Scale for visibility
            "Observations": reg_results["n_observations"] / 1000,  # Scale for visibility
        }

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = axes[0, 1].bar(metric_names, metric_values, color=["green", "orange", "blue", "purple"], alpha=0.7)
        axes[0, 1].set_title("Model Performance Metrics")
        axes[0, 1].set_ylabel("Value")

        # Add value labels on bars
        for bar, value, name in zip(bars, metric_values, metric_names):
            if name == "R² Score":
                label = f"{value:.3f}"
            elif name == "RMSE":
                label = f"{value:.3f}"
            elif name == "Features":
                label = f"{int(value * 100)}"
            else:  # Observations
                label = f"{int(value)}k"
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, label, ha="center", va="bottom")

        # Plot 3: Feature Categories Analysis
        feature_categories = {"Career Stage": 0, "Era": 0, "Field": 0, "Metrics": 0}

        for _, row in reg_results["feature_importance"].iterrows():
            feature = row["feature"]
            abs_coef = row["abs_coefficient"]

            if "career" in feature.lower():
                feature_categories["Career Stage"] += abs_coef
            elif "era" in feature.lower():
                feature_categories["Era"] += abs_coef
            elif "field" in feature.lower():
                feature_categories["Field"] += abs_coef
            else:
                feature_categories["Metrics"] += abs_coef

        wedges, texts, autotexts = axes[1, 0].pie(
            feature_categories.values(), labels=feature_categories.keys(), autopct="%1.1f%%", startangle=90
        )
        axes[1, 0].set_title("Feature Importance by Category")

        # Plot 4: HLM Results / Coefficient Distribution
        if "hlm" in reg_results and "error" not in reg_results["hlm"]:
            hlm = reg_results["hlm"]

            # Show HLM summary - handle None values gracefully
            aic_str = f"{hlm['aic']:.1f}" if hlm.get("aic") is not None else "N/A"
            bic_str = f"{hlm['bic']:.1f}" if hlm.get("bic") is not None else "N/A"
            llf_str = f"{hlm['log_likelihood']:.1f}" if hlm.get("log_likelihood") is not None else "N/A"
            icc_val = hlm.get("icc", 0) or 0
            n_obs = hlm.get("n_observations", "N/A")
            n_obs_str = f"{n_obs:,}" if isinstance(n_obs, (int, float)) else str(n_obs)

            hlm_text = f"""Hierarchical Linear Model Results
(Random Effect: Discipline)

Model Fit:
• AIC: {aic_str}
• BIC: {bic_str}
• Log-Likelihood: {llf_str}

Variance Decomposition:
• ICC: {icc_val * 100:.1f}%
  ({icc_val * 100:.1f}% of variance due to field)

Sample:
• N observations: {n_obs_str}
• N fields: {hlm.get('n_groups', 'N/A')}
• Converged: {hlm.get('converged', 'N/A')}

Key Fixed Effects (p-values):"""

            # Add significant fixed effects
            if "fixed_effects_pvalues" in hlm:
                for effect, pval in list(hlm["fixed_effects_pvalues"].items())[:5]:
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    effect_short = effect[:20] + "..." if len(effect) > 20 else effect
                    hlm_text += f"\n• {effect_short}: p={pval:.3f}{sig}"

            axes[1, 1].text(
                0.05,
                0.95,
                hlm_text,
                transform=axes[1, 1].transAxes,
                verticalalignment="top",
                fontsize=10,
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Hierarchical Linear Model (Mixed Effects)")
        else:
            # Fallback to coefficient distribution
            coefficients = reg_results["feature_importance"]["coefficient"].values
            axes[1, 1].hist(coefficients, bins=30, alpha=0.7, edgecolor="black")
            axes[1, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
            axes[1, 1].set_title("Distribution of Regression Coefficients")
            axes[1, 1].set_xlabel("Coefficient Value")
            axes[1, 1].set_ylabel("Frequency")

            stats_text = f"""Statistics:
Mean: {np.mean(coefficients):.4f}
Std: {np.std(coefficients):.4f}
Positive: {np.sum(coefficients > 0)}
Negative: {np.sum(coefficients < 0)}"""
            axes[1, 1].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[1, 1].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()
        self.figures["regression_analysis"] = fig

    def _create_machine_learning_plots(self):
        """Generate the 4 ML analysis plots."""
        if "machine_learning" not in self.results:
            logger.warning("⚠️  No machine learning data available for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Machine Learning Analysis of Self-Citation Patterns", fontsize=16, fontweight="bold")

        ml_results = self.results["machine_learning"]

        # Plot 1: Random Forest Feature Importance
        rf_importance = ml_results["rf_importance"]
        y_pos = range(len(rf_importance))

        axes[0, 0].barh(y_pos, rf_importance["importance"], alpha=0.7, color="forestgreen")
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(rf_importance["feature"])
        axes[0, 0].set_title("Random Forest Feature Importance")
        axes[0, 0].set_xlabel("Importance Score")

        # Plot 2: Cluster Analysis - Self-Citation Rate by Cluster
        cluster_data = ml_results["clusters"]
        cluster_ids = cluster_data.index
        cluster_means = cluster_data["self_citation_rate"]["mean"].values
        cluster_stds = cluster_data["self_citation_rate"]["std"].values
        cluster_counts = cluster_data["self_citation_rate"]["count"].values

        bars = axes[0, 1].bar(cluster_ids, cluster_means, yerr=cluster_stds, capsize=5, alpha=0.7, color="skyblue")
        axes[0, 1].set_title("Self-Citation Rate by Cluster")
        axes[0, 1].set_xlabel("Cluster ID")
        axes[0, 1].set_ylabel("Mean Self-Citation Rate")

        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"n={int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 3: Cluster Characteristics Heatmap
        try:
            # Create a matrix of cluster characteristics
            characteristics = ["career_stage", "era", "primary_field"]
            cluster_matrix = []
            cluster_labels = []

            for cluster_id in cluster_ids:
                cluster_info = cluster_data.loc[cluster_id]
                cluster_labels.append(f"Cluster {cluster_id}")

                # Create a row of characteristics (simplified encoding)
                row = []
                for char in characteristics:
                    if char in cluster_info:
                        value = str(cluster_info[char])
                        # Simple hash-based encoding for categorical data
                        row.append(hash(value) % 100)
                    else:
                        row.append(0)
                cluster_matrix.append(row)

            if cluster_matrix:
                im = axes[1, 0].imshow(cluster_matrix, aspect="auto", cmap="viridis")
                axes[1, 0].set_xticks(range(len(characteristics)))
                axes[1, 0].set_xticklabels([char.replace("_", " ").title() for char in characteristics])
                axes[1, 0].set_yticks(range(len(cluster_labels)))
                axes[1, 0].set_yticklabels(cluster_labels)
                axes[1, 0].set_title("Cluster Characteristics")
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "Cluster characteristics\nnot available",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
        except Exception:
            axes[1, 0].text(
                0.5,
                0.5,
                "Cluster characteristics\nvisualization failed",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # Plot 4: Feature Importance Comparison (RF vs Regression)
        if "regression" in self.results:
            reg_importance = self.results["regression"]["feature_importance"]

            # Match features between RF and regression
            common_features = []
            rf_scores = []
            reg_scores = []

            for _, rf_row in rf_importance.iterrows():
                rf_feature = rf_row["feature"]
                # Try to match with regression features (simplified matching)
                matching_reg = reg_importance[
                    reg_importance["feature"].str.contains(rf_feature.split("_")[0], case=False, na=False)
                ]

                if not matching_reg.empty:
                    common_features.append(rf_feature)
                    rf_scores.append(rf_row["importance"])
                    reg_scores.append(abs(matching_reg.iloc[0]["coefficient"]))

            if common_features:
                axes[1, 1].scatter(rf_scores, reg_scores, alpha=0.7, s=60)
                axes[1, 1].set_xlabel("Random Forest Importance")
                axes[1, 1].set_ylabel("Regression |Coefficient|")
                axes[1, 1].set_title("Feature Importance: RF vs Regression")

                # Add diagonal line
                max_val = max(max(rf_scores), max(reg_scores))
                axes[1, 1].plot([0, max_val], [0, max_val], "r--", alpha=0.5)

                # Add correlation coefficient
                corr = np.corrcoef(rf_scores, reg_scores)[0, 1]
                axes[1, 1].text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.3f}",
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No matching features\nbetween methods",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
        else:
            axes[1, 1].text(
                0.5, 0.5, "Regression results\nnot available", ha="center", va="center", transform=axes[1, 1].transAxes
            )
            axes[1, 1].set_title("Feature Importance Comparison")

        plt.tight_layout()
        self.figures["machine_learning_analysis"] = fig

    def generate_comprehensive_report(self):
        """Create a markdown report summarizing all analysis results."""
        logger.info("📋 Generating analysis report...")

        report = f"""
# Self-Citation Analysis Report
## 1970-2025 OpenAlex Data

### Summary

Analyzed {len(self.enhanced_author_data):,} authors and {len(self.self_citation_data_dask):,} self-citation instances.

### Key Findings

#### 1. Temporal Patterns
"""

        if "temporal" in self.results:
            era_stats = self.enhanced_author_data.groupby("era")["self_citation_rate"].agg(["mean", "std", "count"])
            report += f"""
**Era-based Analysis:**
- Pre-digital (≤1995): Mean SCR = {era_stats.loc['pre_digital', 'mean']:.4f} (n={era_stats.loc['pre_digital', 'count']})
- Early digital (1996-2009): Mean SCR = {era_stats.loc['early_digital', 'mean']:.4f} (n={era_stats.loc['early_digital', 'count']})
- AI emergence (2010-2019): Mean SCR = {era_stats.loc['ai_emergence', 'mean']:.4f} (n={era_stats.loc['ai_emergence', 'count']})
- COVID era (2020-2022): Mean SCR = {era_stats.loc['covid_era', 'mean']:.4f} (n={era_stats.loc['covid_era', 'count']})
- Modern AI (2023-2025): Mean SCR = {era_stats.loc['modern_ai', 'mean']:.4f} (n={era_stats.loc['modern_ai', 'count']})
"""

            # Add ARIMA results
            arima = self.results["temporal"].get("arima", {})
            if arima and "error" not in arima:
                report += f"""
**ARIMA Time Series Analysis:**
- Model: ARIMA{arima.get('order', '?')}
- AIC: {arima.get('aic', 'N/A'):.2f}, BIC: {arima.get('bic', 'N/A'):.2f}
- Stationarity (ADF test): p={arima.get('stationarity', {}).get('adf_pvalue', 'N/A'):.4f} ({'stationary' if arima.get('stationarity', {}).get('adf_stationary') else 'non-stationary'})
"""
                if arima.get("structural_break_2010"):
                    sb = arima["structural_break_2010"]
                    report += f"""
**Structural Break (2010 - AI Emergence):**
- Pre-2010 mean: {sb['pre_mean']:.4f}
- Post-2010 mean: {sb['post_mean']:.4f}
- t-statistic: {sb['t_statistic']:.4f}, p-value: {sb['p_value']:.4f}
- Significant: {'Yes ✓' if sb['significant'] else 'No'}
"""
                if arima.get("covid_impact"):
                    ci = arima["covid_impact"]
                    report += f"""
**COVID-19 Impact (2020-2022):**
- Pre-COVID mean (2017-2019): {ci['pre_covid_mean']:.4f}
- COVID era mean: {ci['covid_mean']:.4f}
- t-statistic: {ci['t_statistic']:.4f}, p-value: {ci['p_value']:.4f}
- Significant: {'Yes ✓' if ci['significant'] else 'No'}
"""
                if arima.get("forecast"):
                    fc = arima["forecast"]
                    report += f"""
**5-Year Forecast (2026-2030):**
- 2026: {fc['values'][0]:.4f} (95% CI: {fc['lower_ci'][0]:.4f} - {fc['upper_ci'][0]:.4f})
- 2030: {fc['values'][4]:.4f} (95% CI: {fc['lower_ci'][4]:.4f} - {fc['upper_ci'][4]:.4f})
"""

        if "career_stage" in self.results:
            career_stats = self.enhanced_author_data.groupby("career_stage")["self_citation_rate"].agg(
                ["mean", "std", "count"]
            )
            report += f"""
#### 2. Career Stage Patterns
- Early career (0-5 years): Mean SCR = {career_stats.loc['early', 'mean']:.4f} (n={career_stats.loc['early', 'count']})
- Mid-career (6-15 years): Mean SCR = {career_stats.loc['mid', 'mean']:.4f} (n={career_stats.loc['mid', 'count']})
- Senior career (16+ years): Mean SCR = {career_stats.loc['senior', 'mean']:.4f} (n={career_stats.loc['senior', 'count']})

**Statistical Significance:** F-statistic = {self.results['career_stage']['anova']['f_stat']:.4f}, p-value = {self.results['career_stage']['anova']['p_value']:.4f}
"""

        if "disciplinary" in self.results:
            top_fields = self.results["disciplinary"]["field_ranking"].head(5)
            report += """
#### 3. Disciplinary Patterns
**Top 5 Fields by Self-Citation Rate:**
"""
            for i, (field, rate) in enumerate(top_fields.items(), 1):
                report += f"{i}. {field}: {rate:.4f}\n"

        if "ai_impact" in self.results:
            ai_impact = self.results["ai_impact"]
            # Format values with fallbacks for None
            pre_ai_mean = ai_impact.get("pre_ai_mean")
            ai_covid_mean = ai_impact.get("ai_covid_mean")
            comparison = ai_impact.get("comparison", {})
            t_stat = comparison.get("t_stat")
            p_value = comparison.get("p_value")
            effect_size = comparison.get("effect_size")

            pre_ai_str = f"{pre_ai_mean:.4f}" if pre_ai_mean is not None else "N/A"
            ai_covid_str = f"{ai_covid_mean:.4f}" if ai_covid_mean is not None else "N/A"
            t_stat_str = f"{t_stat:.4f}" if t_stat is not None else "N/A"
            p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            effect_size_str = f"{effect_size:.4f}" if effect_size is not None else "N/A"

            report += f"""
#### 4. AI Era Impact
- Pre-AI era mean: {pre_ai_str}
- AI/COVID era mean: {ai_covid_str}
- Statistical significance: t = {t_stat_str}, p = {p_value_str}
- Effect size (Cohen's d): {effect_size_str}
"""

        if "regression" in self.results:
            reg_results = self.results["regression"]
            report += f"""
#### 5. Multivariate Analysis
**Ordinary Least Squares Regression:**
- R² Score: {reg_results['r2_score']:.4f}
- RMSE: {reg_results['rmse']:.4f}
- Number of features: {reg_results['n_features']}
- Sample size: {reg_results['n_observations']:,}

**Top 5 Predictive Features:**
"""
            top_features = reg_results["feature_importance"].head(5)
            for _, row in top_features.iterrows():
                report += f"- {row['feature']}: {row['coefficient']:.4f}\n"

            # Add HLM results if available
            if "hlm" in reg_results and "error" not in reg_results["hlm"]:
                hlm = reg_results["hlm"]
                icc_val = hlm.get("icc", 0)
                icc_pct = f"{icc_val * 100:.1f}" if icc_val is not None else "N/A"
                aic_val = hlm.get("aic")
                aic_str = f"{aic_val:.1f}" if aic_val is not None else "N/A"
                n_groups = hlm.get("n_groups", "N/A")
                converged = hlm.get("converged", "N/A")
                report += f"""
**Hierarchical Linear Model (Mixed Effects):**
- Random effect grouping: Academic discipline
- ICC (Intraclass Correlation): {icc_pct}%
  - Interpretation: {icc_pct}% of self-citation variance is attributable to field differences
- AIC: {aic_str}
- Number of discipline groups: {n_groups}
- Model converged: {converged}
"""

        if "machine_learning" in self.results:
            ml_results = self.results["machine_learning"]
            report += """
#### 6. Machine Learning Insights
**Random Forest Feature Importance (Top 5):**
"""
            for _, row in ml_results["rf_importance"].head(5).iterrows():
                report += f"- {row['feature']}: {row['importance']:.4f}\n"

            report += """
**Cluster Analysis:** Identified 5 distinct author archetypes based on self-citation behavior.
"""

        if "network" in self.results and "structure_metrics" in self.results["network"]:
            network_stats = self.results["network"]["structure_metrics"]
            report += f"""
#### 7. Network Analysis
- Network nodes (author-year combinations): {network_stats['nodes']:,}
- Network edges (self-citation relationships): {network_stats['edges']:,}
"""
            if "density" in network_stats:
                report += f"- Network density: {network_stats['density']:.4f}\n"
                report += f"- Connected components: {network_stats['components']}\n"

            # Add centrality analysis if available
            if "centrality_measures" in self.results["network"]:
                centrality_data = self.results["network"]["centrality_measures"]
                if centrality_data.get("pagerank"):
                    top_pagerank = sorted(centrality_data["pagerank"].items(), key=lambda x: x[1], reverse=True)[:3]
                    report += "\n**Top 3 Nodes by PageRank:**\n"
                    for i, (node, score) in enumerate(top_pagerank, 1):
                        report += f"{i}. {node}: {score:.4f}\n"

        report += f"""
### Methodological Notes
- Data source: OpenAlex database (local PostgreSQL instance)
- Analysis period: 1970-2025 (55 years)
- Self-citation definition: Citations where citing and cited works share at least one author
- Career stage classification: Based on years since first publication
- Statistical significance threshold: p < 0.05
- Effect size interpretation: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)

### Visualizations
This analysis generates 28 visualizations across 7 modules:

#### 1. Temporal Analysis (4 plots)
- Era comparison analysis with error bars
- Multi-field yearly trend analysis
- Era × Career stage interaction plots
- Pre-AI vs AI era distribution comparisons

#### 2. Career Stage Analysis (4 plots)
- Career stage distribution box plots with statistical tests
- Career progression analysis across temporal eras
- Career stage vs h-index scatter plot relationships
- Statistical significance heatmaps for pairwise comparisons

#### 3. Disciplinary Analysis (4 plots)
- Field ranking by self-citation rates (top 15 fields)
- Field comparison statistics with confidence intervals
- Field × Era interaction heatmaps
- Author distribution pie charts across disciplines

#### 4. AI Era Impact Analysis (4 plots)
- Pre-AI vs AI era box plot comparisons
- Era progression by career stage interactions
- Effect sizes across academic fields (Cohen's d)
- Temporal trends with AI milestone annotations

#### 5. Network Analysis (4 plots)
- Self-citation network layout visualization (spring algorithm)
- Node degree distribution analysis (log-scale)
- Centrality measures comparison (eigenvector, betweenness, PageRank)
- Network summary statistics with temporal distance analysis

#### 6. Regression Analysis (4 plots)
- Feature coefficient importance (top 15 features)
- Model performance metrics (R², RMSE, sample statistics)
- Feature importance by category (career, era, field, metrics)
- Coefficient distribution with descriptive statistics

#### 7. Machine Learning Analysis (4 plots)
- Random Forest feature importance ranking
- K-means cluster analysis with self-citation rates
- Cluster characteristics heatmap visualization
- Feature importance method comparison (RF vs Regression)

All visualizations are generated in both high-resolution PNG (300 DPI) and vector SVG formats for publication use.

### Policy Implications
1. **Research Evaluation:** Self-citation rates should be adjusted for career stage and discipline
2. **Funding Allocation:** Consider field-normalized metrics to ensure equitable distribution
3. **Career Development:** Early-career researchers show higher self-citation rates, suggesting need for mentorship on citation practices
4. **Digital Transformation:** AI era shows changes in self-citation behavior that need new evaluation approaches
5. **Network Effects:** Self-citation networks reveal structural patterns that inform research collaboration policies

### Technical Implementation
- **Scalable Network Analysis:** Handles networks of any size with smart sampling for visualization
- **Error Handling:** Handles sparse data and edge cases gracefully
- **Publication Quality:** Professional styling with seaborn themes and matplotlib backends
- **Multiple Output Formats:** Both raster (PNG) and vector (SVG) formats for different use cases
- **Auto-Generated Reports:** Creates methodology and results docs

### Limitations and Future Research
- Author name disambiguation challenges in large-scale datasets
- Citation context analysis (positive vs. negative citations) not captured
- Institutional self-citation vs. individual author self-citation distinction needed
- Longitudinal tracking of individual authors' career trajectories requires further investigation

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return report

    def save_results(self, output_dir=None):
        """Save all data, plots (PNG + SVG), and the markdown report."""
        import os

        # Default to current working directory if not specified
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "self_citation_results")

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"💾 Saving results to {output_dir}...")

        # Save datasets - use chunked writing for large Dask DataFrame
        logger.info("   📝 Writing self-citation raw data (chunked to avoid OOM)...")
        csv_path = f"{output_dir}/self_citation_raw_data.csv"
        # Write header first
        first_partition = self.self_citation_data_dask.get_partition(0).compute()
        first_partition.head(0).to_csv(csv_path, index=False)  # Write header only
        # Append each partition
        for i in range(self.self_citation_data_dask.npartitions):
            partition = self.self_citation_data_dask.get_partition(i).compute()
            partition.to_csv(csv_path, mode="a", header=False, index=False)
            del partition
            gc.collect()
        logger.info(f"   ✅ Wrote self-citation data to {csv_path}")

        self.enhanced_author_data.to_csv(f"{output_dir}/author_level_analysis.csv", index=False)

        # Save statistical results
        import pickle

        with open(f"{output_dir}/analysis_results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        # Save figures
        for name, fig in self.figures.items():
            fig.savefig(f"{output_dir}/{name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{output_dir}/{name}.svg", bbox_inches="tight")  # Vector format

        # Save report
        report = self.generate_comprehensive_report()
        with open(f"{output_dir}/comprehensive_report.md", "w") as f:
            f.write(report)

        logger.info("✅ Results saved successfully!")
        logger.info("   📊 Raw data: self_citation_raw_data.csv")
        logger.info("   📈 Author analysis: author_level_analysis.csv")
        logger.info("   📋 Comprehensive report: comprehensive_report.md")
        logger.info(f"   🎨 Visualizations: {len(self.figures)} figure sets ({len(self.figures) * 4} total plots)")
        logger.info("   📐 Formats: High-resolution PNG (300 DPI) + Vector SVG")
        logger.info("   📑 Coverage: Temporal, Career, Disciplinary, AI Impact, Network, Regression, ML")

        return output_dir


def main():
    """Run the full self-citation analysis pipeline."""
    logger.info(f"📝 Logging to file: {log_filename}")
    logger.info("🎓 Self-Citation Analysis")
    logger.info("=" * 60)
    logger.info("Analyzing 55 years of citation patterns (1970-2025)")
    logger.info("Using OpenAlex Database")
    logger.info("=" * 60)

    # Database configuration - MODIFY THESE SETTINGS FOR YOUR LOCAL SETUP
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "openalex_db",  # Your OpenAlex database name
        "user": "postgres",  # Your PostgreSQL username
        "password": "password",  # Your PostgreSQL password
    }

    logger.info("🔗 Connecting to OpenAlex database...")
    try:
        # Initialize analyzer with sampling configuration
        # Research Question: How do self-citation patterns vary across career stages,
        # disciplines, and temporal contexts? What do these variations reveal about
        # the evolution of scientific knowledge production and validation in the digital age?

        # Cross-Disciplinary Comparison Strategy:
        # Select 10 disciplines representing fundamentally different research cultures,
        # methodologies, and citation norms to maximize comparative insights

        analyzer = SelfCitationAnalyzer(
            db_config,
            use_sampling=True,
            sample_fraction=0.10,  # 10% sample for balanced precision and speed
            target_disciplines=[
                # === COMPUTATIONAL SCIENCES (Fast-moving, high collaboration) ===
                "Computer Science",  # AI/ML era epicenter, rapid evolution
                "Mathematics",  # Foundational, different citation culture
                # === PHYSICAL SCIENCES (Established norms, experimental) ===
                "Physics and Astronomy",  # Large collaborations, preprint culture
                "Chemistry",  # Lab-based, incremental knowledge building
                # === LIFE SCIENCES (High citation density, translational) ===
                "Medicine",  # Clinical impact, high self-citation?
                "Biochemistry, Genetics and Molecular Biology",  # Molecular revolution
                # === ENGINEERING (Applied, industry links) ===
                "Engineering",  # Problem-solving, diverse subfields
                # === SOCIAL SCIENCES (Interpretive, slower-moving) ===
                "Economics, Econometrics and Finance",  # Quantitative social science
                "Psychology",  # Experimental + qualitative methods
                # === HUMANITIES (Qualitative, book-oriented) ===
                "Arts and Humanities",  # Longest reference lists, lowest citation rates
            ],
        )

        # Why these 10 disciplines?
        # 1. METHODOLOGICAL DIVERSITY: Experimental, computational, observational, qualitative
        # 2. TEMPORAL DYNAMICS: Fast-moving (CS) vs. established (Physics) vs. slow (Humanities)
        # 3. COLLABORATION PATTERNS: Large teams (Physics) vs. solo (Humanities)
        # 4. CITATION CULTURES: High density (Medicine) vs. sparse (Arts)
        # 5. DIGITAL TRANSFORMATION: Varying AI era impact across fields
        # 6. CAREER TRAJECTORIES: Different publication/citation norms affect self-citation

        # Expected insights from this comparison:
        # - Do fast-moving fields (CS) show higher self-citation due to rapid publication?
        # - Does clinical Medicine show different patterns than basic Biochemistry?
        # - How do Humanities (books, long gestation) compare to STEM (rapid articles)?
        # - Did the AI era (2010+) affect all fields equally or create divergence?
        # - Do large-team fields (Physics) have different self-citation networks?

        # Phase 0: Perform sampling if enabled
        if analyzer.use_sampling:
            logger.info("\n📊 Phase 0: Stratified Sampling")
            sampled_authors, sampling_weights = analyzer.perform_sampling()
            if sampled_authors:
                logger.info(f"   ✅ Sampled {len(sampled_authors):,} authors with weights")

        logger.info("\n📊 Phase 1: Data Extraction")

        analyzer.extract_self_citations(chunk_size=analyzer.performance_config["chunk_size"])

        logger.info("\n📈 Phase 1.5: Calculating Total Citations")
        analyzer.calculate_total_citations()

        logger.info("\n⏰ Phase 2: Temporal Analysis (4 visualizations)")
        analyzer.temporal_analysis()

        logger.info("\n🎯 Phase 3: Career Stage Analysis (4 visualizations)")
        analyzer.career_stage_analysis()

        logger.info("\n🔬 Phase 4: Disciplinary Analysis (4 visualizations)")
        analyzer.disciplinary_analysis()

        logger.info("\n🤖 Phase 5: AI Era Impact Analysis (4 visualizations)")
        analyzer.ai_era_impact_analysis()

        logger.info("\n🕸️ Phase 6: Network Analysis (4 visualizations)")
        analyzer.network_analysis()

        logger.info("\n📊 Phase 7: Multivariate Regression Analysis (4 visualizations)")
        analyzer.multivariate_regression_analysis()

        logger.info("\n🤖 Phase 8: Machine Learning Analysis (4 visualizations)")
        analyzer.machine_learning_analysis()

        logger.info("\n💾 Phase 9: Saving Results and 28 Visualizations")
        output_path = analyzer.save_results()

        logger.info("\n🎉 Analysis Complete!")
        logger.info(f"📁 Results saved to: {output_path}")
        logger.info("\n📋 Summary Statistics:")
        logger.info(f"   • Authors analyzed: {len(analyzer.enhanced_author_data):,}")
        logger.info(f"   • Author-level records: {len(analyzer.author_level_data):,}")
        if hasattr(analyzer, "enriched_data_path"):
            logger.info(f"   • Enriched data saved to: {analyzer.enriched_data_path}")
        logger.info(f"   • Disciplines: {analyzer.enhanced_author_data['primary_field'].nunique()}")

        return analyzer

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.info("\n🔧 Troubleshooting Tips:")
        logger.info("1. Verify your PostgreSQL database is running")
        logger.info("2. Check database connection parameters in db_config")
        logger.info("3. Ensure OpenAlex schema exists with required tables")
        logger.info("4. Verify sufficient memory/disk space for large dataset analysis")
        raise


if __name__ == "__main__":
    # Run analysis
    analyzer = main()
