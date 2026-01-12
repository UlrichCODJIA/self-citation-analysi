#!/usr/bin/env python3
"""
Sampling module for bibliometric analysis.

Provides stratified and systematic sampling to get representative
subsets from large author populations. Works with PostgreSQL.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

# Configure logging
log_filename = f"sampling_strategy/sampling_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],  # Also print to console
)
logger = logging.getLogger(__name__)


class BiblioSampler:
    """
    Handles sampling from large author populations.
    Supports stratified and systematic sampling with proper weights.
    """

    def __init__(self, db_config: Dict, random_state: int = 42):
        """
        Set up the sampler.

        Args:
            db_config: Dict with host, port, database, user, password
            random_state: Seed for reproducibility (default: 42)
        """
        self.db_config = db_config
        self.random_state = random_state
        np.random.seed(random_state)

        # Create engine with connection pooling for better performance
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}",
            pool_size=10,  # Connection pool size
            max_overflow=20,  # Extra connections if pool exhausted
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            connect_args={"connect_timeout": 30, "application_name": "bibliometric_sampler"},
        )

        # PostgreSQL performance settings for large queries
        # These are applied per-connection for sampling operations
        self.pg_performance_settings = {
            "work_mem": "256MB",  # Memory for sorting/hashing (default: 4MB)
            "maintenance_work_mem": "512MB",  # Memory for maintenance operations
            "effective_cache_size": "4GB",  # Estimate of OS cache (helps planner)
            "random_page_cost": "1.1",  # SSD optimization (default: 4.0 for HDD)
            "effective_io_concurrency": "200",  # SSD concurrent I/O (default: 1)
            "max_parallel_workers_per_gather": "4",  # Parallel query workers
            "max_parallel_workers": "8",  # Max parallel workers total
            "enable_partitionwise_join": "on",  # Better partition handling
            "enable_partitionwise_aggregate": "on",
            "hash_mem_multiplier": "2.0",  # Extra memory for hash operations
            "temp_buffers": "64MB",  # Temporary table buffers
        }

        logger.info("✅ Database connection initialized with performance optimizations")
        logger.info("   Connection pool: 10 base + 20 overflow")
        logger.info(f"   Work memory: {self.pg_performance_settings['work_mem']}")
        logger.info(f"   Parallel workers: {self.pg_performance_settings['max_parallel_workers_per_gather']}")

        # Temporal eras for stratification
        self.temporal_eras = {
            "pre_digital": (1970, 1995),
            "early_digital": (1996, 2009),
            "ai_emergence": (2010, 2019),
            "covid_era": (2020, 2022),
            "modern_ai": (2023, 2025),
        }

        # Career stages for stratification
        self.career_stages = {
            "early": (0, 5),
            "mid": (6, 15),
            "senior": (16, 999),
        }

    def _apply_pg_performance_settings(self, connection):
        """Apply PostgreSQL performance tuning to a connection."""
        for setting, value in self.pg_performance_settings.items():
            try:
                connection.execute(text(f"SET {setting} = '{value}'"))
            except Exception as e:
                logger.warning(f"   ⚠️  Could not set {setting}: {e}")
                # Continue even if some settings fail (permissions, version compatibility)

        logger.debug("   PostgreSQL performance settings applied to connection")

    def list_available_disciplines(self) -> List[str]:
        """Get all unique discipline names from the database."""
        logger.info("📚 Querying available disciplines from database...")

        query = text(
            """
            SELECT DISTINCT t.field_display_name as discipline
            FROM openalex.topics t
            WHERE t.field_display_name IS NOT NULL
            ORDER BY t.field_display_name
        """
        )

        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
                disciplines = result["discipline"].tolist()

            logger.info(f"   ✅ Found {len(disciplines)} unique disciplines")
            return disciplines

        except Exception as e:
            logger.error(f"   ❌ Failed to query disciplines: {e}")
            return []

    def get_discipline_statistics(self, disciplines: Optional[List[str]] = None) -> pd.DataFrame:
        """Get author counts and work counts per discipline."""
        logger.info("📊 Querying discipline statistics...")

        discipline_filter = ""
        if disciplines:
            escaped_disciplines = ["'" + d.replace("'", "''") + "'" for d in disciplines]
            discipline_filter = f"AND t.field_display_name IN ({', '.join(escaped_disciplines)})"

        query = text(
            f"""
            SELECT
                t.field_display_name as discipline,
                COUNT(DISTINCT wa.author_id) as total_authors,
                COUNT(DISTINCT w.id) as total_works,
                MIN(w.publication_year) as year_start,
                MAX(w.publication_year) as year_end,
                AVG(w.cited_by_count) as avg_citations
            FROM openalex.works w
            INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
            LEFT JOIN openalex.works_topics wt ON w.id = wt.work_id
            LEFT JOIN openalex.topics t ON wt.topic_id = t.id
            WHERE w.publication_year >= 1970
              AND w.publication_year <= 2025
              AND t.field_display_name IS NOT NULL
              {discipline_filter}
            GROUP BY t.field_display_name
            ORDER BY total_authors DESC
        """
        )

        try:
            with self.engine.connect() as conn:
                self._apply_pg_performance_settings(conn)
                stats = pd.read_sql(query, conn)

            logger.info(f"   ✅ Retrieved statistics for {len(stats)} disciplines")
            return stats

        except Exception as e:
            logger.error(f"   ❌ Failed to query discipline statistics: {e}")
            return pd.DataFrame()

    def calculate_sample_size(
        self, population_size: int, confidence_level: float = 0.95, margin_error: float = 0.05, proportion: float = 0.5
    ) -> int:
        """Calculate minimum sample size using Cochran's formula."""
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)

        # Cochran's formula
        n = (z**2 * proportion * (1 - proportion)) / (margin_error**2)

        # Finite population correction
        if population_size < 100000:  # Apply correction for smaller populations
            n_adjusted = n / (1 + (n - 1) / population_size)
        else:
            n_adjusted = n

        return int(np.ceil(n_adjusted))

    def stratified_random_sampling(
        self,
        stratification_criteria: List[str],
        sample_fraction: float = 0.10,
        min_stratum_size: int = 30,
        allocation_method: str = "proportional",
        oversample_rare_strata: Optional[Dict[str, int]] = None,
        target_disciplines: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Sample authors by stratifying on era and/or discipline.

        Args:
            stratification_criteria: e.g., ['era', 'discipline']
            sample_fraction: What percent to sample (0.10 = 10%)
            min_stratum_size: Minimum authors per stratum
            allocation_method: 'proportional', 'equal', 'neyman', or 'disproportionate'
            target_disciplines: If set, only sample from these fields

        Returns:
            DataFrame with sampled authors, stratum labels, and weights
        """
        logger.info("🎯 Starting STRATIFIED RANDOM SAMPLING")
        logger.info(f"   Stratification criteria: {stratification_criteria}")
        logger.info(f"   Sample fraction: {sample_fraction * 100:.1f}%")
        logger.info(f"   Allocation method: {allocation_method}")
        if target_disciplines:
            logger.info(f"   🎯 Target disciplines: {len(target_disciplines)} specified")
            for disc in target_disciplines[:5]:
                logger.info(f"      • {disc}")
            if len(target_disciplines) > 5:
                logger.info(f"      ... and {len(target_disciplines) - 5} more")
        else:
            logger.info("   📊 Sampling from all disciplines")

        # Step 1: Get population statistics for stratification
        logger.info("\n📊 Step 1: Analyzing population structure...")

        # Build discipline filter for SQL query
        discipline_filter_sql = ""
        if target_disciplines:
            # Escape single quotes and build IN clause
            escaped_disciplines = ["'" + d.replace("'", "''") + "'" for d in target_disciplines]
            discipline_filter_sql = f"AND t.field_display_name IN ({', '.join(escaped_disciplines)})"

        population_query = text(
            f"""
            SELECT
                w.publication_year,
                t.field_display_name as discipline,
                COUNT(DISTINCT wa.author_id) as author_count,
                COUNT(DISTINCT w.id) as work_count
            FROM openalex.works w
            INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
            LEFT JOIN openalex.works_topics wt ON w.id = wt.work_id
            LEFT JOIN openalex.topics t ON wt.topic_id = t.id
            WHERE w.publication_year >= 1970
              AND w.publication_year <= 2025
              AND wa.author_id IS NOT NULL
              {discipline_filter_sql}
            GROUP BY w.publication_year, t.field_display_name
            ORDER BY w.publication_year, t.field_display_name
        """
        )

        with self.engine.connect() as conn:
            # Apply performance settings for this query
            self._apply_pg_performance_settings(conn)
            population_stats = pd.read_sql(population_query, conn)

        logger.info(f"   Total years: {population_stats['publication_year'].nunique()}")
        logger.info(f"   Total disciplines: {population_stats['discipline'].nunique()}")
        logger.info(f"   Total author-year combinations: {len(population_stats):,}")

        # Validate target disciplines exist in data
        if target_disciplines:
            found_disciplines = set(population_stats["discipline"].unique())
            requested_disciplines = set(target_disciplines)
            missing_disciplines = requested_disciplines - found_disciplines

            if missing_disciplines:
                logger.warning(f"   ⚠️  {len(missing_disciplines)} requested disciplines not found in data:")
                for disc in list(missing_disciplines)[:5]:
                    logger.warning(f"      • {disc}")
                if len(missing_disciplines) > 5:
                    logger.warning(f"      ... and {len(missing_disciplines) - 5} more")
                logger.info(f"   ℹ️  Available disciplines: {sorted(found_disciplines)}")
            else:
                logger.info(f"   ✅ All {len(target_disciplines)} target disciplines found in data")

        # Step 2: Define strata based on criteria
        logger.info("\n📋 Step 2: Defining strata...")

        # Assign temporal era
        def assign_era(year):
            for era_name, (start, end) in self.temporal_eras.items():
                if start <= year <= end:
                    return era_name
            return "other"

        population_stats["era"] = population_stats["publication_year"].apply(assign_era)

        # Create stratification key
        strata_columns = []
        if "era" in stratification_criteria:
            strata_columns.append("era")
        if "discipline" in stratification_criteria:
            strata_columns.append("discipline")

        if not strata_columns:
            raise ValueError("Must specify at least one stratification criterion")

        population_stats["stratum"] = population_stats[strata_columns].apply(lambda x: "_".join(x.astype(str)), axis=1)

        # Step 3: Calculate sample sizes per stratum
        logger.info("\n📐 Step 3: Calculating stratum sample sizes...")

        stratum_sizes = (
            population_stats.groupby("stratum").agg({"author_count": "sum", "work_count": "sum"}).reset_index()
        )

        total_authors = stratum_sizes["author_count"].sum()

        if allocation_method == "proportional":
            # Proportional allocation
            stratum_sizes["sample_size"] = (stratum_sizes["author_count"] * sample_fraction).apply(
                lambda x: max(int(x), min_stratum_size)
            )

        elif allocation_method == "equal":
            # Equal allocation
            stratum_sizes["sample_size"] = min_stratum_size

        elif allocation_method == "neyman":
            # Neyman optimal allocation (requires variance estimates)
            logger.info("   Using Neyman allocation (optimal for minimizing variance)")
            # Simplified: assume variance proportional to stratum size
            stratum_sizes["weight"] = np.sqrt(stratum_sizes["author_count"])
            total_weight = stratum_sizes["weight"].sum()
            target_total_sample = int(total_authors * sample_fraction)
            stratum_sizes["sample_size"] = (stratum_sizes["weight"] / total_weight * target_total_sample).apply(
                lambda x: max(int(x), min_stratum_size)
            )

        elif allocation_method == "disproportionate":
            # Disproportionate allocation - oversample rare but important groups
            logger.info("   Using DISPROPORTIONATE allocation (oversampling rare/important strata)")
            logger.warning("   ⚠️  Remember: Use sampling weights in all analyses!")

            # Start with proportional allocation as baseline
            stratum_sizes["sample_size"] = (stratum_sizes["author_count"] * sample_fraction).apply(
                lambda x: max(int(x), min_stratum_size)
            )

            # Apply oversampling rules
            if oversample_rare_strata:
                for pattern, target_size in oversample_rare_strata.items():
                    # Match strata containing the pattern
                    matching_strata = stratum_sizes["stratum"].str.contains(pattern, case=False, na=False)
                    original_sizes = stratum_sizes.loc[matching_strata, "sample_size"].sum()

                    # Set target sample size for matching strata
                    stratum_sizes.loc[matching_strata, "sample_size"] = stratum_sizes.loc[
                        matching_strata, "sample_size"
                    ].apply(lambda x: max(x, target_size))

                    new_sizes = stratum_sizes.loc[matching_strata, "sample_size"].sum()
                    logger.info(f"      Oversampling '{pattern}' strata: {original_sizes:,} → {new_sizes:,}")
            else:
                logger.warning("      No oversample_rare_strata specified - using proportional baseline")
                logger.info("      Example: oversample_rare_strata={'top_1%': 500, 'superstar': 1000}")

        else:
            raise ValueError(f"Unknown allocation_method: {allocation_method}")

        # Log sampling plan
        logger.info("\n   Sampling plan summary:")
        logger.info(f"   Total population: {total_authors:,} authors")
        total_sample = stratum_sizes["sample_size"].sum()
        logger.info(f"   Total sample: {total_sample:,} authors ({total_sample / total_authors * 100:.2f}%)")
        logger.info(f"   Number of strata: {len(stratum_sizes)}")
        logger.info("\n   Stratum details:")
        for _, row in stratum_sizes.head(10).iterrows():
            logger.info(f"      {row['stratum']}: {row['author_count']:,} → {row['sample_size']:,}")

        # Step 4: Perform stratified sampling from database
        logger.info("\n🎲 Step 4: Performing stratified random sampling...")

        sampled_authors = []

        for _, stratum in stratum_sizes.iterrows():
            stratum_name = stratum["stratum"]
            sample_size = stratum["sample_size"]

            # Parse stratum back to criteria
            if "era" in stratification_criteria and "discipline" in stratification_criteria:
                # Match era name from the start of stratum_name
                era_name = None
                discipline = None
                for era in self.temporal_eras.keys():
                    if stratum_name.startswith(era + "_"):
                        era_name = era
                        discipline = stratum_name[len(era) + 1 :]  # Skip era + underscore
                        break

                if era_name is None:
                    logger.warning(f"      Skipping stratum with unrecognized era: {stratum_name}")
                    continue

                era_range = self.temporal_eras[era_name]

                # Use hash-based sampling: Much faster than ORDER BY RANDOM()
                # hashtext() is deterministic and indexed, modulo gives us a sample
                # For 8M rows needing 1000 samples, we skip ~8000 rows between each sample
                sample_query = text(
                    """
                    SELECT DISTINCT
                        wa.author_id,
                        w.publication_year,
                        t.field_display_name as discipline,
                        :era_name as era,
                        :stratum as stratum
                    FROM openalex.works w
                    INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
                    LEFT JOIN openalex.works_topics wt ON w.id = wt.work_id
                    LEFT JOIN openalex.topics t ON wt.topic_id = t.id
                    WHERE w.publication_year >= :start_year
                      AND w.publication_year <= :end_year
                      AND t.field_display_name = :discipline
                      AND wa.author_id IS NOT NULL
                      AND (hashtext(wa.author_id) % 100) < :sample_percent
                    LIMIT :sample_size_limit
                """
                )

                with self.engine.connect() as conn:
                    # Apply performance settings for this query
                    self._apply_pg_performance_settings(conn)

                    # Calculate sampling percentage (hash modulo 100)
                    # If we need 1000 samples and expect 10000 authors, sample ~10%
                    # Add buffer to ensure we get enough samples
                    sample_percent = min(50, max(1, int(sample_size / max(stratum["author_count"], 1) * 100 * 2)))

                    logger.info(f"      Querying {stratum_name} (sampling ~{sample_percent}% via hash modulo)...")

                    stratum_sample = pd.read_sql(
                        sample_query,
                        conn,
                        params={
                            "start_year": int(era_range[0]),
                            "end_year": int(era_range[1]),
                            "discipline": str(discipline),
                            "era_name": str(era_name),
                            "stratum": str(stratum_name),
                            "sample_percent": int(sample_percent),
                            "sample_size_limit": int(sample_size * 3),  # Get extra, we'll sample down if needed
                        },
                    )

                    # If we got more than needed, randomly sample down to exact size
                    if len(stratum_sample) > sample_size:
                        stratum_sample = stratum_sample.sample(n=sample_size, random_state=self.random_state)

                if len(stratum_sample) > 0:
                    sampled_authors.append(stratum_sample)
                    logger.info(f"      Sampled {len(stratum_sample):,} authors from {stratum_name}")

        # Combine all strata
        if sampled_authors:
            final_sample = pd.concat(sampled_authors, ignore_index=True)
            logger.info("\n✅ Stratified sampling complete!")
            logger.info(f"   Final sample size: {len(final_sample):,} author-year observations")
            logger.info(f"   Unique authors: {final_sample['author_id'].nunique():,}")

            # Save sample metadata
            sample_metadata = {
                "method": "stratified_random_sampling",
                "criteria": stratification_criteria,
                "allocation": allocation_method,
                "sample_fraction": sample_fraction,
                "total_size": len(final_sample),
                "unique_authors": final_sample["author_id"].nunique(),
                "strata_count": len(stratum_sizes),
                "random_state": self.random_state,
            }

            # Save to file
            os.makedirs("sampling_results", exist_ok=True)
            final_sample.to_parquet("sampling_results/stratified_sample.parquet", compression="snappy", index=False)
            pd.DataFrame([sample_metadata]).to_json(
                "sampling_results/stratified_sample_metadata.json", orient="records", indent=2
            )

            return final_sample
        else:
            logger.error("❌ No samples collected!")
            return pd.DataFrame()

    def systematic_sampling(self, sample_interval: int = 10, random_start: bool = True) -> pd.DataFrame:
        """
        Sample every kth author from an ordered list.
        Good for temporal trends but watch out for periodic patterns.
        """
        logger.info("🎯 Starting SYSTEMATIC SAMPLING")
        logger.info(f"   Sample interval: every {sample_interval}th element")
        logger.info(f"   Random start: {random_start}")

        # Get total author count
        count_query = text(
            """
            SELECT COUNT(DISTINCT author_id) as total_authors
            FROM openalex.works_authorships
            WHERE author_id IS NOT NULL
        """
        )

        with self.engine.connect() as conn:
            # Apply performance settings for this query
            self._apply_pg_performance_settings(conn)
            total_authors = pd.read_sql(count_query, conn)["total_authors"].iloc[0]

        expected_sample_size = total_authors // sample_interval
        logger.info(f"   Population size: {total_authors:,} authors")
        logger.info(f"   Expected sample size: {expected_sample_size:,} authors")

        # Determine starting point
        if random_start:
            start_position = np.random.randint(0, sample_interval)
        else:
            start_position = 0

        logger.info(f"   Starting at position: {start_position}")

        # Systematic sampling query with ROW_NUMBER
        sample_query = text(
            """
            WITH numbered_authors AS (
                SELECT
                    author_id,
                    ROW_NUMBER() OVER (ORDER BY author_id) as row_num
                FROM (
                    SELECT DISTINCT author_id
                    FROM openalex.works_authorships
                    WHERE author_id IS NOT NULL
                ) sub
            )
            SELECT author_id
            FROM numbered_authors
            WHERE (row_num - :start_position) % :interval = 0
              AND row_num >= :start_position
        """
        )

        with self.engine.connect() as conn:
            # Apply performance settings for this query
            self._apply_pg_performance_settings(conn)
            sample = pd.read_sql(
                sample_query, conn, params={"start_position": start_position, "interval": sample_interval}
            )

        logger.info("\n✅ Systematic sampling complete!")
        logger.info(f"   Sample size: {len(sample):,} authors")
        logger.info(f"   Actual sampling rate: {len(sample) / total_authors * 100:.2f}%")

        # Save
        os.makedirs("sampling_results", exist_ok=True)
        sample.to_parquet("sampling_results/systematic_sample.parquet", compression="snappy")

        return sample

    def get_sampling_weights(self, sample: pd.DataFrame, stratum_col: str = "stratum") -> pd.DataFrame:
        """
        Add sampling weights to the data.
        Weight = population_size / sample_size for each stratum.
        Use these weights in all analyses to avoid bias.
        """
        logger.info("⚖️  Calculating sampling weights for proper statistical adjustment...")

        # Check if cached population sizes table exists
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'openalex' AND table_name = 'stratum_population_sizes'
                """
                    )
                )
                table_exists = result.scalar() > 0

                if table_exists:
                    logger.info("   ✅ Reusing cached population sizes (openalex.stratum_population_sizes)")
                    population_sizes = pd.read_sql(
                        "SELECT stratum, population_size FROM openalex.stratum_population_sizes", conn
                    )
                    logger.info(f"   Retrieved population sizes for {len(population_sizes)} strata")
                else:
                    logger.info("   Querying population sizes from database (one-time, ~2 hours)...")
                    logger.info("   Results will be cached in openalex.stratum_population_sizes")

                    # Apply performance settings for this query
                    self._apply_pg_performance_settings(conn)

                    # Build query to get population sizes per stratum
                    # This matches the stratification logic used in sampling
                    population_query = text(
                        """
                        WITH stratified_population AS (
                            SELECT
                                wa.author_id,
                                CASE
                                    WHEN w.publication_year BETWEEN 1970 AND 1995 THEN 'pre_digital'
                                    WHEN w.publication_year BETWEEN 1996 AND 2009 THEN 'early_digital'
                                    WHEN w.publication_year BETWEEN 2010 AND 2019 THEN 'ai_emergence'
                                    WHEN w.publication_year BETWEEN 2020 AND 2022 THEN 'covid_era'
                                    WHEN w.publication_year BETWEEN 2023 AND 2025 THEN 'modern_ai'
                                    ELSE 'other'
                                END as era,
                                COALESCE(t.field_display_name, 'Unknown') as discipline
                            FROM openalex.works w
                            INNER JOIN openalex.works_authorships wa ON w.id = wa.work_id
                            LEFT JOIN openalex.works_topics wt ON w.id = wt.work_id
                            LEFT JOIN openalex.topics t ON wt.topic_id = t.id
                            WHERE w.publication_year >= 1970
                              AND w.publication_year <= 2025
                              AND wa.author_id IS NOT NULL
                        )
                        SELECT
                            era || '_' || discipline as stratum,
                            COUNT(DISTINCT author_id) as population_size
                        FROM stratified_population
                        GROUP BY stratum
                        ORDER BY stratum
                    """
                    )

                    population_sizes = pd.read_sql(population_query, conn)
                    logger.info(f"   Retrieved population sizes for {len(population_sizes)} strata")

                    # Cache to permanent table
                    logger.info("   Caching population sizes to permanent table...")
                    conn.execute(
                        text(
                            """
                        CREATE TABLE openalex.stratum_population_sizes (
                            stratum TEXT PRIMARY KEY,
                            population_size BIGINT,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """
                        )
                    )
                    for _, row in population_sizes.iterrows():
                        conn.execute(
                            text(
                                """
                            INSERT INTO openalex.stratum_population_sizes (stratum, population_size)
                            VALUES (:stratum, :pop_size)
                        """
                            ),
                            {"stratum": row["stratum"], "pop_size": int(row["population_size"])},
                        )
                    conn.commit()
                    logger.info("   ✅ Population sizes cached for future runs")

        except Exception as e:
            logger.error(f"❌ Failed to query population sizes: {e}")
            logger.warning("   Falling back to equal weights (weight=1.0)")
            logger.warning("   ⚠️  Results may be inaccurate. Fix database connection and re-run.")

            sample_with_weights = sample.copy()
            sample_with_weights["sampling_weight"] = 1.0
            return sample_with_weights

        # Calculate sample sizes per stratum
        sample_sizes = sample.groupby(stratum_col).size().reset_index(name="sample_size")

        logger.info(f"   Calculated sample sizes for {len(sample_sizes)} strata")

        # Merge population and sample sizes
        weights_df = population_sizes.merge(sample_sizes, left_on="stratum", right_on=stratum_col, how="inner")

        # Check for missing matches
        if len(weights_df) < len(sample_sizes):
            missing_strata = set(sample_sizes[stratum_col]) - set(weights_df["stratum"])
            logger.warning(f"   ⚠️  {len(missing_strata)} strata not found in population query:")
            for stratum in list(missing_strata)[:5]:
                logger.warning(f"      - {stratum}")
            if len(missing_strata) > 5:
                logger.warning(f"      ... and {len(missing_strata) - 5} more")

        # Calculate sampling weights: weight = N_h / n_h
        weights_df["sampling_weight"] = weights_df["population_size"] / weights_df["sample_size"]

        # Add weights to sample
        sample_with_weights = sample.merge(
            weights_df[["stratum", "sampling_weight"]], left_on=stratum_col, right_on="stratum", how="left"
        )

        # Handle any missing weights (set to 1.0 as fallback)
        missing_weights = sample_with_weights["sampling_weight"].isna().sum()
        if missing_weights > 0:
            logger.warning(f"   ⚠️  {missing_weights} observations have missing weights (set to 1.0)")
            sample_with_weights["sampling_weight"].fillna(1.0, inplace=True)

        # Log weight statistics
        logger.info("\n   ✅ Sampling weights calculated!")
        logger.info(
            f"   Weight range: {sample_with_weights['sampling_weight'].min():.2f} to {sample_with_weights['sampling_weight'].max():.2f}"
        )
        logger.info(f"   Weight mean: {sample_with_weights['sampling_weight'].mean():.2f}")
        logger.info(f"   Weight median: {sample_with_weights['sampling_weight'].median():.2f}")

        # Check for extreme weights (potential issues)
        extreme_threshold = 100
        extreme_weights = (sample_with_weights["sampling_weight"] > extreme_threshold).sum()
        if extreme_weights > 0:
            logger.warning(f"   ⚠️  {extreme_weights} observations have weights > {extreme_threshold}")
            logger.warning("      This suggests some strata are severely undersampled")
            logger.warning("      Consider increasing sample size or revising stratification")

        logger.info("\n   💡 Use 'sampling_weight' column in all analyses!")
        logger.info("      See WEIGHTED_ANALYSIS_GUIDE.md for examples")

        return sample_with_weights

    def validate_sample_representativeness(
        self,
        sample: pd.DataFrame,
        population: Optional[pd.DataFrame] = None,
        validation_criteria: List[str] = ["publication_year", "discipline"],
    ) -> Dict:
        """
        Check if the sample looks representative.
        Runs chi-square and KS tests if population data is provided.
        """
        logger.info("\n🔍 Validating sample representativeness...")

        validation_results = {}

        # TEST 1: Coverage checks
        if "publication_year" in sample.columns:
            _ = sample["publication_year"].value_counts(normalize=True).sort_index()
            logger.info("   ✓ Temporal coverage check")
            validation_results["temporal_coverage"] = {
                "min_year": int(sample["publication_year"].min()),
                "max_year": int(sample["publication_year"].max()),
                "span": int(sample["publication_year"].max() - sample["publication_year"].min()),
                "unique_years": int(sample["publication_year"].nunique()),
                "coverage": "complete" if sample["publication_year"].nunique() >= 50 else "partial",
            }

            logger.info(
                f"      Years: {validation_results['temporal_coverage']['min_year']} - "
                f"{validation_results['temporal_coverage']['max_year']} "
                f"({validation_results['temporal_coverage']['unique_years']} unique years)"
            )

        if "discipline" in sample.columns:
            unique_disciplines = sample["discipline"].nunique()
            validation_results["disciplinary_coverage"] = {
                "unique_disciplines": int(unique_disciplines),
                "coverage": (
                    "excellent" if unique_disciplines >= 20 else ("good" if unique_disciplines >= 10 else "limited")
                ),
            }
            logger.info(f"   ✓ Disciplinary coverage: {unique_disciplines} unique disciplines")

        # If no population data provided, stop here
        if population is None:
            logger.info("\n   ℹ️  No population data provided - skipping statistical tests")
            logger.info("      Provide population DataFrame to enable Chi-square and KS tests")
            logger.info("\n✅ Coverage validation complete.")
            # Add is_representative key to prevent KeyError downstream
            validation_results["is_representative"] = True  # Assume representative if no population to compare
            return validation_results

        # TEST 2: Chi-square test for disciplinary distribution
        if (
            "discipline" in validation_criteria
            and "discipline" in sample.columns
            and "discipline" in population.columns
        ):
            logger.info("\n   📊 Chi-square test (disciplinary distribution)...")

            try:
                pop_disc = population["discipline"].value_counts()
                sample_disc = sample["discipline"].value_counts()

                # Find common disciplines
                common_disciplines = pop_disc.index.intersection(sample_disc.index)

                if len(common_disciplines) > 0:
                    # Expected frequencies in sample based on population proportions
                    pop_proportions = pop_disc[common_disciplines] / len(population)
                    expected_counts = pop_proportions * len(sample)
                    observed_counts = sample_disc[common_disciplines]

                    # Chi-square test
                    chi2, p_value = stats.chisquare(observed_counts, expected_counts)

                    validation_results["discipline_chi2"] = {
                        "chi2_statistic": float(chi2),
                        "p_value": float(p_value),
                        "df": len(common_disciplines) - 1,
                        "interpretation": (
                            "PASS - Sample represents population"
                            if p_value > 0.05
                            else "FAIL - Significant difference detected"
                        ),
                        "alpha": 0.05,
                    }

                    logger.info(f"      χ² statistic: {chi2:.2f}")
                    logger.info(f"      p-value: {p_value:.4f}")
                    logger.info(f"      Result: {validation_results['discipline_chi2']['interpretation']}")

                    if p_value <= 0.05:
                        logger.warning("      ⚠️  Sample distribution differs significantly from population!")
                        logger.warning("         Consider revising sampling strategy or increasing sample size")
                else:
                    logger.warning("      ⚠️  No common disciplines found between sample and population")

            except Exception as e:
                logger.error(f"      ❌ Chi-square test failed: {e}")

        # TEST 3: Kolmogorov-Smirnov test for temporal distribution
        if (
            "publication_year" in validation_criteria
            and "publication_year" in sample.columns
            and "publication_year" in population.columns
        ):
            logger.info("\n   📊 Kolmogorov-Smirnov test (temporal distribution)...")

            try:
                # Remove NaN values
                pop_years = population["publication_year"].dropna()
                sample_years = sample["publication_year"].dropna()

                # KS test (two-sample)
                ks_stat, p_value = stats.ks_2samp(pop_years, sample_years)

                validation_results["temporal_ks"] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "interpretation": (
                        "PASS - Distributions are similar"
                        if p_value > 0.05
                        else "FAIL - Distributions differ significantly"
                    ),
                    "alpha": 0.05,
                }

                logger.info(f"      KS statistic (D): {ks_stat:.4f}")
                logger.info(f"      p-value: {p_value:.4f}")
                logger.info(f"      Result: {validation_results['temporal_ks']['interpretation']}")

                if p_value <= 0.05:
                    logger.warning("      ⚠️  Temporal distribution differs significantly from population!")
                    logger.warning("         Sample may not represent the full time span properly")

            except Exception as e:
                logger.error(f"      ❌ KS test failed: {e}")

        # TEST 4: Effect size (Cohen's d) for temporal mean
        if "publication_year" in sample.columns and "publication_year" in population.columns:
            logger.info("\n   📊 Effect size analysis (Cohen's d)...")

            try:
                pop_mean = population["publication_year"].mean()
                sample_mean = sample["publication_year"].mean()
                pop_std = population["publication_year"].std()
                sample_std = sample["publication_year"].std()

                # Pooled standard deviation
                pooled_std = np.sqrt((pop_std**2 + sample_std**2) / 2)

                # Cohen's d
                cohens_d = abs(sample_mean - pop_mean) / pooled_std

                # Interpret effect size
                if cohens_d < 0.2:
                    interpretation = "Negligible (< 0.2) - Excellent match"
                elif cohens_d < 0.5:
                    interpretation = "Small (0.2-0.5) - Acceptable match"
                elif cohens_d < 0.8:
                    interpretation = "Medium (0.5-0.8) - Noticeable difference"
                else:
                    interpretation = "Large (≥ 0.8) - Substantial difference"

                validation_results["effect_size"] = {
                    "cohens_d": float(cohens_d),
                    "population_mean": float(pop_mean),
                    "sample_mean": float(sample_mean),
                    "difference": float(sample_mean - pop_mean),
                    "interpretation": interpretation,
                }

                logger.info(f"      Population mean year: {pop_mean:.2f}")
                logger.info(f"      Sample mean year: {sample_mean:.2f}")
                logger.info(f"      Difference: {sample_mean - pop_mean:.2f} years")
                logger.info(f"      Cohen's d: {cohens_d:.4f}")
                logger.info(f"      Interpretation: {interpretation}")

                if cohens_d >= 0.5:
                    logger.warning("      ⚠️  Effect size indicates noticeable difference between sample and population")

            except Exception as e:
                logger.error(f"      ❌ Effect size calculation failed: {e}")

        # TEST 5: Stratum size validation
        if "stratum" in sample.columns:
            logger.info("\n   📊 Stratum size validation...")

            stratum_sizes = sample.groupby("stratum").size()
            min_size = stratum_sizes.min()
            small_strata = (stratum_sizes < 30).sum()

            validation_results["stratum_validation"] = {
                "total_strata": len(stratum_sizes),
                "min_stratum_size": int(min_size),
                "strata_below_30": int(small_strata),
                "status": "PASS" if small_strata == 0 else f"WARNING - {small_strata} strata have n < 30",
            }

            logger.info(f"      Total strata: {len(stratum_sizes)}")
            logger.info(f"      Minimum stratum size: {min_size}")

            if small_strata > 0:
                logger.warning(f"      ⚠️  {small_strata} strata have fewer than 30 observations")
                logger.warning("         Statistical inference may be unreliable for these strata")
                logger.warning("         Consider increasing sample size or combining small strata")

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("   VALIDATION SUMMARY")
        logger.info("=" * 60)

        passed_tests = 0
        total_tests = 0

        if "discipline_chi2" in validation_results:
            total_tests += 1
            if validation_results["discipline_chi2"]["p_value"] > 0.05:
                passed_tests += 1
                logger.info("   ✅ Chi-square test: PASSED")
            else:
                logger.info("   ❌ Chi-square test: FAILED")

        if "temporal_ks" in validation_results:
            total_tests += 1
            if validation_results["temporal_ks"]["p_value"] > 0.05:
                passed_tests += 1
                logger.info("   ✅ KS test: PASSED")
            else:
                logger.info("   ❌ KS test: FAILED")

        if "effect_size" in validation_results:
            total_tests += 1
            if validation_results["effect_size"]["cohens_d"] < 0.2:
                passed_tests += 1
                logger.info("   ✅ Effect size: PASSED (negligible difference)")
            elif validation_results["effect_size"]["cohens_d"] < 0.5:
                logger.info("   ⚠️  Effect size: ACCEPTABLE (small difference)")
            else:
                logger.info("   ❌ Effect size: CAUTION (noticeable difference)")

        if total_tests > 0:
            logger.info(f"\n   Overall: {passed_tests}/{total_tests} tests passed")

            if passed_tests == total_tests:
                logger.info("\n   ✅ Sample is REPRESENTATIVE of population!")
            elif passed_tests >= total_tests * 0.5:
                logger.info("\n   ⚠️  Sample is MOSTLY representative (some concerns)")
            else:
                logger.info("\n   ❌ Sample may NOT be representative!")
                logger.info("      Consider revising sampling strategy")

        logger.info("=" * 60)

        return validation_results

    def calculate_weighted_statistics(
        self,
        sample: pd.DataFrame,
        value_col: str,
        weight_col: str = "sampling_weight",
        groupby_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate weighted mean, std, and confidence intervals.
        Use this instead of raw mean() when working with stratified samples.
        """
        logger.info(f"\n⚖️  Calculating WEIGHTED statistics for '{value_col}'...")

        if weight_col not in sample.columns:
            logger.error(f"❌ Column '{weight_col}' not found! Cannot compute weighted statistics.")
            raise ValueError(f"Sampling weight column '{weight_col}' missing. Run get_sampling_weights() first.")

        def weighted_stats_for_group(group_data):
            """Calculate weighted statistics for a group"""
            values = group_data[value_col].values
            weights = group_data[weight_col].values

            # Remove NaN values
            mask = ~np.isnan(values)
            values = values[mask]
            weights = weights[mask]

            if len(values) == 0:
                return pd.Series(
                    {
                        "weighted_mean": np.nan,
                        "weighted_std": np.nan,
                        "weighted_se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "n": 0,
                        "effective_n": 0,
                    }
                )

            # Weighted mean
            weighted_mean = np.average(values, weights=weights)

            # Weighted variance
            weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
            weighted_std = np.sqrt(weighted_var)

            # Effective sample size (accounts for unequal weights)
            # Kish's design effect: n_eff = (Σw)² / Σ(w²)
            effective_n = (weights.sum()) ** 2 / (weights**2).sum()

            # Standard error of weighted mean
            # SE = sqrt(Σ(w² × (x - x̄)²) / (Σw)²)
            weighted_se = np.sqrt(np.sum(weights**2 * (values - weighted_mean) ** 2) / (weights.sum()) ** 2)

            # 95% confidence interval
            # Using t-distribution with effective sample size
            from scipy.stats import t as t_dist

            t_critical = t_dist.ppf(0.975, df=max(1, int(effective_n) - 1))
            ci_lower = weighted_mean - t_critical * weighted_se
            ci_upper = weighted_mean + t_critical * weighted_se

            return pd.Series(
                {
                    "weighted_mean": weighted_mean,
                    "weighted_std": weighted_std,
                    "weighted_se": weighted_se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": len(values),
                    "effective_n": effective_n,
                }
            )

        if groupby_col:
            # Grouped weighted statistics
            logger.info(f"   Grouped by: {groupby_col}")
            results = sample.groupby(groupby_col).apply(weighted_stats_for_group).reset_index()
        else:
            # Overall weighted statistics
            results = pd.DataFrame([weighted_stats_for_group(sample)])

        logger.info("   ✅ Weighted statistics calculated!")
        logger.info("   Use these estimates for population-level inference.")

        return results

    def calculate_design_effect(
        self, sample: pd.DataFrame, stratum_col: str = "stratum", weight_col: str = "sampling_weight"
    ) -> float:
        """
        Calculate design effect (DEFF) to measure sampling efficiency.
        DEFF < 1.0 means stratification helped. Typical good value: 0.6-0.8.
        """
        logger.info("\n📊 Calculating Design Effect (DEFF)...")

        if weight_col not in sample.columns:
            logger.error(f"❌ Column '{weight_col}' not found!")
            raise ValueError(f"Sampling weight column '{weight_col}' missing.")

        weights = sample[weight_col].values

        # Kish's design effect formula
        # DEFF = n × Σ(w²) / (Σw)²
        # where w are the sampling weights
        n = len(weights)
        sum_w = weights.sum()
        sum_w2 = (weights**2).sum()

        deff = n * sum_w2 / (sum_w**2)

        logger.info(f"   Design Effect (DEFF): {deff:.3f}")

        if deff < 1.0:
            efficiency_gain = (1.0 - deff) * 100
            logger.info(f"   ✅ Stratification IMPROVED efficiency by {efficiency_gain:.1f}%")
            logger.info("      (Variance reduced vs simple random sampling)")
        elif deff > 1.0:
            efficiency_loss = (deff - 1.0) * 100
            logger.warning(f"   ⚠️  Stratification REDUCED efficiency by {efficiency_loss:.1f}%")
            logger.warning("      Consider revising stratification criteria")
        else:
            logger.info("   Stratification has same efficiency as simple random sampling")

        # Effective sample size
        n_eff = n / deff
        logger.info(f"   Effective sample size: {n_eff:.0f} (vs actual n={n})")

        return deff

    def post_stratification_adjustment(
        self, sample: pd.DataFrame, population_proportions: pd.DataFrame, stratum_col: str = "stratum"
    ) -> pd.DataFrame:
        """
        Reweight sample to match known population proportions.
        Use when sample proportions don't match population (e.g., non-response).
        """
        logger.info("\n🔄 Applying post-stratification adjustment...")

        # Calculate sample proportions
        total_pop = population_proportions["population_size"].sum()
        population_proportions["pop_proportion"] = population_proportions["population_size"] / total_pop

        sample_counts = sample.groupby(stratum_col).size().reset_index(name="sample_size")
        total_sample = sample_counts["sample_size"].sum()
        sample_counts["sample_proportion"] = sample_counts["sample_size"] / total_sample

        # Merge and calculate adjustment factors
        adjustment_factors = population_proportions.merge(
            sample_counts[[stratum_col, "sample_proportion"]], on=stratum_col, how="left"
        )

        adjustment_factors["adjustment_factor"] = (
            adjustment_factors["pop_proportion"] / adjustment_factors["sample_proportion"]
        )

        # Apply to sample
        sample_adjusted = sample.merge(
            adjustment_factors[[stratum_col, "adjustment_factor"]], on=stratum_col, how="left"
        )

        # Calculate post-stratification weight
        if "sampling_weight" in sample_adjusted.columns:
            sample_adjusted["post_strat_weight"] = (
                sample_adjusted["sampling_weight"] * sample_adjusted["adjustment_factor"]
            )
        else:
            sample_adjusted["post_strat_weight"] = sample_adjusted["adjustment_factor"]

        logger.info("   ✅ Post-stratification weights calculated!")
        logger.info("   Use 'post_strat_weight' column in analyses")

        # Log adjustments
        logger.info("\n   Adjustment factors by stratum:")
        for _, row in adjustment_factors.head(10).iterrows():
            logger.info(
                f"      {row[stratum_col]}: {row['adjustment_factor']:.3f} "
                f"(pop: {row['pop_proportion']:.3%}, sample: {row['sample_proportion']:.3%})"
            )

        return sample_adjusted


# Example usage function
def demo_sampling_workflow():
    """Demo the full sampling workflow with weights and validation."""
    # Database configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "openalex_db",
        "user": "postgres",
        "password": "password",
    }

    # Initialize sampler
    sampler = BiblioSampler(db_config, random_state=42)

    print("\n" + "=" * 80)
    print("DEMO: Complete Sampling Workflow for OpenAlex Data")
    print("=" * 80)

    # STEP 1: Stratified random sampling
    print("\n📊 STEP 1: Performing stratified random sampling...")
    stratified_sample = sampler.stratified_random_sampling(
        stratification_criteria=["era", "discipline"],
        sample_fraction=0.10,  # 10% sample
        min_stratum_size=50,
        allocation_method="proportional",
    )

    print(f"   ✅ Sample created: {len(stratified_sample):,} observations")

    # STEP 2: Calculate sampling weights
    print("\n⚖️  STEP 2: Calculating sampling weights...")
    print("   (Needed for accurate analysis)")
    stratified_sample_weighted = sampler.get_sampling_weights(stratified_sample, stratum_col="stratum")

    # Save weighted sample
    os.makedirs("sampling_results", exist_ok=True)
    stratified_sample_weighted.to_parquet(
        "sampling_results/stratified_sample_with_weights.parquet", compression="snappy", index=False
    )
    print("   ✅ Weighted sample saved to: sampling_results/stratified_sample_with_weights.parquet")

    # STEP 3: Validate representativeness
    print("\n🔍 STEP 3: Validating sample representativeness...")
    print("   (Provide population DataFrame for full validation)")

    # Basic validation (without population)
    # validation_results = sampler.validate_sample_representativeness(
    #     stratified_sample_weighted,
    #     population=None,  # Set to population DataFrame for statistical tests
    #     validation_criteria=["publication_year", "discipline"],
    # )

    # To run full validation with Chi-square and KS tests, uncomment:
    population_query = """
        SELECT w.publication_year, t.field_display_name as discipline
        FROM openalex.works w
        LEFT JOIN openalex.works_topics wt ON w.id = wt.work_id
        LEFT JOIN openalex.topics t ON wt.topic_id = t.id
        WHERE w.publication_year BETWEEN 1970 AND 2025
    """
    population = pd.read_sql(population_query, sampler.engine)
    validation_results = sampler.validate_sample_representativeness(stratified_sample_weighted, population=population)

    # STEP 4: CALCULATE DESIGN EFFECT
    print("\n📈 STEP 4: Calculating design effect...")
    deff = sampler.calculate_design_effect(
        stratified_sample_weighted, stratum_col="stratum", weight_col="sampling_weight"
    )

    # STEP 5: DEMONSTRATION - Weighted vs Unweighted Analysis
    print("\n" + "=" * 80)
    print("DEMO: Why Sampling Weights Matter")
    print("=" * 80)

    # Simulate some self-citation data for demonstration
    if "era" in stratified_sample_weighted.columns:
        # Create dummy self-citation rates (for demo only)
        np.random.seed(42)
        stratified_sample_weighted["self_citation_rate"] = np.random.beta(2, 10, len(stratified_sample_weighted)) * 0.20

        print("\n❌ WRONG: Unweighted mean by era")
        unweighted_means = stratified_sample_weighted.groupby("era")["self_citation_rate"].mean()
        print(unweighted_means)

        print("\n✅ CORRECT: Weighted mean by era")
        weighted_means = stratified_sample_weighted.groupby("era").apply(
            lambda x: np.average(x["self_citation_rate"], weights=x["sampling_weight"])
        )
        print(weighted_means)

        print("\n⚠️  Difference (Weighted - Unweighted):")
        difference = weighted_means - unweighted_means
        print(difference)
        print(
            f"\nMaximum bias: {difference.abs().max():.4f} ({difference.abs().max() / unweighted_means.mean() * 100:.1f}% relative error)"
        )

    # OPTIONAL: Systematic sampling for comparison
    print("\n" + "=" * 80)
    print("OPTIONAL: Systematic Sampling (for comparison)")
    print("=" * 80)
    systematic_sample = sampler.systematic_sampling(sample_interval=10, random_start=True)  # Every 10th author
    print(f"   Systematic sample size: {len(systematic_sample):,}")

    # SUMMARY
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"✅ Stratified sample: {len(stratified_sample_weighted):,} observations")
    print(
        f"✅ Sampling weights: Calculated (range: {stratified_sample_weighted['sampling_weight'].min():.2f} - {stratified_sample_weighted['sampling_weight'].max():.2f})"
    )
    print(f"✅ Validation: {len(validation_results)} checks performed")
    print(f"✅ Design effect (DEFF): {deff:.3f}")
    print(f"✅ Systematic sample: {len(systematic_sample):,} (optional)")
    print("\n📁 Files saved to: sampling_results/")
    print("\n📖 Next steps:")
    print("   1. Review WEIGHTED_ANALYSIS_GUIDE.md for analysis examples")
    print("   2. Run: python main/weighted_analysis_examples.py")
    print("   3. Integrate with your self_citation_analysis.py")
    print("\n⚠️  Remember to use 'sampling_weight' in your analyses!")
    print("=" * 80)

    return stratified_sample_weighted, systematic_sample, validation_results


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(" Sampling Strategy Module")
    logger.info(" Stratified sampling for OpenAlex data")
    logger.info("=" * 80)

    # Run demo
    demo_sampling_workflow()
