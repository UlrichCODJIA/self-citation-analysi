# Self-Citation Analysis Project

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18220363.svg)](https://doi.org/10.5281/zenodo.18220363)

Analyzes 55 years (1970-2025) of self-citation patterns in academic research using the OpenAlex database.

## Overview

This project provides a complete pipeline for:

- Extracting citation data from OpenAlex database snapshots
- Processing large datasets on memory-constrained hardware (~15GB RAM)
- Running 7 analysis modules with 28 visualizations
- Generating statistical reports with ARIMA forecasting

## Features

- **Chunked Processing**: Handles massive datasets without running out of memory
- **Stratified Sampling**: Get representative subsets with proper statistical weights
- **7 Analysis Modules**:
  1. Temporal trends (with ARIMA forecasting)
  2. Career stage breakdown (early/mid/senior)
  3. Cross-discipline comparison
  4. AI era impact (pre-2010 vs post-2010)
  5. Network analysis
  6. Regression modeling (OLS + Mixed Effects)
  7. ML clustering (Random Forest + K-means)

## Requirements

- Python 3.11+
- PostgreSQL with OpenAlex data loaded
- ~15GB RAM recommended

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

- `pandas`, `dask`, `numpy`, `pyarrow`
- `psycopg2-binary`, `sqlalchemy`
- `scikit-learn`, `networkx`, `scipy`, `statsmodels`
- `matplotlib`, `seaborn`, `plotly`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/UlrichCODJIA/self-citation-analysi.git
    cd citation-research
    ```

2. Create and activate virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up PostgreSQL database with OpenAlex data:

    ```bash
    psql -d openalex_db -f openalex-pg-schema.sql
    psql -d openalex_db -f copy-openalex-csv.sql
    ```

## Usage

### Run Full Analysis

```bash
python main/self_citation_analysis.py
```

This will:

1. Extract self-citation data (with checkpoints for resume)
2. Run all 7 analysis modules
3. Generate 28 plots (PNG + SVG)
4. Save results to `self_citation_results/`

### Run Sampling Only

```bash
python main/sampling_strategy.py
```

Performs stratified sampling and saves weighted sample to `sampling_results/`.

### Process New OpenAlex Snapshot

```bash
# 1. Extract JSONL to CSV
python flatten-openalex-jsonl.py

# 2. Load into PostgreSQL
psql -d openalex_db -f openalex-pg-schema.sql
psql -d openalex_db -f copy-openalex-csv.sql
```

## Project Structure

```bash
citation-research/
├── main/
│   ├── self_citation_analysis.py   # Main analysis script
│   ├── sampling_strategy.py        # Stratified sampling module
│   └── list_disciplines.py         # List available disciplines
├── manuscript/                      # LaTeX manuscript files
├── self_citation_results/           # Output plots and reports
├── sampling_results/                # Sampling output files
├── temp_self_citation_chunks/       # Checkpoint files (don't delete during runs)
├── temp_enriched_chunks/            # Enriched data cache
├── flatten-openalex-jsonl.py        # JSONL to CSV converter
├── openalex-pg-schema.sql           # PostgreSQL schema
├── copy-openalex-csv.sql            # CSV import script
└── requirements.txt                 # Python dependencies
```

## Configuration

Edit database connection in `main/self_citation_analysis.py`:

```python
db_config = {
    "host": "localhost",
    "port": "5432",
    "database": "openalex_db",
    "user": "postgres",
    "password": "your_password",
}
```

### Sampling Options

```python
analyzer = SelfCitationAnalyzer(
    db_config,
    use_sampling=True,           # Enable stratified sampling
    sample_fraction=0.10,        # 10% sample
    target_disciplines=[...],    # Specific fields to analyze
)
```

## Output Files

After running the analysis:

- `self_citation_results/comprehensive_report.md` - Full analysis report
- `self_citation_results/author_level_analysis.csv` - Author-level data
- `self_citation_results/*.png` / `*.svg` - Visualizations (28 plots)
- `sampling_results/stratified_sample.parquet` - Weighted sample data

## Checkpoints & Resume

The pipeline saves progress to:

- `temp_self_citation_chunks/` - Year-range query results
- `temp_enriched_chunks/` - Enriched data with metadata

If a run fails, just restart - it will pick up where it left off.

## Key Definitions

- **Self-citation**: When citing and cited works share at least one author
- **Career stages**:
  - Early: 0-5 years since first publication
  - Mid: 6-15 years
  - Senior: 16+ years
- **Temporal eras**:
  - Pre-digital: 1970-1995
  - Early digital: 1996-2009
  - AI emergence: 2010-2019
  - COVID era: 2020-2022
  - Modern AI: 2023-2025

## Performance Tips

- Use SSD storage for temp files
- Monitor with `htop` during long runs
- Check PostgreSQL temp file usage with `pg_stat_database`
- Typical full run: 4-8 hours on reference hardware

## Troubleshooting

**Out of memory errors:**

- Reduce `chunk_size` in analysis config
- Enable sampling with smaller `sample_fraction`

**Slow queries:**

- Check PostgreSQL indexes exist
- Run `ANALYZE` on tables after loading

**Resume from failure:**

- Just re-run the script - checkpoints are automatic

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
Self-Citation Analysis Project (2026). https://doi.org/10.5281/zenodo.18220363
```
