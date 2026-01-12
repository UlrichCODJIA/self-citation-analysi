import csv
import glob
import gzip
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import orjson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("openalex_flatten.log")],
)
logger = logging.getLogger(__name__)

SNAPSHOT_DIR = "openalex-snapshot"
CSV_DIR = "csv-files"
CHECKPOINT_DIR = "checkpoints"

if not os.path.exists(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

FILES_PER_ENTITY = int(os.environ.get("OPENALEX_DEMO_FILES_PER_ENTITY", "0"))
NUM_WORKERS = cpu_count()
logger.info(f"System has {cpu_count()} CPUs, using {NUM_WORKERS} workers")
BATCH_SIZE = 10000

csv_files = {
    "authors": {
        "authors": {
            "name": os.path.join(CSV_DIR, "authors.csv.gz"),
            "columns": [
                "id",
                "orcid",
                "display_name",
                "display_name_alternatives",
                "works_count",
                "cited_by_count",
                "last_known_institution",
                "works_api_url",
                "updated_date",
            ],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "authors_ids.csv.gz"),
            "columns": ["author_id", "openalex", "orcid", "scopus", "twitter", "wikipedia", "mag"],
        },
        "counts_by_year": {
            "name": os.path.join(CSV_DIR, "authors_counts_by_year.csv.gz"),
            "columns": ["author_id", "year", "works_count", "cited_by_count", "oa_works_count"],
        },
    },
    "concepts": {
        "concepts": {
            "name": os.path.join(CSV_DIR, "concepts.csv.gz"),
            "columns": [
                "id",
                "wikidata",
                "display_name",
                "level",
                "description",
                "works_count",
                "cited_by_count",
                "image_url",
                "image_thumbnail_url",
                "works_api_url",
                "updated_date",
            ],
        },
        "ancestors": {
            "name": os.path.join(CSV_DIR, "concepts_ancestors.csv.gz"),
            "columns": ["concept_id", "ancestor_id"],
        },
        "counts_by_year": {
            "name": os.path.join(CSV_DIR, "concepts_counts_by_year.csv.gz"),
            "columns": ["concept_id", "year", "works_count", "cited_by_count", "oa_works_count"],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "concepts_ids.csv.gz"),
            "columns": ["concept_id", "openalex", "wikidata", "wikipedia", "umls_aui", "umls_cui", "mag"],
        },
        "related_concepts": {
            "name": os.path.join(CSV_DIR, "concepts_related_concepts.csv.gz"),
            "columns": ["concept_id", "related_concept_id", "score"],
        },
    },
    "topics": {
        "topics": {
            "name": os.path.join(CSV_DIR, "topics.csv.gz"),
            "columns": [
                "id",
                "display_name",
                "subfield_id",
                "subfield_display_name",
                "field_id",
                "field_display_name",
                "domain_id",
                "domain_display_name",
                "description",
                "keywords",
                "works_api_url",
                "wikipedia_id",
                "works_count",
                "cited_by_count",
                "updated_date",
                "siblings",
            ],
        }
    },
    "institutions": {
        "institutions": {
            "name": os.path.join(CSV_DIR, "institutions.csv.gz"),
            "columns": [
                "id",
                "ror",
                "display_name",
                "country_code",
                "type",
                "homepage_url",
                "image_url",
                "image_thumbnail_url",
                "display_name_acronyms",
                "display_name_alternatives",
                "works_count",
                "cited_by_count",
                "works_api_url",
                "updated_date",
            ],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "institutions_ids.csv.gz"),
            "columns": ["institution_id", "openalex", "ror", "grid", "wikipedia", "wikidata", "mag"],
        },
        "geo": {
            "name": os.path.join(CSV_DIR, "institutions_geo.csv.gz"),
            "columns": [
                "institution_id",
                "city",
                "geonames_city_id",
                "region",
                "country_code",
                "country",
                "latitude",
                "longitude",
            ],
        },
        "associated_institutions": {
            "name": os.path.join(CSV_DIR, "institutions_associated_institutions.csv.gz"),
            "columns": ["institution_id", "associated_institution_id", "relationship"],
        },
        "counts_by_year": {
            "name": os.path.join(CSV_DIR, "institutions_counts_by_year.csv.gz"),
            "columns": ["institution_id", "year", "works_count", "cited_by_count", "oa_works_count"],
        },
    },
    "publishers": {
        "publishers": {
            "name": os.path.join(CSV_DIR, "publishers.csv.gz"),
            "columns": [
                "id",
                "display_name",
                "alternate_titles",
                "country_codes",
                "hierarchy_level",
                "parent_publisher",
                "works_count",
                "cited_by_count",
                "sources_api_url",
                "updated_date",
            ],
        },
        "counts_by_year": {
            "name": os.path.join(CSV_DIR, "publishers_counts_by_year.csv.gz"),
            "columns": ["publisher_id", "year", "works_count", "cited_by_count", "oa_works_count"],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "publishers_ids.csv.gz"),
            "columns": ["publisher_id", "openalex", "ror", "wikidata"],
        },
    },
    "sources": {
        "sources": {
            "name": os.path.join(CSV_DIR, "sources.csv.gz"),
            "columns": [
                "id",
                "issn_l",
                "issn",
                "display_name",
                "publisher",
                "works_count",
                "cited_by_count",
                "is_oa",
                "is_in_doaj",
                "homepage_url",
                "works_api_url",
                "updated_date",
            ],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "sources_ids.csv.gz"),
            "columns": ["source_id", "openalex", "issn_l", "issn", "mag", "wikidata", "fatcat"],
        },
        "counts_by_year": {
            "name": os.path.join(CSV_DIR, "sources_counts_by_year.csv.gz"),
            "columns": ["source_id", "year", "works_count", "cited_by_count", "oa_works_count"],
        },
    },
    "works": {
        "works": {
            "name": os.path.join(CSV_DIR, "works.csv.gz"),
            "columns": [
                "id",
                "doi",
                "title",
                "display_name",
                "publication_year",
                "publication_date",
                "type",
                "cited_by_count",
                "is_retracted",
                "is_paratext",
                "cited_by_api_url",
                "abstract_inverted_index",
                "language",
            ],
        },
        "primary_locations": {
            "name": os.path.join(CSV_DIR, "works_primary_locations.csv.gz"),
            "columns": ["work_id", "source_id", "landing_page_url", "pdf_url", "is_oa", "version", "license"],
        },
        "locations": {
            "name": os.path.join(CSV_DIR, "works_locations.csv.gz"),
            "columns": ["work_id", "source_id", "landing_page_url", "pdf_url", "is_oa", "version", "license"],
        },
        "best_oa_locations": {
            "name": os.path.join(CSV_DIR, "works_best_oa_locations.csv.gz"),
            "columns": ["work_id", "source_id", "landing_page_url", "pdf_url", "is_oa", "version", "license"],
        },
        "authorships": {
            "name": os.path.join(CSV_DIR, "works_authorships.csv.gz"),
            "columns": ["work_id", "author_position", "author_id", "institution_id", "raw_affiliation_string"],
        },
        "biblio": {
            "name": os.path.join(CSV_DIR, "works_biblio.csv.gz"),
            "columns": ["work_id", "volume", "issue", "first_page", "last_page"],
        },
        "topics": {"name": os.path.join(CSV_DIR, "works_topics.csv.gz"), "columns": ["work_id", "topic_id", "score"]},
        "concepts": {
            "name": os.path.join(CSV_DIR, "works_concepts.csv.gz"),
            "columns": ["work_id", "concept_id", "score"],
        },
        "ids": {
            "name": os.path.join(CSV_DIR, "works_ids.csv.gz"),
            "columns": ["work_id", "openalex", "doi", "mag", "pmid", "pmcid"],
        },
        "mesh": {
            "name": os.path.join(CSV_DIR, "works_mesh.csv.gz"),
            "columns": [
                "work_id",
                "descriptor_ui",
                "descriptor_name",
                "qualifier_ui",
                "qualifier_name",
                "is_major_topic",
            ],
        },
        "open_access": {
            "name": os.path.join(CSV_DIR, "works_open_access.csv.gz"),
            "columns": ["work_id", "is_oa", "oa_status", "oa_url", "any_repository_has_fulltext"],
        },
        "referenced_works": {
            "name": os.path.join(CSV_DIR, "works_referenced_works.csv.gz"),
            "columns": ["work_id", "referenced_work_id"],
        },
        "related_works": {
            "name": os.path.join(CSV_DIR, "works_related_works.csv.gz"),
            "columns": ["work_id", "related_work_id"],
        },
    },
}


def load_checkpoint(entity_name):
    """Load processed files from checkpoint."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{entity_name}_checkpoint.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_checkpoint(entity_name, processed_file):
    """Save processed file to checkpoint."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{entity_name}_checkpoint.txt")
    with open(checkpoint_file, "a") as f:
        f.write(f"{processed_file}\n")


def get_file_stats(files):
    """Get total size of files for progress tracking."""
    total_size = 0
    for file in files:
        try:
            total_size += os.path.getsize(file)
        except Exception:
            pass
    return total_size


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes):
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}PB"


def process_concept_file(jsonl_file_name):
    """Process a single concept JSONL file."""
    concepts_data = []
    ids_data = []
    ancestors_data = []
    counts_by_year_data = []
    related_concepts_data = []
    seen_concept_ids = set()

    try:
        with gzip.open(jsonl_file_name, "r") as concepts_jsonl:
            for concept_json in concepts_jsonl:
                if not concept_json.strip():
                    continue

                concept = orjson.loads(concept_json)

                if not (concept_id := concept.get("id")) or concept_id in seen_concept_ids:
                    continue

                seen_concept_ids.add(concept_id)
                concepts_data.append(concept)

                if concept_ids := concept.get("ids"):
                    concept_ids["concept_id"] = concept_id
                    concept_ids["umls_aui"] = orjson.dumps(concept_ids.get("umls_aui")).decode("utf-8")
                    concept_ids["umls_cui"] = orjson.dumps(concept_ids.get("umls_cui")).decode("utf-8")
                    ids_data.append(concept_ids)

                if ancestors := concept.get("ancestors"):
                    for ancestor in ancestors:
                        if ancestor_id := ancestor.get("id"):
                            ancestors_data.append({"concept_id": concept_id, "ancestor_id": ancestor_id})

                if counts_by_year := concept.get("counts_by_year"):
                    for count_by_year in counts_by_year:
                        count_by_year["concept_id"] = concept_id
                        counts_by_year_data.append(count_by_year)

                if related_concepts := concept.get("related_concepts"):
                    for related_concept in related_concepts:
                        if related_concept_id := related_concept.get("id"):
                            related_concepts_data.append(
                                {
                                    "concept_id": concept_id,
                                    "related_concept_id": related_concept_id,
                                    "score": related_concept.get("score"),
                                }
                            )

        return (
            concepts_data,
            ids_data,
            ancestors_data,
            counts_by_year_data,
            related_concepts_data,
            os.path.getsize(jsonl_file_name),
        )
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], [], [], [], [], 0


def flatten_concepts():
    """Flatten concepts using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting concepts flattening...")

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "concepts", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("concepts")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All concepts files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(csv_files["concepts"]["concepts"]["name"], "at", encoding="utf-8") as concepts_csv, gzip.open(
            csv_files["concepts"]["ancestors"]["name"], "at", encoding="utf-8"
        ) as ancestors_csv, gzip.open(
            csv_files["concepts"]["counts_by_year"]["name"], "at", encoding="utf-8"
        ) as counts_by_year_csv, gzip.open(
            csv_files["concepts"]["ids"]["name"], "at", encoding="utf-8"
        ) as ids_csv, gzip.open(
            csv_files["concepts"]["related_concepts"]["name"], "at", encoding="utf-8"
        ) as related_concepts_csv:

            if len(processed_files) == 0:
                concepts_writer = csv.DictWriter(
                    concepts_csv, fieldnames=csv_files["concepts"]["concepts"]["columns"], extrasaction="ignore"
                )
                concepts_writer.writeheader()

                ancestors_writer = csv.DictWriter(
                    ancestors_csv, fieldnames=csv_files["concepts"]["ancestors"]["columns"]
                )
                ancestors_writer.writeheader()

                counts_by_year_writer = csv.DictWriter(
                    counts_by_year_csv, fieldnames=csv_files["concepts"]["counts_by_year"]["columns"]
                )
                counts_by_year_writer.writeheader()

                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["concepts"]["ids"]["columns"])
                ids_writer.writeheader()

                related_concepts_writer = csv.DictWriter(
                    related_concepts_csv, fieldnames=csv_files["concepts"]["related_concepts"]["columns"]
                )
                related_concepts_writer.writeheader()
            else:
                concepts_writer = csv.DictWriter(
                    concepts_csv, fieldnames=csv_files["concepts"]["concepts"]["columns"], extrasaction="ignore"
                )
                ancestors_writer = csv.DictWriter(
                    ancestors_csv, fieldnames=csv_files["concepts"]["ancestors"]["columns"]
                )
                counts_by_year_writer = csv.DictWriter(
                    counts_by_year_csv, fieldnames=csv_files["concepts"]["counts_by_year"]["columns"]
                )
                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["concepts"]["ids"]["columns"])
                related_concepts_writer = csv.DictWriter(
                    related_concepts_csv, fieldnames=csv_files["concepts"]["related_concepts"]["columns"]
                )

            total_concepts = 0
            for i, (concepts_data, ids_data, ancestors_data, counts_data, related_data, file_size) in enumerate(
                pool.imap(process_concept_file, remaining_files)
            ):
                concepts_writer.writerows(concepts_data)
                ids_writer.writerows(ids_data)
                ancestors_writer.writerows(ancestors_data)
                counts_by_year_writer.writerows(counts_data)
                related_concepts_writer.writerows(related_data)
                total_concepts += len(concepts_data)
                processed_size += file_size
                save_checkpoint("concepts", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_concepts} | ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Concepts completed: {total_concepts} records in {format_time(time.time() - start_time)}")


def process_institution_file(jsonl_file_name):
    """Process a single institution JSONL file."""
    institutions_data = []
    ids_data = []
    geo_data = []
    associated_institutions_data = []
    counts_by_year_data = []
    seen_institution_ids = set()

    try:
        with gzip.open(jsonl_file_name, "r") as institutions_jsonl:
            for institution_json in institutions_jsonl:
                if not institution_json.strip():
                    continue

                institution = orjson.loads(institution_json)

                if not (institution_id := institution.get("id")) or institution_id in seen_institution_ids:
                    continue

                seen_institution_ids.add(institution_id)

                institution["display_name_acronyms"] = orjson.dumps(institution.get("display_name_acronyms")).decode(
                    "utf-8"
                )
                institution["display_name_alternatives"] = orjson.dumps(
                    institution.get("display_name_alternatives")
                ).decode("utf-8")
                institutions_data.append(institution)

                if institution_ids := institution.get("ids"):
                    institution_ids["institution_id"] = institution_id
                    ids_data.append(institution_ids)

                if institution_geo := institution.get("geo"):
                    institution_geo["institution_id"] = institution_id
                    geo_data.append(institution_geo)

                if associated_institutions := institution.get(
                    "associated_institutions", institution.get("associated_insitutions")
                ):
                    for associated_institution in associated_institutions:
                        if associated_institution_id := associated_institution.get("id"):
                            associated_institutions_data.append(
                                {
                                    "institution_id": institution_id,
                                    "associated_institution_id": associated_institution_id,
                                    "relationship": associated_institution.get("relationship"),
                                }
                            )

                if counts_by_year := institution.get("counts_by_year"):
                    for count_by_year in counts_by_year:
                        count_by_year["institution_id"] = institution_id
                        counts_by_year_data.append(count_by_year)

        return (
            institutions_data,
            ids_data,
            geo_data,
            associated_institutions_data,
            counts_by_year_data,
            os.path.getsize(jsonl_file_name),
        )
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], [], [], [], [], 0


def flatten_institutions():
    """Flatten institutions using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting institutions flattening...")
    file_spec = csv_files["institutions"]

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "institutions", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("institutions")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All institutions files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(file_spec["institutions"]["name"], "at", encoding="utf-8") as institutions_csv, gzip.open(
            file_spec["ids"]["name"], "at", encoding="utf-8"
        ) as ids_csv, gzip.open(file_spec["geo"]["name"], "at", encoding="utf-8") as geo_csv, gzip.open(
            file_spec["associated_institutions"]["name"], "at", encoding="utf-8"
        ) as associated_csv, gzip.open(
            file_spec["counts_by_year"]["name"], "at", encoding="utf-8"
        ) as counts_csv:

            if len(processed_files) == 0:
                institutions_writer = csv.DictWriter(
                    institutions_csv, fieldnames=file_spec["institutions"]["columns"], extrasaction="ignore"
                )
                institutions_writer.writeheader()

                ids_writer = csv.DictWriter(ids_csv, fieldnames=file_spec["ids"]["columns"])
                ids_writer.writeheader()

                geo_writer = csv.DictWriter(geo_csv, fieldnames=file_spec["geo"]["columns"])
                geo_writer.writeheader()

                associated_writer = csv.DictWriter(
                    associated_csv, fieldnames=file_spec["associated_institutions"]["columns"]
                )
                associated_writer.writeheader()

                counts_writer = csv.DictWriter(counts_csv, fieldnames=file_spec["counts_by_year"]["columns"])
                counts_writer.writeheader()
            else:
                institutions_writer = csv.DictWriter(
                    institutions_csv, fieldnames=file_spec["institutions"]["columns"], extrasaction="ignore"
                )
                ids_writer = csv.DictWriter(ids_csv, fieldnames=file_spec["ids"]["columns"])
                geo_writer = csv.DictWriter(geo_csv, fieldnames=file_spec["geo"]["columns"])
                associated_writer = csv.DictWriter(
                    associated_csv, fieldnames=file_spec["associated_institutions"]["columns"]
                )
                counts_writer = csv.DictWriter(counts_csv, fieldnames=file_spec["counts_by_year"]["columns"])

            total_institutions = 0
            for i, (inst_data, ids_data, geo_data, assoc_data, counts_data, file_size) in enumerate(
                pool.imap(process_institution_file, remaining_files)
            ):
                institutions_writer.writerows(inst_data)
                ids_writer.writerows(ids_data)
                geo_writer.writerows(geo_data)
                associated_writer.writerows(assoc_data)
                counts_writer.writerows(counts_data)
                total_institutions += len(inst_data)
                processed_size += file_size
                save_checkpoint("institutions", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_institutions} | ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Institutions completed: {total_institutions} records in {format_time(time.time() - start_time)}")


def process_publisher_file(jsonl_file_name):
    """Process a single publisher JSONL file."""
    publishers_data = []
    ids_data = []
    counts_by_year_data = []
    seen_publisher_ids = set()

    try:
        with gzip.open(jsonl_file_name, "r") as publishers_jsonl:
            for publisher_json in publishers_jsonl:
                if not publisher_json.strip():
                    continue

                publisher = orjson.loads(publisher_json)

                if not (publisher_id := publisher.get("id")) or publisher_id in seen_publisher_ids:
                    continue

                seen_publisher_ids.add(publisher_id)

                publisher["alternate_titles"] = orjson.dumps(publisher.get("alternate_titles")).decode("utf-8")
                publisher["country_codes"] = orjson.dumps(publisher.get("country_codes")).decode("utf-8")
                publishers_data.append(publisher)

                if publisher_ids := publisher.get("ids"):
                    publisher_ids["publisher_id"] = publisher_id
                    ids_data.append(publisher_ids)

                if counts_by_year := publisher.get("counts_by_year"):
                    for count_by_year in counts_by_year:
                        count_by_year["publisher_id"] = publisher_id
                        counts_by_year_data.append(count_by_year)

        return publishers_data, ids_data, counts_by_year_data, os.path.getsize(jsonl_file_name)
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], [], [], 0


def flatten_publishers():
    """Flatten publishers using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting publishers flattening...")

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "publishers", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("publishers")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All publishers files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(
            csv_files["publishers"]["publishers"]["name"], "at", encoding="utf-8"
        ) as publishers_csv, gzip.open(
            csv_files["publishers"]["counts_by_year"]["name"], "at", encoding="utf-8"
        ) as counts_csv, gzip.open(
            csv_files["publishers"]["ids"]["name"], "at", encoding="utf-8"
        ) as ids_csv:

            if len(processed_files) == 0:
                publishers_writer = csv.DictWriter(
                    publishers_csv, fieldnames=csv_files["publishers"]["publishers"]["columns"], extrasaction="ignore"
                )
                publishers_writer.writeheader()

                counts_writer = csv.DictWriter(
                    counts_csv, fieldnames=csv_files["publishers"]["counts_by_year"]["columns"]
                )
                counts_writer.writeheader()

                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["publishers"]["ids"]["columns"])
                ids_writer.writeheader()
            else:
                publishers_writer = csv.DictWriter(
                    publishers_csv, fieldnames=csv_files["publishers"]["publishers"]["columns"], extrasaction="ignore"
                )
                counts_writer = csv.DictWriter(
                    counts_csv, fieldnames=csv_files["publishers"]["counts_by_year"]["columns"]
                )
                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["publishers"]["ids"]["columns"])

            total_publishers = 0
            for i, (pub_data, ids_data, counts_data, file_size) in enumerate(
                pool.imap(process_publisher_file, remaining_files)
            ):
                publishers_writer.writerows(pub_data)
                ids_writer.writerows(ids_data)
                counts_writer.writerows(counts_data)
                total_publishers += len(pub_data)
                processed_size += file_size
                save_checkpoint("publishers", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_publishers} | ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Publishers completed: {total_publishers} records in {format_time(time.time() - start_time)}")


def process_source_file(jsonl_file_name):
    """Process a single source JSONL file."""
    sources_data = []
    ids_data = []
    counts_by_year_data = []
    seen_source_ids = set()

    try:
        with gzip.open(jsonl_file_name, "r") as sources_jsonl:
            for source_json in sources_jsonl:
                if not source_json.strip():
                    continue

                source = orjson.loads(source_json)

                if not (source_id := source.get("id")) or source_id in seen_source_ids:
                    continue

                seen_source_ids.add(source_id)

                source["issn"] = orjson.dumps(source.get("issn")).decode("utf-8")
                sources_data.append(source)

                if source_ids := source.get("ids"):
                    source_ids["source_id"] = source_id
                    source_ids["issn"] = orjson.dumps(source_ids.get("issn")).decode("utf-8")
                    ids_data.append(source_ids)

                if counts_by_year := source.get("counts_by_year"):
                    for count_by_year in counts_by_year:
                        count_by_year["source_id"] = source_id
                        counts_by_year_data.append(count_by_year)

        return sources_data, ids_data, counts_by_year_data, os.path.getsize(jsonl_file_name)
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], [], [], 0


def flatten_sources():
    """Flatten sources using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting sources flattening...")

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "sources", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("sources")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All sources files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(csv_files["sources"]["sources"]["name"], "at", encoding="utf-8") as sources_csv, gzip.open(
            csv_files["sources"]["ids"]["name"], "at", encoding="utf-8"
        ) as ids_csv, gzip.open(csv_files["sources"]["counts_by_year"]["name"], "at", encoding="utf-8") as counts_csv:

            if len(processed_files) == 0:
                sources_writer = csv.DictWriter(
                    sources_csv, fieldnames=csv_files["sources"]["sources"]["columns"], extrasaction="ignore"
                )
                sources_writer.writeheader()

                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["sources"]["ids"]["columns"])
                ids_writer.writeheader()

                counts_writer = csv.DictWriter(counts_csv, fieldnames=csv_files["sources"]["counts_by_year"]["columns"])
                counts_writer.writeheader()
            else:
                sources_writer = csv.DictWriter(
                    sources_csv, fieldnames=csv_files["sources"]["sources"]["columns"], extrasaction="ignore"
                )
                ids_writer = csv.DictWriter(ids_csv, fieldnames=csv_files["sources"]["ids"]["columns"])
                counts_writer = csv.DictWriter(counts_csv, fieldnames=csv_files["sources"]["counts_by_year"]["columns"])

            total_sources = 0
            for i, (src_data, ids_data, counts_data, file_size) in enumerate(
                pool.imap(process_source_file, remaining_files)
            ):
                sources_writer.writerows(src_data)
                ids_writer.writerows(ids_data)
                counts_writer.writerows(counts_data)
                total_sources += len(src_data)
                processed_size += file_size
                save_checkpoint("sources", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_sources} | ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Sources completed: {total_sources} records in {format_time(time.time() - start_time)}")


def process_author_file(jsonl_file_name):
    """Process a single author JSONL file and return data as lists."""
    authors_data = []
    ids_data = []
    counts_by_year_data = []

    try:
        with gzip.open(jsonl_file_name, "r") as authors_jsonl:
            for author_json in authors_jsonl:
                if not author_json.strip():
                    continue

                author = orjson.loads(author_json)

                if not (author_id := author.get("id")):
                    continue

                author["display_name_alternatives"] = orjson.dumps(author.get("display_name_alternatives")).decode(
                    "utf-8"
                )
                author["last_known_institution"] = (author.get("last_known_institution") or {}).get("id")
                authors_data.append(author)

                if author_ids := author.get("ids"):
                    author_ids["author_id"] = author_id
                    ids_data.append(author_ids)

                if counts_by_year := author.get("counts_by_year"):
                    for count_by_year in counts_by_year:
                        count_by_year["author_id"] = author_id
                        counts_by_year_data.append(count_by_year)

        return authors_data, ids_data, counts_by_year_data, os.path.getsize(jsonl_file_name)
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], [], [], 0


def flatten_authors():
    """Flatten authors using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting authors flattening...")
    file_spec = csv_files["authors"]

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "authors", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("authors")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All authors files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    logger.info(f"Total data to process: {format_size(total_size)}")

    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(file_spec["authors"]["name"], "at", encoding="utf-8") as authors_csv, gzip.open(
            file_spec["ids"]["name"], "at", encoding="utf-8"
        ) as ids_csv, gzip.open(file_spec["counts_by_year"]["name"], "at", encoding="utf-8") as counts_by_year_csv:

            if len(processed_files) == 0:
                authors_writer = csv.DictWriter(
                    authors_csv, fieldnames=file_spec["authors"]["columns"], extrasaction="ignore"
                )
                authors_writer.writeheader()

                ids_writer = csv.DictWriter(ids_csv, fieldnames=file_spec["ids"]["columns"])
                ids_writer.writeheader()

                counts_by_year_writer = csv.DictWriter(
                    counts_by_year_csv, fieldnames=file_spec["counts_by_year"]["columns"]
                )
                counts_by_year_writer.writeheader()
            else:
                authors_writer = csv.DictWriter(
                    authors_csv, fieldnames=file_spec["authors"]["columns"], extrasaction="ignore"
                )
                ids_writer = csv.DictWriter(ids_csv, fieldnames=file_spec["ids"]["columns"])
                counts_by_year_writer = csv.DictWriter(
                    counts_by_year_csv, fieldnames=file_spec["counts_by_year"]["columns"]
                )

            total_authors = 0
            for i, (authors_data, ids_data, counts_data, file_size) in enumerate(
                pool.imap(process_author_file, remaining_files)
            ):
                authors_writer.writerows(authors_data)
                ids_writer.writerows(ids_data)
                counts_by_year_writer.writerows(counts_data)
                total_authors += len(authors_data)

                processed_size += file_size
                save_checkpoint("authors", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_authors} | "
                        f"Rate: {format_size(rate)}/s | "
                        f"ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Authors completed: {total_authors} records in {format_time(time.time() - start_time)}")


def process_topic_file(jsonl_file_name):
    """Process a single topic JSONL file."""
    topics_data = []
    seen_topic_ids = set()

    try:
        with gzip.open(jsonl_file_name, "r") as topics_jsonl:
            for line in topics_jsonl:
                if not line.strip():
                    continue
                topic = orjson.loads(line)
                topic["keywords"] = "; ".join(topic.get("keywords", ""))
                if not (topic_id := topic.get("id")) or topic_id in seen_topic_ids:
                    continue
                seen_topic_ids.add(topic_id)
                for key in ("subfield", "field", "domain"):
                    topic[f"{key}_id"] = topic[key]["id"]
                    topic[f"{key}_display_name"] = topic[key]["display_name"]
                    del topic[key]
                topic["updated_date"] = topic["updated"]
                del topic["updated"]
                topic["wikipedia_id"] = topic["ids"].get("wikipedia")
                del topic["ids"]
                del topic["created_date"]
                topics_data.append(topic)

        return topics_data, os.path.getsize(jsonl_file_name)
    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return [], 0


def flatten_topics():
    """Flatten topics using parallel processing."""
    logger.info("=" * 80)
    logger.info("Starting topics flattening...")

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "topics", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("topics")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All topics files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    start_time = time.time()
    processed_size = 0

    with Pool(processes=NUM_WORKERS) as pool:
        with gzip.open(csv_files["topics"]["topics"]["name"], "at", encoding="utf-8") as topics_csv:
            if len(processed_files) == 0:
                topics_writer = csv.DictWriter(topics_csv, fieldnames=csv_files["topics"]["topics"]["columns"])
                topics_writer.writeheader()
            else:
                topics_writer = csv.DictWriter(topics_csv, fieldnames=csv_files["topics"]["topics"]["columns"])

            total_topics = 0
            for i, (topics_data, file_size) in enumerate(pool.imap(process_topic_file, remaining_files)):
                topics_writer.writerows(topics_data)
                total_topics += len(topics_data)
                processed_size += file_size
                save_checkpoint("topics", remaining_files[i])

                if (i + 1) % 10 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Records: {total_topics} | ETA: {format_time(eta)}"
                    )

    logger.info(f"✓ Topics completed: {total_topics} records in {format_time(time.time() - start_time)}")


def process_work_file(jsonl_file_name):
    """
    Process a single work JSONL file and write to temporary, gzipped CSV files.
    Returns a dictionary of temporary filenames.
    """
    try:
        # Generate unique temp filenames for this specific task
        file_hash = hash(jsonl_file_name)
        temp_files = {}
        writers = {}
        open_files = []

        file_spec = csv_files["works"]
        for key in file_spec.keys():
            temp_name = os.path.join(CSV_DIR, f"temp_works_{key}_{file_hash}.csv.gz")
            temp_files[key] = temp_name

            # Open file, create writer, and write header
            f = gzip.open(temp_name, "at", encoding="utf-8")
            open_files.append(f)
            writers[key] = csv.DictWriter(
                f,
                fieldnames=file_spec[key]["columns"],
                extrasaction="ignore" if key in ["works", "ids"] else "raise",
            )
            writers[key].writeheader()

        # Batch lists
        batches = {key: [] for key in file_spec.keys()}

        with gzip.open(jsonl_file_name, "r") as works_jsonl:
            for work_json in works_jsonl:
                if not work_json.strip():
                    continue

                work = orjson.loads(work_json)

                if not (work_id := work.get("id")):
                    continue

                # --- Process Work (Copied from your original function) ---
                if (abstract := work.get("abstract_inverted_index")) is not None:
                    work["abstract_inverted_index"] = orjson.dumps(abstract).decode("utf-8")
                batches["works"].append(work)

                if primary_location := (work.get("primary_location") or {}):
                    if primary_location.get("source") and primary_location.get("source").get("id"):
                        batches["primary_locations"].append(
                            {
                                "work_id": work_id,
                                "source_id": primary_location["source"]["id"],
                                "landing_page_url": primary_location.get("landing_page_url"),
                                "pdf_url": primary_location.get("pdf_url"),
                                "is_oa": primary_location.get("is_oa"),
                                "version": primary_location.get("version"),
                                "license": primary_location.get("license"),
                            }
                        )

                if locations := work.get("locations"):
                    for location in locations:
                        if location.get("source") and location.get("source").get("id"):
                            batches["locations"].append(
                                {
                                    "work_id": work_id,
                                    "source_id": location["source"]["id"],
                                    "landing_page_url": location.get("landing_page_url"),
                                    "pdf_url": location.get("pdf_url"),
                                    "is_oa": location.get("is_oa"),
                                    "version": location.get("version"),
                                    "license": location.get("license"),
                                }
                            )

                if best_oa_location := (work.get("best_oa_location") or {}):
                    if best_oa_location.get("source") and best_oa_location.get("source").get("id"):
                        batches["best_oa_locations"].append(
                            {
                                "work_id": work_id,
                                "source_id": best_oa_location["source"]["id"],
                                "landing_page_url": best_oa_location.get("landing_page_url"),
                                "pdf_url": best_oa_location.get("pdf_url"),
                                "is_oa": best_oa_location.get("is_oa"),
                                "version": best_oa_location.get("version"),
                                "license": best_oa_location.get("license"),
                            }
                        )

                if authorships := work.get("authorships"):
                    for authorship in authorships:
                        if author_id := authorship.get("author", {}).get("id"):
                            institutions = authorship.get("institutions")
                            institution_ids = [i.get("id") for i in institutions]
                            institution_ids = [i for i in institution_ids if i]
                            institution_ids = institution_ids or [None]

                            for institution_id in institution_ids:
                                batches["authorships"].append(
                                    {
                                        "work_id": work_id,
                                        "author_position": authorship.get("author_position"),
                                        "author_id": author_id,
                                        "institution_id": institution_id,
                                        "raw_affiliation_string": authorship.get("raw_affiliation_string"),
                                    }
                                )

                if biblio := work.get("biblio"):
                    biblio["work_id"] = work_id
                    batches["biblio"].append(biblio)

                for topic in work.get("topics", []):
                    if topic_id := topic.get("id"):
                        batches["topics"].append(
                            {"work_id": work_id, "topic_id": topic_id, "score": topic.get("score")}
                        )

                for concept in work.get("concepts", []):
                    if concept_id := concept.get("id"):
                        batches["concepts"].append(
                            {
                                "work_id": work_id,
                                "concept_id": concept_id,
                                "score": concept.get("score"),
                            }
                        )

                if ids := work.get("ids"):
                    ids["work_id"] = work_id
                    batches["ids"].append(ids)

                for mesh in work.get("mesh", []):
                    mesh["work_id"] = work_id
                    batches["mesh"].append(mesh)

                if open_access := work.get("open_access"):
                    open_access["work_id"] = work_id
                    batches["open_access"].append(open_access)

                for referenced_work in work.get("referenced_works", []):
                    if referenced_work:
                        batches["referenced_works"].append({"work_id": work_id, "referenced_work_id": referenced_work})

                for related_work in work.get("related_works", []):
                    if related_work:
                        batches["related_works"].append({"work_id": work_id, "related_work_id": related_work})

                # --- Check if batches are full and write ---
                # Use 'works' as the trigger for batch size
                if len(batches["works"]) >= BATCH_SIZE:
                    for key, writer in writers.items():
                        if batches[key]:
                            writer.writerows(batches[key])
                            batches[key] = []  # Clear the batch

            # --- Write any remaining data after the loop ---
            for key, writer in writers.items():
                if batches[key]:
                    writer.writerows(batches[key])
                    batches[key] = []

        # Close all temp files
        for f in open_files:
            f.close()

        # Return the names of the temp files and the processed size
        return (temp_files, os.path.getsize(jsonl_file_name))

    except Exception as e:
        logger.error(f"Error processing {jsonl_file_name}: {e}", exc_info=True)
        return (None, 0)  # Return None on failure


def flatten_works():
    """Flatten works using parallel processing with progress tracking."""
    logger.info("=" * 80)
    logger.info("Starting works flattening...")
    file_spec = csv_files["works"]

    all_files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "data", "works", "*", "*.gz")))

    if FILES_PER_ENTITY:
        all_files = all_files[:FILES_PER_ENTITY]

    processed_files = load_checkpoint("works")
    remaining_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total files: {len(all_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}"
    )

    if not remaining_files:
        logger.info("All works files already processed, skipping...")
        return

    total_size = get_file_stats(remaining_files)
    logger.info(f"Total data to process: {format_size(total_size)}")

    start_time = time.time()
    processed_size = 0

    all_temp_file_dicts = []  # Store all dicts of temp files

    with Pool(processes=NUM_WORKERS) as pool:
        try:
            total_works = 0  # This count is no longer accurate, change logging

            # Use chunksize=1 to provide results one-by-one
            for i, (temp_files, file_size) in enumerate(pool.imap(process_work_file, remaining_files, chunksize=1)):
                if temp_files is None:  # Worker failed
                    logger.warning(f"Skipping failed file: {remaining_files[i]}")
                    continue

                all_temp_file_dicts.append(temp_files)
                processed_size += file_size
                save_checkpoint("works", remaining_files[i])

                # Progress update every 5 files or at the end
                if (i + 1) % 5 == 0 or i == len(remaining_files) - 1:
                    elapsed = time.time() - start_time
                    pct = (processed_size / total_size * 100) if total_size > 0 else 0
                    rate = processed_size / elapsed if elapsed > 0 else 0
                    eta = (total_size - processed_size) / rate if rate > 0 else 0
                    files_per_sec = (i + 1) / elapsed if elapsed > 0 else 0

                    logger.info(
                        f"Progress: {i+1}/{len(remaining_files)} files ({pct:.1f}%) | "
                        f"Works: {total_works:,} | "
                        f"Rate: {format_size(rate)}/s ({files_per_sec:.2f} files/s) | "
                        f"Elapsed: {format_time(elapsed)} | "
                        f"ETA: {format_time(eta)}"
                    )

        except Exception as e:
            logger.error(f"Error in worker pool: {e}", exc_info=True)
            # Handle partial completion?
        finally:
            logger.info("Worker pool finished. Now concatenating files...")

    # --- NEW CONCATENATION STEP ---
    # This runs *after* the pool is closed.

    if not all_temp_file_dicts:
        logger.info("No temporary files were generated. Exiting.")
        return

    for key in file_spec.keys():
        logger.info(f"Concatenating temporary files for '{key}'...")
        final_filename = file_spec[key]["name"]
        temp_files_for_key = [d[key] for d in all_temp_file_dicts if key in d]

        if not temp_files_for_key:
            logger.warning(f"No temp files found for {key}, skipping.")
            continue

        # Open the final file
        with gzip.open(final_filename, "at", encoding="utf-8") as final_csv:
            # Copy the *first* file (header and all)
            first_file = temp_files_for_key[0]
            with gzip.open(first_file, "rt", encoding="utf-8") as f_in:
                final_csv.write(f_in.read())
            os.remove(first_file)  # Clean up

            # Copy all *other* files (skipping their headers)
            for temp_file in temp_files_for_key[1:]:
                try:
                    with gzip.open(temp_file, "rt", encoding="utf-8") as f_in:
                        next(f_in)  # Skip the header line
                        final_csv.write(f_in.read())
                    os.remove(temp_file)  # Clean up
                except Exception as e:
                    logger.error(f"Error processing temp file {temp_file}: {e}")

    logger.info(f"✓ Works completed and concatenated in {format_time(time.time() - start_time)}")


if __name__ == "__main__":

    logger.info("=" * 80)
    logger.info("Starting OpenAlex flattening process with checkpoint support")
    logger.info(f"Workers: {NUM_WORKERS}")
    logger.info(f"Files per entity limit: {FILES_PER_ENTITY or 'unlimited'}")
    logger.info("=" * 80)

    overall_start = time.time()

    try:
        # flatten_topics()
        # flatten_authors()
        # flatten_concepts()
        # flatten_institutions()
        # flatten_publishers()
        # flatten_sources()
        flatten_works()

        elapsed_time = time.time() - overall_start
        logger.info("=" * 80)
        logger.info(f"✓ Successfully completed all flattening operations in {format_time(elapsed_time)}")
        logger.info("=" * 80)
    except KeyboardInterrupt:
        logger.warning("\n⚠ Process interrupted by user. Progress has been saved to checkpoints.")
        logger.info("You can resume by running the script again.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
