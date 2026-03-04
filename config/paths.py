from pathlib import Path
from config.settings import DATA_DIR

# Raw data
RAW_DIR = DATA_DIR / "raw"
RAW_PDFS = RAW_DIR / "pdfs"

# Intermediate data
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
CLASSIFIED_DIR = INTERMEDIATE_DIR / "classified"
EXTRACTED_DIR = INTERMEDIATE_DIR / "extracted"

# Processed data
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
NORMALIZED_DIR = PROCESSED_DIR / "normalized"
QUALITY_REPORTS_DIR = PROCESSED_DIR / "quality_reports"

# Vector stores
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
CHROMA_DIR = VECTOR_STORES_DIR / "chroma"

def ensure_directories():
    dirs = [
        RAW_PDFS, CLASSIFIED_DIR, EXTRACTED_DIR,
        CHUNKS_DIR, NORMALIZED_DIR, QUALITY_REPORTS_DIR,
        CHROMA_DIR
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)