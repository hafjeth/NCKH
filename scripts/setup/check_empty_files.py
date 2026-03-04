"""
Kiểm tra files trống và so sánh với files cũ
"""
import os
from pathlib import Path

def check_empty_files():
    print("=" * 60)
    print("KIỂM TRA FILES TRỐNG")
    print("=" * 60 + "\n")
    
    # Mapping files mới -> files cũ
    file_mappings = {
        # PDF Processing
        'src/pipeline/pdf_processing/detector.py': 'src/knowledge/pdf_pineline/detect_pdf_type.py',
        'src/pipeline/pdf_processing/text_extractor.py': 'src/knowledge/pdf_pineline/extract_text_from_pdf.py',
        'src/pipeline/pdf_processing/ocr_processor.py': 'src/knowledge/pdf_pineline/ocr_scan_pdf.py',
        'src/pipeline/pdf_processing/quality_checker.py': 'src/knowledge/pdf_pineline/check_text_quality.py',
        'src/pipeline/pdf_processing/normalizer.py': 'src/knowledge/pdf_pineline/normalize_text.py',
        
        # Chunking
        'src/pipeline/chunking/business_chunker.py': 'src/knowledge/chunking_embedding/business_paragraph_chunking.py',
        'src/pipeline/chunking/legal_chunker.py': 'src/knowledge/chunking_embedding/legal_chunking.py',
        'src/pipeline/chunking/business_semantic.py': 'src/knowledge/chunking_embedding/business_paragraph_semantic.py',
        'src/pipeline/chunking/legal_semantic.py': 'src/knowledge/chunking_embedding/legal_semantic_tagging.py',
        
        # Embeddings
        'src/knowledge/embeddings/embedding_store.py': 'src/knowledge/chunking_embedding/legal_embedding_store.py',
        
        # Retrieval
        'src/knowledge/retrieval/retriever.py': 'src/knowledge/retrieval.py',
        
        # Vector DB
        'src/knowledge/vector_db/chroma_client.py': 'src/knowledge/chromadb_client.py',
        'src/knowledge/vector_db/collection_manager.py': 'src/knowledge/count_collection.py',
        
        # Core
        'src/core/personas.py': 'src/knowledge/personas.py',
        
        # Evaluation
        'src/evaluation/metrics/retrieval_metrics.py': 'src/evaluation/metrics.py',
        
        # Scripts
        'scripts/data_processing/ingest_business_paragraphs.py': 'src/knowledge/ingestion/ingest_business_paragraphs.py',
        'scripts/data_processing/ingest_legal_semantic_chunks.py': 'src/knowledge/ingestion/ingest_legal_semantic_chunks.py',
    }
    
    empty_files = []
    missing_content = []
    
    for new_file, old_file in file_mappings.items():
        new_exists = os.path.exists(new_file)
        old_exists = os.path.exists(old_file)
        
        if not new_exists:
            print(f"❌ NEW FILE MISSING: {new_file}")
            missing_content.append(new_file)
            continue
            
        new_size = os.path.getsize(new_file)
        
        if new_size == 0:
            print(f"⚠️  EMPTY: {new_file}")
            if old_exists:
                old_size = os.path.getsize(old_file)
                print(f"   → Old file exists ({old_size} bytes): {old_file}")
                empty_files.append((new_file, old_file))
            else:
                print(f"   → Old file also missing!")
        else:
            status = "✓" if new_size > 100 else "⚠️"
            print(f"{status} {new_file} ({new_size} bytes)")
            
            if old_exists and new_size < 100:
                old_size = os.path.getsize(old_file)
                if old_size > new_size:
                    print(f"   → Old file is larger ({old_size} bytes)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if empty_files:
        print(f"\n⚠️  Found {len(empty_files)} empty files that need content:")
        for new_f, old_f in empty_files:
            print(f"  - {new_f}")
            print(f"    Copy from: {old_f}")
    else:
        print("\n✓ All files have content!")
    
    return empty_files

def generate_copy_commands(empty_files):
    if not empty_files:
        return
    
    print("\n" + "=" * 60)
    print("COPY COMMANDS (PowerShell)")
    print("=" * 60 + "\n")
    
    for new_file, old_file in empty_files:
        print(f"Copy-Item -Force {old_file} {new_file}")

if __name__ == '__main__':
    empty_files = check_empty_files()
    generate_copy_commands(empty_files)