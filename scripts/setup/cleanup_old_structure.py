"""
Script dọn dẹp cấu trúc cũ
"""
import os
import shutil

def cleanup():
    print("Cleaning up old structure...")
    print("=" * 60)
    
    items_to_remove = [
        'src/knowledge/pdf_pineline',
        'src/knowledge/chunking_embedding',
        'src/knowledge/deprecate',
        'src/knowledge/ingestion',
        'src/knowledge/chromadb_client.py',
        'src/knowledge/count_collection.py',
        'src/knowledge/retrieval.py',
        'src/evaluation/metrics_old.py',
        'data/processed/business_paragraphs',
        'data/processed/business_paragraphs_semantic',
        'data/processed/legal_chunks',
        'data/processed/legal_chunks_semantic',
        'data/processed/normalized_text',
    ]
    
    removed_count = 0
    
    for item in items_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"✓ Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"✓ Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {item}: {e}")
        else:
            print(f"- Already removed: {item}")
    
    print("=" * 60)
    print(f"\nRemoved {removed_count} items")
    print("\n⚠ Note: Backup was created as NCKH_backup (if you ran that command)")
    print("If everything works fine, you can delete NCKH_backup folder")

if __name__ == '__main__':
    response = input("Are you sure you want to cleanup old structure? (yes/no): ")
    if response.lower() == 'yes':
        cleanup()
        print("\n✓ Cleanup completed!")
    else:
        print("Cleanup cancelled")