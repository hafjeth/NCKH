import os
import re
from pathlib import Path

# Mapping các import cũ sang mới
IMPORT_MAPPINGS = {
    'from config.settings import': 'from config.settings import',
    'from src.core.base_agent import': 'from src.core.base_agent import',
    'from src.knowledge.vector_db.chroma_client import': 'from src.knowledge.vector_db.chroma_client import',
    'from src.knowledge.retrieval.retriever import': 'from src.knowledge.retrieval.retriever import',
    'from src.core.personas import': 'from src.core.personas import',
    'from src.knowledge.vector_db.collection_manager import': 'from src.knowledge.vector_db.collection_manager import',
    'from src.knowledge.vector_db import chroma_client': 'from src.knowledge.vector_db import chroma_client',
    'from src.knowledge.retrieval import retriever': 'from src.knowledge.retrieval import retriever',
}

def update_imports_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old, new in IMPORT_MAPPINGS.items():
            if old in content:
                content = content.replace(old, new)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'✓ Updated: {file_path}')
            return True
        return False
    except Exception as e:
        print(f'✗ Error in {file_path}: {e}')
        return False

def update_all_imports(base_dirs):
    updated_count = 0
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            # Skip các thư mục không cần thiết
            if '__pycache__' in root or 'venv' in root or '.git' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if update_imports_in_file(file_path):
                        updated_count += 1
    return updated_count

if __name__ == '__main__':
    print('Updating imports...')
    print('-' * 50)
    dirs_to_update = ['src', 'scripts', 'tests', 'config']
    count = update_all_imports(dirs_to_update)
    print('-' * 50)
    print(f'\nCompleted! Updated {count} files.')