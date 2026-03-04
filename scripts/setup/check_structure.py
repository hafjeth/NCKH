"""
Script kiểm tra cấu trúc project
"""
import os
from pathlib import Path
import importlib.util

# Màu cho output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_directory_structure():
    """Kiểm tra cấu trúc thư mục"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KIỂM TRA CẤU TRÚC THư MỤC{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    required_dirs = {
        'config': ['__init__.py', 'settings.py', 'paths.py'],
        'src/core': ['__init__.py', 'base_agent.py', 'debate_manager.py', 'moderator.py', 'personas.py'],
        'src/pipeline/pdf_processing': ['__init__.py'],
        'src/pipeline/chunking': ['__init__.py'],
        'src/knowledge/embeddings': ['__init__.py'],
        'src/knowledge/retrieval': ['__init__.py', 'retriever.py'],
        'src/knowledge/vector_db': ['__init__.py', 'chroma_client.py'],
        'src/agents': ['__init__.py'],
        'src/evaluation/metrics': ['__init__.py', 'retrieval_metrics.py'],
        'src/evaluation/judges': ['__init__.py'],
        'src/utils': ['__init__.py', 'logger.py'],
        'scripts/setup': [],
        'scripts/data_processing': [],
        'scripts/evaluation': [],
        'tests/unit': ['__init__.py'],
        'tests/integration': ['__init__.py'],
        'logs/app': [],
        'logs/pipeline': [],
        'data/raw/pdfs': [],
        'data/processed/chunks': [],
        'data/vector_stores/chroma': [],
    }
    
    missing_dirs = []
    missing_files = []
    
    for dir_path, files in required_dirs.items():
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"{Colors.RED}✗{Colors.END} Missing directory: {dir_path}")
        else:
            print(f"{Colors.GREEN}✓{Colors.END} {dir_path}")
            for file in files:
                file_path = os.path.join(dir_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    print(f"  {Colors.RED}✗{Colors.END} Missing file: {file}")
                else:
                    print(f"  {Colors.GREEN}✓{Colors.END} {file}")
    
    return missing_dirs, missing_files

def check_imports():
    """Kiểm tra các import chính"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KIỂM TRA IMPORTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    imports_to_check = [
        ('config.settings', 'Config'),
        ('config.paths', 'ensure_directories'),
        ('src.utils.logger', 'get_logger'),
        ('src.knowledge.retrieval', 'KnowledgeRetriever'),
        ('src.knowledge.vector_db.chroma_client', None),
        ('src.core.base_agent', 'BaseAgent'),
        ('src.core.debate_manager', 'DebateManager'),
        ('src.evaluation.metrics', 'MetricsCalculator'),
    ]
    
    failed_imports = []
    
    for module_name, attr_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
            if attr_name:
                getattr(module, attr_name)
            print(f"{Colors.GREEN}✓{Colors.END} {module_name}" + (f".{attr_name}" if attr_name else ""))
        except ImportError as e:
            failed_imports.append((module_name, attr_name, str(e)))
            print(f"{Colors.RED}✗{Colors.END} {module_name}" + (f".{attr_name}" if attr_name else ""))
            print(f"  Error: {e}")
        except AttributeError as e:
            failed_imports.append((module_name, attr_name, str(e)))
            print(f"{Colors.RED}✗{Colors.END} {module_name}.{attr_name}")
            print(f"  Error: {e}")
    
    return failed_imports

def check_config_values():
    """Kiểm tra giá trị config"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KIỂM TRA CONFIG VALUES{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    try:
        from config.settings import Config
        
        required_attrs = [
            'BASE_DIR', 'DATA_DIR', 'LOGS_DIR',
            'ANTHROPIC_API_KEY', 'OPENAI_API_KEY',
            'MODEL_NAME', 'DEFAULT_MODEL',
            'CHROMA_PERSIST_DIR', 'COLLECTION_NAME'
        ]
        
        missing_attrs = []
        
        for attr in required_attrs:
            if hasattr(Config, attr):
                value = getattr(Config, attr)
                if value is None and 'API_KEY' in attr:
                    print(f"{Colors.YELLOW}⚠{Colors.END} {attr}: Not set (check .env)")
                else:
                    print(f"{Colors.GREEN}✓{Colors.END} {attr}: {value}")
            else:
                missing_attrs.append(attr)
                print(f"{Colors.RED}✗{Colors.END} Missing: {attr}")
        
        return missing_attrs
    except Exception as e:
        print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
        return []

def count_python_files():
    """Đếm số file Python"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}THỐNG KÊ FILES{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    total_files = 0
    dir_counts = {}
    
    for root, dirs, files in os.walk('.'):
        # Skip venv và __pycache__
        dirs[:] = [d for d in dirs if d not in ['venv', 'venv310', '__pycache__', '.git', 'node_modules']]
        
        py_files = [f for f in files if f.endswith('.py')]
        if py_files:
            total_files += len(py_files)
            relative_path = os.path.relpath(root, '.')
            dir_counts[relative_path] = len(py_files)
    
    for dir_path, count in sorted(dir_counts.items()):
        print(f"  {dir_path}: {count} files")
    
    print(f"\n{Colors.GREEN}Total Python files: {total_files}{Colors.END}")
    
    return total_files

def check_old_structure():
    """Kiểm tra các thư mục/file cũ cần xóa"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KIỂM TRA CẤU TRÚC CŨ (CẦN DỌN DẸP){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    old_items = [
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
    
    found_old = []
    
    for item in old_items:
        if os.path.exists(item):
            found_old.append(item)
            if os.path.isdir(item):
                print(f"{Colors.YELLOW}⚠{Colors.END} Old directory found: {item}")
            else:
                print(f"{Colors.YELLOW}⚠{Colors.END} Old file found: {item}")
    
    if not found_old:
        print(f"{Colors.GREEN}✓ No old structure found - Clean!{Colors.END}")
    
    return found_old

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}NCKH PROJECT STRUCTURE CHECKER{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    # 1. Kiểm tra thư mục
    missing_dirs, missing_files = check_directory_structure()
    
    # 2. Kiểm tra imports
    failed_imports = check_imports()
    
    # 3. Kiểm tra config
    missing_attrs = check_config_values()
    
    # 4. Thống kê files
    total_files = count_python_files()
    
    # 5. Kiểm tra cấu trúc cũ
    old_items = check_old_structure()
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    if not missing_dirs and not missing_files and not failed_imports and not missing_attrs:
        print(f"{Colors.GREEN}✓✓✓ ALL CHECKS PASSED! ✓✓✓{Colors.END}")
        print(f"{Colors.GREEN}Project structure is correct!{Colors.END}")
    else:
        print(f"{Colors.RED}Some issues found:{Colors.END}")
        if missing_dirs:
            print(f"  - Missing {len(missing_dirs)} directories")
        if missing_files:
            print(f"  - Missing {len(missing_files)} files")
        if failed_imports:
            print(f"  - {len(failed_imports)} import errors")
        if missing_attrs:
            print(f"  - {len(missing_attrs)} missing config attributes")
    
    if old_items:
        print(f"\n{Colors.YELLOW}⚠ Found {len(old_items)} old items that can be cleaned up{Colors.END}")
    
    print(f"\nTotal Python files: {total_files}")
    print()

if __name__ == '__main__':
    main()