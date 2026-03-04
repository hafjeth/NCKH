import os
import re
from pathlib import Path

def analyze_imports(file_path):
    """Phân tích các import trong file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tìm tất cả imports
        imports = re.findall(r'^from\s+[\w.]+\s+import\s+.+$', content, re.MULTILINE)
        imports += re.findall(r'^import\s+[\w.]+$', content, re.MULTILINE)
        
        return imports
    except Exception as e:
        return []

def find_all_python_files(base_dir):
    """Tìm tất cả Python files"""
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip venv và __pycache__
        dirs[:] = [d for d in dirs if d not in ['venv', 'venv310', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    print("Analyzing imports in project...")
    print("=" * 60)
    
    # Phân tích tất cả imports
    all_imports = {}
    for file_path in find_all_python_files('src'):
        imports = analyze_imports(file_path)
        if imports:
            all_imports[file_path] = imports
    
    # In ra các imports có vấn đề
    problematic = []
    for file_path, imports in all_imports.items():
        for imp in imports:
            if 'src.core.config' in imp or 'src.retrieval' in imp or 'src.chromadb' in imp:
                problematic.append((file_path, imp))
    
    if problematic:
        print("\nProblematic imports found:")
        print("-" * 60)
        for file_path, imp in problematic:
            print(f"\n{file_path}")
            print(f"  → {imp}")
    else:
        print("\n✓ No problematic imports found!")
    
    print("\n" + "=" * 60)
    print(f"Total files analyzed: {len(all_imports)}")
    print(f"Problematic imports: {len(problematic)}")

if __name__ == '__main__':
    main()