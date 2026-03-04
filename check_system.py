import sys, json, glob
from pathlib import Path

print("="*60)
print("SYSTEM INFORMATION REPORT")
print("="*60)

# Python version
print(f"\\nPython: {sys.version}")

# Count files
pdf_files = glob.glob("data/**/*.pdf", recursive=True)
txt_files = glob.glob("data/**/*.txt", recursive=True)
json_files = glob.glob("data/**/*.json", recursive=True)

print(f"\\n📁 DATA FILES:")
print(f"  PDF files : {len(pdf_files)}")
print(f"  TXT files : {len(txt_files)}")
print(f"  JSON files: {len(json_files)}")

# Sample files
if pdf_files:
    print(f"\\n📄 SAMPLE PDFS (first 5):")
    for f in pdf_files[:5]:
        print(f"  - {f}")

print("\\n✅ Done!")
