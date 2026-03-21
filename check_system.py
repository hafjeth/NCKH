"""
check_system.py
===============
Chạy script này để kiểm tra toàn bộ hệ thống trước khi chạy main.py

Usage:
    python check_system.py
"""

import os
import sys
from pathlib import Path

# Đảm bảo import từ project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("🔍 MAD-POLICY SYSTEM CHECK")
print("=" * 60)

errors   = []
warnings = []

# ======================================================
# CHECK 1: .env và API Key
# ======================================================
print("\n📋 CHECK 1: API Key")
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        print(f"  ✅ OPENAI_API_KEY found: sk-...{api_key[-6:]}")
    elif api_key:
        print(f"  ⚠️  OPENAI_API_KEY found but format looks unusual")
        warnings.append("OPENAI_API_KEY format unusual")
    else:
        print("  ❌ OPENAI_API_KEY not found in .env")
        errors.append("Missing OPENAI_API_KEY")
except Exception as e:
    print(f"  ❌ Error loading .env: {e}")
    errors.append(str(e))

# ======================================================
# CHECK 2: Config
# ======================================================
print("\n📋 CHECK 2: Config")
try:
    from src.core.config import Config
    print(f"  ✅ Config loaded")
    print(f"     API Provider : {Config.API_PROVIDER}")
    print(f"     Model        : {Config.MODEL_NAME}")
    print(f"     FALLBACK_CONFIG exists: {'FALLBACK_CONFIG' in dir(Config)}")
    if not hasattr(Config, 'FALLBACK_CONFIG'):
        errors.append("Config missing FALLBACK_CONFIG")
except Exception as e:
    print(f"  ❌ Config error: {e}")
    errors.append(f"Config: {e}")

# ======================================================
# CHECK 3: ChromaDB connection + mode detection
# ======================================================
print("\n📋 CHECK 3: ChromaDB")
chroma_mode = None

# Try HTTP mode first (Docker)
try:
    import chromadb
    from chromadb.config import Settings
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=Settings(anonymized_telemetry=False)
    )
    client.heartbeat()
    chroma_mode = "http"
    print(f"  ✅ ChromaDB HTTP mode (localhost:8000) — running via Docker")
except Exception as e_http:
    print(f"  ⚠️  HTTP mode failed: {e_http}")

    # Try local persistent mode
    try:
        persist_dir = PROJECT_ROOT / "data" / "vector_stores" / "chroma"
        if persist_dir.exists():
            client = chromadb.PersistentClient(path=str(persist_dir))
            chroma_mode = "persistent"
            print(f"  ✅ ChromaDB Persistent mode: {persist_dir}")
        else:
            print(f"  ❌ Persistent dir not found: {persist_dir}")
            errors.append("ChromaDB not reachable (HTTP or Persistent)")
    except Exception as e_local:
        print(f"  ❌ Persistent mode also failed: {e_local}")
        errors.append("ChromaDB not reachable")

# ======================================================
# CHECK 4: Collection & data
# ======================================================
if chroma_mode:
    print("\n📋 CHECK 4: Collection data")
    try:
        collection_name = "carbon_policy_textile_vn"
        collection = client.get_or_create_collection(collection_name)
        count = collection.count()
        print(f"  Collection: '{collection_name}'")
        if count > 0:
            print(f"  ✅ Documents in collection: {count}")
        else:
            print(f"  ⚠️  Collection is EMPTY — cần chạy ingest scripts trước!")
            warnings.append("ChromaDB collection is empty — run ingest scripts first")

        # List all collections
        all_cols = client.list_collections()
        print(f"  All collections: {[c.name for c in all_cols]}")
    except Exception as e:
        print(f"  ❌ Collection error: {e}")
        errors.append(f"Collection: {e}")
else:
    print("\n📋 CHECK 4: Skipped (ChromaDB not connected)")

# ======================================================
# CHECK 5: Core imports
# ======================================================
print("\n📋 CHECK 5: Core imports")
imports_to_check = [
    ("src.core.base_agent",              "BaseAgent"),
    ("src.core.debate_manager",          "DebateManager"),
    ("src.core.moderator",               "ModeratorAgent"),
    ("src.core.fallback_manager",        "FallbackManager"),
    ("src.core.personas",                "AGENT_PERSONAS"),
    ("src.agents.government.carbon_policy_agent",    "CarbonPolicyAgent"),
    ("src.agents.enterprise.textile_industry_agent", "TextileIndustryAgent"),
    ("src.agents.expert.expert_council_agent",       "ExpertCouncilAgent"),
    ("src.knowledge.retrieval.retriever",            "KnowledgeRetriever"),
    ("experiments.evaluation.judges.llm_judge",      "LLMJudge"),
    ("experiments.evaluation.metrics.retrieval_metrics", "MetricsCalculator"),
]

for module_path, class_name in imports_to_check:
    try:
        mod = __import__(module_path, fromlist=[class_name])
        getattr(mod, class_name)
        print(f"  ✅ {module_path}.{class_name}")
    except ImportError as e:
        print(f"  ❌ ImportError: {module_path} — {e}")
        errors.append(f"Import {module_path}: {e}")
    except AttributeError as e:
        print(f"  ⚠️  Module OK but class missing: {module_path}.{class_name}")
        warnings.append(f"Missing class {class_name} in {module_path}")
    except Exception as e:
        print(f"  ⚠️  {module_path}: {e}")
        warnings.append(f"{module_path}: {e}")

# ======================================================
# CHECK 6: Quick OpenAI API test
# ======================================================
print("\n📋 CHECK 6: OpenAI API test (1 request)")
try:
    from openai import OpenAI
    client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client_oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
        temperature=0
    )
    reply = resp.choices[0].message.content.strip()
    print(f"  ✅ OpenAI API working — response: '{reply}'")
except Exception as e:
    print(f"  ❌ OpenAI API failed: {e}")
    errors.append(f"OpenAI API: {e}")

# ======================================================
# SUMMARY
# ======================================================
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if not errors and not warnings:
    print("✅ ALL CHECKS PASSED — Bạn có thể chạy main.py ngay!")
    print("\n🚀 Lệnh chạy:")
    print('   python main.py --topic "Tác động của thuế carbon đến ngành dệt may Việt Nam" --rounds 2')

elif not errors:
    print(f"⚠️  {len(warnings)} WARNING(S) — Có thể chạy nhưng lưu ý:")
    for w in warnings:
        print(f"   • {w}")
    print("\n🚀 Lệnh chạy:")
    print('   python main.py --topic "Tác động của thuế carbon đến ngành dệt may Việt Nam" --rounds 2')

else:
    print(f"❌ {len(errors)} ERROR(S) — Cần fix trước khi chạy:")
    for err in errors:
        print(f"   • {err}")
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"   • {w}")
    print("\n💡 Gửi output này cho Claude để được hướng dẫn fix tiếp.")

print("=" * 60)