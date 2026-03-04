# tests/test_cache_direct.py
"""
Test cache trực tiếp không qua fallback
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cache_manager import get_cache_manager


def test_cache_direct():
    print("="*60)
    print("TEST CACHE DIRECTLY")
    print("="*60)
    
    cache = get_cache_manager()
    
    # Test messages
    messages = [
        {"role": "system", "content": "Bạn là chuyên gia phân tích chính sách."},
        {"role": "user", "content": "Thuế carbon là gì? Trả lời ngắn gọn."}
    ]
    
    model = "gpt-4o-mini"
    temperature = 0.3
    test_response = "Đây là response test từ cache"
    
    print(f"\n1️⃣ Cache directory: {cache.cache_dir.absolute()}")
    
    # Test set
    print("\n2️⃣ Testing cache SET...")
    cache.set(messages, model, temperature, test_response)
    
    # Debug sau khi set
    print("\n3️⃣ After SET:")
    cache.debug_key(messages, model, temperature)
    
    # Test get
    print("\n4️⃣ Testing cache GET...")
    retrieved = cache.get(messages, model, temperature)
    
    if retrieved:
        print(f"✅ SUCCESS! Retrieved: {retrieved[:50]}...")
    else:
        print("❌ FAILED! Could not retrieve from cache")
    
    # Stats
    print("\n5️⃣ Cache Stats:")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_cache_direct()