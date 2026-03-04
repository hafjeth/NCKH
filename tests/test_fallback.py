# tests/test_fallback.py
"""
Test fallback and cache mechanism
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.fallback_manager import get_fallback_manager
from src.utils.cache_manager import get_cache_manager


def test_system():
    print("="*60)
    print("TESTING FALLBACK & CACHE SYSTEM")
    print("="*60)
    
    # Test messages
    messages = [
        {"role": "system", "content": "Bạn là chuyên gia phân tích chính sách."},
        {"role": "user", "content": "Thuế carbon là gì? Trả lời ngắn gọn."}
    ]
    
    fallback = get_fallback_manager(Config)
    cache = get_cache_manager()
    
    # Clear cache trước khi test
    print("\n0️⃣ Clearing old cache...")
    cache.clear_all()
    
    # DEBUG: Xem key sẽ tạo ra
    print("\n🔍 DEBUG - Cache key for test:")
    cache.debug_key(messages, "gpt-4o-mini", 0.3)
    
    # Test 1: Normal call (có cache)
    print("\n1️⃣ Test 1: Normal call (with cache)")
    result1 = fallback.call_with_fallback(
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        use_cache=True  # Bật cache ngay từ đầu
    )
    print(f"Model used: {result1['model_used']}")
    print(f"From cache: {result1.get('from_cache', False)}")
    print(f"Response: {result1['response'][:100]}...")
    print(f"Attempts: {result1['attempts']}")
    
    # DEBUG: Kiểm tra key sau khi set
    print("\n🔍 DEBUG - After test 1:")
    cache.debug_key(messages, "gpt-4o-mini", 0.3)
    
    # Test 2: Cached call (phải HIT)
    print("\n2️⃣ Test 2: Cached call (should HIT)")
    result2 = fallback.call_with_fallback(
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        use_cache=True
    )
    print(f"Model used: {result2['model_used']}")
    print(f"From cache: {result2.get('from_cache', False)}")
    print(f"Response: {result2['response'][:100]}...")
    
    # Test 3: Different temperature (nên MISS)
    print("\n3️⃣ Test 3: Different temperature (should MISS)")
    result3 = fallback.call_with_fallback(
        messages=messages,
        temperature=0.5,
        max_tokens=500,
        use_cache=True
    )
    print(f"Model used: {result3['model_used']}")
    print(f"From cache: {result3.get('from_cache', False)}")
    
    # Test 4: Cache stats
    print("\n4️⃣ Cache Stats")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_system()