from src.knowledge.retrieval.retriever import KnowledgeRetriever

def main():
    print("🔌 Connecting to ChromaDB...")
    retriever = KnowledgeRetriever()
    
    print("\n📊 Collection stats:", retriever.get_collection_stats())
    
    print("\n🔍 Testing query...")
    results = retriever.retrieve(
        query="xử phạt xả nước thải ra môi trường",
        agent="government",
        k=5
    )
    
    print(f"\n✅ Retrieved {len(results)} chunks\n")
    print("=" * 60)
    
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}")
        print(f"    Law: {r['metadata'].get('law', 'N/A')}")
        print(f"    Type: {r['metadata'].get('clause_type', 'N/A')}")
        print(f"    Article: {r['metadata'].get('article', 'N/A')}")
        print(f"    Text: {r['text'][:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()