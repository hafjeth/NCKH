from chromadb_client import ChromaDBClient

client = ChromaDBClient(
    host="localhost",
    port=8000,
    tenant="default_tenant",
    database="default_database"
)

collection = client.get_or_create_collection("carbon_policy_textile_vn")

count = collection.count()
print(f"🔢 Total documents in collection: {count}")
