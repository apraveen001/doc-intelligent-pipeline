from core.vector_store import get_collection

collection = get_collection()
print("Total records:", collection.count())
print("Sample:", collection.peek(limit=2))
