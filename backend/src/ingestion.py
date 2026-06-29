from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core import parse_and_chunk, embed_chunks, check_ollama
from core.vector_store import store_embeddings, query_collection, get_collection
from core.query_pipeline import run_query

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if not await check_ollama():
        raise HTTPException(status_code=503, detail="Ollama not running.")

    contents = await file.read()

    print(f"[1] Parsing {file.filename}...")
    result = parse_and_chunk(contents, file.filename)
    print(f"[2] Got {result['total_chunks']} chunks")

    print("[3] Embedding chunks...")
    embedded_chunks = await embed_chunks(result["chunks"])
    print(f"[4] Embedded {len(embedded_chunks)} chunks")

    print("[5] Storing in ChromaDB...")
    try:
        stored = store_embeddings(embedded_chunks, file.filename)
        print(f"[6] Stored {stored} chunks successfully")
    except Exception as e:
        print(f"[ERROR] ChromaDB store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "filename": result["filename"],
        "total_words": result["total_words"],
        "total_chunks": result["total_chunks"],
        "chunks_stored": stored,
    })
    
    
# 2nd endpoint for deleting a file's chunks from ChromaDB
@app.delete("/collection/clear")
async def clear_collection():
    try:
        collection = get_collection()
        count_before = collection.count()
        
        # Get all IDs and delete them
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        
        return JSONResponse(content={
            "message": "Collection cleared successfully.",
            "deleted": count_before,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# 3rd endpoint for list the documents in the collection

@app.get("/documents")
async def list_documents():
    try:
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        
        # Extract unique filenames from metadata
        seen = set()
        documents = []
        for metadata in results["metadatas"]:
            filename = metadata["filename"]
            if filename not in seen:
                seen.add(filename)
                documents.append(filename)
        
        return JSONResponse(content={
            "total_documents": len(documents),
            "documents": documents,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 4th endpoint for querying the collection
 
@app.get("/query")
async def query_documents(
    q: str,
    n_results: int = 5,
    filename: str | None = None,
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = await run_query(q, n_results=n_results, filename=filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))