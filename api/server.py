from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import os
from embeddings.clip_encoder import embed_text, embed_image
from embeddings.indexer import ImageIndexer

app = FastAPI()

indexer = ImageIndexer(dim=512)
indexer.load()

@app.post("/query/text")
async def query_text(text: str = Form(...)):
    emb = embed_text(text)
    results = indexer.search(emb, top_k=5)
    return JSONResponse(content={"query": text, "results": results})

@app.post("/query/photo")
async def query_photo(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    emb = embed_image(temp_path)
    os.remove(temp_path)
    results = indexer.search(emb, top_k=5)
    return JSONResponse(content={"results": results})
