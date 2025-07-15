#!/usr/bin/env python3
"""
Production-ready FastAPI web service for Custom GPT integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from config import Config
import uvicorn

app = FastAPI(
    title="Course Materials API",
    description="API for searching construction contract course materials",
    version="1.0.0"
)

# Add CORS middleware for Custom GPT access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(Config.PINECONE_INDEX_NAME)

# Use OpenAI API key from environment
embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    score: float
    document_name: str
    module: str
    text: str
    chunk_index: int

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

@app.post("/search", response_model=QueryResponse)
async def search_course_materials(request: QueryRequest):
    """
    Search through course materials using semantic search
    """
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(request.query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        # Format results
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                score=match.score,
                document_name=match.metadata.get('document_name', 'Unknown'),
                module=match.metadata.get('module', 'Unknown'),
                text=match.metadata.get('text', ''),
                chunk_index=match.metadata.get('chunk_index', 0)
            ))
        
        return QueryResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Course Materials API"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Course Materials API",
        "version": "1.0.0",
        "description": "Search construction contract course materials",
        "endpoints": {
            "search": "POST /search - Search course materials",
            "docs": "GET /docs - API documentation"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
