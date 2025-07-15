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
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Environment variable configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "course-documents")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# Global variables for services
pc = None
index = None
embeddings = None

# Initialize services
def initialize_services():
    global pc, index, embeddings
    
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
            
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        logger.info("Services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        return False

# Try to initialize services
services_ready = initialize_services()

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
    global services_ready, embeddings, index
    
    # Check if services are ready
    if not services_ready:
        # Try to reinitialize services
        services_ready = initialize_services()
        if not services_ready:
            raise HTTPException(status_code=503, detail="Services not available. Please check environment variables.")
    
    if not embeddings or not index:
        raise HTTPException(status_code=503, detail="Search services not initialized")
    
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
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global services_ready, embeddings, index
    
    # Check service status
    status = "healthy" if services_ready and embeddings and index else "unhealthy"
    
    return {
        "status": status,
        "service": "Course Materials API",
        "services_ready": services_ready,
        "embeddings_ready": embeddings is not None,
        "index_ready": index is not None,
        "environment_variables": {
            "OPENAI_API_KEY": "set" if OPENAI_API_KEY else "missing",
            "PINECONE_API_KEY": "set" if PINECONE_API_KEY else "missing",
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
            "GOOGLE_DRIVE_FOLDER_ID": "set" if GOOGLE_DRIVE_FOLDER_ID else "missing"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Course Materials API",
        "version": "1.0.0",
        "description": "Search construction contract course materials",
        "endpoints": {
            "search": "POST /search - Search course materials",
            "docs": "GET /docs - API documentation",
            "schema": "GET /custom-openapi.json - Custom GPT OpenAPI schema"
        }
    }

@app.get("/custom-openapi.json")
async def custom_openapi():
    """Custom OpenAPI schema with server configuration for Custom GPT"""
    import json
    try:
        with open('openapi_schema.json', 'r') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        logger.error(f"Failed to load custom OpenAPI schema: {str(e)}")
        # Fallback to auto-generated schema with server info
        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title="Course Materials API",
            version="1.0.0",
            description="API for searching construction contract course materials",
            routes=app.routes,
        )
        openapi_schema["servers"] = [
            {
                "url": "https://microcred.onrender.com",
                "description": "Production server"
            }
        ]
        return openapi_schema

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
