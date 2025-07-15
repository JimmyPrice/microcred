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
import hashlib
import time
from functools import lru_cache

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
            "schema": "GET /custom-openapi.json - Custom GPT OpenAPI schema",
            "privacy": "GET /privacy - Privacy policy",
            "terms": "GET /terms - Terms of service"
        }
    }

@app.get("/privacy")
async def privacy_policy():
    """Privacy policy endpoint required for public GPT actions"""
    return {
        "title": "Privacy Policy - Course Materials API",
        "last_updated": "2024-01-01",
        "content": {
            "data_collection": "We collect search queries to provide relevant course material results. No personal information is stored.",
            "data_usage": "Search queries are used only to return relevant construction contract course materials. Query logs may be kept for service improvement.",
            "data_sharing": "We do not share user data with third parties. All data remains within our secure infrastructure.",
            "data_retention": "Search queries are retained for 30 days for service improvement purposes, then automatically deleted.",
            "user_rights": "Users can request deletion of their data by contacting support.",
            "contact": "For privacy concerns, contact us through the API documentation."
        }
    }

@app.get("/terms")
async def terms_of_service():
    """Terms of service endpoint"""
    return {
        "title": "Terms of Service - Course Materials API",
        "last_updated": "2024-01-01",
        "content": {
            "service_description": "This API provides access to construction contract course materials for educational purposes.",
            "acceptable_use": "The service is intended for educational and research purposes related to construction contract management.",
            "limitations": "The service is provided 'as is' without warranties. Usage is subject to reasonable rate limits.",
            "intellectual_property": "Course materials remain the property of their respective authors and institutions.",
            "termination": "Access may be terminated for misuse or violation of these terms.",
            "governing_law": "These terms are governed by applicable local laws."
        }
    }

@app.get("/custom-openapi.json")
async def custom_openapi():
    """Custom OpenAPI schema with server configuration for Custom GPT"""
    import json
    try:
        with open('gpt_schema.json', 'r') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        logger.error(f"Failed to load GPT schema: {str(e)}")
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

@app.post("/reindex")
async def reindex_documents():
    """Re-index all documents from Google Drive"""
    global services_ready, embeddings, index
    
    if not services_ready:
        raise HTTPException(status_code=503, detail="Services not available")
    
    try:
        # Import here to avoid circular imports
        from google_drive_client import GoogleDriveClient
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import re
        
        # Initialize components
        drive_client = GoogleDriveClient()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Expected document structure - using actual module names from index
        expected_modules = {
            '1 - Progress Claims Management (micro)': [
                'Certification and payment',
                'Dispute management', 
                'Introduction to progress claims',
                'Preparation and Submission Process'
            ],
            '2 - Variation Claims Handling': [
                'Preparing Variation Claims',
                'The Documentation and Record Keeping of Variations',
                'The Negotiation and Agreement of Variations',
                'Types Of Variation'
            ],
            '3 - Frustration, Delay, and Extension of Time Claims ': [
                'Frustration vs. Delays under NZS 3910',
                'Frustration, Delay, and Extension of Time Claims',
                'Mitigation Strategies in Construction Projects under NZS 3910',
                'Negotiation Processes in Construction Projects Administered under NZS 3910'
            ],
            '4 - Dispute Resolution Techniques': [
                'Dispute Resolution Techniques'
            ],
            '5 - Cost Control in Post-Contract Administration': [
                'Cost Control Methods of Monitoring and Reporting in a Construction Project Administered Using the NZS 3910 Form of Contract',
                'Cost-Saving Measures in a Construction Project Administered Using the NZS 3910 Form of Contract',
                'The Principles of Cost Control in a Construction Project Using the NZS 3910 Form of Contract'
            ]
        }
        
        # Get all documents
        all_docs = drive_client.get_all_documents()
        
        results = {
            "total_documents_found": len(all_docs),
            "indexed_documents": [],
            "missing_documents": [],
            "errors": []
        }
        
        # Process each expected document
        for module, expected_docs in expected_modules.items():
            for expected_doc in expected_docs:
                matched_doc = None
                
                # Find matching document
                for doc in all_docs:
                    if (doc['name'] == expected_doc or 
                        doc['name'].lower() == expected_doc.lower() or
                        expected_doc.lower() in doc['name'].lower()):
                        matched_doc = doc
                        break
                
                if matched_doc:
                    try:
                        # Delete existing chunks
                        existing_query = index.query(
                            vector=[0] * 1536,
                            filter={'document_name': matched_doc['name']},
                            top_k=1000,
                            include_metadata=True
                        )
                        
                        if existing_query.matches:
                            ids_to_delete = [match.id for match in existing_query.matches]
                            index.delete(ids=ids_to_delete)
                        
                        # Create new chunks
                        chunks = text_splitter.split_text(matched_doc['content'])
                        
                        # Create embeddings and index
                        vectors_to_upsert = []
                        for i, chunk in enumerate(chunks):
                            embedding = embeddings.embed_query(chunk)
                            vectors_to_upsert.append({
                                'id': f"{matched_doc['name']}_{i}",
                                'values': embedding,
                                'metadata': {
                                    'document_name': matched_doc['name'],
                                    'module': module,
                                    'text': chunk,
                                    'chunk_index': i
                                }
                            })
                        
                        # Upsert to Pinecone
                        if vectors_to_upsert:
                            index.upsert(vectors=vectors_to_upsert)
                            results["indexed_documents"].append({
                                "name": matched_doc['name'],
                                "module": module,
                                "chunks": len(vectors_to_upsert)
                            })
                        
                    except Exception as e:
                        results["errors"].append({
                            "document": matched_doc['name'],
                            "error": str(e)
                        })
                else:
                    results["missing_documents"].append({
                        "expected_name": expected_doc,
                        "module": module
                    })
        
        return results
        
    except Exception as e:
        logger.error(f"Reindexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
