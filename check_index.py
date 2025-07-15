#!/usr/bin/env python3
"""
Check what documents are currently in the Pinecone index
"""

from config import Config
from pinecone import Pinecone
from collections import defaultdict

def check_index():
    # Initialize Pinecone
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    # Query for a sample of documents to see what's in there
    query_response = index.query(
        vector=[0] * 1536,  # Dummy vector
        top_k=100,
        include_metadata=True
    )
    
    # Group documents by module
    modules = defaultdict(set)
    document_counts = defaultdict(int)
    
    for match in query_response.matches:
        if 'module' in match.metadata:
            module = match.metadata['module']
            doc_name = match.metadata.get('document_name', 'Unknown')
            modules[module].add(doc_name)
            document_counts[doc_name] += 1
    
    print("\n=== Current Index Contents ===")
    for module, docs in modules.items():
        print(f"\n{module}:")
        for doc in sorted(docs):
            chunk_count = document_counts[doc]
            print(f"  - {doc} ({chunk_count} chunks)")
    
    print(f"\nTotal documents found: {len(document_counts)}")
    print(f"Total chunks: {sum(document_counts.values())}")

if __name__ == '__main__':
    check_index()
