import os
import hashlib
import re
from config import Config
from google_drive_client import GoogleDriveClient
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentIndexer:
    def __init__(self):
        self.drive_client = GoogleDriveClient()
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        self.pinecone_index = Pinecone(api_key=Config.PINECONE_API_KEY).Index(Config.PINECONE_INDEX_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def update_index(self):
        # Define modules with complete course structure
        modules = {
            'Module 1 - Progress Claims Management': [
                'Certification and payment',
                'Dispute management',
                'Introduction to progress claims',
                'Preparation and Submission Process'
            ],
            'Module 2 - Variation Claims Handling': [
                'Preparing Variation Claims',
                'The Documentation and Record Keeping of Variations',
                'The Negotiation and Agreement of Variations',
                'Types Of Variation'
            ],
            'Module 3 - Frustration, Delay, and Extension of Time Claims': [
                'Frustration vs. Delays under NZS 3910',
                'Frustration, Delay, and Extension of Time Claims',
                'Mitigation Strategies in Construction Projects under NZS 3910',
                'Negotiation Processes in Construction Projects Administered under NZS 3910'
            ],
            'Module 4 - Dispute Resolution Techniques': [
                'Dispute Resolution Techniques'
            ],
            'Module 5 - Cost Control in Post-Contract Administration': [
                'Cost Control Methods of Monitoring and Reporting in a Construction Project Administered Using the NZS 3910 Form of Contract',
                'Cost-Saving Measures in a Construction Project Administered Using the NZS 3910 Form of Contract',
                'The Principles of Cost Control in a Construction Project Using the NZS 3910 Form of Contract'
            ]
        }

        # Get all documents from Google Drive
        all_docs = self.drive_client.get_all_documents()
        print(f"Found {len(all_docs)} documents from Google Drive")
        
        # Process each module
        for module, expected_documents in modules.items():
            print(f"\nProcessing {module}...")
            
            for expected_doc in expected_documents:
                matched_doc = self.find_matching_document(all_docs, expected_doc, module)
                
                if matched_doc:
                    print(f"  Found: {matched_doc['name']}")
                    self.index_document(matched_doc, module)
                else:
                    print(f"  Missing: {expected_doc}")
        
        print("\nIndex update completed!")
    
    def find_matching_document(self, all_docs, expected_name, module):
        """Find document that matches expected name, handling variations"""
        # First try exact match
        for doc in all_docs:
            if doc['name'] == expected_name:
                return doc
        
        # Try case-insensitive match
        for doc in all_docs:
            if doc['name'].lower() == expected_name.lower():
                return doc
        
        # Try partial match (removing file extensions, etc.)
        for doc in all_docs:
            doc_name_clean = re.sub(r'\.(docx?|pdf)$', '', doc['name'], flags=re.IGNORECASE)
            expected_clean = re.sub(r'\.(docx?|pdf)$', '', expected_name, flags=re.IGNORECASE)
            if doc_name_clean.lower() == expected_clean.lower():
                return doc
        
        # Try fuzzy match (contains)
        for doc in all_docs:
            if expected_name.lower() in doc['name'].lower() or doc['name'].lower() in expected_name.lower():
                return doc
        
        return None
    
    def index_document(self, doc_info, module):
        """Index a single document"""
        try:
            # Delete existing chunks for this document
            self.delete_document_chunks(doc_info['name'])
            
            # Create new chunks
            chunks = self.text_splitter.split_text(doc_info['content'])
            
            # Create embeddings and upload to Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                vectors_to_upsert.append({
                    'id': f"{doc_info['name']}_{i}",
                    'values': embedding,
                    'metadata': {
                        'document_name': doc_info['name'],
                        'module': module,
                        'text': chunk,
                        'chunk_index': i
                    }
                })
            
            # Batch upsert for efficiency
            if vectors_to_upsert:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                print(f"    Indexed {len(vectors_to_upsert)} chunks")
            
        except Exception as e:
            print(f"    Error indexing {doc_info['name']}: {str(e)}")
    
    def delete_document_chunks(self, document_name):
        """Delete existing chunks for a document"""
        try:
            # Query for existing chunks
            query_response = self.pinecone_index.query(
                vector=[0] * 1536,  # Dummy vector for metadata filtering
                filter={'document_name': document_name},
                top_k=10000,
                include_metadata=True
            )
            
            # Delete existing chunks
            if query_response.matches:
                ids_to_delete = [match.id for match in query_response.matches]
                self.pinecone_index.delete(ids=ids_to_delete)
                print(f"    Deleted {len(ids_to_delete)} existing chunks")
                
        except Exception as e:
            print(f"    Warning: Could not delete existing chunks: {str(e)}")

    def chunk_document(self, text, chunk_size=1000, overlap=200):
        """Chunk text into segments for embedding"""
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(' '.join(chunk))
        return chunks

if __name__ == '__main__':
    indexer = DocumentIndexer()
    indexer.update_index()

