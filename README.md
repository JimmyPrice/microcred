# Construction Contract Course Materials API

This API provides semantic search capabilities for construction contract course materials, specifically focused on NZS 3910 contract administration.

## Features

- **Semantic Search**: Search through 17 course documents across 5 modules
- **Module Coverage**:
  - Progress Claims Management
  - Variation Claims Handling
  - Frustration, Delay, and Extension of Time Claims
  - Dispute Resolution Techniques
  - Cost Control in Post-Contract Administration

## API Endpoints

### POST /search
Search through course materials using semantic search.

**Request Body:**
```json
{
  "query": "What are progress claims?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What are progress claims?",
  "results": [
    {
      "score": 0.877,
      "document_name": "Introduction to progress claims",
      "module": "1 - Progress Claims Management (micro)",
      "text": "A progress claim is a formal payment request...",
      "chunk_index": 0
    }
  ],
  "total_results": 5
}
```

### GET /health
Health check endpoint.

### GET /
API information and available endpoints.

## Deployment

### Environment Variables Required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index (default: microcred)
- `GOOGLE_DRIVE_FOLDER_ID`: Google Drive folder ID containing course materials

### Deploy to Render:
1. Fork this repository
2. Connect to Render
3. Set environment variables
4. Deploy

### Build Commands:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`

## Usage with Custom GPT

This API is designed to work with OpenAI's Custom GPT feature. Import the OpenAPI schema and configure your Custom GPT to use the `/search` endpoint for accessing course materials.

## License

MIT License
