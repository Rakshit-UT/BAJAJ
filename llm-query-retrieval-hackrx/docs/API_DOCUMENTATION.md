# API Documentation

## Overview

The LLM Query-Retrieval System provides a RESTful API for processing documents and answering natural language questions with contextual, explainable responses.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798
```

## Endpoints

### POST /hackrx/run

Process documents and answer questions.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period?",
    "What are the exclusions?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period is 30 days for premium payment.",
    "Exclusions include pre-existing diseases for 36 months."
  ]
}
```

### POST /hackrx/run-advanced

Process with detailed citations and rationale.

**Request:** Same as `/hackrx/run`

**Response:**
```json
{
  "answers": ["Answer 1", "Answer 2"],
  "citations": [
    [{"chunk_id": "chunk_1", "source": "document.pdf", "page": "2"}],
    [{"chunk_id": "chunk_5", "source": "document.pdf", "page": "4"}]
  ],
  "rationale": [
    ["Found relevant clause in section 3", "Clause specifies 30-day period"],
    ["Located exclusions in section 7", "Pre-existing condition clause applies"]
  ]
}
```

### GET /health

System health check.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "document_processor": true,
    "embedding_service": true,
    "llm_service": true
  }
}
```

### GET /status

API status and information.

**Response:**
```json
{
  "status": "operational",
  "version": "1.0.0",
  "endpoints": [
    "/api/v1/hackrx/run",
    "/api/v1/hackrx/run-advanced",
    "/api/v1/status"
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid input data or missing fields"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid authentication token"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to process request: <error_message>"
}
```

## Rate Limits

- Default: 10 concurrent requests
- Timeout: 300 seconds per request
- Configure via environment variables

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is covered under this policy?"]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What are the key terms?"]
  }'
```
