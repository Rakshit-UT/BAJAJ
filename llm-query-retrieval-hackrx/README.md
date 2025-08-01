# LLM-Powered Intelligent Query–Retrieval System

A sophisticated document processing and query system designed for **insurance, legal, HR, and compliance domains**. This system processes large documents (PDFs, DOCX, emails) and provides contextual, explainable answers to natural language queries.

## 🚀 Features

- **Multi-format Document Processing**: PDF, DOCX, and email parsing
- **Semantic Search**: FAISS/Pinecone-powered vector similarity search
- **Clause-level Extraction**: Intelligent extraction of policy/contract clauses
- **LLM Integration**: GPT-4 powered query understanding and response generation
- **Explainable AI**: Provides rationale and citations for all answers
- **RESTful API**: Easy integration with existing systems
- **Real-time Processing**: Efficient handling of large documents

## 🏗 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   LLM Parser     │    │ Embedding Search│
│   (PDF/DOCX)    │───▶│ (Query Intent)   │───▶│   (FAISS)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  JSON Output    │◀───│ Logic Evaluation │◀───│ Clause Matching │
│ (Structured)    │    │  (Business Rules)│    │ (Semantic Sim.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.10+
- OpenAI API Key
- 4GB+ RAM (for embedding models)
- Internet connection for document downloads

## 🛠 Installation

### 1. Clone and Setup
```bash
# Extract the project
cd llm-query-retrieval-hackrx

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. Download Required Models
```bash
# Download spaCy model (optional, for enhanced text processing)
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

### Start the Server
```bash
python start.py
```

The server will start on `http://localhost:8000`

### API Usage

#### Basic Request
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the exclusions in this policy?"
    ]
  }'
```

#### Response Format
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "The policy excludes pre-existing diseases for the first 36 months of coverage."
  ]
}
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All requests require a Bearer token:
```
Authorization: Bearer c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798
```

### Endpoints

#### `POST /hackrx/run`
Process documents and answer questions.

**Request Body:**
```json
{
  "documents": "string (URL)",
  "questions": ["array of strings"]
}
```

**Response:**
```json
{
  "answers": ["array of strings"]
}
```

#### `POST /hackrx/run-advanced`
Advanced processing with citations and rationale.

**Response:**
```json
{
  "answers": ["array of strings"],
  "citations": [{"doc_id": "string", "page": "string"}],
  "rationale": [["step1", "step2"]]
}
```

#### `GET /health`
System health check.

#### `GET /status`
API status and version information.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `BEARER_TOKEN` | API authentication token | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-4` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Document chunk size (chars) | `1000` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | `0.5` |
| `MAX_CONTEXT_CHUNKS` | Max chunks per query | `5` |

### Advanced Configuration

#### Using Pinecone (Optional)
```bash
USE_PINECONE=true
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=llm-query-retrieval
```

## 📊 Supported Document Types

### PDF Documents
- Insurance policies
- Legal contracts
- Compliance documents
- Multi-page reports

### DOCX Documents
- HR policies
- Standard operating procedures
- Template documents

### Email Content
- Compliance communications
- Policy updates
- Query responses

## 🎯 Domain-Specific Features

### Insurance Domain
- Premium calculation clauses
- Coverage definitions
- Exclusion identification
- Waiting period extraction
- Claim process details

### Legal Domain
- Contract clause analysis
- Obligation identification
- Liability assessment
- Jurisdiction requirements

### HR Domain
- Policy interpretation
- Benefit explanations
- Procedure guidelines
- Compliance requirements

### Compliance Domain
- Regulatory requirement mapping
- Audit trail maintenance
- Risk assessment
- Policy adherence checking

## 🧪 Testing

### Manual Testing
```bash
# Test with sample documents
python -m pytest tests/ -v
```

### API Testing
```bash
# Test API endpoints
curl -X GET "http://localhost:8000/health"
```

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify API key in `.env`
   - Check API usage limits
   - Ensure model availability

2. **Memory Issues**
   - Reduce `CHUNK_SIZE`
   - Lower `MAX_CONTEXT_CHUNKS`
   - Use smaller embedding model

3. **Document Processing Errors**
   - Verify document URL accessibility
   - Check document format support
   - Review file size limitations

### Logs
```bash
# View application logs
tail -f logs/app.log
```

## 📈 Performance Optimization

### For Large Documents
- Increase `CHUNK_OVERLAP` for better context
- Use Pinecone for better scaling
- Implement document caching

### For High Traffic
- Increase `API_WORKERS`
- Use Redis for session management
- Implement request queuing

## 🛡 Security

- API authentication via Bearer tokens
- Input validation and sanitization
- Rate limiting (configure as needed)
- Secure credential management

## 📄 License

This project is developed for HackRx hackathon purposes.

## 🤝 Contributing

This is a hackathon project. For improvements:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📞 Support

For hackathon support and questions:
- Review the documentation
- Check the troubleshooting section
- Examine the example requests

## 🏆 HackRx Implementation

This system specifically addresses the HackRx challenge requirements:

✅ **Document Processing**: Multi-format support (PDF, DOCX, email)  
✅ **Semantic Search**: FAISS-powered vector similarity  
✅ **LLM Integration**: GPT-4 for intelligent responses  
✅ **Domain Expertise**: Insurance, legal, HR, compliance focus  
✅ **Explainable AI**: Citations and reasoning provided  
✅ **RESTful API**: Standard HTTP interface  
✅ **Real-time Processing**: Efficient document handling  

---

**Built for HackRx 2025** 🚀
