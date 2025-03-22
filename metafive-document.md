# Metafive: Comprehensive Technical Documentation

## Project Overview

Metafive is an interactive learning platform that accelerates cross-domain knowledge transfer through AI-driven analogies. The system maps concepts between different software applications, allowing users to leverage their existing expertise to learn new tools more efficiently. By identifying conceptual parallels between domains (e.g., "Color Filter" in After Effects corresponding to "EQ Eight" in Ableton Live), Metafive creates intuitive learning pathways that build on established mental models.

The architecture implements a retrieval-augmented generation (RAG) approach that combines:
1. Static curated analogies from domain experts
2. Dynamic AI-generated explanations via Ollama LLM integration
3. Contextual documentation retrieval through SearXNG metasearch
4. Vector similarity matching for novel cross-domain connections

## System Architecture

```
┌────────────────┐       ┌──────────────────┐       ┌───────────────────┐
│  React Frontend│◄─────►│  FastAPI Backend │◄─────►│ SearXNG @ 9090    │
└────────────────┘       └───────┬───────────┘       └───────────────────┘
                                 │
                                 ▼
                         ┌───────────────────┐
                         │ Ollama @ 45721    │
                         └───────────────────┘
```

### Component Interaction Flow

1. **User Selection**: User selects source software (known domain), target software (learning domain), and a specific feature/concept
2. **Analogy Retrieval**: Backend searches for the optimal analogy through a tiered approach:
   - First attempts retrieval from curated database
   - If found, enhances with LLM-generated explanation
   - Falls back to vector similarity search when no direct mapping exists
3. **Documentation Enrichment**: SearXNG retrieves relevant documentation for both source and target concepts
4. **Presentation**: Frontend displays the analogy with color-coded indication of source (curated vs. AI-generated)

## Technical Implementation Details

### Frontend (React)

The frontend application is built with React and utilizes a component structure optimized for interactive learning experiences:

```
frontend/
├── public/                   # Static assets
│   └── index.html            # Entry point HTML
├── src/
│   ├── components/           # React components
│   │   └── ui/               # UI component library
│   │       └── card.jsx      # Card component for displaying analogies
│   ├── App.js                # Main application component
│   ├── index.css             # Global styles with Tailwind
│   └── index.js              # React entry point
├── package.json              # Dependencies and scripts
├── craco.config.js           # Path alias configuration
├── tailwind.config.js        # Tailwind CSS configuration
├── postcss.config.js         # PostCSS configuration
├── nginx/                    # Nginx configuration for production
│   └── default.conf          # Reverse proxy configuration
└── Dockerfile                # Frontend container configuration
```

**Key React Components:**

1. **SoftwareAnalogyTool (App.js)**: The main application component managing:
   - State management for software selection and analogy results
   - Service status monitoring (Ollama, SearXNG)
   - Model selection interface
   - Analogy display with visual feedback

2. **UI Components**:
   - Card/CardContent: Display containers for analogy explanations
   - ServiceStatus: Visual indicators for external service availability
   - ModelSelector: Interface for choosing Ollama models
   - DocumentationLinks: Displays relevant documentation from SearXNG

**State Management:**

```javascript
// Primary state variables
const [sourceSoftware, setSourceSoftware] = useState("");           // Source domain
const [targetSoftware, setTargetSoftware] = useState("");           // Target domain
const [sourceFeatures, setSourceFeatures] = useState([]);           // Available features
const [selectedFeature, setSelectedFeature] = useState("");         // Selected feature
const [analogyResult, setAnalogyResult] = useState(null);           // Analogy response

// Service integration state
const [ollamaStatus, setOllamaStatus] = useState(null);             // Ollama availability
const [ollamaModels, setOllamaModels] = useState([]);               // Available models
const [selectedModel, setSelectedModel] = useState("");             // Current model
const [searxngStatus, setSearxngStatus] = useState(null);           // SearXNG availability
```

**API Integration:**

The frontend communicates with the backend through RESTful endpoints:
- Software and feature fetching endpoints use GET requests
- Analogy generation uses POST with the source/target/feature parameters
- Dynamic model switching via the `/ollama-model` endpoint

**Styling and Presentation:**

Visual feedback is implemented through conditional styling:
- Analogy source is indicated by gradient background colors (blue: curated, green: AI-generated)
- Service status indicators show real-time availability of external systems
- Documentation links are presented in a structured format with snippets

### Backend (FastAPI)

The backend implements a tiered analogy generation system with multiple fallback mechanisms:

```
backend/
├── main.py                   # FastAPI application and endpoints
├── ollama_integration.py     # Ollama LLM client
├── searxng_integration.py    # SearXNG search client
├── vector_search.py          # Vector similarity engine
├── requirements.txt          # Python dependencies
└── Dockerfile                # Backend container configuration
```

**API Endpoints:**

```python
@app.get("/source-software")                         # List available source software
@app.get("/target-software/{source}")                # List compatible target software
@app.get("/features/{source}")                       # List features for source software
@app.post("/analogy")                                # Get analogy for specific feature
@app.get("/ollama-status")                           # Check Ollama availability
@app.get("/ollama-models")                           # List available Ollama models
@app.post("/ollama-model")                           # Set active Ollama model
@app.get("/searxng-status")                          # Check SearXNG availability
```

**Data Models:**

```python
class DocumentationLink(BaseModel):
    title: str                                       # Documentation title
    url: str                                         # URL to documentation
    snippet: Optional[str] = None                    # Text snippet

class AnalogyRequest(BaseModel):
    source_software: str                             # Source domain
    target_software: str                             # Target domain
    source_feature: str                              # Feature to find analogy for

class AnalogyResponse(BaseModel):
    target_feature: str                              # Analogous feature
    explanation: str                                 # Explanation text
    similarity_score: float = 1.0                    # Confidence score (0-1)
    generated_by: str = "static"                     # Source: 'static', 'vector', 'llm'
    source_documentation: List[DocumentationLink] = [] # Source docs from SearXNG
    target_documentation: List[DocumentationLink] = [] # Target docs from SearXNG
```

**Analogy Generation Pipeline:**

1. **Static Database Lookup**: First attempts to retrieve from curated database
   ```python
   if source in analogy_db and feature in analogy_db[source] and target in analogy_db[source][feature]:
       static_analogy = analogy_db[source][feature][target]
   ```

2. **Documentation Enhancement**:
   ```python
   if settings.USE_SEARXNG:
       source_docs = searxng_client.get_software_documentation(source, feature)
       target_docs = searxng_client.get_software_documentation(target, target_feature)
   ```

3. **Dynamic Explanation Generation**:
   ```python
   if settings.USE_OLLAMA:
       ollama_explanation = await ollama_client.generate_explanation(
           source, feature, target, target_feature,
           additional_context=documentation_context
       )
   ```

4. **Response Assembly**:
   ```python
   return AnalogyResponse(
       target_feature=target_feature,
       explanation=explanation_text,  # From LLM or static source
       similarity_score=score,
       generated_by=source_type,
       source_documentation=source_docs,
       target_documentation=target_docs
   )
   ```

### External Service Integration

#### Ollama LLM Client

The `OllamaClient` class provides a robust interface to the Ollama inference service:

```python
class OllamaClient:
    def __init__(self, base_url: str, model: str = "llama2"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
    def list_models(self) -> List[str]:
        # Fetches available models from Ollama
        
    def set_model(self, model_name: str) -> bool:
        # Changes the active model for generation
        
    def generate_explanation(self, 
                           source_software: str, 
                           source_feature: str,
                           target_software: str,
                           target_feature: str,
                           temperature: float = 0.7,
                           max_tokens: int = 300,
                           additional_context: str = None) -> Optional[str]:
        # Generates analogy explanations with carefully engineered prompts
        
    def health_check(self) -> Dict[str, Union[bool, str]]:
        # Verifies Ollama availability and returns current model
```

**Prompt Engineering for Analogies:**

```python
prompt = f"""
Create an explanation of how "{source_feature}" in {source_software} is analogous to 
"{target_feature}" in {target_software}.

Focus on:
1. How the concepts are similar in purpose
2. How understanding one helps understand the other
3. Key differences to be aware of

Keep the explanation straightforward and informative.
"""

# Add documentation context if available
if additional_context:
    prompt += f"\n\nHere is some additional documentation that may help:\n{additional_context}\n"
```

#### SearXNG Integration

The `SearXNGClient` provides a structured interface to the SearXNG metasearch engine:

```python
class SearXNGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.search_endpoint = f"{self.base_url}/search"
    
    def search(self, 
               query: str, 
               categories: List[str] = None, 
               engines: List[str] = None,
               language: str = "en",
               max_results: int = 5) -> Optional[Dict]:
        # Performs a search using SearXNG with configurable parameters
    
    def get_software_documentation(self, 
                                  software_name: str, 
                                  feature_name: str = None) -> List[Dict]:
        # Retrieves documentation specific to software features
        # Constructs optimized queries based on software and feature names
        
    def health_check(self) -> bool:
        # Verifies SearXNG availability
```

**Documentation Query Construction:**

```python
query = software_name
if feature_name:
    query += f" {feature_name} documentation tutorial guide"
else:
    query += " documentation guide"

results = self.search(
    query=query,
    categories=["it", "science"],
    engines=["duckduckgo", "bing", "brave", "qwant"]
)
```

### Vector Similarity Search

The `FeatureEmbedding` class implements vector similarity matching for finding analogous concepts:

```python
class FeatureEmbedding:
    def __init__(self):
        # Initializes with pre-computed embeddings and feature metadata
        self.embeddings = self._load_mock_embeddings()
        self.feature_info = self._load_feature_info()
    
    def get_most_similar_features(self, query_key: str, target_software: str, top_k: int = 3) -> List[Dict]:
        # Finds most similar features in target software using cosine similarity
        
    def generate_analogy_explanation(self, source_key: str, target_key: str) -> str:
        # Generates explanatory text for vector-matched analogies
```

**Similarity Calculation:**

```python
# Extract the query embedding
query_embedding = self.embeddings[query_key]

# Find all features from the target software
target_keys = [k for k in self.embeddings.keys() if k.startswith(f"{target_software}:")]

# Get embeddings for target features
target_embeddings = np.array([self.embeddings[k] for k in target_keys])

# Calculate similarity scores using cosine similarity
similarities = cosine_similarity([query_embedding], target_embeddings)[0]

# Sort by similarity
sorted_indices = np.argsort(-similarities)  # Descending order
```

## Deployment Architecture

The application uses Docker and Docker Compose for containerized deployment:

```
docker-compose.yml         # Multi-container orchestration
├── backend                # FastAPI container
│   └── Dockerfile         # Python environment
├── frontend               # React+Nginx container
│   └── Dockerfile         # Build and serve frontend
└── analogy-network        # Internal Docker network
```

**Environment Configuration:**

Configuration is managed through environment variables, allowing flexible deployment options:

```
# Ollama Configuration
OLLAMA_URL=http://47.177.58.4:45721
OLLAMA_MODEL=openchat:latest
USE_OLLAMA=true

# SearXNG Configuration
SEARXNG_URL=http://47.177.58.4:9090
USE_SEARXNG=true

# General Configuration
FALLBACK_TO_STATIC=true
```

## Database Structures and Knowledge Representation

### Static Analogy Database

The system maintains a structured database of curated analogies in a nested dictionary format:

```python
{
    "Source Software": {
        "Source Feature": {
            "Target Software": {
                "target_feature": "Target Feature Name",
                "explanation": "Detailed explanation text...",
                "similarity_score": 0.85  # Confidence/relevance score
            }
        }
    }
}
```

Current domain mappings include:
- Adobe After Effects → Ableton Live
- Adobe After Effects → Blender
- Adobe Photoshop → Ableton Live
- Microsoft Excel → Python

### Vector Embedding Structure

Vector representations use a format that combines software and feature names as keys:

```python
embeddings = {
    "Adobe After Effects:Color Filter": vector1,  # 128-dimensional vector
    "Ableton Live:EQ Eight": vector2,             # Semantically similar to vector1
    # Additional embeddings...
}
```

Feature relationships are encoded by adjusting vector proximity for analogous concepts:

```python
# Create semantic similarity by making certain pairs more similar
color_base = np.random.rand(128)  # Base vector for color manipulation concept
embeddings["Adobe After Effects:Color Filter"] = color_base + noise_vector * 0.3
embeddings["Ableton Live:EQ Eight"] = color_base + noise_vector * 0.3
```

## Dependencies and Requirements

### Backend Requirements

```
fastapi==0.109.2           # API framework
uvicorn[standard]==0.27.1  # ASGI server
pydantic==2.5.3            # Data validation
scikit-learn==1.3.2        # Vector operations
numpy==1.26.3              # Numerical processing
requests==2.31.0           # HTTP client
```

### Frontend Dependencies

```json
{
  "dependencies": {
    "@craco/craco": "^7.1.0",      // Configuration override for CRA
    "react": "^18.2.0",            // UI framework
    "react-dom": "^18.2.0",        // DOM rendering
    "react-scripts": "5.0.1",      // Build scripts
    "axios": "^1.6.2",             // HTTP client
    "tailwindcss": "^3.3.0",       // Utility CSS framework
    "postcss": "^8.4.21",          // CSS processing
    "autoprefixer": "^10.4.14"     // CSS compatibility
  }
}
```

## Performance Considerations

### Memory Management

- **Backend**: The FastAPI application uses asynchronous patterns to handle multiple concurrent requests efficiently
- **LLM Integration**: Requests to Ollama use a 30-second timeout to accommodate generation time
- **Vector Operations**: NumPy arrays are used for efficient mathematical operations on embeddings

### Threading Model

- **Uvicorn Server**: Can be configured with multiple workers for CPU-intensive operations
- **React Frontend**: Uses the browser's event loop for non-blocking UI updates

### Caching Strategies

- **Static Analogies**: Loaded once at startup into memory
- **Service Status**: Checked on initial page load with status indicators

## Security Considerations

- **CORS Configuration**: Middleware allows configurable cross-origin access
- **Input Validation**: Request bodies are validated using Pydantic models
- **Error Handling**: Exceptions are caught and returned as structured HTTP responses

## Extension Points

The system architecture is designed for extensibility:

1. **Additional Domains**: Expanding the `analogy_db` dictionary with new software mappings
2. **Enhanced Vector Search**: Replacing mock embeddings with a true vector database
3. **Custom LLM Integration**: Adapting the Ollama client for other LLM APIs
4. **Feedback Loop**: Adding user rating system to improve analogy quality over time

## Development Workflow

### Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm start
```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

## Troubleshooting Guide

### Common Issues

1. **Docker Networking**: Ensure containers can communicate over the `analogy-network`
2. **Component Dependencies**: ShadowUI components require proper path aliases
3. **External Services**: Verify connectivity to Ollama and SearXNG instances
4. **Environment Variables**: Check that `.env` values match actual service locations

### Diagnostic Steps

1. **Backend API**: Verify availability at http://localhost:8000/docs
2. **Ollama Connection**: Test with curl: `curl -I http://47.177.58.4:45721/api/tags`
3. **SearXNG Connection**: Test with curl: `curl -I http://47.177.58.4:9090`
4. **Frontend Build**: Check for build errors in logs: `docker-compose logs frontend`

## Future Development Roadmap

1. **Knowledge Graph Integration**: Implementing Neo4j for complex concept relationships
2. **User Accounts and Personalization**: Saving learning paths and progress
3. **ML-Enhanced Analogies**: Fine-tuning domain-specific models for better analogies
4. **Interactive Visualization**: Graph-based navigation of concept relationships
5. **Voice Integration**: Adding voice commands and explanations for accessibility

## Conclusion

Metafive represents a novel approach to knowledge transfer that leverages:
- Structured analogical reasoning through curated mappings
- Dynamic explanations through LLM generation
- Contextual enrichment through documentation retrieval
- Visual interaction through intuitive UI components

This multi-modal system creates an environment where domain expertise becomes transferable across software boundaries, reducing the learning curve and accelerating skill acquisition.