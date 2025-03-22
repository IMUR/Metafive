# Vector-First Approach to Analogical Learning System

## Streamlined Architecture

You've hit on an excellent approach! Rather than building a complex system upfront, we can create a learning platform that improves through usage:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ User Interface  │──────┤ Analogy Engine  │──────┤ Feedback Loop   │
└─────────────────┘      └───────┬─────────┘      └─────────────────┘
                                 │                          │
                         ┌───────▼─────────┐                │
                         │ Ollama + SearXNG│                │
                         └───────┬─────────┘                │
                                 │                          │
                         ┌───────▼─────────────────────────▼┐
                         │     Vector Database (Growing)    │
                         └───────────────────────────────────┘
```

## Implementation Strategy

### 1. Core Analogy Generation (2 days)

```python
# main.py - Core analogy generation logic
from fastapi import FastAPI, HTTPException
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel

class AnalogyRequest(BaseModel):
    source_domain: str           # Software/domain the user knows
    source_concept: str          # Concept the user understands
    target_domain: str           # Software/domain the user wants to learn

class AnalogyResponse(BaseModel):
    target_concept: str          # Identified analogous concept
    explanation: str             # Detailed explanation
    similarity_score: float      # Confidence score (0-1)
    user_rating: Optional[int]   # User feedback (1-5 stars)

async def generate_analogy(request: AnalogyRequest) -> Dict:
    """Generate analogy using Ollama with SearXNG context enrichment"""
    
    # Step 1: Get contextual information via SearXNG
    source_context = await search_documentation(
        f"{request.source_domain} {request.source_concept} how it works", 
        max_results=2
    )
    
    target_context = await search_documentation(
        f"{request.target_domain} main features concepts", 
        max_results=3
    )
    
    # Step 2: Construct optimized prompt for Ollama
    prompt = f"""
    Create an educational analogy mapping from {request.source_domain} to {request.target_domain}.
    
    SOURCE CONCEPT: {request.source_concept} in {request.source_domain}
    TARGET DOMAIN: {request.target_domain}
    
    Your task:
    1. Identify the most analogous concept/feature in {request.target_domain}
    2. Create a structured explanation of the conceptual mapping
    3. Format your response exactly as follows:
    
    ANALOGOUS CONCEPT: [name of the analogous concept in {request.target_domain}]
    
    MAPPING:
    * [key aspect 1]: [explanation]
    * [key aspect 2]: [explanation]
    * [key aspect 3]: [explanation]
    
    DIFFERENCES:
    * [important difference 1]
    * [important difference 2]
    
    LEARNING PATHWAY:
    [1-2 sentences on how understanding the source helps understand the target]
    
    Context about source concept:
    {source_context}
    
    Context about target domain:
    {target_context}
    """
    
    # Step 3: Generate with Ollama
    response = await call_ollama_api(prompt, model="openchat:latest", max_tokens=800)
    
    # Step 4: Parse structured response
    parsed = parse_structured_response(response)
    
    return {
        "target_concept": parsed["analogous_concept"],
        "explanation": parsed["full_explanation"],
        "similarity_score": 0.85,  # Default for fresh generations
        "metadata": {
            "generation_method": "ollama_dynamic",
            "model_used": "openchat:latest",
            "timestamp": current_timestamp()
        }
    }
```

### 2. Vector Database Integration (1 day)

```python
# vector_store.py - Growing database of analogies
import numpy as np
from typing import Dict, List, Optional
import asyncio
from datetime import datetime

class VectorStore:
    def __init__(self, embedding_dimension=1536):
        # Start with empty collections
        self.embeddings = []         # Vector representations
        self.metadata = []           # Associated metadata
        self.ratings = {}            # User ratings
    
    async def add_analogy(self, analogy_data: Dict, user_rating: Optional[int] = None):
        """Add new analogy to the database with embedding"""
        
        # Create composite key for the analogy
        key = f"{analogy_data['source_domain']}:{analogy_data['source_concept']}→{analogy_data['target_domain']}:{analogy_data['target_concept']}"
        
        # Generate embedding (could use Ollama or dedicated embedding model)
        text_to_embed = f"{analogy_data['source_concept']} {analogy_data['target_concept']} {analogy_data['explanation']}"
        embedding = await generate_embedding(text_to_embed)
        
        # Store in our simple vector store
        self.embeddings.append(embedding)
        self.metadata.append({
            "key": key,
            "source_domain": analogy_data["source_domain"],
            "source_concept": analogy_data["source_concept"],
            "target_domain": analogy_data["target_domain"],
            "target_concept": analogy_data["target_concept"],
            "explanation": analogy_data["explanation"],
            "created_at": datetime.now().isoformat(),
            "rating_count": 1 if user_rating else 0,
            "rating_avg": user_rating if user_rating else None
        })
        
        # Store rating if provided
        if user_rating:
            self.ratings[key] = [user_rating]
    
    async def find_similar(self, query_text: str, min_similarity: float = 0.8):
        """Find similar analogies using vector similarity"""
        
        # Early return if database is empty
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = await generate_embedding(query_text)
        
        # Calculate similarities (cosine similarity)
        similarities = [
            np.dot(query_embedding, stored_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            for stored_embedding in self.embeddings
        ]
        
        # Get matching analogies above threshold
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                result = self.metadata[i].copy()
                result["similarity"] = float(similarity)
                results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
```

### 3. Minimal API Implementation (1 day)

```python
# Simplified FastAPI implementation
app = FastAPI(title="Metafive Learning Bridges")

@app.post("/analogy", response_model=AnalogyResponse)
async def get_analogy(request: AnalogyRequest):
    """Get an analogy with vector lookup or dynamic generation"""
    
    try:
        # Step 1: Check if we have similar analogies in the vector store
        query = f"{request.source_domain} {request.source_concept} {request.target_domain}"
        similar_analogies = await vector_store.find_similar(query, min_similarity=0.85)
        
        # If we have a high-quality match, return it
        if similar_analogies and similar_analogies[0].get("rating_avg", 0) >= 4.0:
            best_match = similar_analogies[0]
            return {
                "target_concept": best_match["target_concept"],
                "explanation": best_match["explanation"],
                "similarity_score": best_match["similarity"],
                "generation_method": "vector_retrieval"
            }
        
        # Step 2: Otherwise, generate a new analogy
        generated = await generate_analogy(request)
        
        # Step 3: Store the generated analogy (without rating yet)
        await vector_store.add_analogy({
            "source_domain": request.source_domain,
            "source_concept": request.source_concept,
            "target_domain": request.target_domain,
            "target_concept": generated["target_concept"],
            "explanation": generated["explanation"]
        })
        
        return generated
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analogy generation failed: {str(e)}")

@app.post("/rate-analogy")
async def rate_analogy(rating_data: Dict):
    """Store user ratings to improve the system"""
    
    # Update vector database with user feedback
    key = f"{rating_data['source_domain']}:{rating_data['source_concept']}→{rating_data['target_domain']}:{rating_data['target_concept']}"
    
    # Update ratings
    if key in vector_store.ratings:
        vector_store.ratings[key].append(rating_data["rating"])
    else:
        vector_store.ratings[key] = [rating_data["rating"]]
    
    # Update metadata with new average rating
    for i, metadata in enumerate(vector_store.metadata):
        if metadata["key"] == key:
            ratings = vector_store.ratings[key]
            vector_store.metadata[i]["rating_count"] = len(ratings)
            vector_store.metadata[i]["rating_avg"] = sum(ratings) / len(ratings)
            break
    
    return {"status": "success", "message": "Rating stored successfully"}
```

### 4. Simple React Frontend (1-2 days)

```jsx
// App.jsx - Simplified React component
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AnalogyGenerator = () => {
  // State
  const [sourceDomain, setSourceDomain] = useState('');
  const [sourceOptions, setSourceOptions] = useState([
    'Adobe After Effects', 'Adobe Photoshop', 'Microsoft Excel', 'Python', 'Unity'
  ]);
  
  const [sourceConcept, setSourceConcept] = useState('');
  const [conceptOptions, setConceptOptions] = useState([]);
  
  const [targetDomain, setTargetDomain] = useState('');
  const [targetOptions, setTargetOptions] = useState([
    'Ableton Live', 'Blender', 'Unreal Engine', 'Figma', 'JavaScript'
  ]);
  
  const [analogy, setAnalogy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [userRating, setUserRating] = useState(0);
  
  // Helper for domain-specific concept suggestions
  const getConceptSuggestions = (domain) => {
    // Simplified example - in production, fetch from API
    const conceptMap = {
      'Adobe After Effects': ['Color Filter', 'Keyframes', 'Precomposing', 'Masks'],
      'Adobe Photoshop': ['Layers', 'Adjustment Layers', 'Filters'],
      'Microsoft Excel': ['Formulas', 'Pivot Tables', 'Charts'],
      // Add more mappings as needed
    };
    
    return conceptMap[domain] || [];
  };
  
  // When source domain changes, update concept options
  useEffect(() => {
    if (sourceDomain) {
      setConceptOptions(getConceptSuggestions(sourceDomain));
      setSourceConcept('');
    } else {
      setConceptOptions([]);
    }
  }, [sourceDomain]);
  
  // Function to generate analogy
  const generateAnalogy = async () => {
    if (!sourceDomain || !sourceConcept || !targetDomain) {
      alert('Please select all required fields');
      return;
    }
    
    setLoading(true);
    setAnalogy(null);
    
    try {
      const response = await axios.post('/api/analogy', {
        source_domain: sourceDomain,
        source_concept: sourceConcept,
        target_domain: targetDomain
      });
      
      setAnalogy(response.data);
      setUserRating(0); // Reset rating for new analogy
    } catch (error) {
      console.error('Error generating analogy:', error);
      alert('Failed to generate analogy. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Submit user rating
  const submitRating = async (rating) => {
    if (!analogy) return;
    
    setUserRating(rating);
    
    try {
      await axios.post('/api/rate-analogy', {
        source_domain: sourceDomain,
        source_concept: sourceConcept,
        target_domain: targetDomain,
        target_concept: analogy.target_concept,
        rating: rating
      });
      
      // Show confirmation
      alert('Thanks for your feedback!');
    } catch (error) {
      console.error('Error submitting rating:', error);
    }
  };
  
  // Render star rating component
  const renderStarRating = () => {
    return (
      <div className="rating-container">
        <p>How helpful was this analogy?</p>
        <div className="stars">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className={star <= userRating ? 'star active' : 'star'}
              onClick={() => submitRating(star)}
            >
              ★
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="analogy-generator">
      <h1>Metafive Learning Bridges</h1>
      
      <div className="input-section">
        <div className="form-group">
          <label>I know</label>
          <select value={sourceDomain} onChange={(e) => setSourceDomain(e.target.value)}>
            <option value="">Select software you know...</option>
            {sourceOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        
        {sourceDomain && (
          <div className="form-group">
            <label>Specifically, I understand</label>
            <select value={sourceConcept} onChange={(e) => setSourceConcept(e.target.value)}>
              <option value="">Select a concept...</option>
              {conceptOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        )}
        
        <div className="form-group">
          <label>I want to learn</label>
          <select value={targetDomain} onChange={(e) => setTargetDomain(e.target.value)}>
            <option value="">Select software to learn...</option>
            {targetOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
        
        <button 
          className="generate-button" 
          onClick={generateAnalogy}
          disabled={!sourceDomain || !sourceConcept || !targetDomain || loading}
        >
          {loading ? 'Generating...' : 'Find Analogy'}
        </button>
      </div>
      
      {analogy && (
        <div className="result-container">
          <h2>
            <span className="source">{sourceConcept}</span> in {sourceDomain} 
            is like 
            <span className="target"> {analogy.target_concept}</span> in {targetDomain}
          </h2>
          
          <div className="explanation">
            {analogy.explanation}
          </div>
          
          {renderStarRating()}
        </div>
      )}
    </div>
  );
};

export default AnalogyGenerator;
```

## Deployment Strategy 

1. **FastAPI Backend with Vector Store**:
   ```bash
   # requirements.txt
   fastapi==0.109.2
   uvicorn[standard]==0.27.1
   numpy==1.26.3
   requests==2.31.0
   scikit-learn==1.3.2  # For vector operations
   faiss-cpu==1.7.4     # Simple vector database (upgrade to Weaviate/Qdrant later)
   ```

2. **Docker Setup**:
   ```yaml
   # docker-compose.yml (simplified)
   version: '3.8'
   
   services:
     backend:
       build: ./backend
       ports:
         - "8000:8000"
       environment:
         - OLLAMA_URL=http://47.177.58.4:45721
         - SEARXNG_URL=http://47.177.58.4:9090
     
     frontend:
       build: ./frontend
       ports:
         - "3000:80"
       depends_on:
         - backend
   ```

## Implementation Timeline

| Day | Focus | Tasks |
|-----|-------|-------|
| 1 | Core Integration | Connect to Ollama/SearXNG, implement analogy prompt |
| 2 | Vector Storage | Build simple vector database, implement similarity search |
| 3 | API Endpoints | Create FastAPI endpoints for generation and rating |
| 4-5 | Frontend | Build React UI with domain selection and rating system |
| 6 | Deployment | Docker setup, environment configuration |
| 7 | Testing | End-to-end testing, user feedback collection |

## Advantages of This Approach

1. **Start Small, Grow Smart**: Begin with pure LLM generation, build database through usage
2. **Continuous Improvement**: System gets better as users rate analogies
3. **Reduced Upfront Development**: No need for extensive pre-built analogy database
4. **Real User Validation**: Vector database only stores what users find valuable

This approach gives you a working system in days rather than weeks, focusing on the core value proposition while creating a foundation that improves with each user interaction.

Would you like me to elaborate on any specific aspect of this implementation plan?