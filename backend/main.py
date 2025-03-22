# backend/main.py
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from functools import lru_cache

# Import our custom modules
from ollama_integration import OllamaClient
from searxng_integration import SearXNGClient

app = FastAPI(title="Software Analogy API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DocumentationLink(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class AnalogyRequest(BaseModel):
    source_software: str
    target_software: str
    source_feature: str

class AnalogyResponse(BaseModel):
    target_feature: str
    explanation: str
    similarity_score: float = 1.0
    generated_by: str = "static"  # 'static', 'vector', or 'llm'
    source_documentation: List[DocumentationLink] = []
    target_documentation: List[DocumentationLink] = []

# Configuration
class Settings:
    # Ollama settings
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://47.177.58.4:45721")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
    
    # SearXNG settings
    SEARXNG_URL = os.getenv("SEARXNG_URL", "http://47.177.58.4:9090")
    USE_SEARXNG = os.getenv("USE_SEARXNG", "true").lower() == "true"
    
    # General settings
    FALLBACK_TO_STATIC = os.getenv("FALLBACK_TO_STATIC", "true").lower() == "true"

settings = Settings()

# Dependencies
@lru_cache(maxsize=1)
def get_ollama_client():
    """Get or create an Ollama client"""
    return OllamaClient(settings.OLLAMA_URL, settings.OLLAMA_MODEL)

@lru_cache(maxsize=1)
def get_searxng_client():
    """Get or create a SearXNG client"""
    return SearXNGClient(settings.SEARXNG_URL)

# Status endpoints
@app.get("/ollama-status")
async def check_ollama_status(ollama_client: OllamaClient = Depends(get_ollama_client)):
    """Check if Ollama is available and get current model"""
    status = ollama_client.health_check()
    return status

@app.get("/searxng-status")
async def check_searxng_status(searxng_client: SearXNGClient = Depends(get_searxng_client)):
    """Check if SearXNG is available"""
    is_available = searxng_client.health_check()
    return {"available": is_available}

# Model management endpoints
@app.get("/ollama-models", response_model=List[str])
async def get_ollama_models(ollama_client: OllamaClient = Depends(get_ollama_client)):
    """Get a list of available Ollama models"""
    return ollama_client.list_models()

@app.post("/ollama-model")
async def set_ollama_model(
    model_request: Dict[str, str],
    ollama_client: OllamaClient = Depends(get_ollama_client)
):
    """Set the active Ollama model"""
    model = model_request.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    success = ollama_client.set_model(model)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set model {model}. Model may not be available."
        )
    return {"status": "success", "model": model}

# Analogy endpoints
@app.post("/analogy", response_model=AnalogyResponse)
async def get_analogy(
    request: AnalogyRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
    searxng_client: SearXNGClient = Depends(get_searxng_client)
):
    """Get an analogy for a specific feature with documentation"""
    source = request.source_software
    target = request.target_software
    feature = request.source_feature
    
    # First try to get from static database
    if source in analogy_db and feature in analogy_db[source] and target in analogy_db[source][feature]:
        static_analogy = analogy_db[source][feature][target]
        target_feature = static_analogy["target_feature"]
        explanation = static_analogy["explanation"]
        similarity_score = static_analogy["similarity_score"]
        generated_by = "static"
        
        # Get documentation if SearXNG is enabled
        source_docs = []
        target_docs = []
        documentation_context = ""
        
        if settings.USE_SEARXNG:
            try:
                source_docs = [
                    DocumentationLink(**doc) for doc in 
                    searxng_client.get_software_documentation(source, feature)
                ]
                target_docs = [
                    DocumentationLink(**doc) for doc in 
                    searxng_client.get_software_documentation(target, target_feature)
                ]
                
                # Prepare documentation context for Ollama
                if source_docs or target_docs:
                    documentation_context = "Source software documentation:\n"
                    for doc in source_docs[:2]:
                        documentation_context += f"- {doc.title}: {doc.snippet}\n"
                    
                    documentation_context += "\nTarget software documentation:\n"
                    for doc in target_docs[:2]:
                        documentation_context += f"- {doc.title}: {doc.snippet}\n"
            except Exception as e:
                print(f"Error fetching documentation: {e}")
        
        # Try to generate a dynamic explanation with Ollama
        if settings.USE_OLLAMA:
            ollama_explanation = await ollama_client.generate_explanation(
                source, feature, target, target_feature,
                additional_context=documentation_context if documentation_context else None
            )
            
            if ollama_explanation:
                # Return with the dynamically generated explanation
                return AnalogyResponse(
                    target_feature=target_feature,
                    explanation=ollama_explanation,
                    similarity_score=similarity_score,
                    generated_by="llm",
                    source_documentation=source_docs,
                    target_documentation=target_docs
                )
        
        # If Ollama generation failed or is disabled, return the static explanation
        return AnalogyResponse(
            target_feature=target_feature,
            explanation=explanation,
            similarity_score=similarity_score,
            generated_by="static",
            source_documentation=source_docs,
            target_documentation=target_docs
        )
    
    # If not found in static database, raise an error
    # In a more advanced implementation, this could use vector search
    fallback_message = f"No analogy found for {feature} in {target}. We're expanding our database to include more analogies."
    
    raise HTTPException(
        status_code=404, 
        detail=fallback_message
    )

# Load the static analogy database (same as before)
def load_analogy_database():
    # Static database implementation (same as in your original code)
    return {
        "Adobe After Effects": {
            "Color Filter": {
                "Ableton Live": {
                    "target_feature": "EQ Eight",
                    "explanation": "Just as a color filter in After Effects adjusts specific color channels to transform the visual tone of your footage, the EQ Eight in Ableton Live isolates and adjusts specific frequency bands to shape the tonal quality of your audio. Both tools allow for precise manipulation of the spectrum (visual vs audio) using similar curve/band-based interfaces.",
                    "similarity_score": 0.85
                },
                # Other entries...
            },
            # Other features...
        },
        # Other software...
    }

analogy_db = load_analogy_database()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Software Analogy API is running with SearXNG and Ollama integration"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)