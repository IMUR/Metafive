# backend/ollama_integration.py
import requests
import json
import logging
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API to generate analogy explanations
    """
    
    def __init__(self, base_url: str, model: str = "llama2"):
        """
        Initialize the Ollama client
        
        Args:
            base_url: The base URL of the Ollama API (e.g., "http://47.177.58.4:45721")
            model: The model to use for generations (default: llama2)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        logger.info(f"Initialized Ollama client with URL: {self.base_url}, model: {self.model}")
    
    def list_models(self) -> List[str]:
        """
        Get a list of available models from Ollama
        
        Returns:
            List of model names available on the Ollama instance
        """
        try:
            logger.info(f"Fetching available models from Ollama")
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            available_models = []
            
            if "models" in models_data:
                available_models = [model.get("name", "") for model in models_data["models"]]
            
            logger.info(f"Found {len(available_models)} available models: {', '.join(available_models)}")
            return available_models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the active model
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Attempting to set active model to: {model_name}")
            
            # Verify the model exists
            available_models = self.list_models()
            if model_name not in available_models:
                logger.warning(f"Model '{model_name}' not available in Ollama")
                return False
                
            self.model = model_name
            logger.info(f"Successfully set active model to: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False
    
    def generate_explanation(self, 
                           source_software: str, 
                           source_feature: str,
                           target_software: str,
                           target_feature: str,
                           temperature: float = 0.7,
                           max_tokens: int = 300,
                           additional_context: str = None) -> Optional[str]:
        """
        Generate an explanation of the analogy between two software features
        
        Args:
            source_software: Name of the source software
            source_feature: Name of the feature in source software
            target_software: Name of the target software
            target_feature: Name of the analogous feature in target software
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            additional_context: Optional additional context or documentation snippets
            
        Returns:
            Generated explanation or None if generation failed
        """
        prompt = f"""
        Create an explanation of how "{source_feature}" in {source_software} is analogous to 
        "{target_feature}" in {target_software}.
        
        Focus on:
        1. How the concepts are similar in purpose
        2. How understanding one helps understand the other
        3. Key differences to be aware of
        
        Keep the explanation straightforward and informative.
        """
        
        # Add additional context if provided
        if additional_context:
            prompt += f"\n\nHere is some additional documentation that may help:\n{additional_context}\n"
        
        try:
            logger.info(f"Generating analogy explanation using model: {self.model}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=30  # Longer timeout for generation
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text and clean it up
            explanation = result.get("response", "").strip()
            logger.info(f"Successfully generated explanation ({len(explanation)} chars)")
            return explanation
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    def health_check(self) -> Dict[str, Union[bool, str]]:
        """
        Check if the Ollama service is available
        
        Returns:
            Dict with status and current model
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            is_available = response.status_code == 200
            
            return {
                "available": is_available,
                "current_model": self.model if is_available else None
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama health check failed: {e}")
            return {"available": False, "current_model": None}