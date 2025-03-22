# backend/searxng_integration.py
import requests
from typing import List, Dict, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearXNGClient:
    """Client for interacting with SearXNG metasearch engine"""
    
    def __init__(self, base_url: str):
        """
        Initialize the SearXNG client
        
        Args:
            base_url: The base URL of the SearXNG instance (e.g., "http://47.177.58.4:9090")
        """
        self.base_url = base_url.rstrip('/')
        self.search_endpoint = f"{self.base_url}/search"
        logger.info(f"Initialized SearXNG client with endpoint: {self.search_endpoint}")
    
    def search(self, 
               query: str, 
               categories: List[str] = None, 
               engines: List[str] = None,
               language: str = "en",
               max_results: int = 5,
               timeout: int = 10) -> Optional[Dict]:
        """
        Perform a search using SearXNG
        
        Args:
            query: Search query string
            categories: Optional list of categories to search
            engines: Optional list of engines to use
            language: Language code
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            
        Returns:
            Search results or None if search failed
        """
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "pageno": 1,
            "safesearch": 1,
            "time_range": "",
        }
        
        if categories:
            params["categories"] = ",".join(categories)
        
        if engines:
            params["engines"] = ",".join(engines)
        
        try:
            logger.info(f"Searching SearXNG for: {query}")
            response = requests.get(
                self.search_endpoint,
                params=params,
                headers={"User-Agent": "Software-Analogy-Tool/1.0"},
                timeout=timeout
            )
            
            response.raise_for_status()
            results = response.json()
            
            # Limit the number of results if needed
            if "results" in results and len(results["results"]) > max_results:
                results["results"] = results["results"][:max_results]
            
            logger.info(f"Found {len(results.get('results', []))} results")
            return results
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error performing SearXNG search: {e}")
            return None
    
    def get_software_documentation(self, 
                                  software_name: str, 
                                  feature_name: str = None) -> List[Dict]:
        """
        Search for documentation about a specific software or feature
        
        Args:
            software_name: Name of the software
            feature_name: Optional specific feature name
            
        Returns:
            List of relevant documentation entries
        """
        query = software_name
        if feature_name:
            query += f" {feature_name} documentation tutorial guide"
        else:
            query += " documentation guide"
        
        logger.info(f"Searching for documentation: {query}")
        results = self.search(
            query=query,
            categories=["it", "science"],
            engines=["duckduckgo", "bing", "brave", "qwant"]
        )
        
        if not results or "results" not in results:
            return []
        
        # Process and clean up results
        docs = []
        for result in results["results"]:
            # Skip results without title or URL
            if not result.get("title") or not result.get("url"):
                continue
                
            # Format document entry
            docs.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:150] + "..." if result.get("content") else ""
            })
        
        return docs
    
    def health_check(self) -> bool:
        """Check if the SearXNG service is available"""
        try:
            # Try to access the main page or health endpoint
            response = requests.get(f"{self.base_url}", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"SearXNG health check failed: {e}")
            return False