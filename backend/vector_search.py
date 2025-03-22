from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# This file would be imported into the main FastAPI app
# We're showing the vector search component separately for clarity

class FeatureEmbedding:
    """Class to handle feature embeddings and similarity search"""
    
    def __init__(self):
        # In a real system, these would be generated using a proper embedding model
        # For this MVP, we'll use mock pre-computed embeddings
        self.embeddings = self._load_mock_embeddings()
        self.feature_info = self._load_feature_info()
    
    def _load_mock_embeddings(self) -> Dict[str, np.ndarray]:
        """Load mock embeddings for features (in reality, use a proper embedding model)"""
        
        # Each feature is represented by a vector (simplified for demo)
        # Format: {"software:feature": embedding_vector}
        
        # These are randomly generated for demo purposes
        # In a real system, these would be meaningful semantic embeddings
        np.random.seed(42)  # For reproducibility
        
        embeddings = {
            # After Effects features
            "Adobe After Effects:Color Filter": np.random.rand(128),
            "Adobe After Effects:Keyframes": np.random.rand(128),
            "Adobe After Effects:Precomposing": np.random.rand(128),
            "Adobe After Effects:Masks": np.random.rand(128),
            "Adobe After Effects:Effects Panel": np.random.rand(128),
            
            # Photoshop features
            "Adobe Photoshop:Layers": np.random.rand(128),
            "Adobe Photoshop:Adjustment Layers": np.random.rand(128),
            "Adobe Photoshop:Filters": np.random.rand(128),
            "Adobe Photoshop:Selection Tools": np.random.rand(128),
            
            # Excel features
            "Microsoft Excel:Formulas": np.random.rand(128),
            "Microsoft Excel:Pivot Tables": np.random.rand(128),
            "Microsoft Excel:Charts": np.random.rand(128),
            
            # Ableton Live features
            "Ableton Live:EQ Eight": np.random.rand(128),
            "Ableton Live:Automation Curves": np.random.rand(128),
            "Ableton Live:Groups and Sends": np.random.rand(128),
            "Ableton Live:Tracks": np.random.rand(128),
            "Ableton Live:Return Tracks": np.random.rand(128),
            "Ableton Live:MIDI Editor": np.random.rand(128),
            
            # Blender features
            "Blender:Color Ramp Node": np.random.rand(128),
            "Blender:Collections": np.random.rand(128),
            "Blender:Animation Timeline": np.random.rand(128),
            "Blender:Material Nodes": np.random.rand(128),
            
            # Python features
            "Python:Functions": np.random.rand(128),
            "Python:pandas GroupBy": np.random.rand(128),
            "Python:List Comprehensions": np.random.rand(128)
        }
        
        # Create semantic similarity by making certain pairs more similar
        # Simulating semantic proximity for analogous concepts
        
        # Color manipulation concepts
        color_base = np.random.rand(128)
        embeddings["Adobe After Effects:Color Filter"] = color_base + np.random.rand(128) * 0.3
        embeddings["Ableton Live:EQ Eight"] = color_base + np.random.rand(128) * 0.3
        embeddings["Blender:Color Ramp Node"] = color_base + np.random.rand(128) * 0.3
        
        # Timeline/keyframe concepts
        timeline_base = np.random.rand(128)
        embeddings["Adobe After Effects:Keyframes"] = timeline_base + np.random.rand(128) * 0.3
        embeddings["Ableton Live:Automation Curves"] = timeline_base + np.random.rand(128) * 0.3
        embeddings["Blender:Animation Timeline"] = timeline_base + np.random.rand(128) * 0.3
        
        # Grouping concepts
        grouping_base = np.random.rand(128)
        embeddings["Adobe After Effects:Precomposing"] = grouping_base + np.random.rand(128) * 0.3
        embeddings["Ableton Live:Groups and Sends"] = grouping_base + np.random.rand(128) * 0.3
        embeddings["Blender:Collections"] = grouping_base + np.random.rand(128) * 0.3
        
        # Layering concepts
        layer_base = np.random.rand(128)
        embeddings["Adobe Photoshop:Layers"] = layer_base + np.random.rand(128) * 0.3
        embeddings["Ableton Live:Tracks"] = layer_base + np.random.rand(128) * 0.3
        
        # Data transformation concepts
        data_base = np.random.rand(128)
        embeddings["Microsoft Excel:Formulas"] = data_base + np.random.rand(128) * 0.3
        embeddings["Python:Functions"] = data_base + np.random.rand(128) * 0.3
        
        # Normalize the vectors
        for key in embeddings:
            embeddings[key] = embeddings[key] / np.linalg.norm(embeddings[key])
            
        return embeddings
    
    def _load_feature_info(self) -> Dict[str, Dict]:
        """Load descriptive information about features"""
        return {
            # After Effects features
            "Adobe After Effects:Color Filter": {
                "name": "Color Filter",
                "description": "Adjusts colors in video footage by manipulating color channels and tones"
            },
            "Adobe After Effects:Keyframes": {
                "name": "Keyframes",
                "description": "Marks points where you specify values for properties like position, scale, and rotation"
            },
            "Adobe After Effects:Precomposing": {
                "name": "Precomposing",
                "description": "Groups layers into a nested composition to organize and apply effects collectively"
            },
            
            # Ableton Live features
            "Ableton Live:EQ Eight": {
                "name": "EQ Eight",
                "description": "Audio equalizer that isolates and adjusts specific frequency bands in audio"
            },
            "Ableton Live:Automation Curves": {
                "name": "Automation Curves",
                "description": "Controls how parameters change over time using editable curves on the timeline"
            },
            "Ableton Live:Groups and Sends": {
                "name": "Groups and Sends",
                "description": "Organizes multiple tracks and routes them through shared processing"
            },
            
            # More features would be defined here...
        }
    
    def get_most_similar_features(self, query_key: str, target_software: str, top_k: int = 3) -> List[Dict]:
        """
        Find the most similar features from the target software
        
        Args:
            query_key: String in format "software:feature"
            target_software: Target software to find analogies in
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with similarity results
        """
        if query_key not in self.embeddings:
            raise ValueError(f"Unknown feature: {query_key}")
        
        query_embedding = self.embeddings[query_key]
        
        # Find all features from the target software
        target_keys = [k for k in self.embeddings.keys() if k.startswith(f"{target_software}:")]
        
        if not target_keys:
            return []
        
        # Get embeddings for target features
        target_embeddings = np.array([self.embeddings[k] for k in target_keys])
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], target_embeddings)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(-similarities)  # Descending order
        
        results = []
        for idx in sorted_indices[:top_k]:
            target_key = target_keys[idx]
            software, feature = target_key.split(":", 1)
            
            # Get feature info or use defaults
            feature_info = self.feature_info.get(target_key, {
                "name": feature,
                "description": "No description available"
            })
            
            results.append({
                "software": software,
                "feature": feature_info["name"],
                "description": feature_info["description"],
                "similarity_score": float(similarities[idx])
            })
        
        return results

    def generate_analogy_explanation(self, source_key: str, target_key: str) -> str:
        """
        Generate an explanation of the analogy between two features
        
        In a real system, this would use an LLM. Here we'll use templates.
        """
        source_software, source_feature = source_key.split(":", 1)
        target_software, target_feature = target_key.split(":", 1)
        
        source_info = self.feature_info.get(source_key, {"description": "No description available"})
        target_info = self.feature_info.get(target_key, {"description": "No description available"})
        
        templates = [
            f"In {source_software}, {source_feature} {source_info['description']}. Similarly, in {target_software}, {target_feature} {target_info['description']}. Both tools allow you to manipulate core elements in their respective domains.",
            
            f"{source_feature} in {source_software} is analogous to {target_feature} in {target_software} because both serve similar functions in their domains. While one works with {source_software.split()[-1].lower()} elements, the other handles {target_software.split()[-1].lower()} components.",
            
            f"The concept of {source_feature} from {source_software} translates to {target_feature} in {target_software}. If you understand how to use {source_feature}, you already grasp the fundamental concept behind {target_feature}."
        ]
        
        # In a real system, choose based on context or use an LLM
        import random
        return random.choice(templates)


# Example usage (this would be integrated with the main API)
feature_embedder = FeatureEmbedding()

def find_analogies_vector(source_software: str, source_feature: str, target_software: str):
    """Find analogies using vector similarity"""
    query_key = f"{source_software}:{source_feature}"
    
    try:
        similar_features = feature_embedder.get_most_similar_features(
            query_key, target_software, top_k=3
        )
        
        if not similar_features:
            return {
                "target_feature": "No analogy found",
                "explanation": f"We couldn't find features in {target_software} analogous to {source_feature}.",
                "similarity_score": 0.0
            }
        
        # Get the top match
        top_match = similar_features[0]
        target_key = f"{top_match['software']}:{top_match['feature']}"
        
        # Generate explanation
        explanation = feature_embedder.generate_analogy_explanation(query_key, target_key)
        
        return {
            "target_feature": top_match["feature"],
            "explanation": explanation,
            "similarity_score": top_match["similarity_score"]
        }
    
    except ValueError as e:
        return {
            "target_feature": "Error",
            "explanation": str(e),
            "similarity_score": 0.0
        }

# Example of how to integrate this with the main FastAPI app:
# 
# @app.post("/vector-analogy")
# async def get_vector_analogy(request: AnalogyRequest):
#     return find_analogies_vector(
#         request.source_software, 
#         request.source_feature,
#         request.target_software
#     )