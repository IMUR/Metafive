import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';

const SoftwareAnalogyTool = () => {
  // API endpoint
  const API_BASE_URL = 'http://localhost:8000';
  
  // Software options
  const [sourceOptions, setSourceOptions] = useState([]);
  const [targetOptions, setTargetOptions] = useState([]);

  // State management
  const [sourceSoftware, setSourceSoftware] = useState("");
  const [targetSoftware, setTargetSoftware] = useState("");
  const [sourceFeatures, setSourceFeatures] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState("");
  const [analogyResult, setAnalogyResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for Ollama models
  const [ollamaStatus, setOllamaStatus] = useState(null);
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  
  // State for SearXNG
  const [searxngStatus, setSearxngStatus] = useState(null);

  // Load Ollama status and models on initial render
  useEffect(() => {
    const checkServices = async () => {
      try {
        // Check Ollama status
        const ollamaResponse = await fetch(`${API_BASE_URL}/ollama-status`);
        const ollamaData = await ollamaResponse.json();
        setOllamaStatus(ollamaData);
        
        if (ollamaData.available) {
          // Fetch available models
          const modelsResponse = await fetch(`${API_BASE_URL}/ollama-models`);
          if (modelsResponse.ok) {
            const modelsData = await modelsResponse.json();
            setOllamaModels(modelsData);
            
            // Set selected model to current or first available
            if (ollamaData.current_model && modelsData.includes(ollamaData.current_model)) {
              setSelectedModel(ollamaData.current_model);
            } else if (modelsData.length > 0) {
              setSelectedModel(modelsData[0]);
            }
          }
        }
        
        // Check SearXNG status
        const searxngResponse = await fetch(`${API_BASE_URL}/searxng-status`);
        const searxngData = await searxngResponse.json();
        setSearxngStatus(searxngData.available);
      } catch (error) {
        console.error('Error checking services:', error);
      }
    };
    
    checkServices();
  }, []);

  // Fetch source software options on load
  useEffect(() => {
    const fetchSourceSoftware = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/source-software`);
        if (!response.ok) {
          throw new Error('Failed to fetch source software');
        }
        const data = await response.json();
        setSourceOptions(data);
      } catch (error) {
        console.error('Error fetching source software:', error);
        // Fallback to static options if API fails
        setSourceOptions([
          "Adobe After Effects",
          "Adobe Photoshop",
          "Final Cut Pro",
          "Microsoft Excel",
          "Unity"
        ]);
      }
    };

    fetchSourceSoftware();
  }, []);

  // Update target software options when source software changes
  useEffect(() => {
    const fetchTargetSoftware = async () => {
      if (!sourceSoftware) {
        setTargetOptions([]);
        return;
      }

      try {
        const response = await fetch(`${API_BASE_URL}/target-software/${encodeURIComponent(sourceSoftware)}`);
        if (!response.ok) {
          throw new Error('Failed to fetch target software');
        }
        const data = await response.json();
        setTargetOptions(data);
      } catch (error) {
        console.error('Error fetching target software:', error);
        // Fallback to static options if API fails
        setTargetOptions([
          "Ableton Live",
          "Blender",
          "Figma",
          "Python",
          "Unreal Engine"
        ]);
      }
    };

    fetchTargetSoftware();
    setTargetSoftware(""); // Reset selection when source changes
  }, [sourceSoftware]);

  // Fetch features when source software changes
  useEffect(() => {
    const fetchFeatures = async () => {
      if (!sourceSoftware) {
        setSourceFeatures([]);
        return;
      }

      try {
        const response = await fetch(`${API_BASE_URL}/features/${encodeURIComponent(sourceSoftware)}`);
        if (!response.ok) {
          throw new Error('Failed to fetch features');
        }
        const data = await response.json();
        setSourceFeatures(data);
      } catch (error) {
        console.error('Error fetching features:', error);
        setSourceFeatures([]);
      }
    };

    fetchFeatures();
    setSelectedFeature(""); // Reset selection when source changes
  }, [sourceSoftware]);

  // Find analogy when all selections are made
  useEffect(() => {
    const findAnalogy = async () => {
      if (!sourceSoftware || !targetSoftware || !selectedFeature) {
        return;
      }

      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${API_BASE_URL}/analogy`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            source_software: sourceSoftware,
            target_software: targetSoftware,
            source_feature: selectedFeature,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to find analogy');
        }

        const data = await response.json();
        setAnalogyResult(data);
      } catch (error) {
        console.error('Error finding analogy:', error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    };

    findAnalogy();
  }, [sourceSoftware, targetSoftware, selectedFeature]);

  // Change Ollama model
  const handleModelChange = async (model) => {
    try {
      const response = await fetch(`${API_BASE_URL}/ollama-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model }),
      });
      
      if (response.ok) {
        setSelectedModel(model);
        // Show success indicator
        // ...
      } else {
        // Show error indicator
        // ...
      }
    } catch (error) {
      console.error('Error setting model:', error);
    }
  };
  
  // Render service status indicators
  const renderServiceStatus = () => (
    <div className="flex flex-wrap gap-2 justify-center my-2">
      <span className={`inline-flex items-center px-3 py-1 text-sm rounded-full ${ollamaStatus?.available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        <span className={`w-2 h-2 mr-2 rounded-full ${ollamaStatus?.available ? 'bg-green-500' : 'bg-red-500'}`}></span>
        Ollama {ollamaStatus?.available ? 'Connected' : 'Unavailable'} 
        {ollamaStatus?.current_model && ` (${ollamaStatus.current_model})`}
      </span>
      
      <span className={`inline-flex items-center px-3 py-1 text-sm rounded-full ${searxngStatus ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        <span className={`w-2 h-2 mr-2 rounded-full ${searxngStatus ? 'bg-green-500' : 'bg-red-500'}`}></span>
        SearXNG {searxngStatus ? 'Connected' : 'Unavailable'}
      </span>
    </div>
  );
  
  // Render model selector
  const renderModelSelector = () => {
    if (!ollamaStatus?.available || ollamaModels.length === 0) return null;
    
    return (
      <div className="mb-6 p-4 border rounded-md bg-blue-50">
        <label className="block text-md font-medium mb-2">
          Ollama Model Selection:
        </label>
        <div className="flex flex-wrap gap-2">
          {ollamaModels.map((model) => (
            <button
              key={model}
              onClick={() => handleModelChange(model)}
              className={`px-3 py-1 rounded-md text-sm ${
                selectedModel === model 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-white text-blue-600 border border-blue-300 hover:bg-blue-50'
              }`}
            >
              {model}
            </button>
          ))}
        </div>
      </div>
    );
  };
  
  // Render documentation links
  const renderDocumentationLinks = (docs, title) => {
    if (!docs || docs.length === 0) return null;
    
    return (
      <div className="mt-3">
        <h4 className="font-medium text-gray-700">{title}:</h4>
        <ul className="mt-1 space-y-1">
          {docs.map((doc, index) => (
            <li key={index} className="ml-4 text-sm">
              <a 
                href={doc.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline flex items-start"
              >
                <span className="inline-block w-4 h-4 mt-0.5 mr-1">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" 
                      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                </span>
                {doc.title}
              </a>
              {doc.snippet && (
                <p className="text-xs text-gray-600 ml-5 mt-1">{doc.snippet}</p>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Get background color based on generation method
  const getResultBackgroundColor = () => {
    if (!analogyResult) return "from-blue-50 to-indigo-50";
    
    switch(analogyResult.generated_by) {
      case 'llm':
        return "from-green-50 to-teal-50";
      case 'vector':
        return "from-purple-50 to-indigo-50";
      case 'static':
      default:
        return "from-blue-50 to-indigo-50";
    }
  };

  // Get badge text based on generation method
  const getGenerationBadge = () => {
    if (!analogyResult) return null;
    
    switch(analogyResult.generated_by) {
      case 'llm':
        return {
          text: "AI-Generated",
          color: "bg-green-100 text-green-800"
        };
      case 'vector':
        return {
          text: "Vector Match",
          color: "bg-purple-100 text-purple-800"
        };
      case 'static':
      default:
        return {
          text: "Curated",
          color: "bg-blue-100 text-blue-800"
        };
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">Software Analogy Discovery Tool</h1>
      
      <div className="mb-8 text-center">
        <p className="text-lg mb-2">
          Select software you know and software you want to learn. 
          Then choose a feature you understand to find its analogy.
        </p>
        
        {renderServiceStatus()}
      </div>
      
      {renderModelSelector()}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <label className="block text-lg font-medium mb-2">Software I Know:</label>
          <select
            className="w-full p-2 border rounded-md bg-white"
            value={sourceSoftware}
            onChange={(e) => setSourceSoftware(e.target.value)}
          >
            <option value="">Select software...</option>
            {sourceOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-lg font-medium mb-2">Software I Want to Learn:</label>
          <select
            className="w-full p-2 border rounded-md bg-white"
            value={targetSoftware}
            onChange={(e) => setTargetSoftware(e.target.value)}
            disabled={!sourceSoftware}
          >
            <option value="">Select software...</option>
            {targetOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {sourceSoftware && sourceFeatures.length > 0 && (
        <div className="mb-8">
          <label className="block text-lg font-medium mb-2">
            Select a feature from {sourceSoftware} you're familiar with:
          </label>
          <select
            className="w-full p-2 border rounded-md bg-white"
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            disabled={!targetSoftware}
          >
            <option value="">Select a feature...</option>
            {sourceFeatures.map((feature) => (
              <option key={feature} value={feature}>
                {feature}
              </option>
            ))}
          </select>
        </div>
      )}
      
      {isLoading && (
        <div className="text-center py-8">
          <p className="text-lg">Finding analogies...</p>
        </div>
      )}
      
      {error && (
        <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-700">
          <p>{error}</p>
        </div>
      )}
      
      {!isLoading && analogyResult && (
        <Card className={`mt-8 bg-gradient-to-r ${getResultBackgroundColor()} border-2 border-blue-200`}>
          <CardContent className="p-6">
            <div className="flex justify-between items-start">
              <h2 className="text-2xl font-bold mb-4">
                <span className="text-blue-600">{selectedFeature}</span> in {sourceSoftware} is like <span className="text-indigo-600">{analogyResult.target_feature}</span> in {targetSoftware}
              </h2>
              
              {getGenerationBadge() && (
                <span className={`ml-2 px-3 py-1 text-sm rounded-full ${getGenerationBadge().color}`}>
                  {getGenerationBadge().text}
                </span>
              )}
            </div>
            
            <div className="mt-4 text-lg">
              <p>{analogyResult.explanation}</p>
            </div>
            
            <div className="mt-6 text-sm text-gray-600">
              <p>Similarity score: {(analogyResult.similarity_score * 100).toFixed(0)}%</p>
            </div>
            
            {/* Documentation sections */}
            {(analogyResult.source_documentation?.length > 0 || 
              analogyResult.target_documentation?.length > 0) && (
              <div className="mt-6 pt-4 border-t border-blue-200">
                <h3 className="text-lg font-semibold mb-2">Related Documentation</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {renderDocumentationLinks(
                    analogyResult.source_documentation,
                    `${sourceSoftware} Resources`
                  )}
                  
                  {renderDocumentationLinks(
                    analogyResult.target_documentation,
                    `${targetSoftware} Resources`
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
      
      <div className="mt-12 text-center text-sm text-gray-500">
        <p>This is an MVP version with a limited database of analogies.</p>
        <p className="mt-1">Connect to Ollama for AI-generated explanations.</p>
      </div>
    </div>
  );
};

export default SoftwareAnalogyTool;