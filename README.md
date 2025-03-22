# Metafive

## Overview

The Metafive is an interactive learning application that helps users leverage their existing knowledge in one software domain to learn another through analogies. By mapping familiar concepts from software you know to corresponding features in software you want to learn, the tool accelerates the learning curve through contextual understanding.

## Key Features

- **Cross-Domain Analogies**: Maps concepts between different software applications (e.g., color filters in After Effects to EQ in Ableton Live)
- **AI-Generated Explanations**: Uses Ollama for dynamic generation of detailed analogy explanations
- **Multiple LLM Model Support**: Switch between different Ollama models for varied explanation styles
- **Documentation Integration**: Retrieves relevant documentation via SearXNG to supplement explanations
- **Visual Feedback**: Color-coded indicators show whether analogies are curated or AI-generated
- **Offline Fallback**: Includes a static database of curated analogies when services are unavailable

## Architecture

The application consists of three main components:

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

1. **React Frontend**: User interface for software selection and analogy display
2. **FastAPI Backend**: API service handling analogy generation and external service coordination
3. **External Services**:
   - **Ollama**: LLM service for generating dynamic explanations
   - **SearXNG**: Metasearch engine for retrieving relevant documentation

## Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Node.js 16+ and npm (for frontend development)
- Python 3.8+ (for backend development)
- Access to Ollama instance (running at 47.177.58.4:45721)
- Access to SearXNG instance (running at 47.177.58.4:9090)

## Project Structure

```
software-analogy-tool/
├── backend/                      # FastAPI application
│   ├── main.py                   # API endpoints and backend logic
│   ├── ollama_integration.py     # Ollama LLM client
│   ├── searxng_integration.py    # SearXNG search client
│   ├── vector_search.py          # Vector-based similarity search
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Backend container configuration
├── frontend/                     # React application
│   ├── public/                   # Static assets
│   ├── src/
│   │   ├── App.js                # Main React component
│   │   └── index.js              # React entry point
│   ├── package.json              # Frontend dependencies
│   ├── nginx/                    # Nginx configuration for production
│   │   └── default.conf
│   └── Dockerfile                # Frontend container configuration
└── docker-compose.yml            # Multi-container orchestration
```

## Quick Start Guide

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/IMUR/Metafive.git
   cd Metafive
   ```

2. Configure external services in `docker-compose.yml`:
   ```yaml
   environment:
     - OLLAMA_URL=http://47.177.58.4:45721
     - SEARXNG_URL=http://47.177.58.4:9090
   ```

3. Start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API docs: http://localhost:8000/docs

### Manual Development Setup

#### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## API Endpoints

The backend provides the following API endpoints:

- `GET /source-software` - List available source software
- `GET /target-software/{source}` - List compatible target software
- `GET /features/{source}` - List features for a given source software
- `POST /analogy` - Get an analogy for a specific feature
- `GET /ollama-status` - Check Ollama availability and current model
- `GET /ollama-models` - List available Ollama models
- `POST /ollama-model` - Set the active Ollama model
- `GET /searxng-status` - Check SearXNG availability

## Configuration

### Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | URL of the Ollama instance | http://47.177.58.4:45721 |
| `OLLAMA_MODEL` | Default Ollama model to use | llama2 |
| `USE_OLLAMA` | Enable/disable Ollama integration | true |
| `SEARXNG_URL` | URL of the SearXNG instance | http://47.177.58.4:9090 |
| `USE_SEARXNG` | Enable/disable SearXNG integration | true |
| `FALLBACK_TO_STATIC` | Use static analogies when services unavailable | true |

### Available Software Pairs

The MVP includes the following software mappings:

- Adobe After Effects → Ableton Live
- Adobe After Effects → Blender 
- Adobe Photoshop → Ableton Live
- Microsoft Excel → Python

## Extension Points

The application is designed for easy extension:

1. **Adding New Software Analogies**: Edit the `analogy_db` dictionary in `main.py`
2. **Custom LLM Integration**: The Ollama client can be adapted for other LLM APIs
3. **Vector Database Integration**: The vector search module can be connected to proper vector databases
4. **User Feedback Collection**: Add rating system for analogies to improve quality over time

## Hardware Requirements

- Standard web server capable of running Docker containers
- 2GB+ RAM for running all components
- Network access to Ollama and SearXNG instances
- Modern web browser for frontend access

## Voice AI Integration Notes

This application demonstrates key principles for voice AI integration:

- **Modular Architecture**: Components are loosely coupled for independent development
- **Asynchronous Processing**: All API calls run asynchronously to prevent UI blocking
- **Graceful Degradation**: Falls back to static content when external services are unavailable
- **Service Status Monitoring**: Visual indicators show connectivity to external AI services
- **Multiple Model Support**: Interface for switching between different LLM models

## Troubleshooting

### Common Issues

- **Connection Errors**: Ensure the Ollama and SearXNG services are running and accessible
- **Missing Analogies**: The static database is limited; consider adding more entries or enabling vector search
- **Slow Response Times**: LLM generation may take time; consider using smaller models for faster responses

### Logs

- Backend logs: `docker-compose logs backend`
- Frontend logs: `docker-compose logs frontend`

## License

This project is licensed under the MIT License - see the LICENSE file for details.