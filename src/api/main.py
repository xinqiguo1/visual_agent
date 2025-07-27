"""
Visual Analytics Agent - FastAPI Backend

Main application file that serves the API endpoints and integrates with the data analysis agents.
"""

import os
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
from contextlib import asynccontextmanager
import logging
from typing import Optional

# Import route modules
from .routes import data, analysis, visualization, health
from .middleware.error_handler import ErrorHandlerMiddleware
from .utils.session_manager import SessionManager
from .utils.file_manager import FileManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
session_manager = SessionManager()
file_manager = FileManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting Visual Analytics Agent API...")
    
    # Initialize services
    await session_manager.initialize()
    await file_manager.initialize()
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("✅ API initialization complete")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down API...")
    await session_manager.cleanup()
    await file_manager.cleanup()
    logger.info("✅ API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Visual Analytics Agent API",
    description="""
    🚀 **Visual Analytics Agent API**
    
    A powerful data analysis and visualization API powered by GPT-4o and LangChain.
    
    ## Features
    - 📊 **Data Upload & Processing**: Support for CSV, Excel, JSON files
    - 🤖 **AI-Powered Analysis**: Natural language queries with GPT-4o
    - 📈 **Interactive Visualizations**: Plotly-powered charts and dashboards
    - 🔍 **Smart Insights**: Automated pattern detection and recommendations
    - 💻 **Code Generation**: Python code for custom analysis
    - 🌐 **Web-Ready**: RESTful API with WebSocket support
    
    ## Getting Started
    1. Upload your data using `/api/data/upload`
    2. Ask questions using `/api/analysis/query`
    3. Generate visualizations using `/api/visualization/create`
    4. Download results and dashboards
    
    ## Authentication
    Set your OpenAI API key in the environment variable `OPENAI_API_KEY` for enhanced features.
    """,
    version="2.0.0",
    contact={
        "name": "Visual Analytics Agent",
        "url": "https://github.com/your-repo/visual-agent"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom error handler middleware
app.add_middleware(ErrorHandlerMiddleware)

# Mount static files (for serving generated visualizations)
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Include API routes
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(data.router, prefix="/api/data", tags=["Data Management"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["Visualization"])

# Global exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with detailed error information."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": str(pd.Timestamp.now()),
                "path": str(request.url)
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)} - {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "detail": "Internal server error",
                "message": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
                "timestamp": str(pd.Timestamp.now()),
                "path": str(request.url)
            }
        }
    )


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visual Analytics Agent API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #333; text-align: center; }
            .feature { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .links { text-align: center; margin-top: 30px; }
            .links a { margin: 0 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .links a:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Visual Analytics Agent API</h1>
            <p>Welcome to the Visual Analytics Agent API! This powerful backend provides AI-powered data analysis and visualization capabilities.</p>
            
            <div class="feature">
                <h3>📊 Data Processing</h3>
                <p>Upload CSV, Excel, or JSON files and get instant data profiling and quality assessment.</p>
            </div>
            
            <div class="feature">
                <h3>🤖 AI Analysis</h3>
                <p>Ask questions in natural language and get intelligent analysis powered by GPT-4o and LangChain.</p>
            </div>
            
            <div class="feature">
                <h3>📈 Interactive Visualizations</h3>
                <p>Generate beautiful, interactive charts and dashboards using Plotly that work on any device.</p>
            </div>
            
            <div class="feature">
                <h3>💻 Code Generation</h3>
                <p>Get Python code for custom analysis and reproducible data science workflows.</p>
            </div>
            
            <div class="links">
                <a href="/docs">📚 API Documentation</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/api/health">🏥 Health Check</a>
            </div>
        </div>
    </body>
    </html>
    """


# Dependency injection helpers
def get_session_manager():
    """Get the session manager instance."""
    return session_manager


def get_file_manager():
    """Get the file manager instance."""
    return file_manager


# Development server
if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        log_level="info"
    ) 