"""
FastAPI Server Startup Script

Convenient script to start the Visual Analytics Agent API server.
"""

import os
import sys
import uvicorn
import argparse

def main():
    """Start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Start Visual Analytics Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print("🚀 Starting Visual Analytics Agent API Server")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 50)
    
    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key configured - Enhanced AI features enabled")
    else:
        print("⚠️  OpenAI API key not found - Fallback mode only")
        print("   Set OPENAI_API_KEY environment variable for full features")
    
    print("\n📚 API Documentation:")
    print(f"   • Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   • ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"   • Health Check: http://{args.host}:{args.port}/api/health")
    print(f"   • Main Page: http://{args.host}:{args.port}/")
    
    print("\n🔧 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 