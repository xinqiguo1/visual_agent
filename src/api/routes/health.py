"""
Health Check Routes

Health monitoring and status endpoints.
"""

import os
import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..models.base_models import StatusResponse, SuccessResponse
from ..utils.session_manager import SessionManager
from ..utils.file_manager import FileManager

router = APIRouter()

# Store application start time
START_TIME = time.time()


@router.get("/", response_model=StatusResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    current_time = time.time()
    uptime = current_time - START_TIME
    
    return StatusResponse(
        status="success",
        message="Visual Analytics Agent API is healthy",
        service_name="Visual Analytics Agent API",
        version="2.0.0",
        uptime=uptime,
        health_checks={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_formatted": _format_uptime(uptime)
        }
    )


@router.get("/detailed", response_model=SuccessResponse)
async def detailed_health_check(
    session_manager: SessionManager = Depends(lambda: None),
    file_manager: FileManager = Depends(lambda: None)
):
    """
    Detailed health check with system information.
    
    Returns:
        Detailed health status and system information
    """
    current_time = time.time()
    uptime = current_time - START_TIME
    
    # System information
    try:
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
    except:
        system_info = {"error": "Unable to fetch system information"}
    
    # Service health checks
    health_checks = {
        "api": {"status": "healthy", "message": "API is running"},
        "system": {
            "status": "healthy" if system_info.get("memory_percent", 0) < 90 else "warning",
            "info": system_info
        }
    }
    
    # Check session manager
    if session_manager:
        try:
            session_stats = await session_manager.get_global_stats()
            health_checks["session_manager"] = {
                "status": "healthy",
                "stats": session_stats
            }
        except Exception as e:
            health_checks["session_manager"] = {
                "status": "error",
                "error": str(e)
            }
    
    # Check file manager
    if file_manager:
        try:
            storage_stats = await file_manager.get_storage_stats()
            health_checks["file_manager"] = {
                "status": "healthy",
                "stats": storage_stats
            }
        except Exception as e:
            health_checks["file_manager"] = {
                "status": "error", 
                "error": str(e)
            }
    
    # Check environment
    env_checks = {
        "openai_api_key": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured",
        "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
        "python_version": f"{psutil.version_info}",
        "working_directory": os.getcwd()
    }
    
    return SuccessResponse(
        status="success",
        message="Detailed health check completed",
        data={
            "service": {
                "name": "Visual Analytics Agent API",
                "version": "2.0.0",
                "uptime_seconds": uptime,
                "uptime_formatted": _format_uptime(uptime),
                "start_time": datetime.fromtimestamp(START_TIME).isoformat()
            },
            "health_checks": health_checks,
            "environment": env_checks,
            "timestamp": datetime.now().isoformat()
        }
    )


@router.get("/readiness")
async def readiness_check():
    """
    Kubernetes-style readiness check.
    
    Returns:
        Ready status for load balancers
    """
    # Check if essential services are ready
    ready = True
    checks = {}
    
    # Check if directories exist
    required_dirs = ["uploads", "outputs", "data", "cache", "sessions"]
    for directory in required_dirs:
        if os.path.exists(directory):
            checks[f"dir_{directory}"] = "ready"
        else:
            checks[f"dir_{directory}"] = "not_ready"
            ready = False
    
    # Check memory usage
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent < 95:
            checks["memory"] = "ready"
        else:
            checks["memory"] = "not_ready"
            ready = False
    except:
        checks["memory"] = "unknown"
    
    status_code = 200 if ready else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": ready,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    )


@router.get("/liveness")
async def liveness_check():
    """
    Kubernetes-style liveness check.
    
    Returns:
        Live status for container orchestration
    """
    # Simple liveness check - if this endpoint responds, the app is alive
    return JSONResponse(
        status_code=200,
        content={
            "alive": True,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - START_TIME
        }
    )


@router.get("/metrics")
async def metrics_endpoint(
    session_manager: SessionManager = Depends(lambda: None),
    file_manager: FileManager = Depends(lambda: None)
):
    """
    Basic metrics endpoint.
    
    Returns:
        Application metrics
    """
    metrics = {
        "uptime_seconds": time.time() - START_TIME,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add session metrics
    if session_manager:
        try:
            session_stats = await session_manager.get_global_stats()
            metrics["sessions"] = session_stats
        except:
            metrics["sessions"] = {"error": "Unable to fetch session metrics"}
    
    # Add storage metrics
    if file_manager:
        try:
            storage_stats = await file_manager.get_storage_stats()
            metrics["storage"] = storage_stats
        except:
            metrics["storage"] = {"error": "Unable to fetch storage metrics"}
    
    # Add system metrics
    try:
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024**2,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except:
        metrics["system"] = {"error": "Unable to fetch system metrics"}
    
    return JSONResponse(content=metrics)


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days" 