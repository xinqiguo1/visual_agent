"""
Error Handler Middleware

Custom error handling middleware for the FastAPI application.
"""

import logging
import traceback
from typing import Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Custom error handling middleware that catches and formats errors.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_map = {
            ValueError: {"status_code": 400, "error_type": "validation_error"},
            FileNotFoundError: {"status_code": 404, "error_type": "file_not_found"},
            PermissionError: {"status_code": 403, "error_type": "permission_denied"},
            ConnectionError: {"status_code": 503, "error_type": "service_unavailable"},
            TimeoutError: {"status_code": 504, "error_type": "timeout"},
            MemoryError: {"status_code": 507, "error_type": "insufficient_storage"},
            NotImplementedError: {"status_code": 501, "error_type": "not_implemented"},
        }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and handle any errors.
        
        Args:
            request: HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful requests
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.4f}s"
            )
            
            return response
            
        except Exception as exc:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"{request.method} {request.url.path} - "
                f"Error: {str(exc)} - "
                f"Time: {process_time:.4f}s"
            )
            
            # Return formatted error response
            return await self._handle_error(request, exc)
    
    async def _handle_error(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle and format errors.
        
        Args:
            request: HTTP request
            exc: Exception that occurred
            
        Returns:
            JSON error response
        """
        # Get error details
        error_details = self._get_error_details(exc)
        
        # Create error response
        error_response = {
            "error": {
                "type": error_details["error_type"],
                "message": str(exc),
                "timestamp": time.time(),
                "path": str(request.url.path),
                "method": request.method
            }
        }
        
        # Add debug info in development
        if logger.isEnabledFor(logging.DEBUG):
            error_response["error"]["debug"] = {
                "traceback": traceback.format_exc(),
                "exception_type": exc.__class__.__name__,
                "args": exc.args if hasattr(exc, 'args') else None
            }
        
        # Add request info for certain errors
        if error_details["status_code"] >= 500:
            error_response["error"]["request_info"] = {
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "client": str(request.client) if request.client else None
            }
        
        return JSONResponse(
            status_code=error_details["status_code"],
            content=error_response
        )
    
    def _get_error_details(self, exc: Exception) -> Dict[str, Any]:
        """
        Get error details based on exception type.
        
        Args:
            exc: Exception
            
        Returns:
            Dictionary with error details
        """
        # Check for specific error types
        for error_type, details in self.error_map.items():
            if isinstance(exc, error_type):
                return details
        
        # Check for HTTP exceptions
        if hasattr(exc, 'status_code'):
            return {
                "status_code": exc.status_code,
                "error_type": "http_error"
            }
        
        # Check for specific error messages
        error_message = str(exc).lower()
        
        if "not found" in error_message:
            return {"status_code": 404, "error_type": "not_found"}
        elif "unauthorized" in error_message or "authentication" in error_message:
            return {"status_code": 401, "error_type": "unauthorized"}
        elif "forbidden" in error_message or "permission" in error_message:
            return {"status_code": 403, "error_type": "forbidden"}
        elif "timeout" in error_message:
            return {"status_code": 504, "error_type": "timeout"}
        elif "too large" in error_message or "size" in error_message:
            return {"status_code": 413, "error_type": "payload_too_large"}
        elif "rate limit" in error_message:
            return {"status_code": 429, "error_type": "rate_limit_exceeded"}
        elif "invalid" in error_message or "malformed" in error_message:
            return {"status_code": 400, "error_type": "bad_request"}
        elif "database" in error_message or "connection" in error_message:
            return {"status_code": 503, "error_type": "service_unavailable"}
        
        # Default to internal server error
        return {"status_code": 500, "error_type": "internal_server_error"}


class APIError(Exception):
    """Base API error class."""
    
    def __init__(self, message: str, status_code: int = 500, error_type: str = "api_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)


class ValidationError(APIError):
    """Validation error."""
    
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, 400, "validation_error")


class DataError(APIError):
    """Data processing error."""
    
    def __init__(self, message: str):
        super().__init__(message, 422, "data_error")


class AgentError(APIError):
    """Agent execution error."""
    
    def __init__(self, message: str, agent_type: str = None):
        self.agent_type = agent_type
        super().__init__(message, 500, "agent_error")


class FileError(APIError):
    """File processing error."""
    
    def __init__(self, message: str, filename: str = None):
        self.filename = filename
        super().__init__(message, 400, "file_error")


class SessionError(APIError):
    """Session management error."""
    
    def __init__(self, message: str, session_id: str = None):
        self.session_id = session_id
        super().__init__(message, 400, "session_error")


class ConfigurationError(APIError):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message, 500, "configuration_error")


class ResourceError(APIError):
    """Resource limitation error."""
    
    def __init__(self, message: str, resource_type: str = None):
        self.resource_type = resource_type
        super().__init__(message, 429, "resource_error")


class ExternalServiceError(APIError):
    """External service error."""
    
    def __init__(self, message: str, service_name: str = None):
        self.service_name = service_name
        super().__init__(message, 503, "external_service_error")


# Error handler functions
def handle_validation_error(exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "field": exc.field,
                "timestamp": time.time()
            }
        }
    )


def handle_data_error(exc: DataError) -> JSONResponse:
    """Handle data processing errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "suggestions": [
                    "Check your data format and structure",
                    "Verify all required columns are present",
                    "Ensure data types are correct"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_agent_error(exc: AgentError) -> JSONResponse:
    """Handle agent execution errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "agent_type": exc.agent_type,
                "suggestions": [
                    "Check your query format",
                    "Verify your data is loaded correctly",
                    "Try a simpler query first"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_file_error(exc: FileError) -> JSONResponse:
    """Handle file processing errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "filename": exc.filename,
                "suggestions": [
                    "Check file format and size",
                    "Ensure file is not corrupted",
                    "Verify file permissions"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_session_error(exc: SessionError) -> JSONResponse:
    """Handle session management errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "session_id": exc.session_id,
                "suggestions": [
                    "Create a new session",
                    "Check session ID format",
                    "Verify session hasn't expired"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_configuration_error(exc: ConfigurationError) -> JSONResponse:
    """Handle configuration errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "config_key": exc.config_key,
                "suggestions": [
                    "Check environment variables",
                    "Verify configuration file",
                    "Contact administrator"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_resource_error(exc: ResourceError) -> JSONResponse:
    """Handle resource limitation errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "resource_type": exc.resource_type,
                "suggestions": [
                    "Try reducing data size",
                    "Wait before retrying",
                    "Contact support for higher limits"
                ],
                "timestamp": time.time()
            }
        }
    )


def handle_external_service_error(exc: ExternalServiceError) -> JSONResponse:
    """Handle external service errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "service_name": exc.service_name,
                "suggestions": [
                    "Check service availability",
                    "Verify API credentials",
                    "Try again later"
                ],
                "timestamp": time.time()
            }
        }
    ) 