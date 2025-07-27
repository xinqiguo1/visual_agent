"""
Base Pydantic Models

Common response structures and base classes for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: StatusEnum = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseResponse):
    """Success response model."""
    status: StatusEnum = Field(StatusEnum.SUCCESS, description="Success status")
    data: Optional[Any] = Field(None, description="Response data")
    
    
class ErrorResponse(BaseResponse):
    """Error response model."""
    status: StatusEnum = Field(StatusEnum.ERROR, description="Error status")
    error_code: Optional[str] = Field(None, description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    

class StatusResponse(BaseResponse):
    """Status response model."""
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    health_checks: Optional[Dict[str, Any]] = Field(None, description="Health check results")


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    data: List[Any] = Field(default_factory=list, description="Paginated data")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")


class FileInfo(BaseModel):
    """File information model."""
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    file_id: str = Field(..., description="Unique file identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskStatus(BaseModel):
    """Task status model for long-running operations."""
    task_id: str = Field(..., description="Unique task identifier")
    status: StatusEnum = Field(..., description="Task status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Task progress percentage")
    result: Optional[Any] = Field(None, description="Task result (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    started_at: datetime = Field(default_factory=datetime.now, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(..., description="Field name that failed validation")
    message: str = Field(..., description="Validation error message")
    rejected_value: Optional[Any] = Field(None, description="Value that was rejected")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response model."""
    validation_errors: List[ValidationError] = Field(default_factory=list, description="List of validation errors") 