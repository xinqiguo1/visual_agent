"""
Data Models

Pydantic models for data upload, management, and dataset operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

from .base_models import BaseResponse, FileInfo


class DataType(str, Enum):
    """Data type enumeration."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORY = "category"
    UNKNOWN = "unknown"


class ColumnInfo(BaseModel):
    """Column information model."""
    name: str = Field(..., description="Column name")
    data_type: DataType = Field(..., description="Column data type")
    non_null_count: int = Field(..., ge=0, description="Number of non-null values")
    null_count: int = Field(..., ge=0, description="Number of null values")
    unique_count: Optional[int] = Field(None, ge=0, description="Number of unique values")
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    
    # Statistics for numeric columns
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value")
    mean_value: Optional[float] = Field(None, description="Mean value")
    median_value: Optional[float] = Field(None, description="Median value")
    std_deviation: Optional[float] = Field(None, description="Standard deviation")
    
    # Info for categorical columns
    most_frequent_value: Optional[Any] = Field(None, description="Most frequent value")
    frequency_count: Optional[int] = Field(None, ge=0, description="Frequency of most common value")
    categories: Optional[List[str]] = Field(None, description="List of categories (if categorical)")


class DatasetSummary(BaseModel):
    """Dataset summary model."""
    rows: int = Field(..., ge=0, description="Number of rows")
    columns: int = Field(..., ge=0, description="Number of columns")
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
    column_info: List[ColumnInfo] = Field(default_factory=list, description="Column information")
    
    # Data quality metrics
    total_missing_values: int = Field(..., ge=0, description="Total missing values")
    missing_percentage: float = Field(..., ge=0, le=100, description="Percentage of missing values")
    duplicate_rows: int = Field(..., ge=0, description="Number of duplicate rows")
    
    # Column type distribution
    numeric_columns: List[str] = Field(default_factory=list, description="Numeric column names")
    categorical_columns: List[str] = Field(default_factory=list, description="Categorical column names")
    datetime_columns: List[str] = Field(default_factory=list, description="DateTime column names")
    
    # Automatic insights
    data_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Data quality score")
    recommendations: List[str] = Field(default_factory=list, description="Data quality recommendations")


class DataInfo(BaseModel):
    """Data information model."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    file_info: FileInfo = Field(..., description="Original file information")
    summary: DatasetSummary = Field(..., description="Dataset summary")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataUploadResponse(BaseResponse):
    """Data upload response model."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    file_info: FileInfo = Field(..., description="Uploaded file information")
    summary: DatasetSummary = Field(..., description="Dataset summary")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class DataPreviewRequest(BaseModel):
    """Data preview request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    start_row: int = Field(0, ge=0, description="Starting row index")
    end_row: Optional[int] = Field(None, ge=0, description="Ending row index")
    columns: Optional[List[str]] = Field(None, description="Specific columns to preview")
    
    @validator('end_row')
    def validate_end_row(cls, v, values):
        if v is not None and 'start_row' in values and v <= values['start_row']:
            raise ValueError('end_row must be greater than start_row')
        return v


class DataPreviewResponse(BaseResponse):
    """Data preview response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    data: List[Dict[str, Any]] = Field(..., description="Preview data")
    start_row: int = Field(..., ge=0, description="Starting row index")
    end_row: int = Field(..., ge=0, description="Ending row index")
    total_rows: int = Field(..., ge=0, description="Total number of rows")
    columns: List[str] = Field(..., description="Column names")


class DataFilterRequest(BaseModel):
    """Data filter request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    filters: Dict[str, Any] = Field(..., description="Filter conditions")
    limit: Optional[int] = Field(None, ge=1, le=10000, description="Maximum rows to return")
    offset: Optional[int] = Field(None, ge=0, description="Row offset")


class DataFilterResponse(BaseResponse):
    """Data filter response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    filtered_data: List[Dict[str, Any]] = Field(..., description="Filtered data")
    total_matches: int = Field(..., ge=0, description="Total number of matching rows")
    applied_filters: Dict[str, Any] = Field(..., description="Applied filter conditions")


class DataTransformRequest(BaseModel):
    """Data transformation request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    transformation_type: str = Field(..., description="Type of transformation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transformation parameters")
    create_new_dataset: bool = Field(False, description="Whether to create a new dataset")


class DataTransformResponse(BaseResponse):
    """Data transformation response model."""
    original_dataset_id: str = Field(..., description="Original dataset identifier")
    new_dataset_id: Optional[str] = Field(None, description="New dataset identifier (if created)")
    transformation_applied: str = Field(..., description="Transformation that was applied")
    rows_affected: int = Field(..., ge=0, description="Number of rows affected")
    summary: Optional[DatasetSummary] = Field(None, description="New dataset summary")


class DataExportRequest(BaseModel):
    """Data export request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    format: str = Field(..., description="Export format (csv, json, excel)")
    filename: Optional[str] = Field(None, description="Custom filename")
    columns: Optional[List[str]] = Field(None, description="Specific columns to export")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter conditions")
    
    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['csv', 'json', 'excel', 'xlsx']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Format must be one of: {allowed_formats}')
        return v.lower()


class DataExportResponse(BaseResponse):
    """Data export response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    export_format: str = Field(..., description="Export format")
    file_path: str = Field(..., description="Path to exported file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    rows_exported: int = Field(..., ge=0, description="Number of rows exported")
    columns_exported: List[str] = Field(..., description="Column names exported")


class DataDeleteRequest(BaseModel):
    """Data deletion request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    confirm: bool = Field(False, description="Confirmation flag")
    
    @validator('confirm')
    def validate_confirm(cls, v):
        if not v:
            raise ValueError('Confirmation is required for deletion')
        return v


class DataDeleteResponse(BaseResponse):
    """Data deletion response model."""
    dataset_id: str = Field(..., description="Deleted dataset identifier")
    files_deleted: List[str] = Field(default_factory=list, description="List of deleted files")
    cleanup_performed: bool = Field(..., description="Whether cleanup was performed")


class DataListResponse(BaseResponse):
    """Data list response model."""
    datasets: List[DataInfo] = Field(default_factory=list, description="List of available datasets")
    total_count: int = Field(..., ge=0, description="Total number of datasets")


# Additional models for API routes compatibility
class DatasetInfo(BaseModel):
    """Dataset information model (simplified for API)."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    upload_time: str = Field(..., description="Upload timestamp")
    rows: int = Field(..., ge=0, description="Number of rows")
    columns: int = Field(..., ge=0, description="Number of columns")
    file_type: str = Field(..., description="File type")


class DatasetListResponse(BaseResponse):
    """Dataset list response model."""
    datasets: List[DatasetInfo] = Field(default_factory=list, description="List of datasets")
    total_count: int = Field(..., ge=0, description="Total number of datasets")


class DatasetPreview(BaseResponse):
    """Dataset preview response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    columns: List[str] = Field(..., description="Column names")
    data_types: Dict[str, str] = Field(..., description="Data types")
    preview_data: List[Dict[str, Any]] = Field(..., description="Preview data")
    total_rows: int = Field(..., ge=0, description="Total number of rows")
    preview_rows: int = Field(..., ge=0, description="Number of preview rows")


class DatasetUploadResponse(BaseResponse):
    """Dataset upload response model."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    rows: int = Field(..., ge=0, description="Number of rows")
    columns: int = Field(..., ge=0, description="Number of columns")
    columns_list: List[str] = Field(..., description="Column names")
    data_types: Dict[str, str] = Field(..., description="Data types")
    missing_values: Dict[str, int] = Field(..., description="Missing values per column")
    auto_insights: List[str] = Field(default_factory=list, description="Automatic insights")


class DatasetStatsResponse(BaseResponse):
    """Dataset statistics response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    numeric_stats: Dict[str, Any] = Field(..., description="Numeric column statistics")
    categorical_stats: Dict[str, Any] = Field(..., description="Categorical column statistics")
    missing_values: Dict[str, int] = Field(..., description="Missing values per column")
    data_types: Dict[str, str] = Field(..., description="Data types")
    correlation_matrix: Dict[str, Any] = Field(..., description="Correlation matrix")
    total_size_mb: float = Field(..., ge=0, description="Total size of all datasets in MB") 