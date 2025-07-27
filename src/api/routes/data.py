"""
Data Management Routes

Routes for data upload, management, and dataset operations.
"""

import os
import json
import uuid
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..models.data_models import (
    DatasetInfo, DatasetPreview, DatasetUploadResponse, 
    DatasetListResponse, DatasetStatsResponse
)
from ..models.base_models import SuccessResponse, ErrorResponse
from ..utils.file_manager import FileManager, convert_numpy_types
from ..utils.session_manager import SessionManager
from ...agents.data_analyst import DataAnalyst

router = APIRouter()

# Dependencies
def get_file_manager() -> FileManager:
    return FileManager()

def get_session_manager() -> SessionManager:
    return SessionManager()

def get_data_analyst() -> DataAnalyst:
    return DataAnalyst()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Query(None, description="Session ID for tracking"),
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager),
    data_analyst: DataAnalyst = Depends(get_data_analyst)
):
    """
    Upload a dataset file (CSV, JSON, Excel).
    
    Args:
        file: The uploaded file
        session_id: Optional session ID for tracking
        
    Returns:
        Dataset upload response with metadata
    """
    try:
        # Validate file type
        allowed_types = ['.csv', '.json', '.xlsx', '.xls']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type {file_ext}. Supported types: {', '.join(allowed_types)}"
            )
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Create session if not provided
        if not session_id:
            session_id = await session_manager.create_session()
        
        # Read file content
        file_content = await file.read()
        
        # Save file using file manager with correct parameters
        file_id, file_info = await file_manager.save_uploaded_file(
            file_content, 
            file.filename, 
            file.content_type or 'application/octet-stream'
        )
        
        # Load dataset from uploaded file
        dataset_id, dataset_metadata = await file_manager.load_dataset(
            file_id, 
            file.filename.split('.')[0],  # Use filename as dataset name
            f"Uploaded file: {file.filename}"
        )
        
        # Load dataset DataFrame for analysis
        dataset_df = await file_manager.load_dataframe(dataset_id)
        
        # Generate dataset summary using data analyst
        dataset_summary = data_analyst.load_dataset(dataset_df, {
            "filename": file.filename,
            "file_size": file_info.size,
            "upload_time": datetime.now().isoformat(),
            "file_type": file_ext
        })
        
        # Store dataset metadata in session
        await session_manager.store_dataset_metadata(
            session_id, 
            dataset_id, 
            {
                "dataset_id": dataset_id,
                "filename": file.filename,
                "file_size": file_info.size,
                "rows": dataset_summary.get("shape", [0, 0])[0],
                "columns": dataset_summary.get("shape", [0, 0])[1],
                "upload_time": datetime.now().isoformat()
            }
        )
        
        return DatasetUploadResponse(
            status="success",
            message=f"Dataset {file.filename} uploaded successfully",
            dataset_id=dataset_id,
            session_id=session_id,
            filename=file.filename,
            file_size=file_info.size,
            rows=dataset_summary.get("shape", [0, 0])[0],
            columns=dataset_summary.get("shape", [0, 0])[1],
            columns_list=dataset_summary.get("columns", []),
            data_types=dataset_summary.get("dtypes", {}),
            missing_values=dataset_summary.get("missing_values", {}),
            auto_insights=dataset_summary.get("auto_insights", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    session_id: str = Query(None, description="Session ID to filter datasets"),
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    List all available datasets.
    
    Args:
        session_id: Optional session ID to filter datasets
        
    Returns:
        List of available datasets
    """
    try:
        # Initialize file manager to ensure it's set up
        await file_manager.initialize()
        
        # Get all datasets from file manager
        all_datasets = await file_manager.list_datasets()
        dataset_list = []
        
        for dataset_metadata in all_datasets:
            # Extract info from DatasetMetadata object
            dataset_list.append(DatasetInfo(
                dataset_id=dataset_metadata.dataset_id,
                filename=dataset_metadata.file_info.filename if dataset_metadata.file_info else "Unknown",
                file_size=dataset_metadata.file_info.size if dataset_metadata.file_info else 0,
                upload_time=dataset_metadata.created_at.isoformat() if dataset_metadata.created_at else "",
                rows=dataset_metadata.summary.rows if dataset_metadata.summary else 0,
                columns=dataset_metadata.summary.columns if dataset_metadata.summary else 0,
                file_type=Path(dataset_metadata.file_info.filename).suffix if dataset_metadata.file_info else "unknown"
            ))
        
        return DatasetListResponse(
            status="success",
            message=f"Found {len(dataset_list)} datasets",
            datasets=dataset_list,
            total_count=len(dataset_list)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/datasets/{dataset_id}", response_model=SuccessResponse)
async def get_dataset_details(
    dataset_id: str,
    file_manager: FileManager = Depends(get_file_manager),
    data_analyst: DataAnalyst = Depends(get_data_analyst)
):
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_id: The dataset ID
        
    Returns:
        Detailed dataset information
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or could not be loaded")
        
        # Get comprehensive analysis
        dataset_summary = data_analyst.load_dataset(dataset_df)
        
        # Get file metadata
        file_info = await file_manager.get_dataset(dataset_id)
        
        return SuccessResponse(
            status="success",
            message=f"Dataset {dataset_id} details retrieved",
            data={
                "dataset_id": dataset_id,
                "file_info": file_info,
                "analysis": dataset_summary,
                "preview": [convert_numpy_types(record) for record in dataset_df.head(10).to_dict('records')]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: str,
    rows: int = Query(10, ge=1, le=100, description="Number of rows to preview"),
    file_manager: FileManager = Depends(get_file_manager)
):
    """
    Get a preview of the dataset.
    
    Args:
        dataset_id: The dataset ID
        rows: Number of rows to preview (1-100)
        
    Returns:
        Dataset preview
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or could not be loaded")
        
        # Get preview data
        preview_data = dataset_df.head(rows).to_dict('records')
        
        return DatasetPreview(
            status="success",
            message=f"Dataset preview ({rows} rows)",
            dataset_id=dataset_id,
            columns=list(dataset_df.columns),
            data_types={col: str(dtype) for col, dtype in dataset_df.dtypes.items()},
            preview_data=[convert_numpy_types(record) for record in preview_data],
            total_rows=len(dataset_df),
            preview_rows=len(preview_data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")


@router.get("/datasets/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def get_dataset_statistics(
    dataset_id: str,
    file_manager: FileManager = Depends(get_file_manager)
):
    """
    Get statistical summary of the dataset.
    
    Args:
        dataset_id: The dataset ID
        
    Returns:
        Dataset statistics
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or could not be loaded")
        
        # Calculate statistics
        numeric_columns = dataset_df.select_dtypes(include=['number']).columns
        numeric_stats = convert_numpy_types(dataset_df.describe().to_dict()) if len(numeric_columns) > 0 else {}
        
        # Only calculate correlation for numeric columns
        correlation_matrix = {}
        if len(numeric_columns) > 1:
            try:
                numeric_df = dataset_df[numeric_columns]
                correlation_matrix = convert_numpy_types(numeric_df.corr().to_dict())
            except Exception as e:
                print(f"Warning: Could not calculate correlation matrix: {e}")
                correlation_matrix = {}
        
        return DatasetStatsResponse(
            status="success",
            message=f"Statistics for dataset {dataset_id}",
            dataset_id=dataset_id,
            numeric_stats=numeric_stats,
            categorical_stats=convert_numpy_types({
                col: dataset_df[col].value_counts().head(10).to_dict()
                for col in dataset_df.select_dtypes(include=['object']).columns
            }),
            missing_values=convert_numpy_types(dataset_df.isnull().sum().to_dict()),
            data_types={col: str(dtype) for col, dtype in dataset_df.dtypes.items()},
            correlation_matrix=correlation_matrix,
            total_size_mb=convert_numpy_types(round(dataset_df.memory_usage(deep=True).sum() / 1024**2, 2))
        )
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")


@router.delete("/datasets/{dataset_id}", response_model=SuccessResponse)
async def delete_dataset(
    dataset_id: str,
    session_id: str = Query(None, description="Session ID to remove dataset from"),
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Delete a dataset.
    
    Args:
        dataset_id: The dataset ID
        session_id: Optional session ID to remove from
        
    Returns:
        Success response
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Delete from file manager
        await file_manager.delete_dataset(dataset_id)
        
        # Remove from session if specified
        if session_id:
            await session_manager.remove_dataset_from_session(session_id, dataset_id)
        
        return SuccessResponse(
            status="success",
            message=f"Dataset {dataset_id} deleted successfully",
            data={"dataset_id": dataset_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}") 