"""
File Manager

Handles file uploads, storage, and dataset management.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import pandas as pd
import json
import logging
from dataclasses import dataclass, field
import hashlib
import mimetypes
import numpy as np

from ..models.base_models import FileInfo
from ..models.data_models import DataType, ColumnInfo, DatasetSummary

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'name') and hasattr(obj, 'type'):  # pandas dtype objects
        return str(obj)
    elif str(type(obj)).startswith("<class 'pandas"):  # any pandas objects
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


@dataclass
class DatasetMetadata:
    """Dataset metadata structure."""
    dataset_id: str
    name: str
    description: Optional[str] = None
    file_info: Optional[FileInfo] = None
    summary: Optional[DatasetSummary] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None
    cached_data_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Handle FileInfo serialization with datetime conversion
        file_info_dict = None
        if self.file_info:
            file_info_dict = self.file_info.dict()
            # Convert datetime to string if present
            if 'upload_time' in file_info_dict and hasattr(file_info_dict['upload_time'], 'isoformat'):
                file_info_dict['upload_time'] = file_info_dict['upload_time'].isoformat()
        
        # Handle Summary serialization 
        summary_dict = None
        if self.summary:
            summary_dict = convert_numpy_types(self.summary.dict())
        
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'file_info': file_info_dict,
            'summary': summary_dict,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'file_path': self.file_path,
            'cached_data_path': self.cached_data_path
        }


class FileManager:
    """
    Manages file uploads, storage, and dataset operations.
    """
    
    def __init__(self, 
                 upload_dir: str = "uploads",
                 data_dir: str = "data", 
                 cache_dir: str = "cache",
                 max_file_size: int = 100 * 1024 * 1024):  # 100MB
        """
        Initialize file manager.
        
        Args:
            upload_dir: Directory for uploaded files
            data_dir: Directory for processed data
            cache_dir: Directory for cached data
            max_file_size: Maximum file size in bytes
        """
        self.upload_dir = Path(upload_dir)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.max_file_size = max_file_size
        
        # Dataset metadata storage
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.metadata_file = self.data_dir / "datasets.json"
        
        # Supported file types
        self.supported_types = {
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.json': 'application/json',
            '.parquet': 'application/octet-stream',
            '.tsv': 'text/tab-separated-values'
        }
        
        logger.info(f"FileManager initialized with max_file_size={max_file_size/1024/1024:.1f}MB")
    
    async def initialize(self):
        """Initialize the file manager."""
        logger.info("Starting file manager...")
        
        # Create directories
        self.upload_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load existing datasets
        await self._load_datasets()
        
        logger.info("File manager started successfully")
    
    async def cleanup(self):
        """Cleanup file manager."""
        logger.info("Stopping file manager...")
        
        # Save datasets metadata
        await self._save_datasets()
        
        logger.info("File manager stopped")
    
    async def save_uploaded_file(self, file_content: bytes, filename: str, content_type: str) -> Tuple[str, FileInfo]:
        """
        Save uploaded file.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            content_type: MIME content type
            
        Returns:
            Tuple of (file_id, FileInfo)
        """
        # Validate file size
        if len(file_content) > self.max_file_size:
            raise ValueError(f"File too large. Maximum size: {self.max_file_size/1024/1024:.1f}MB")
        
        # Validate file type
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.supported_types:
            raise ValueError(f"Unsupported file type. Supported: {list(self.supported_types.keys())}")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create file info
        file_info = FileInfo(
            filename=filename,
            size=len(file_content),
            content_type=content_type,
            file_id=file_id
        )
        
        # Save file
        file_path = self.upload_dir / f"{file_id}_{filename}"
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded file: {filename} ({len(file_content)} bytes)")
        return file_id, file_info
    
    async def load_dataset(self, file_id: str, dataset_name: str, description: str = None) -> Tuple[str, DatasetMetadata]:
        """
        Load dataset from uploaded file.
        
        Args:
            file_id: File identifier
            dataset_name: Dataset name
            description: Dataset description
            
        Returns:
            Tuple of (dataset_id, DatasetMetadata)
        """
        # Find uploaded file
        uploaded_files = list(self.upload_dir.glob(f"{file_id}_*"))
        if not uploaded_files:
            raise ValueError(f"File not found: {file_id}")
        
        file_path = uploaded_files[0]
        
        # Load data
        try:
            df = await self._load_dataframe(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
        
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Create file info
        file_info = FileInfo(
            filename=file_path.name,
            size=file_path.stat().st_size,
            content_type=mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream',
            file_id=file_id
        )
        
        # Generate summary
        summary = await self._generate_dataset_summary(df)
        
        # Cache processed data
        cached_path = self.cache_dir / f"{dataset_id}.parquet"
        df.to_parquet(cached_path)
        
        # Create dataset metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=dataset_name,
            description=description,
            file_info=file_info,
            summary=summary,
            file_path=str(file_path),
            cached_data_path=str(cached_path)
        )
        
        # Store dataset
        self.datasets[dataset_id] = metadata
        await self._save_datasets()
        
        logger.info(f"Loaded dataset: {dataset_name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return dataset_id, metadata
    
    async def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset metadata or None if not found
        """
        return self.datasets.get(dataset_id)
    
    async def load_dataframe(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            DataFrame or None if not found
        """
        metadata = self.datasets.get(dataset_id)
        if not metadata:
            return None
        
        try:
            # Load from cache first
            if metadata.cached_data_path and Path(metadata.cached_data_path).exists():
                return pd.read_parquet(metadata.cached_data_path)
            
            # Load from original file
            if metadata.file_path and Path(metadata.file_path).exists():
                df = await self._load_dataframe(Path(metadata.file_path))
                
                # Update cache
                if metadata.cached_data_path:
                    df.to_parquet(metadata.cached_data_path)
                
                return df
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
        
        return None
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete dataset and associated files.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            True if deleted, False if not found
        """
        metadata = self.datasets.get(dataset_id)
        if not metadata:
            return False
        
        # Delete files
        files_deleted = []
        
        # Delete cached data
        if metadata.cached_data_path and Path(metadata.cached_data_path).exists():
            Path(metadata.cached_data_path).unlink()
            files_deleted.append(metadata.cached_data_path)
        
        # Delete original file
        if metadata.file_path and Path(metadata.file_path).exists():
            Path(metadata.file_path).unlink()
            files_deleted.append(metadata.file_path)
        
        # Remove from metadata
        del self.datasets[dataset_id]
        await self._save_datasets()
        
        logger.info(f"Deleted dataset {dataset_id}, files: {files_deleted}")
        return True
    
    async def list_datasets(self) -> List[DatasetMetadata]:
        """
        List all datasets.
        
        Returns:
            List of dataset metadata
        """
        return list(self.datasets.values())
    
    async def get_dataset_preview(self, dataset_id: str, start_row: int = 0, end_row: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get dataset preview.
        
        Args:
            dataset_id: Dataset identifier
            start_row: Starting row index
            end_row: Ending row index
            
        Returns:
            List of records or None if not found
        """
        df = await self.load_dataframe(dataset_id)
        if df is None:
            return None
        
        # Get preview slice
        preview_df = df.iloc[start_row:end_row]
        records = preview_df.to_dict('records')
        # Convert numpy types in the records
        return [convert_numpy_types(record) for record in records]
    
    async def export_dataset(self, dataset_id: str, format: str = 'csv', filename: str = None) -> Optional[str]:
        """
        Export dataset to specified format.
        
        Args:
            dataset_id: Dataset identifier
            format: Export format (csv, json, excel, parquet)
            filename: Custom filename
            
        Returns:
            Export file path or None if error
        """
        df = await self.load_dataframe(dataset_id)
        if df is None:
            return None
        
        metadata = self.datasets.get(dataset_id)
        if not metadata:
            return None
        
        # Generate filename
        if not filename:
            filename = f"{metadata.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Export file
        export_path = self.data_dir / f"{filename}.{format}"
        
        try:
            if format == 'csv':
                df.to_csv(export_path, index=False)
            elif format == 'json':
                df.to_json(export_path, orient='records', indent=2)
            elif format == 'excel':
                df.to_excel(export_path, index=False)
            elif format == 'parquet':
                df.to_parquet(export_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported dataset {dataset_id} to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting dataset {dataset_id}: {e}")
            return None
    
    async def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load DataFrame from file."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Unable to read CSV file with any supported encoding")
        
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        
        elif file_ext == '.json':
            # Enhanced JSON loading to handle complex structures
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Try direct pandas conversion first
                try:
                    return pd.read_json(file_path)
                except (ValueError, TypeError):
                    # Handle complex nested JSON structures
                    if isinstance(json_data, dict):
                        # Look for array-like data in the JSON
                        for key, value in json_data.items():
                            if isinstance(value, list) and len(value) > 0:
                                # Try to normalize the nested data
                                try:
                                    df = pd.json_normalize(value)
                                    # Convert any remaining list columns to strings to avoid unhashable type errors
                                    for col in df.columns:
                                        if df[col].apply(lambda x: isinstance(x, list)).any():
                                            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                                    return df
                                except:
                                    pass
                        
                        # If no arrays found, try to convert the dict directly
                        try:
                            df = pd.DataFrame([json_data])
                            # Convert any list columns to strings
                            for col in df.columns:
                                if df[col].apply(lambda x: isinstance(x, list)).any():
                                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                            return df
                        except:
                            # Flatten the dict and create a single-row DataFrame
                            flattened = pd.json_normalize(json_data)
                            # Convert any list columns to strings
                            for col in flattened.columns:
                                if flattened[col].apply(lambda x: isinstance(x, list)).any():
                                    flattened[col] = flattened[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                            return flattened
                    
                    elif isinstance(json_data, list):
                        # Direct list conversion
                        df = pd.json_normalize(json_data)
                        # Convert any list columns to strings
                        for col in df.columns:
                            if df[col].apply(lambda x: isinstance(x, list)).any():
                                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                        return df
                    
                    else:
                        # Fallback: create DataFrame from the raw data
                        return pd.DataFrame([{'data': str(json_data)}])
                    
            except Exception as e:
                raise ValueError(f"Unable to read JSON file: {str(e)}")
        
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        
        elif file_ext == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    async def _generate_dataset_summary(self, df: pd.DataFrame) -> DatasetSummary:
        """Generate dataset summary."""
        # Basic info
        rows, columns = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        # Column info
        column_info = []
        for col in df.columns:
            col_data = df[col]
            
            # Determine data type
            if pd.api.types.is_integer_dtype(col_data):
                data_type = DataType.INTEGER
            elif pd.api.types.is_float_dtype(col_data):
                data_type = DataType.FLOAT
            elif pd.api.types.is_bool_dtype(col_data):
                data_type = DataType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = DataType.DATETIME
            elif pd.api.types.is_categorical_dtype(col_data):
                data_type = DataType.CATEGORY
            else:
                data_type = DataType.STRING
            
            # Basic stats
            non_null_count = col_data.notna().sum()
            null_count = col_data.isna().sum()
            unique_count = col_data.nunique()
            
            col_info = ColumnInfo(
                name=col,
                data_type=data_type,
                non_null_count=int(non_null_count),
                null_count=int(null_count),
                unique_count=int(unique_count)
            )
            
            # Add numeric stats
            if data_type in [DataType.INTEGER, DataType.FLOAT]:
                try:
                    col_info.min_value = convert_numpy_types(col_data.min())
                    col_info.max_value = convert_numpy_types(col_data.max())
                    col_info.mean_value = convert_numpy_types(col_data.mean())
                    col_info.median_value = convert_numpy_types(col_data.median())
                    col_info.std_deviation = convert_numpy_types(col_data.std())
                except:
                    pass
            
            # Add categorical stats
            elif data_type in [DataType.STRING, DataType.CATEGORY]:
                try:
                    value_counts = col_data.value_counts()
                    if len(value_counts) > 0:
                        col_info.most_frequent_value = str(value_counts.index[0])
                        col_info.frequency_count = int(value_counts.iloc[0])
                        if len(value_counts) <= 20:  # Only store categories if not too many
                            col_info.categories = [str(cat) for cat in value_counts.index[:20]]
                except:
                    pass
            
            column_info.append(col_info)
        
        # Data quality metrics
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (rows * columns)) * 100 if rows * columns > 0 else 0
        duplicate_rows = df.duplicated().sum()
        
        # Column type distribution
        numeric_columns = list(df.select_dtypes(include=['number']).columns)
        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
        datetime_columns = list(df.select_dtypes(include=['datetime']).columns)
        
        # Data quality score (simple heuristic)
        quality_score = 100.0
        quality_score -= min(missing_percentage, 50)  # Penalize missing values
        quality_score -= min((duplicate_rows / rows) * 50, 25) if rows > 0 else 0  # Penalize duplicates
        quality_score = max(0, quality_score)
        
        # Recommendations
        recommendations = []
        if missing_percentage > 10:
            recommendations.append("Consider handling missing values")
        if duplicate_rows > 0:
            recommendations.append("Consider removing duplicate rows")
        if len(numeric_columns) < 2:
            recommendations.append("Limited numeric data for statistical analysis")
        if columns > 50:
            recommendations.append("Consider dimensionality reduction for large datasets")
        
        return DatasetSummary(
            rows=rows,
            columns=columns,
            memory_usage_mb=memory_usage_mb,
            column_info=column_info,
            total_missing_values=int(total_missing),
            missing_percentage=missing_percentage,
            duplicate_rows=int(duplicate_rows),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            data_quality_score=quality_score,
            recommendations=recommendations
        )
    
    async def _save_datasets(self):
        """Save datasets metadata to file."""
        try:
            datasets_data = {
                dataset_id: metadata.to_dict() 
                for dataset_id, metadata in self.datasets.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(datasets_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving datasets metadata: {e}")
    
    async def _load_datasets(self):
        """Load datasets metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    datasets_data = json.load(f)
                
                for dataset_id, data in datasets_data.items():
                    try:
                        # Reconstruct metadata
                        metadata = DatasetMetadata(
                            dataset_id=data['dataset_id'],
                            name=data['name'],
                            description=data.get('description'),
                            created_at=datetime.fromisoformat(data['created_at']),
                            updated_at=datetime.fromisoformat(data['updated_at']),
                            file_path=data.get('file_path'),
                            cached_data_path=data.get('cached_data_path')
                        )
                        
                        # Reconstruct file info
                        if data.get('file_info'):
                            metadata.file_info = FileInfo(**data['file_info'])
                        
                        # Reconstruct summary
                        if data.get('summary'):
                            summary_data = data['summary']
                            
                            # Reconstruct column info
                            column_info = []
                            for col_data in summary_data.get('column_info', []):
                                column_info.append(ColumnInfo(**col_data))
                            
                            metadata.summary = DatasetSummary(
                                rows=summary_data['rows'],
                                columns=summary_data['columns'],
                                memory_usage_mb=summary_data['memory_usage_mb'],
                                column_info=column_info,
                                total_missing_values=summary_data['total_missing_values'],
                                missing_percentage=summary_data['missing_percentage'],
                                duplicate_rows=summary_data['duplicate_rows'],
                                numeric_columns=summary_data['numeric_columns'],
                                categorical_columns=summary_data['categorical_columns'],
                                datetime_columns=summary_data['datetime_columns'],
                                data_quality_score=summary_data.get('data_quality_score', 0),
                                recommendations=summary_data.get('recommendations', [])
                            )
                        
                        self.datasets[dataset_id] = metadata
                        
                    except Exception as e:
                        logger.error(f"Error loading dataset {dataset_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading datasets metadata: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # Calculate sizes
        upload_size = sum(f.stat().st_size for f in self.upload_dir.glob('*') if f.is_file())
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob('*') if f.is_file())
        data_size = sum(f.stat().st_size for f in self.data_dir.glob('*') if f.is_file())
        
        return {
            'total_datasets': len(self.datasets),
            'upload_size_mb': upload_size / 1024**2,
            'cache_size_mb': cache_size / 1024**2,
            'data_size_mb': data_size / 1024**2,
            'total_size_mb': (upload_size + cache_size + data_size) / 1024**2,
            'max_file_size_mb': self.max_file_size / 1024**2,
            'supported_formats': list(self.supported_types.keys())
        } 