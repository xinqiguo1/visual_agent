"""
Visualization Routes

Routes for creating charts, dashboards, and visualizations.
"""

import uuid
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

from ..models.visualization_models import (
    VisualizationRequest, VisualizationResponse, ChartRequest, ChartResponse,
    DashboardRequestSimple, DashboardResponseSimple, ExportRequest, ExportResponse
)
from ..models.base_models import SuccessResponse, ErrorResponse
from ..utils.file_manager import FileManager
from ..utils.session_manager import SessionManager
from ...agents.web_visualizer import WebVisualizer
from ...agents.visualizer import Visualizer

router = APIRouter()

# Dependencies
def get_file_manager() -> FileManager:
    return FileManager()

def get_session_manager() -> SessionManager:
    return SessionManager()

def get_web_visualizer() -> WebVisualizer:
    return WebVisualizer()

def get_visualizer() -> Visualizer:
    return Visualizer()


@router.post("/create", response_model=VisualizationResponse)
async def create_visualization(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager),
    web_visualizer: WebVisualizer = Depends(get_web_visualizer)
):
    """
    Create a visualization from dataset.
    
    Args:
        request: Visualization request with chart type and parameters
        background_tasks: Background tasks for async processing
        
    Returns:
        Created visualization details
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(request.dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found or could not be loaded")
        
        # Validate columns exist
        all_columns = set(dataset_df.columns)
        requested_columns = {request.x_column, request.y_column, request.color_column} - {None}
        
        if not requested_columns.issubset(all_columns):
            missing_cols = requested_columns - all_columns
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found in dataset: {missing_cols}"
            )
        
        # Create visualization
        viz_result = web_visualizer.create_web_chart(
            data=dataset_df,
            chart_type=request.chart_type,
            x_column=request.x_column,
            y_column=request.y_column,
            color_column=request.color_column,
            title=request.title,
            **request.chart_parameters
        )
        
        if not viz_result["success"]:
            raise HTTPException(status_code=500, detail=viz_result["error"])
        
        # Generate visualization ID
        viz_id = str(uuid.uuid4())
        
        # Store visualization metadata
        viz_metadata = {
            "visualization_id": viz_id,
            "dataset_id": request.dataset_id,
            "chart_type": request.chart_type,
            "x_column": request.x_column,
            "y_column": request.y_column,
            "color_column": request.color_column,
            "title": request.title,
            "chart_parameters": request.chart_parameters,
            "html_path": viz_result["html_path"],
            "chart_id": viz_result["chart_id"],
            "created_at": datetime.now().isoformat()
        }
        
        await session_manager.store_visualization(
            request.session_id,
            viz_id,
            viz_metadata
        )
        
        # Schedule background optimization
        background_tasks.add_task(
            _optimize_visualization_for_web,
            viz_id,
            viz_result["html_path"]
        )
        
        return VisualizationResponse(
            status="success",
            message=f"Visualization created successfully",
            visualization_id=viz_id,
            dataset_id=request.dataset_id,
            chart_type=request.chart_type,
            html_content=viz_result["html_content"],
            div_content=viz_result["div_content"],
            json_data=viz_result["json_data"],
            config=viz_result["config"],
            file_path=viz_result["html_path"],
            title=viz_result["title"],
            x_column=request.x_column,
            y_column=request.y_column,
            color_column=request.color_column
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization creation failed: {str(e)}")


@router.post("/chart", response_model=ChartResponse)
async def create_chart(
    request: ChartRequest,
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager),
    web_visualizer: WebVisualizer = Depends(get_web_visualizer)
):
    """
    Create a specific chart type with custom parameters.
    
    Args:
        request: Chart request with specific chart parameters
        
    Returns:
        Created chart details
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(request.dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found or could not be loaded")
        
        # Apply filters if specified
        if request.filters:
            dataset_df = _apply_filters(dataset_df, request.filters)
        
        # Create chart
        chart_result = web_visualizer.create_web_chart(
            data=dataset_df,
            chart_type=request.chart_type,
            x_column=request.x_column,
            y_column=request.y_column,
            color_column=request.color_column,
            title=request.title,
            **request.styling_options
        )
        
        if not chart_result["success"]:
            raise HTTPException(status_code=500, detail=chart_result["error"])
        
        # Generate chart ID
        chart_id = str(uuid.uuid4())
        
        # Store chart metadata
        await session_manager.store_chart(
            request.session_id,
            chart_id,
            {
                "chart_id": chart_id,
                "dataset_id": request.dataset_id,
                "chart_type": request.chart_type,
                "parameters": request.dict(),
                "file_path": chart_result["html_path"],
                "created_at": datetime.now().isoformat()
            }
        )
        
        return ChartResponse(
            status="success",
            message=f"Chart created successfully",
            chart_id=chart_id,
            dataset_id=request.dataset_id,
            chart_type=request.chart_type,
            html_content=chart_result["html_content"],
            json_data=chart_result["json_data"],
            file_path=chart_result["html_path"],
            title=chart_result["title"],
            data_points=len(dataset_df),
            columns_used=[col for col in [request.x_column, request.y_column, request.color_column] if col is not None],
            styling_applied=request.styling_options
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart creation failed: {str(e)}")


@router.post("/dashboard", response_model=DashboardResponseSimple)
async def create_dashboard(
    request: DashboardRequestSimple,
    background_tasks: BackgroundTasks,
    file_manager: FileManager = Depends(get_file_manager),
    session_manager: SessionManager = Depends(get_session_manager),
    web_visualizer: WebVisualizer = Depends(get_web_visualizer)
):
    """
    Create a dashboard with multiple visualizations.
    
    Args:
        request: Dashboard request with multiple chart specifications
        background_tasks: Background tasks for async processing
        
    Returns:
        Created dashboard details
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(request.dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found or could not be loaded")
        
        # Create individual charts
        charts = []
        for chart_spec in request.charts:
            try:
                # Apply filters if specified
                chart_data = dataset_df
                if chart_spec.get("filters"):
                    chart_data = _apply_filters(chart_data, chart_spec["filters"])
                
                # Create chart
                chart_result = web_visualizer.create_web_chart(
                    data=chart_data,
                    chart_type=chart_spec["chart_type"],
                    x_column=chart_spec.get("x_column"),
                    y_column=chart_spec.get("y_column"),
                    color_column=chart_spec.get("color_column"),
                    title=chart_spec.get("title"),
                    **chart_spec.get("parameters", {})
                )
                
                if chart_result["success"]:
                    charts.append({
                        "chart_id": chart_result["chart_id"],
                        "chart_type": chart_spec["chart_type"],
                        "html_content": chart_result["html_content"],
                        "div_content": chart_result["div_content"],
                        "title": chart_result["title"]
                    })
                    
            except Exception as e:
                # Log error but continue with other charts
                charts.append({
                    "error": f"Failed to create chart: {str(e)}",
                    "chart_type": chart_spec["chart_type"]
                })
        
        # Create dashboard HTML
        dashboard_html = web_visualizer.create_dashboard(
            charts=charts,
            title=request.title or "Data Analysis Dashboard"
        )
        
        # Generate dashboard ID
        dashboard_id = str(uuid.uuid4())
        
        # Store dashboard metadata
        dashboard_metadata = {
            "dashboard_id": dashboard_id,
            "dataset_id": request.dataset_id,
            "title": request.title,
            "charts": charts,
            "layout": request.layout,
            "created_at": datetime.now().isoformat(),
            "html_path": f"outputs/dashboard_{dashboard_id}.html"
        }
        
        await session_manager.store_dashboard(
            request.session_id,
            dashboard_id,
            dashboard_metadata
        )
        
        # Schedule background optimization
        background_tasks.add_task(
            _optimize_dashboard_for_web,
            dashboard_id,
            dashboard_html
        )
        
        return DashboardResponseSimple(
            status="success",
            message=f"Dashboard created with {len(charts)} charts",
            dashboard_id=dashboard_id,
            dataset_id=request.dataset_id,
            title=request.title,
            html_content=dashboard_html,
            charts_created=len([c for c in charts if "error" not in c]),
            charts_failed=len([c for c in charts if "error" in c]),
            layout=request.layout,
            file_path=dashboard_metadata["html_path"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")


@router.get("/charts/{chart_id}", response_model=SuccessResponse)
async def get_chart(
    chart_id: str,
    session_id: str = Query(..., description="Session ID"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get details of a specific chart.
    
    Args:
        chart_id: The chart ID
        session_id: Session ID for access control
        
    Returns:
        Chart details
    """
    try:
        # Get chart metadata
        chart_data = await session_manager.get_chart(session_id, chart_id)
        
        if not chart_data:
            raise HTTPException(status_code=404, detail="Chart not found")
        
        return SuccessResponse(
            status="success",
            message=f"Chart {chart_id} retrieved",
            data=chart_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart: {str(e)}")


@router.get("/dashboards/{dashboard_id}", response_model=SuccessResponse)
async def get_dashboard(
    dashboard_id: str,
    session_id: str = Query(..., description="Session ID"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get details of a specific dashboard.
    
    Args:
        dashboard_id: The dashboard ID
        session_id: Session ID for access control
        
    Returns:
        Dashboard details
    """
    try:
        # Get dashboard metadata
        dashboard_data = await session_manager.get_dashboard(session_id, dashboard_id)
        
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return SuccessResponse(
            status="success",
            message=f"Dashboard {dashboard_id} retrieved",
            data=dashboard_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard: {str(e)}")


@router.get("/charts/{chart_id}/html", response_class=HTMLResponse)
async def get_chart_html(
    chart_id: str,
    session_id: str = Query(..., description="Session ID"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get HTML content of a chart for embedding.
    
    Args:
        chart_id: The chart ID
        session_id: Session ID for access control
        
    Returns:
        HTML content
    """
    try:
        # Get chart metadata
        chart_data = await session_manager.get_chart(session_id, chart_id)
        
        if not chart_data:
            raise HTTPException(status_code=404, detail="Chart not found")
        
        # Read HTML file
        html_path = chart_data.get("file_path")
        if html_path and os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            raise HTTPException(status_code=404, detail="Chart HTML file not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart HTML: {str(e)}")


@router.post("/export", response_model=ExportResponse)
async def export_visualization(
    request: ExportRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Export visualization in various formats.
    
    Args:
        request: Export request with format and options
        
    Returns:
        Export details and file path
    """
    try:
        # Get visualization metadata
        if request.visualization_type == "chart":
            viz_data = await session_manager.get_chart(request.session_id, request.visualization_id)
        elif request.visualization_type == "dashboard":
            viz_data = await session_manager.get_dashboard(request.session_id, request.visualization_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid visualization type")
        
        if not viz_data:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Generate export file path
        export_filename = f"{request.visualization_type}_{request.visualization_id}.{request.format}"
        export_path = f"outputs/{export_filename}"
        
        # Export based on format
        if request.format == "html":
            # Copy existing HTML file
            source_path = viz_data.get("file_path") or viz_data.get("html_path")
            if source_path and os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, export_path)
            else:
                raise HTTPException(status_code=404, detail="Source file not found")
        
        elif request.format == "png":
            # This would require additional libraries like kaleido for plotly
            raise HTTPException(status_code=501, detail="PNG export not implemented yet")
        
        elif request.format == "pdf":
            # This would require additional libraries
            raise HTTPException(status_code=501, detail="PDF export not implemented yet")
        
        elif request.format == "json":
            # Export metadata as JSON
            import json
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2)
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        return ExportResponse(
            status="success",
            message=f"Visualization exported successfully",
            export_id=str(uuid.uuid4()),
            visualization_id=request.visualization_id,
            format=request.format,
            file_path=export_path,
            file_size=os.path.getsize(export_path) if os.path.exists(export_path) else 0,
            exported_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Download exported visualization files.
    
    Args:
        file_path: Path to the file to download
        
    Returns:
        File download response
    """
    try:
        full_path = f"outputs/{file_path}"
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=full_path,
            filename=os.path.basename(full_path),
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/suggest", response_model=SuccessResponse)
async def suggest_visualizations(
    dataset_id: str,
    file_manager: FileManager = Depends(get_file_manager),
    web_visualizer: WebVisualizer = Depends(get_web_visualizer)
):
    """
    Get visualization suggestions for a dataset.
    
    Args:
        dataset_id: The dataset ID
        
    Returns:
        Suggested visualizations
    """
    try:
        # Initialize file manager
        await file_manager.initialize()
        
        # Load dataset
        dataset_df = await file_manager.load_dataframe(dataset_id)
        
        if dataset_df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or could not be loaded")
        
        # Get suggestions
        suggestions = web_visualizer.suggest_web_visualizations(dataset_df)
        
        return SuccessResponse(
            status="success",
            message=f"Generated {len(suggestions)} visualization suggestions",
            data={
                "dataset_id": dataset_id,
                "suggestions": suggestions,
                "total_suggestions": len(suggestions)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")


# Helper functions
def _apply_filters(df, filters: Dict[str, Any]):
    """Apply filters to dataframe."""
    filtered_df = df.copy()
    
    for column, filter_spec in filters.items():
        if column not in df.columns:
            continue
        
        if isinstance(filter_spec, dict):
            if "min" in filter_spec:
                filtered_df = filtered_df[filtered_df[column] >= filter_spec["min"]]
            if "max" in filter_spec:
                filtered_df = filtered_df[filtered_df[column] <= filter_spec["max"]]
            if "values" in filter_spec:
                filtered_df = filtered_df[filtered_df[column].isin(filter_spec["values"])]
        else:
            # Simple value filter
            filtered_df = filtered_df[filtered_df[column] == filter_spec]
    
    return filtered_df


async def _optimize_visualization_for_web(viz_id: str, html_path: str):
    """Background task to optimize visualization for web display."""
    # This would implement various optimizations like:
    # - Minifying HTML/CSS/JS
    # - Compressing images
    # - Adding caching headers
    pass


async def _optimize_dashboard_for_web(dashboard_id: str, html_content: str):
    """Background task to optimize dashboard for web display."""
    # This would implement dashboard-specific optimizations
    pass 