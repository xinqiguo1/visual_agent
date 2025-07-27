"""
Visualization Models

Pydantic models for chart creation, dashboards, and visualization management.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

from .base_models import BaseResponse


class ChartType(str, Enum):
    """Chart type enumeration."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    VIOLIN = "violin"
    AREA = "area"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR = "radar"
    WATERFALL = "waterfall"
    GANTT = "gantt"


class OutputFormat(str, Enum):
    """Output format enumeration."""
    HTML = "html"
    JSON = "json"
    PNG = "png"
    PDF = "pdf"
    SVG = "svg"
    PLOTLY_JSON = "plotly_json"


class ChartConfig(BaseModel):
    """Chart configuration model."""
    chart_type: ChartType = Field(..., description="Type of chart to create")
    title: Optional[str] = Field(None, max_length=200, description="Chart title")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color coding column")
    size_column: Optional[str] = Field(None, description="Size column (for scatter plots)")
    
    # Chart styling
    width: int = Field(800, ge=300, le=2000, description="Chart width in pixels")
    height: int = Field(600, ge=200, le=1500, description="Chart height in pixels")
    theme: str = Field("plotly_white", description="Chart theme")
    color_palette: Optional[str] = Field(None, description="Color palette name")
    
    # Axis configuration
    x_title: Optional[str] = Field(None, description="X-axis title")
    y_title: Optional[str] = Field(None, description="Y-axis title")
    x_log_scale: bool = Field(False, description="Use log scale for X-axis")
    y_log_scale: bool = Field(False, description="Use log scale for Y-axis")
    
    # Interactive features
    show_legend: bool = Field(True, description="Show legend")
    show_grid: bool = Field(True, description="Show grid")
    enable_zoom: bool = Field(True, description="Enable zoom")
    enable_pan: bool = Field(True, description="Enable pan")
    
    # Data filtering
    filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")
    limit: Optional[int] = Field(None, ge=1, le=50000, description="Maximum data points")
    
    # Advanced options
    aggregation: Optional[str] = Field(None, description="Data aggregation method")
    groupby: Optional[str] = Field(None, description="Group by column")
    sort_by: Optional[str] = Field(None, description="Sort by column")
    ascending: bool = Field(True, description="Sort order")
    
    @validator('theme')
    def validate_theme(cls, v):
        allowed_themes = [
            "plotly", "plotly_white", "plotly_dark", "ggplot2", 
            "seaborn", "simple_white", "presentation", "xgridoff",
            "ygridoff", "gridon", "none"
        ]
        if v not in allowed_themes:
            raise ValueError(f'Theme must be one of: {allowed_themes}')
        return v


class VisualizationRequest(BaseModel):
    """Visualization request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_config: ChartConfig = Field(..., description="Chart configuration")
    output_format: OutputFormat = Field(OutputFormat.HTML, description="Output format")
    filename: Optional[str] = Field(None, max_length=100, description="Custom filename")
    
    # AI enhancement options
    auto_enhance: bool = Field(True, description="Auto-enhance chart with AI suggestions")
    smart_labels: bool = Field(True, description="Generate smart labels and titles")
    optimize_for_mobile: bool = Field(True, description="Optimize for mobile viewing")
    
    @validator('filename')
    def validate_filename(cls, v):
        if v:
            # Remove invalid characters
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                if char in v:
                    raise ValueError(f'Filename cannot contain: {invalid_chars}')
        return v


class ChartResult(BaseModel):
    """Chart generation result model."""
    chart_id: str = Field(..., description="Unique chart identifier")
    chart_type: ChartType = Field(..., description="Chart type")
    title: str = Field(..., description="Chart title")
    
    # Output files
    html_path: Optional[str] = Field(None, description="HTML file path")
    html_content: Optional[str] = Field(None, description="HTML content")
    json_data: Optional[str] = Field(None, description="Plotly JSON data")
    image_path: Optional[str] = Field(None, description="Image file path")
    
    # Chart metadata
    data_points: int = Field(..., ge=0, description="Number of data points")
    columns_used: List[str] = Field(default_factory=list, description="Columns used in chart")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    # AI enhancements
    ai_suggestions_applied: List[str] = Field(default_factory=list, description="AI suggestions applied")
    quality_score: float = Field(..., ge=0, le=1, description="Chart quality score")
    
    # Interactive features
    supports_interaction: bool = Field(..., description="Whether chart supports interaction")
    export_options: List[str] = Field(default_factory=list, description="Available export options")


class VisualizationResponse(BaseResponse):
    """Visualization response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_result: ChartResult = Field(..., description="Chart generation result")
    alternative_suggestions: List[ChartConfig] = Field(default_factory=list, description="Alternative chart suggestions")


class DashboardLayout(BaseModel):
    """Dashboard layout configuration."""
    layout_type: str = Field("grid", description="Layout type (grid, tabs, single)")
    columns: int = Field(2, ge=1, le=4, description="Number of columns")
    responsive: bool = Field(True, description="Responsive layout")
    spacing: int = Field(20, ge=0, le=100, description="Spacing between charts")
    
    @validator('layout_type')
    def validate_layout_type(cls, v):
        allowed_types = ['grid', 'tabs', 'single', 'custom']
        if v not in allowed_types:
            raise ValueError(f'Layout type must be one of: {allowed_types}')
        return v


class DashboardRequest(BaseModel):
    """Dashboard creation request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    title: str = Field(..., max_length=200, description="Dashboard title")
    description: Optional[str] = Field(None, max_length=1000, description="Dashboard description")
    
    # Charts to include
    chart_configs: List[ChartConfig] = Field(..., min_items=1, max_items=12, description="Chart configurations")
    layout: DashboardLayout = Field(default_factory=DashboardLayout, description="Dashboard layout")
    
    # Styling
    theme: str = Field("plotly_white", description="Dashboard theme")
    color_scheme: Optional[str] = Field(None, description="Color scheme")
    custom_css: Optional[str] = Field(None, description="Custom CSS styles")
    
    # Features
    include_filters: bool = Field(True, description="Include interactive filters")
    include_export: bool = Field(True, description="Include export options")
    include_refresh: bool = Field(True, description="Include refresh button")
    
    # Auto-generation options
    auto_generate: bool = Field(False, description="Auto-generate charts based on data")
    max_charts: int = Field(6, ge=1, le=12, description="Maximum charts for auto-generation")
    
    @validator('chart_configs')
    def validate_chart_configs(cls, v):
        if len(v) == 0:
            raise ValueError('At least one chart configuration is required')
        return v


class DashboardResult(BaseModel):
    """Dashboard generation result model."""
    dashboard_id: str = Field(..., description="Unique dashboard identifier")
    title: str = Field(..., description="Dashboard title")
    
    # Output files
    html_path: str = Field(..., description="Dashboard HTML file path")
    html_content: str = Field(..., description="Dashboard HTML content")
    
    # Chart results
    charts: List[ChartResult] = Field(..., description="Individual chart results")
    successful_charts: int = Field(..., ge=0, description="Number of successful charts")
    failed_charts: int = Field(..., ge=0, description="Number of failed charts")
    
    # Dashboard metadata
    total_data_points: int = Field(..., ge=0, description="Total data points across all charts")
    processing_time: float = Field(..., ge=0, description="Total processing time")
    layout: DashboardLayout = Field(..., description="Applied layout")
    
    # Quality metrics
    overall_quality_score: float = Field(..., ge=0, le=1, description="Overall dashboard quality")
    mobile_friendly: bool = Field(..., description="Whether dashboard is mobile-friendly")
    
    # Features
    interactive_features: List[str] = Field(default_factory=list, description="Available interactive features")
    export_formats: List[str] = Field(default_factory=list, description="Available export formats")


class DashboardResponse(BaseResponse):
    """Dashboard response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    dashboard_result: DashboardResult = Field(..., description="Dashboard generation result")


class VisualizationSuggestionsRequest(BaseModel):
    """Visualization suggestions request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    max_suggestions: int = Field(5, ge=1, le=10, description="Maximum number of suggestions")
    include_advanced: bool = Field(True, description="Include advanced chart types")
    purpose: Optional[str] = Field(None, description="Analysis purpose (exploratory, presentation, etc.)")
    target_audience: Optional[str] = Field(None, description="Target audience (technical, business, etc.)")


class VisualizationSuggestion(BaseModel):
    """Single visualization suggestion model."""
    chart_config: ChartConfig = Field(..., description="Suggested chart configuration")
    rationale: str = Field(..., description="Why this visualization is recommended")
    use_case: str = Field(..., description="Best use case for this visualization")
    priority: float = Field(..., ge=0, le=1, description="Suggestion priority")
    difficulty: str = Field(..., description="Implementation difficulty (easy, medium, hard)")
    estimated_insight: str = Field(..., description="Expected insights from this visualization")
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        allowed_difficulties = ['easy', 'medium', 'hard']
        if v not in allowed_difficulties:
            raise ValueError(f'Difficulty must be one of: {allowed_difficulties}')
        return v


class VisualizationSuggestionsResponse(BaseResponse):
    """Visualization suggestions response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    suggestions: List[VisualizationSuggestion] = Field(..., description="List of visualization suggestions")
    analysis_summary: str = Field(..., description="Data analysis summary")
    recommended_workflow: List[str] = Field(default_factory=list, description="Recommended analysis workflow")


class ChartTemplateRequest(BaseModel):
    """Chart template request model."""
    name: str = Field(..., max_length=100, description="Template name")
    description: Optional[str] = Field(None, max_length=500, description="Template description")
    chart_config: ChartConfig = Field(..., description="Template chart configuration")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(True, description="Whether template is public")
    
    @validator('category')
    def validate_category(cls, v):
        allowed_categories = [
            'business', 'scientific', 'financial', 'marketing', 
            'operations', 'academic', 'presentation', 'exploratory'
        ]
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {allowed_categories}')
        return v


class ChartTemplateResponse(BaseResponse):
    """Chart template response model."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    chart_config: ChartConfig = Field(..., description="Template chart configuration")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_public: bool = Field(..., description="Whether template is public")
    created_at: datetime = Field(..., description="Creation timestamp")
    usage_count: int = Field(0, ge=0, description="Number of times used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChartUpdateRequest(BaseModel):
    """Chart update request model."""
    chart_id: str = Field(..., description="Chart identifier")
    chart_config: ChartConfig = Field(..., description="Updated chart configuration")
    regenerate: bool = Field(True, description="Whether to regenerate the chart")


class ChartUpdateResponse(BaseResponse):
    """Chart update response model."""
    chart_id: str = Field(..., description="Chart identifier")
    chart_result: ChartResult = Field(..., description="Updated chart result")
    changes_applied: List[str] = Field(default_factory=list, description="List of changes applied")


class ChartExportRequest(BaseModel):
    """Chart export request model."""
    chart_id: str = Field(..., description="Chart identifier")
    format: OutputFormat = Field(..., description="Export format")
    width: Optional[int] = Field(None, ge=300, le=2000, description="Export width")
    height: Optional[int] = Field(None, ge=200, le=1500, description="Export height")
    dpi: int = Field(300, ge=72, le=600, description="DPI for image exports")
    
    
class ChartExportResponse(BaseResponse):
    """Chart export response model."""
    chart_id: str = Field(..., description="Chart identifier")
    export_format: OutputFormat = Field(..., description="Export format")
    file_path: str = Field(..., description="Path to exported file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    download_url: str = Field(..., description="Download URL")


class ChartListRequest(BaseModel):
    """Chart list request model."""
    dataset_id: Optional[str] = Field(None, description="Filter by dataset")
    chart_type: Optional[ChartType] = Field(None, description="Filter by chart type")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of charts")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('Sort order must be asc or desc')
        return v


class ChartListResponse(BaseResponse):
    """Chart list response model."""
    charts: List[ChartResult] = Field(..., description="List of charts")
    total_count: int = Field(..., ge=0, description="Total number of charts")
    has_more: bool = Field(..., description="Whether there are more charts")
    page_info: Dict[str, Any] = Field(..., description="Pagination information")


# Additional models for API routes compatibility
class VisualizationRequest(BaseModel):
    """Visualization request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    chart_type: str = Field(..., description="Chart type")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color column")
    title: Optional[str] = Field(None, description="Chart title")
    chart_parameters: Dict[str, Any] = Field(default_factory=dict, description="Chart parameters")


class VisualizationResponse(BaseResponse):
    """Visualization response model."""
    visualization_id: str = Field(..., description="Visualization identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: str = Field(..., description="Chart type")
    html_content: str = Field(..., description="HTML content")
    div_content: str = Field(..., description="Div content")
    json_data: str = Field(..., description="JSON data")
    config: Dict[str, Any] = Field(..., description="Chart configuration")
    file_path: str = Field(..., description="File path")
    title: str = Field(..., description="Chart title")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color column")


class ChartRequest(BaseModel):
    """Chart request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    chart_type: str = Field(..., description="Chart type")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color column")
    title: Optional[str] = Field(None, description="Chart title")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    styling_options: Dict[str, Any] = Field(default_factory=dict, description="Styling options")


class ChartResponse(BaseResponse):
    """Chart response model."""
    chart_id: str = Field(..., description="Chart identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: str = Field(..., description="Chart type")
    html_content: str = Field(..., description="HTML content")
    json_data: str = Field(..., description="JSON data")
    file_path: str = Field(..., description="File path")
    title: str = Field(..., description="Chart title")
    data_points: int = Field(..., description="Number of data points")
    columns_used: List[str] = Field(..., description="Columns used")
    styling_applied: Dict[str, Any] = Field(..., description="Styling applied")


class DashboardRequestSimple(BaseModel):
    """Dashboard request model (simple)."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Dashboard title")
    charts: List[Dict[str, Any]] = Field(..., description="Chart specifications")
    layout: Dict[str, Any] = Field(default_factory=dict, description="Layout configuration")


class DashboardResponseSimple(BaseResponse):
    """Dashboard response model (simple)."""
    dashboard_id: str = Field(..., description="Dashboard identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    title: Optional[str] = Field(None, description="Dashboard title")
    html_content: str = Field(..., description="HTML content")
    charts_created: int = Field(..., description="Number of charts created")
    charts_failed: int = Field(..., description="Number of charts failed")
    layout: Dict[str, Any] = Field(..., description="Layout configuration")
    file_path: str = Field(..., description="File path")


class ExportRequest(BaseModel):
    """Export request model."""
    visualization_id: str = Field(..., description="Visualization identifier")
    session_id: str = Field(..., description="Session identifier")
    visualization_type: str = Field(..., description="Visualization type (chart/dashboard)")
    format: str = Field(..., description="Export format")


class ExportResponse(BaseResponse):
    """Export response model."""
    export_id: str = Field(..., description="Export identifier")
    visualization_id: str = Field(..., description="Visualization identifier")
    format: str = Field(..., description="Export format")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    exported_at: str = Field(..., description="Export timestamp") 