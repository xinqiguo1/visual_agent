"""
Analysis Models

Pydantic models for data analysis, queries, and insights.
"""

from pydantic import BaseModel, Field, validator, model_validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum

from .base_models import BaseResponse


class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    COMPREHENSIVE = "comprehensive"
    DESCRIPTIVE = "descriptive"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    VISUALIZATION = "visualization"
    PREDICTION = "prediction"
    CLUSTERING = "clustering"
    OUTLIER_DETECTION = "outlier_detection"
    TREND_ANALYSIS = "trend_analysis"
    EXPLORATORY = "exploratory"
    CUSTOM = "custom"


class QueryType(str, Enum):
    """Query type enumeration."""
    NATURAL_LANGUAGE = "natural_language"
    SQL = "sql"
    PYTHON = "python"
    STRUCTURED = "structured"


class AnalysisRequest(BaseModel):
    """Analysis request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    query: Optional[str] = Field(None, min_length=1, max_length=2000, description="Natural language query")
    query_type: QueryType = Field(QueryType.NATURAL_LANGUAGE, description="Type of query")
    analysis_type: Optional[AnalysisType] = Field(None, description="Specific analysis type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    # Agent configuration
    use_ai_agent: bool = Field(True, description="Whether to use AI agent for analysis")
    agent_temperature: float = Field(0.1, ge=0, le=1, description="AI agent temperature")
    max_tokens: int = Field(2000, ge=100, le=4000, description="Maximum tokens for AI response")
    
    # Output preferences
    include_code: bool = Field(True, description="Include generated Python code")
    include_insights: bool = Field(True, description="Include automated insights")
    include_visualizations: bool = Field(True, description="Include visualization suggestions")
    
    @validator('query')
    def validate_query(cls, v):
        if v is not None and not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip() if v else None
    
    @model_validator(mode='after')
    def validate_query_or_analysis_type(self):
        if not self.query and not self.analysis_type:
            raise ValueError('Either query or analysis_type must be provided')
        
        return self


class CodeResult(BaseModel):
    """Code generation result model."""
    code: str = Field(..., description="Generated Python code")
    language: str = Field("python", description="Programming language")
    explanation: str = Field(..., description="Code explanation")
    is_executable: bool = Field(..., description="Whether code is executable")
    execution_time: Optional[float] = Field(None, ge=0, description="Execution time in seconds")
    output: Optional[str] = Field(None, description="Code execution output")
    error: Optional[str] = Field(None, description="Code execution error")


class StatisticalResult(BaseModel):
    """Statistical analysis result model."""
    metric: str = Field(..., description="Statistical metric name")
    value: Union[int, float, str] = Field(..., description="Metric value")
    description: str = Field(..., description="Metric description")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval")
    p_value: Optional[float] = Field(None, description="P-value (if applicable)")
    interpretation: Optional[str] = Field(None, description="Statistical interpretation")


class CorrelationResult(BaseModel):
    """Correlation analysis result model."""
    variable1: str = Field(..., description="First variable")
    variable2: str = Field(..., description="Second variable")
    correlation_coefficient: float = Field(..., ge=-1, le=1, description="Correlation coefficient")
    correlation_type: str = Field(..., description="Type of correlation (pearson, spearman, etc.)")
    p_value: Optional[float] = Field(None, description="P-value")
    significance: Optional[str] = Field(None, description="Statistical significance")
    interpretation: str = Field(..., description="Correlation interpretation")


class InsightResult(BaseModel):
    """Insight generation result model."""
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    importance: float = Field(..., ge=0, le=1, description="Insight importance score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in insight")
    supporting_data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")


class VisualizationSuggestion(BaseModel):
    """Visualization suggestion model."""
    chart_type: str = Field(..., description="Suggested chart type")
    title: str = Field(..., description="Suggested chart title")
    description: str = Field(..., description="Visualization description")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color column")
    priority: float = Field(..., ge=0, le=1, description="Suggestion priority")
    rationale: str = Field(..., description="Why this visualization is suggested")


class QueryResult(BaseModel):
    """Query execution result model."""
    query_id: str = Field(..., description="Unique query identifier")
    original_query: str = Field(..., description="Original query text")
    interpreted_query: str = Field(..., description="Interpreted/processed query")
    query_type: QueryType = Field(..., description="Detected query type")
    analysis_type: AnalysisType = Field(..., description="Detected analysis type")
    
    # Results
    data_result: Optional[List[Dict[str, Any]]] = Field(None, description="Query data result")
    statistical_results: List[StatisticalResult] = Field(default_factory=list, description="Statistical results")
    correlation_results: List[CorrelationResult] = Field(default_factory=list, description="Correlation results")
    insights: List[InsightResult] = Field(default_factory=list, description="Generated insights")
    visualization_suggestions: List[VisualizationSuggestion] = Field(default_factory=list, description="Visualization suggestions")
    
    # Generated code
    code_result: Optional[CodeResult] = Field(None, description="Generated code")
    
    # Execution metadata
    execution_time: float = Field(..., ge=0, description="Total execution time")
    agent_used: bool = Field(..., description="Whether AI agent was used")
    fallback_used: bool = Field(..., description="Whether fallback method was used")
    
    # Quality metrics
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in results")
    completeness_score: float = Field(..., ge=0, le=1, description="Result completeness score")


class AnalysisResponse(BaseResponse):
    """Analysis response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    query_result: QueryResult = Field(..., description="Query execution result")
    session_id: Optional[str] = Field(None, description="Session identifier")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Follow-up query suggestions")


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    queries: List[str] = Field(..., min_items=1, max_items=10, description="List of queries")
    analysis_type: Optional[AnalysisType] = Field(None, description="Analysis type for all queries")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Common parameters")
    
    @validator('queries')
    def validate_queries(cls, v):
        if not v:
            raise ValueError('At least one query is required')
        for query in v:
            if not query.strip():
                raise ValueError('All queries must be non-empty')
        return [q.strip() for q in v]


class BatchAnalysisResponse(BaseResponse):
    """Batch analysis response model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    results: List[QueryResult] = Field(..., description="List of query results")
    batch_id: str = Field(..., description="Batch identifier")
    total_execution_time: float = Field(..., ge=0, description="Total batch execution time")
    successful_queries: int = Field(..., ge=0, description="Number of successful queries")
    failed_queries: int = Field(..., ge=0, description="Number of failed queries")


class AnalysisHistoryRequest(BaseModel):
    """Analysis history request model."""
    dataset_id: Optional[str] = Field(None, description="Dataset identifier (optional)")
    session_id: Optional[str] = Field(None, description="Session identifier (optional)")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Result offset")
    query_type: Optional[QueryType] = Field(None, description="Filter by query type")
    analysis_type: Optional[AnalysisType] = Field(None, description="Filter by analysis type")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        if v and 'start_date' in values and values['start_date'] and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class AnalysisHistoryResponse(BaseResponse):
    """Analysis history response model."""
    queries: List[QueryResult] = Field(..., description="Historical query results")
    total_count: int = Field(..., ge=0, description="Total number of queries")
    page_info: Dict[str, Any] = Field(..., description="Pagination information")


class AnalysisExportRequest(BaseModel):
    """Analysis export request model."""
    query_ids: List[str] = Field(..., min_items=1, description="List of query IDs to export")
    format: str = Field("json", description="Export format")
    include_code: bool = Field(True, description="Include generated code")
    include_data: bool = Field(True, description="Include result data")
    
    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['json', 'csv', 'excel', 'pdf']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Format must be one of: {allowed_formats}')
        return v.lower()


class AnalysisExportResponse(BaseResponse):
    """Analysis export response model."""
    export_id: str = Field(..., description="Export identifier")
    file_path: str = Field(..., description="Path to exported file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    queries_exported: int = Field(..., ge=0, description="Number of queries exported")
    format: str = Field(..., description="Export format")


class SavedQueryRequest(BaseModel):
    """Saved query request model."""
    name: str = Field(..., min_length=1, max_length=100, description="Query name")
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    description: Optional[str] = Field(None, max_length=500, description="Query description")
    tags: List[str] = Field(default_factory=list, description="Query tags")
    is_template: bool = Field(False, description="Whether this is a template query")
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.strip().lower() for tag in v if tag.strip()]


class SavedQueryResponse(BaseResponse):
    """Saved query response model."""
    query_id: str = Field(..., description="Saved query identifier")
    name: str = Field(..., description="Query name")
    query: str = Field(..., description="Query text")
    description: Optional[str] = Field(None, description="Query description")
    tags: List[str] = Field(default_factory=list, description="Query tags")
    is_template: bool = Field(..., description="Whether this is a template query")
    created_at: datetime = Field(..., description="Creation timestamp")
    usage_count: int = Field(0, ge=0, description="Number of times used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Additional models for API routes compatibility
class QueryRequest(BaseModel):
    """Query request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Natural language query")


class QueryResponse(BaseResponse):
    """Query response model."""
    query_id: str = Field(..., description="Query identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Query answer")
    data_results: List[Dict[str, Any]] = Field(default_factory=list, description="Data results")
    visualizations: List[Dict[str, Any]] = Field(default_factory=list, description="Visualizations")
    code_generated: str = Field("", description="Generated code")
    confidence_score: float = Field(0.0, ge=0, le=1, description="Confidence score")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class InsightRequest(BaseModel):
    """Insight request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    insight_type: str = Field("automated", description="Type of insights to generate")
    top_n: int = Field(5, ge=1, le=20, description="Number of top insights to return")


class InsightResponse(BaseResponse):
    """Insight response model."""
    insight_id: str = Field(..., description="Insight identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    insight_type: str = Field(..., description="Type of insights")
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="Generated insights")
    total_insights: int = Field(0, ge=0, description="Total number of insights")
    categories: List[str] = Field(default_factory=list, description="Insight categories")


class ReportRequest(BaseModel):
    """Report request model."""
    dataset_id: str = Field(..., description="Dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    report_type: str = Field("comprehensive", description="Type of report")
    include_visualizations: bool = Field(True, description="Include visualizations")
    include_code: bool = Field(True, description="Include code")


class ReportResponse(BaseResponse):
    """Report response model."""
    report_id: str = Field(..., description="Report identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    report_type: str = Field(..., description="Type of report")
    report_data: Dict[str, Any] = Field(..., description="Report data")
    sections: List[str] = Field(default_factory=list, description="Report sections")
    generated_at: str = Field(..., description="Report generation timestamp") 