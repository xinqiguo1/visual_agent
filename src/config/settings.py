"""
Settings and Configuration

Contains LLM configuration, agent settings, and other application configurations.
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.llms.base import BaseLLM


class AgentConfig:
    """Configuration settings for the analytics agents."""
    
    # LLM Settings
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2000
    
    # Agent Settings
    MAX_CONVERSATION_HISTORY = 10
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30
    
    # Analysis Settings
    CORRELATION_THRESHOLD = 0.3
    OUTLIER_THRESHOLD = 2.0
    MAX_CHART_SUGGESTIONS = 5
    
    # Safety Settings
    ENABLE_CODE_VALIDATION = True
    ENABLE_SAFETY_CHECKS = True
    ALLOWED_IMPORTS = [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", 
        "scipy", "sklearn", "warnings"
    ]
    
    # Tool Settings
    ENABLE_INTERACTIVE_CHARTS = True
    DEFAULT_CHART_DPI = 150
    MAX_DATA_SIZE_MB = 500


def get_llm(api_key: Optional[str] = None, **kwargs) -> BaseLLM:
    """
    Get configured LLM instance.
    
    Args:
        api_key: OpenAI API key (if not provided, uses environment variable)
        **kwargs: Additional LLM parameters
        
    Returns:
        Configured ChatOpenAI instance
    """
    # Get API key from parameter or environment
    openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        print("You can add your API key by setting: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
    
    # Default LLM configuration
    llm_config = {
        "model_name": kwargs.get("model_name", AgentConfig.MODEL_NAME),
        "temperature": kwargs.get("temperature", AgentConfig.TEMPERATURE),
        "max_tokens": kwargs.get("max_tokens", AgentConfig.MAX_TOKENS),
        "request_timeout": kwargs.get("timeout", AgentConfig.TIMEOUT_SECONDS),
        "max_retries": kwargs.get("max_retries", AgentConfig.MAX_RETRIES),
    }
    
    # Only add API key if provided
    if openai_api_key:
        llm_config["openai_api_key"] = openai_api_key
    
    try:
        return ChatOpenAI(**llm_config)
    except Exception as e:
        print(f"Warning: Failed to initialize ChatOpenAI: {e}")
        print("Returning None - agents will use fallback methods")
        return None


def get_agent_prompts() -> Dict[str, str]:
    """Get prompt templates for different agent functions."""
    return {
        "intent_classification": """
You are a data analysis assistant. Analyze the user's query and classify the intent.

Query: "{query}"
Available columns in dataset: {columns}
Data types: {dtypes}

Intent categories:
- exploration: asking about data structure, columns, shape, overview, first few rows
- statistics: calculating means, sums, correlations, aggregations, summary statistics
- visualization: creating charts, plots, graphs, showing distributions
- filtering: selecting subsets, filtering data, querying specific conditions
- transformation: data cleaning, normalization, encoding, feature engineering
- insights: requesting automated insights, patterns, anomaly detection
- general: help, unclear requests, or conversational queries

Respond with just the intent category (one word).
Intent:""",

        "parameter_extraction": """
Extract parameters from this data analysis query for execution.

Query: "{query}"
Available columns: {columns}
Numeric columns: {numeric_columns}
Categorical columns: {categorical_columns}

Extract and return as JSON:
{{
    "chart_type": "type of chart if visualization requested (bar, line, scatter, histogram, box, heatmap, pie)",
    "x_column": "column name for x-axis if specified",
    "y_column": "column name for y-axis if specified", 
    "columns": ["list of columns mentioned"],
    "filters": "any filtering conditions mentioned",
    "aggregation": "any grouping or aggregation requests",
    "operation": "specific statistical operation requested (mean, sum, describe, correlation, etc.)"
}}

Parameters:""",

        "query_analysis": """
You are an expert data analyst. Analyze this query and provide a comprehensive response plan.

Query: "{query}"
Dataset info: {dataset_info}
Available tools: data_exploration, statistical_analysis, visualization, code_generation, insight_generation

Break down the query into:
1. What the user wants to know
2. What tools/analysis are needed
3. What parameters are required
4. Expected output format

Provide a clear analysis plan.
""",

        "code_explanation": """
Explain this generated Python code in simple terms for the user.

Code:
{code}

Query context: {query}

Provide a clear, non-technical explanation of:
1. What the code does
2. What results to expect
3. How to interpret the output

Explanation:""",

        "insight_summary": """
Create a concise, actionable summary of these data insights for the user.

Insights: {insights}
Original query: {query}

Provide:
1. Key findings (2-3 main points)
2. What this means for the data
3. Recommended next steps

Summary:"""
    }


def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions for LangChain tools."""
    return {
        "data_exploration": "Get information about dataset structure, columns, data types, shape, and basic overview. Use this when users ask about the dataset itself.",
        
        "statistical_analysis": "Perform statistical calculations like mean, median, correlation, summary statistics. Use this for numerical analysis requests.",
        
        "visualization": "Create charts, plots, and visualizations from data. Use this when users want to see graphical representations of their data.",
        
        "code_generation": "Generate Python code for data analysis tasks. Use this when users want to see the code or perform custom analysis.",
        
        "insight_generation": "Generate automated insights, patterns, and recommendations from the data. Use this for discovery and pattern finding.",
        
        "data_filtering": "Filter and subset data based on conditions. Use this when users want to look at specific portions of their data."
    }


# Environment setup helper
def setup_environment():
    """Setup environment variables and configurations."""
    # Set default environment variables if not present
    if not os.getenv("OPENAI_API_KEY"):
        print("📝 To use the full AI capabilities, please set your OpenAI API key:")
        print("   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
        print("   or export OPENAI_API_KEY=your-api-key-here")
    
    # Configure warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Set matplotlib backend for better compatibility
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        pass 