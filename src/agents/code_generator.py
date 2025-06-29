"""
Code Generator Agent

Agent responsible for generating Python code for data analysis,
statistical operations, and data manipulations.
"""

from typing import Dict, Any, List, Optional
import ast
import textwrap
from datetime import datetime


class CodeGenerator:
    """
    Agent responsible for generating Python code for data analysis tasks.
    """
    
    def __init__(self):
        """Initialize the code generator agent."""
        self.code_templates = self._load_code_templates()
        self.imports = [
            "import pandas as pd",
            "import numpy as np", 
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from scipy import stats"
        ]
        self.safety_checks = True
        
    def generate_analysis_code(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        data_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate Python code for a specific analysis task.
        
        Args:
            task_type: Type of analysis (statistics, visualization, filtering, etc.)
            parameters: Task-specific parameters
            data_info: Information about the dataset
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            if task_type == "statistics":
                return self._generate_statistics_code(parameters, data_info)
            elif task_type == "visualization":
                return self._generate_visualization_code(parameters, data_info)
            elif task_type == "filtering":
                return self._generate_filtering_code(parameters, data_info)
            elif task_type == "aggregation":
                return self._generate_aggregation_code(parameters, data_info)
            elif task_type == "transformation":
                return self._generate_transformation_code(parameters, data_info)
            else:
                return self._generate_generic_code(task_type, parameters, data_info)
                
        except Exception as e:
            return {
                "error": f"Code generation failed: {str(e)}",
                "task_type": task_type
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate generated Python code for syntax and safety.
        
        Args:
            code: Python code to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": False,
            "syntax_errors": [],
            "safety_warnings": [],
            "suggestions": []
        }
        
        # Check syntax
        try:
            ast.parse(code)
            validation_result["is_valid"] = True
        except SyntaxError as e:
            validation_result["syntax_errors"].append({
                "line": e.lineno,
                "message": str(e),
                "text": e.text
            })
        
        # Safety checks
        if self.safety_checks:
            safety_issues = self._check_code_safety(code)
            validation_result["safety_warnings"] = safety_issues
        
        return validation_result
    
    def _generate_statistics_code(self, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for statistical analysis."""
        operation = parameters.get("operation", "describe")
        columns = parameters.get("columns", [])
        
        if operation == "describe":
            code = """
# Statistical summary of the dataset
summary_stats = data.describe()
print("Summary Statistics:")
print(summary_stats)
result = summary_stats
"""
        elif operation == "correlation":
            code = """
# Correlation analysis
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
result = correlation_matrix
"""
        elif operation == "mean":
            if columns:
                cols_str = str(columns)
                code = f"""
# Calculate mean for specific columns
selected_columns = {cols_str}
mean_values = data[selected_columns].mean()
print("Mean values:")
print(mean_values)
result = mean_values
"""
            else:
                code = """
# Calculate mean for all numeric columns
numeric_data = data.select_dtypes(include=['number'])
mean_values = numeric_data.mean()
print("Mean values:")
print(mean_values)
result = mean_values
"""
        else:
            code = """
# Basic statistical analysis
result = data.describe()
print(result)
"""
        
        return self._format_code_response("statistics", code, parameters)
    
    def _generate_visualization_code(self, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for creating visualizations."""
        chart_type = parameters.get("chart_type", "scatter")
        x_column = parameters.get("x_column")
        y_column = parameters.get("y_column")
        
        if chart_type == "scatter" and x_column and y_column:
            code = f"""
# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['{x_column}'], data['{y_column}'], alpha=0.6)
plt.xlabel('{x_column}')
plt.ylabel('{y_column}')
plt.title('{y_column} vs {x_column}')
plt.grid(True, alpha=0.3)
plt.show()
result = "Scatter plot created successfully"
"""
        elif chart_type == "histogram" and x_column:
            code = f"""
# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(data['{x_column}'], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('{x_column}')
plt.ylabel('Frequency')
plt.title('Distribution of {x_column}')
plt.grid(True, alpha=0.3)
plt.show()
result = "Histogram created successfully"
"""
        elif chart_type == "bar" and x_column:
            code = f"""
# Create bar chart
plt.figure(figsize=(10, 6))
value_counts = data['{x_column}'].value_counts()
value_counts.plot(kind='bar')
plt.xlabel('{x_column}')
plt.ylabel('Count')
plt.title('Count of {x_column}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
result = "Bar chart created successfully"
"""
        else:
            code = """
# Default visualization - data overview
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=['number'])
if len(numeric_data.columns) >= 2:
    plt.subplot(2, 2, 1)
    plt.hist(numeric_data.iloc[:, 0], alpha=0.7)
    plt.title(f'Distribution of {numeric_data.columns[0]}')
    
    plt.subplot(2, 2, 2)
    plt.hist(numeric_data.iloc[:, 1], alpha=0.7)
    plt.title(f'Distribution of {numeric_data.columns[1]}')
    
    plt.subplot(2, 2, 3)
    plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], alpha=0.6)
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])
    plt.title('Scatter plot')
    
    plt.subplot(2, 2, 4)
    correlation_matrix = numeric_data.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
result = "Data overview visualization created"
"""
        
        return self._format_code_response("visualization", code, parameters)
    
    def _generate_filtering_code(self, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for data filtering operations."""
        condition = parameters.get("condition", "")
        columns = parameters.get("columns", [])
        
        if condition:
            code = f"""
# Filter data based on condition
filtered_data = data.query('{condition}')
print(f"Original data shape: {{data.shape}}")
print(f"Filtered data shape: {{filtered_data.shape}}")
print("\\nFiltered data preview:")
print(filtered_data.head())
result = filtered_data
"""
        elif columns:
            cols_str = str(columns)
            code = f"""
# Select specific columns
selected_columns = {cols_str}
filtered_data = data[selected_columns]
print(f"Selected columns: {cols_str}")
print("\\nSelected data preview:")
print(filtered_data.head())
result = filtered_data
"""
        else:
            code = """
# Show data info and basic filtering options
print("Data shape:", data.shape)
print("\\nColumns:", list(data.columns))
print("\\nData types:")
print(data.dtypes)
print("\\nFirst 5 rows:")
print(data.head())
result = data.head()
"""
        
        return self._format_code_response("filtering", code, parameters)
    
    def _generate_aggregation_code(self, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for data aggregation operations."""
        group_by = parameters.get("group_by", [])
        agg_functions = parameters.get("agg_functions", ["count"])
        
        if group_by:
            group_cols = str(group_by)
            agg_funcs = str(agg_functions)
            code = f"""
# Group by analysis
group_columns = {group_cols}
agg_functions = {agg_funcs}

grouped_data = data.groupby(group_columns).agg(agg_functions)
print("Grouped analysis results:")
print(grouped_data)
result = grouped_data
"""
        else:
            code = """
# Basic aggregation analysis
print("Dataset aggregation summary:")
print("\\nNumeric columns summary:")
numeric_data = data.select_dtypes(include=['number'])
print(numeric_data.agg(['count', 'mean', 'median', 'min', 'max']))

print("\\nCategorical columns value counts:")
categorical_data = data.select_dtypes(include=['object'])
for col in categorical_data.columns[:3]:  # Limit to first 3 columns
    print(f"\\n{col}:")
    print(data[col].value_counts().head())
    
result = numeric_data.agg(['count', 'mean', 'median', 'min', 'max'])
"""
        
        return self._format_code_response("aggregation", code, parameters)
    
    def _generate_transformation_code(self, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for data transformation operations."""
        operation = parameters.get("operation", "normalize")
        columns = parameters.get("columns", [])
        
        if operation == "normalize" and columns:
            cols_str = str(columns)
            code = f"""
# Normalize specified columns
from sklearn.preprocessing import StandardScaler

columns_to_normalize = {cols_str}
scaler = StandardScaler()
data_normalized = data.copy()
data_normalized[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

print("Normalization completed")
print("\\nOriginal data statistics:")
print(data[columns_to_normalize].describe())
print("\\nNormalized data statistics:")
print(data_normalized[columns_to_normalize].describe())
result = data_normalized
"""
        else:
            code = """
# Basic data transformation info
print("Data transformation options:")
print("\\nNumeric columns (suitable for scaling/normalization):")
numeric_cols = data.select_dtypes(include=['number']).columns
print(list(numeric_cols))

print("\\nCategorical columns (suitable for encoding):")
categorical_cols = data.select_dtypes(include=['object']).columns
print(list(categorical_cols))

print("\\nMissing values per column:")
print(data.isnull().sum())
result = {"numeric_columns": list(numeric_cols), "categorical_columns": list(categorical_cols)}
"""
        
        return self._format_code_response("transformation", code, parameters)
    
    def _generate_generic_code(self, task_type: str, parameters: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic code for unknown task types."""
        code = f"""
# Generic data analysis for task: {task_type}
print("Data overview:")
print("Shape:", data.shape)
print("\\nColumns:", list(data.columns))
print("\\nData types:")
print(data.dtypes)
print("\\nFirst 5 rows:")
print(data.head())
print("\\nBasic statistics:")
print(data.describe())
result = data.describe()
"""
        return self._format_code_response(task_type, code, parameters)
    
    def _format_code_response(self, task_type: str, code: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Format the code generation response."""
        # Add imports to the beginning
        full_code = "\\n".join(self.imports) + "\\n\\n" + textwrap.dedent(code).strip()
        
        return {
            "task_type": task_type,
            "code": full_code,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "imports_included": True,
            "validation": self.validate_code(full_code)
        }
    
    def _check_code_safety(self, code: str) -> List[Dict[str, str]]:
        """Check code for potential safety issues."""
        warnings = []
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            ("import os", "OS operations detected"),
            ("import subprocess", "Subprocess operations detected"),
            ("exec(", "Dynamic code execution detected"),
            ("eval(", "Dynamic code evaluation detected"),
            ("open(", "File operations detected"),
            ("__import__", "Dynamic imports detected")
        ]
        
        for pattern, message in dangerous_patterns:
            if pattern in code:
                warnings.append({
                    "type": "security",
                    "message": message,
                    "severity": "medium"
                })
        
        return warnings
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for common operations."""
        return {
            "basic_stats": """
data.describe()
""",
            "correlation": """
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
""",
            "value_counts": """
data['{column}'].value_counts()
""",
            "histogram": """
plt.figure(figsize=(10, 6))
plt.hist(data['{column}'], bins=30)
plt.title('Distribution of {column}')
plt.show()
""",
            "scatter_plot": """
plt.figure(figsize=(10, 6))
plt.scatter(data['{x_column}'], data['{y_column}'])
plt.xlabel('{x_column}')
plt.ylabel('{y_column}')
plt.title('{y_column} vs {x_column}')
plt.show()
"""
        }
    
    def get_code_suggestions(self, query: str, data_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get code suggestions based on user query and data characteristics.
        
        Args:
            query: Natural language query
            data_info: Information about the dataset
            
        Returns:
            List of suggested code operations
        """
        suggestions = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["average", "mean"]):
            suggestions.append({
                "operation": "statistics",
                "description": "Calculate average/mean values",
                "code_snippet": "data.mean()"
            })
        
        if any(word in query_lower for word in ["plot", "chart", "visualize"]):
            suggestions.append({
                "operation": "visualization", 
                "description": "Create visualizations",
                "code_snippet": "plt.scatter(data['x'], data['y'])"
            })
        
        if any(word in query_lower for word in ["filter", "where", "subset"]):
            suggestions.append({
                "operation": "filtering",
                "description": "Filter the dataset",
                "code_snippet": "data[data['column'] > value]"
            })
        
        if any(word in query_lower for word in ["group", "by", "aggregate"]):
            suggestions.append({
                "operation": "aggregation",
                "description": "Group and aggregate data",
                "code_snippet": "data.groupby('column').agg('function')"
            })
        
        return suggestions 