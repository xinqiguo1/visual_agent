"""
Visualizer Agent

Agent responsible for creating charts, plots, and other visualizations
based on data and user requests.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import time
from pathlib import Path

# Import the new web visualizer
from .web_visualizer import WebVisualizer


class Visualizer:
    """
    Agent responsible for generating visualizations from data.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the visualizer agent."""
        self.available_chart_types = [
            "bar", "line", "scatter", "histogram", "box", 
            "heatmap", "pie", "area", "violin", "sunburst"
        ]
        self.default_style = "seaborn"
        self.color_palette = "viridis"
        
        # Initialize web visualizer for enhanced capabilities
        self.web_visualizer = WebVisualizer(output_dir=output_dir)
        
        # Set up outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_chart(
        self, 
        data: pd.DataFrame, 
        chart_type: str, 
        x_column: str = None,
        y_column: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chart based on the specified parameters.
        
        Args:
            data: DataFrame containing the data to plot
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            **kwargs: Additional chart parameters
            
        Returns:
            Dictionary with chart information and encoded image
        """
        try:
            if chart_type not in self.available_chart_types:
                return {
                    "error": f"Chart type '{chart_type}' not supported.",
                    "available_types": self.available_chart_types
                }
            
            # Set style
            plt.style.use(self.default_style)
            
            # Create the appropriate chart
            if chart_type == "bar":
                chart_data = self._create_bar_chart(data, x_column, y_column, **kwargs)
            elif chart_type == "line":
                chart_data = self._create_line_chart(data, x_column, y_column, **kwargs)
            elif chart_type == "scatter":
                chart_data = self._create_scatter_plot(data, x_column, y_column, **kwargs)
            elif chart_type == "histogram":
                chart_data = self._create_histogram(data, x_column, **kwargs)
            elif chart_type == "box":
                chart_data = self._create_box_plot(data, x_column, y_column, **kwargs)
            elif chart_type == "heatmap":
                chart_data = self._create_heatmap(data, **kwargs)
            elif chart_type == "pie":
                chart_data = self._create_pie_chart(data, x_column, **kwargs)
            else:
                chart_data = self._create_default_chart(data, x_column, y_column)
            
            return chart_data
            
        except Exception as e:
            return {
                "error": f"Error creating chart: {str(e)}",
                "chart_type": chart_type
            }
    
    def create_web_chart(
        self, 
        data: pd.DataFrame, 
        chart_type: str, 
        x_column: str = None,
        y_column: str = None,
        color_column: str = None,
        title: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a web-ready interactive chart using Plotly.
        
        Args:
            data: DataFrame containing the data to plot
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column to use for coloring
            title: Custom title for the chart
            **kwargs: Additional chart parameters
            
        Returns:
            Dictionary with chart information, HTML file path, and web-ready formats
        """
        try:
            # Use the web visualizer for interactive charts
            result = self.web_visualizer.create_web_chart(
                data=data,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                title=title,
                **kwargs
            )
            
            # Add backward compatibility fields
            if result.get("success"):
                result["file_path"] = result["html_path"]
                result["interactive"] = True
                result["format"] = "html"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error creating web chart: {str(e)}",
                "chart_type": chart_type,
                "success": False
            }
    
    def create_chart_with_options(
        self, 
        data: pd.DataFrame, 
        chart_type: str,
        x_column: str = None,
        y_column: str = None,
        color_column: str = None,
        output_format: str = "web",  # "web", "static", "both"
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create chart with multiple output format options.
        
        Args:
            data: DataFrame containing the data to plot
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column to use for coloring
            output_format: Format to output ("web", "static", "both")
            **kwargs: Additional chart parameters
            
        Returns:
            Dictionary with chart information and outputs
        """
        results = {"chart_type": chart_type, "outputs": {}}
        
        try:
            if output_format in ["web", "both"]:
                # Create interactive web chart
                web_result = self.create_web_chart(
                    data, chart_type, x_column, y_column, color_column, **kwargs
                )
                results["outputs"]["web"] = web_result
                
            if output_format in ["static", "both"]:
                # Create static matplotlib chart
                static_result = self.create_chart(
                    data, chart_type, x_column, y_column, **kwargs
                )
                results["outputs"]["static"] = static_result
            
            # Determine overall success
            results["success"] = any(
                output.get("success", False) for output in results["outputs"].values()
            )
            
            return results
            
        except Exception as e:
            return {
                "error": f"Error creating chart with options: {str(e)}",
                "chart_type": chart_type,
                "success": False
            }
    
    def suggest_visualizations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations based on data characteristics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of suggested visualizations
        """
        suggestions = []
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Suggest based on data types
        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": "histogram",
                "description": "Distribution of numeric variables",
                "columns": list(numeric_cols)[:3]  # Limit to first 3
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "description": "Relationship between numeric variables",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1]
            })
            
            suggestions.append({
                "type": "heatmap",
                "description": "Correlation matrix of numeric variables",
                "columns": list(numeric_cols)
            })
        
        if len(categorical_cols) >= 1:
            suggestions.append({
                "type": "bar",
                "description": "Count of categorical variables",
                "columns": list(categorical_cols)[:2]
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": "box",
                "description": "Distribution of numeric variable by category",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0]
            })
        
        return suggestions
    
    def suggest_web_visualizations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Suggest web-optimized interactive visualizations.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of suggested web visualizations
        """
        return self.web_visualizer.suggest_web_visualizations(data)
    
    def create_dashboard(self, charts: List[Dict[str, Any]], title: str = "Data Analysis Dashboard") -> str:
        """
        Create a multi-chart dashboard.
        
        Args:
            charts: List of chart result dictionaries
            title: Dashboard title
            
        Returns:
            HTML string for complete dashboard
        """
        return self.web_visualizer.create_dashboard(charts, title)
    
    def _create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str = None, **kwargs) -> Dict[str, Any]:
        """Create a bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if y_col is None:
            # Count plot
            data[x_col].value_counts().plot(kind='bar', ax=ax, color=self.color_palette)
            ax.set_title(f'Count of {x_col}')
            ax.set_ylabel('Count')
        else:
            # Bar plot with specific y values
            data.plot(x=x_col, y=y_col, kind='bar', ax=ax, color=self.color_palette)
            ax.set_title(f'{y_col} by {x_col}')
        
        ax.set_xlabel(x_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._save_plot_to_dict("bar", x_col, y_col)
    
    def _create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> Dict[str, Any]:
        """Create a line chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data.plot(x=x_col, y=y_col, kind='line', ax=ax, color=self.color_palette)
        ax.set_title(f'{y_col} over {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        
        return self._save_plot_to_dict("line", x_col, y_col)
    
    def _create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> Dict[str, Any]:
        """Create a scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(data[x_col], data[y_col], alpha=0.6, c=self.color_palette)
        ax.set_title(f'{y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        
        return self._save_plot_to_dict("scatter", x_col, y_col)
    
    def _create_histogram(self, data: pd.DataFrame, x_col: str, **kwargs) -> Dict[str, Any]:
        """Create a histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data[x_col].hist(bins=kwargs.get('bins', 30), ax=ax, color=self.color_palette, alpha=0.7)
        ax.set_title(f'Distribution of {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        
        return self._save_plot_to_dict("histogram", x_col)
    
    def _create_box_plot(self, data: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> Dict[str, Any]:
        """Create a box plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data.boxplot(column=y_col, by=x_col, ax=ax)
        ax.set_title(f'{y_col} by {x_col}')
        plt.tight_layout()
        
        return self._save_plot_to_dict("box", x_col, y_col)
    
    def _create_heatmap(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create a correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        numeric_data = data.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        
        return self._save_plot_to_dict("heatmap")
    
    def _create_pie_chart(self, data: pd.DataFrame, x_col: str, **kwargs) -> Dict[str, Any]:
        """Create a pie chart."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        value_counts = data[x_col].value_counts()
        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        ax.set_title(f'Distribution of {x_col}')
        
        return self._save_plot_to_dict("pie", x_col)
    
    def _create_default_chart(self, data: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Create a default chart when specific type fails."""
        return self._create_scatter_plot(data, x_col, y_col)
    
    def _save_plot_to_dict(self, chart_type: str, x_col: str = None, y_col: str = None) -> Dict[str, Any]:
        """Save the current plot to a base64 encoded string."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()  # Close the figure to free memory
        
        return {
            "chart_type": chart_type,
            "image_base64": image_base64,
            "x_column": x_col,
            "y_column": y_col,
            "success": True
        }
    
    def create_interactive_chart(
        self, 
        data: pd.DataFrame, 
        chart_type: str,
        x_column: str = None,
        y_column: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an interactive chart using Plotly.
        
        Args:
            data: DataFrame containing the data
            chart_type: Type of interactive chart
            x_column: Column for x-axis
            y_column: Column for y-axis
            
        Returns:
            Dictionary with Plotly chart JSON
        """
        try:
            if chart_type == "scatter":
                fig = px.scatter(data, x=x_column, y=y_column, title=f'{y_column} vs {x_column}')
            elif chart_type == "line":
                fig = px.line(data, x=x_column, y=y_column, title=f'{y_column} over {x_column}')
            elif chart_type == "bar":
                if y_column:
                    fig = px.bar(data, x=x_column, y=y_column, title=f'{y_column} by {x_column}')
                else:
                    value_counts = data[x_column].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f'Count of {x_column}')
            elif chart_type == "histogram":
                fig = px.histogram(data, x=x_column, title=f'Distribution of {x_column}')
            else:
                fig = px.scatter(data, x=x_column, y=y_column, title="Default Interactive Chart")
            
            return {
                "chart_type": f"interactive_{chart_type}",
                "plotly_json": fig.to_json(),
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Error creating interactive chart: {str(e)}",
                "chart_type": chart_type
            }
    
    def get_chart_recommendations(self, query: str, data: pd.DataFrame) -> List[str]:
        """
        Get chart type recommendations based on user query.
        
        Args:
            query: User's natural language query
            data: Available data
            
        Returns:
            List of recommended chart types
        """
        query_lower = query.lower()
        recommendations = []
        
        if any(word in query_lower for word in ["trend", "over time", "timeline", "change"]):
            recommendations.append("line")
        
        if any(word in query_lower for word in ["compare", "comparison", "vs", "versus"]):
            recommendations.append("bar")
        
        if any(word in query_lower for word in ["relationship", "correlation", "against"]):
            recommendations.append("scatter")
        
        if any(word in query_lower for word in ["distribution", "spread", "histogram"]):
            recommendations.append("histogram")
        
        if any(word in query_lower for word in ["proportion", "percentage", "share"]):
            recommendations.append("pie")
        
        # Default recommendations if none match
        if not recommendations:
            suggestions = self.suggest_visualizations(data)
            recommendations = [s["type"] for s in suggestions[:2]]
        
        return recommendations[:3]  # Return top 3 recommendations 