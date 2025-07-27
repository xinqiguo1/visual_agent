"""
Web Visualizer Agent

Enhanced visualizer optimized for web display using Plotly.
Creates interactive charts that can be embedded in web pages.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class WebVisualizer:
    """
    Enhanced visualizer that creates interactive, web-ready charts using Plotly.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the web visualizer.
        
        Args:
            output_dir: Directory to save HTML files and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default theme for web
        pio.templates.default = "plotly_white"
        
        # Web-optimized color schemes
        self.color_schemes = {
            'default': px.colors.qualitative.Set2,
            'professional': px.colors.qualitative.Safe,
            'vibrant': px.colors.qualitative.Vivid,
            'minimal': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        # Chart type mapping
        self.chart_creators = {
            'scatter': self._create_scatter,
            'line': self._create_line,
            'bar': self._create_bar,
            'histogram': self._create_histogram,
            'box': self._create_box,
            'heatmap': self._create_heatmap,
            'pie': self._create_pie,
            'violin': self._create_violin,
            'area': self._create_area,
            'sunburst': self._create_sunburst
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
        Create a web-ready interactive chart.
        
        Args:
            data: DataFrame containing the data
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column to use for coloring
            title: Custom title for the chart
            **kwargs: Additional chart parameters
            
        Returns:
            Dictionary with chart data and web-ready formats
        """
        try:
            # Auto-select columns if not specified
            if x_column is None or y_column is None:
                x_column, y_column, color_column = self._auto_select_columns(data, chart_type, x_column, y_column, color_column)
            
            # Get chart creator function
            chart_creator = self.chart_creators.get(chart_type, self._create_scatter)
            
            # Create the chart
            fig = chart_creator(data, x_column, y_column, color_column, title, **kwargs)
            
            # Apply web optimizations
            fig = self._optimize_for_web(fig)
            
            # Generate outputs
            timestamp = int(time.time())
            chart_id = f"{chart_type}_{timestamp}"
            
            # Save HTML file
            html_filename = f"chart_{chart_id}.html"
            html_path = self.output_dir / html_filename
            fig.write_html(html_path)
            
            return {
                "success": True,
                "chart_type": chart_type,
                "chart_id": chart_id,
                "html_path": str(html_path),
                "html_content": fig.to_html(include_plotlyjs='cdn'),
                "json_data": fig.to_json(),
                "div_content": fig.to_html(div_id=f"chart-{chart_id}", include_plotlyjs=False),
                "config": self._get_chart_config(),
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
                "title": title or self._generate_title(chart_type, x_column, y_column)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "chart_type": chart_type,
                "fallback_html": f"<div class='error'>Error creating chart: {str(e)}</div>"
            }
    
    def _create_scatter(self, data: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive scatter plot."""
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title or f"{y_col} vs {x_col}",
            hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_line(self, data: pd.DataFrame, x_col: str, y_col: str, 
                     color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive line chart."""
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title or f"{y_col} over {x_col}",
            hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_bar(self, data: pd.DataFrame, x_col: str, y_col: str = None, 
                    color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive bar chart."""
        if y_col is None:
            # Count plot
            counts = data[x_col].value_counts()
            fig = px.bar(
                x=counts.index, 
                y=counts.values,
                title=title or f"Count of {x_col}",
                labels={'x': x_col, 'y': 'Count'},
                color_discrete_sequence=self.color_schemes['default']
            )
        else:
            fig = px.bar(
                data, 
                x=x_col, 
                y=y_col, 
                color=color_col,
                title=title or f"{y_col} by {x_col}",
                hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
                color_discrete_sequence=self.color_schemes['default']
            )
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, x_col: str, y_col: str = None, 
                         color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive histogram."""
        fig = px.histogram(
            data, 
            x=x_col, 
            color=color_col,
            title=title or f"Distribution of {x_col}",
            nbins=kwargs.get('bins', 30),
            hover_data=self._get_hover_data(data, [x_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_box(self, data: pd.DataFrame, x_col: str, y_col: str, 
                    color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive box plot."""
        fig = px.box(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title or f"{y_col} by {x_col}",
            hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, x_col: str = None, y_col: str = None, 
                       color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive correlation heatmap."""
        numeric_data = data.select_dtypes(include=['number'])
        if len(numeric_data.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for heatmap")
        
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title or "Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto",
            labels=dict(color="Correlation")
        )
        
        # Add correlation values as text
        fig.update_traces(
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        return fig
    
    def _create_pie(self, data: pd.DataFrame, x_col: str, y_col: str = None, 
                    color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive pie chart."""
        if y_col is None:
            # Count-based pie chart
            counts = data[x_col].value_counts()
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=title or f"Distribution of {x_col}",
                color_discrete_sequence=self.color_schemes['default']
            )
        else:
            # Value-based pie chart
            fig = px.pie(
                data,
                values=y_col,
                names=x_col,
                title=title or f"{y_col} by {x_col}",
                color_discrete_sequence=self.color_schemes['default']
            )
        return fig
    
    def _create_violin(self, data: pd.DataFrame, x_col: str, y_col: str, 
                      color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive violin plot."""
        fig = px.violin(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title or f"{y_col} distribution by {x_col}",
            hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_area(self, data: pd.DataFrame, x_col: str, y_col: str, 
                     color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive area chart."""
        fig = px.area(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title or f"{y_col} area over {x_col}",
            hover_data=self._get_hover_data(data, [x_col, y_col, color_col]),
            color_discrete_sequence=self.color_schemes['default']
        )
        return fig
    
    def _create_sunburst(self, data: pd.DataFrame, x_col: str, y_col: str = None, 
                        color_col: str = None, title: str = None, **kwargs) -> go.Figure:
        """Create an interactive sunburst chart."""
        if y_col is None:
            # Single level sunburst
            counts = data[x_col].value_counts()
            fig = px.sunburst(
                names=counts.index,
                values=counts.values,
                title=title or f"Sunburst of {x_col}"
            )
        else:
            # Multi-level sunburst
            fig = px.sunburst(
                data,
                path=[x_col, y_col] if color_col is None else [x_col, y_col, color_col],
                title=title or f"Hierarchical view of {x_col} and {y_col}"
            )
        return fig
    
    def _optimize_for_web(self, fig: go.Figure) -> go.Figure:
        """Optimize chart for web display."""
        fig.update_layout(
            # Responsive design
            autosize=True,
            margin=dict(l=60, r=60, t=60, b=60),
            
            # Web-friendly fonts
            font=dict(family="Arial, sans-serif", size=12),
            
            # Interactive features
            hovermode='closest',
            
            # Clean appearance
            plot_bgcolor='white',
            paper_bgcolor='white',
            
            # Mobile responsive
            height=500,
            
            # Toolbar configuration
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes for better web display
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _get_hover_data(self, data: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
        """Get relevant columns for hover information."""
        exclude_cols = [col for col in exclude_cols if col is not None]
        hover_cols = []
        
        for col in data.columns:
            if col not in exclude_cols and len(hover_cols) < 5:
                if data[col].dtype in ['object', 'int64', 'float64', 'bool']:
                    hover_cols.append(col)
        
        return hover_cols
    
    def _generate_title(self, chart_type: str, x_col: str, y_col: str) -> str:
        """Generate a default title for the chart."""
        if chart_type == 'scatter':
            return f"{y_col} vs {x_col}" if y_col else f"Scatter plot of {x_col}"
        elif chart_type == 'line':
            return f"{y_col} over {x_col}" if y_col else f"Line chart of {x_col}"
        elif chart_type == 'bar':
            return f"{y_col} by {x_col}" if y_col else f"Count of {x_col}"
        elif chart_type == 'histogram':
            return f"Distribution of {x_col}"
        elif chart_type == 'heatmap':
            return "Correlation Heatmap"
        elif chart_type == 'pie':
            return f"Distribution of {x_col}"
        else:
            return f"{chart_type.title()} Chart"
    
    def _get_chart_config(self) -> Dict[str, Any]:
        """Get configuration for chart display."""
        return {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": 500,
                "width": 800,
                "scale": 1
            }
        }
    
    def create_dashboard(self, charts: List[Dict[str, Any]], title: str = "Data Analysis Dashboard") -> str:
        """
        Create a multi-chart dashboard.
        
        Args:
            charts: List of chart result dictionaries
            title: Dashboard title
            
        Returns:
            HTML string for complete dashboard
        """
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }}
                .chart {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }}
                .error {{ color: #d32f2f; padding: 20px; background: #ffebee; border-radius: 5px; }}
                .success {{ color: #2e7d32; padding: 10px; background: #e8f5e8; border-radius: 5px; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>📊 {title}</h1>
                    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="charts">
        """
        
        for i, chart in enumerate(charts):
            if chart.get("success"):
                dashboard_html += f"""
                <div class="chart">
                    <div class="chart-title">{chart.get('title', f'Chart {i+1}')}</div>
                    <div class="success">✅ {chart.get('chart_type', 'chart').title()} created successfully</div>
                    <div id="chart-{i}"></div>
                    <script>
                        Plotly.newPlot('chart-{i}', {chart['json_data']}, {{}}, {json.dumps(chart.get('config', {}))});
                    </script>
                </div>
                """
            else:
                dashboard_html += f"""
                <div class="chart">
                    <div class="chart-title">Chart {i+1} - Error</div>
                    <div class="error">❌ {chart.get('error', 'Unknown error')}</div>
                </div>
                """
        
        dashboard_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_path = self.output_dir / f"dashboard_{int(time.time())}.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return dashboard_html
    
    def _auto_select_columns(self, data: pd.DataFrame, chart_type: str, 
                           x_column: str = None, y_column: str = None, 
                           color_column: str = None) -> tuple:
        """
        Auto-select appropriate columns for chart creation.
        
        Args:
            data: DataFrame containing the data
            chart_type: Type of chart to create
            x_column: Existing x_column (if any)
            y_column: Existing y_column (if any)
            color_column: Existing color_column (if any)
            
        Returns:
            Tuple of (x_column, y_column, color_column)
        """
        numeric_columns = list(data.select_dtypes(include=['number']).columns)
        categorical_columns = list(data.select_dtypes(include=['object', 'category']).columns)
        all_columns = list(data.columns)
        
        # Auto-select based on chart type
        if chart_type == "histogram":
            # For histogram, we only need x_column (numeric preferred)
            if x_column is None:
                x_column = numeric_columns[0] if numeric_columns else all_columns[0]
            return x_column, None, color_column
        
        elif chart_type == "pie":
            # For pie chart, we need a categorical column
            if x_column is None:
                x_column = categorical_columns[0] if categorical_columns else all_columns[0]
            return x_column, None, color_column
        
        elif chart_type == "heatmap":
            # For heatmap, use all numeric columns
            return None, None, None
        
        else:
            # For most charts (scatter, line, bar, etc.), we need x and y
            if x_column is None:
                # Prefer categorical for x-axis in bar charts, numeric for others
                if chart_type == "bar" and categorical_columns:
                    x_column = categorical_columns[0]
                else:
                    x_column = numeric_columns[0] if numeric_columns else all_columns[0]
            
            if y_column is None:
                # Prefer numeric for y-axis
                available_numeric = [col for col in numeric_columns if col != x_column]
                if available_numeric:
                    y_column = available_numeric[0]
                else:
                    # Fallback to any column that's not x_column
                    available_cols = [col for col in all_columns if col != x_column]
                    y_column = available_cols[0] if available_cols else None
            
            # Auto-select color column if not specified
            if color_column is None and categorical_columns:
                available_categorical = [col for col in categorical_columns if col not in [x_column, y_column]]
                if available_categorical:
                    color_column = available_categorical[0]
        
        return x_column, y_column, color_column
    
    def suggest_web_visualizations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate web visualizations based on data characteristics."""
        suggestions = []
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Interactive scatter plots for numeric relationships
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "description": "Interactive scatter plot showing relationships",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1],
                "color_column": categorical_cols[0] if len(categorical_cols) > 0 else None,
                "priority": "high"
            })
        
        # Interactive histograms for distributions
        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": "histogram",
                "description": "Interactive distribution plot",
                "x_column": numeric_cols[0],
                "color_column": categorical_cols[0] if len(categorical_cols) > 0 else None,
                "priority": "medium"
            })
        
        # Interactive bar charts for categories
        if len(categorical_cols) >= 1:
            suggestions.append({
                "type": "bar",
                "description": "Interactive bar chart of categories",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0] if len(numeric_cols) > 0 else None,
                "priority": "medium"
            })
        
        # Correlation heatmap for numeric data
        if len(numeric_cols) >= 3:
            suggestions.append({
                "type": "heatmap",
                "description": "Interactive correlation heatmap",
                "priority": "high"
            })
        
        # Box plots for category vs numeric
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": "box",
                "description": "Interactive box plot showing distributions by category",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0],
                "priority": "medium"
            })
        
        return suggestions 