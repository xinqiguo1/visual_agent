"""
Insight Generator Agent

Agent responsible for automatically discovering patterns and insights in data.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class InsightGenerator:
    """
    Agent responsible for automatically discovering insights from data.
    """
    
    def __init__(self):
        """Initialize the insight generator agent."""
        self.insight_types = [
            "data_quality",
            "statistical_summary", 
            "correlations",
            "outliers",
            "distributions",
            "trends",
            "patterns"
        ]
        self.min_correlation_threshold = 0.3
        self.outlier_threshold = 2.0
        
    def generate_insights(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive insights from a dataset.
        
        Args:
            data: DataFrame to analyze
            target_column: Optional target column for focused analysis
            
        Returns:
            Dictionary with categorized insights
        """
        insights = {
            "data_overview": self._analyze_data_overview(data),
            "data_quality": self._analyze_data_quality(data),
            "statistical_insights": self._analyze_statistical_patterns(data),
            "correlations": self._analyze_correlations(data),
            "outliers": self._detect_outliers(data),
            "distributions": self._analyze_distributions(data),
            "recommendations": self._generate_recommendations(data),
            "summary": ""
        }
        
        # Generate summary
        insights["summary"] = self._create_insight_summary(insights)
        
        return insights
    
    def _analyze_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic data characteristics."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        return {
            "shape": data.shape,
            "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024**2, 2),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "total_missing_values": data.isnull().sum().sum(),
            "missing_percentage": round((data.isnull().sum().sum() / data.size) * 100, 2)
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data quality issues and provide insights."""
        quality_insights = []
        
        # Missing values analysis
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            high_missing_cols = missing_counts[missing_counts > len(data) * 0.1].index
            if len(high_missing_cols) > 0:
                quality_insights.append({
                    "type": "missing_data",
                    "severity": "high",
                    "message": f"Columns with >10% missing values: {list(high_missing_cols)}",
                    "recommendation": "Consider imputation or removal of these columns"
                })
        
        # Duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            quality_insights.append({
                "type": "duplicates",
                "severity": "medium",
                "message": f"Found {duplicate_count} duplicate rows ({duplicate_count/len(data)*100:.1f}%)",
                "recommendation": "Review and consider removing duplicate entries"
            })
        
        # Data type consistency
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for mixed types
                try:
                    pd.to_numeric(data[col], errors='raise')
                    quality_insights.append({
                        "type": "data_type",
                        "severity": "low",
                        "message": f"Column '{col}' contains numeric data but stored as text",
                        "recommendation": f"Consider converting '{col}' to numeric type"
                    })
                except:
                    pass
        
        return quality_insights
    
    def _analyze_statistical_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze statistical patterns in the data."""
        insights = []
        numeric_data = data.select_dtypes(include=['number'])
        
        if len(numeric_data.columns) == 0:
            return [{"message": "No numeric columns found for statistical analysis"}]
        
        # Analyze each numeric column
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Basic statistics
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            # Skewness analysis
            skewness = stats.skew(col_data)
            if abs(skewness) > 1:
                skew_type = "highly skewed" if abs(skewness) > 2 else "moderately skewed"
                direction = "right" if skewness > 0 else "left"
                insights.append({
                    "type": "distribution",
                    "column": col,
                    "message": f"{col} is {skew_type} to the {direction} (skewness: {skewness:.2f})",
                    "recommendation": "Consider transformation for normalization"
                })
            
            # Variability analysis
            cv = std_val / mean_val if mean_val != 0 else 0
            if cv > 1:
                insights.append({
                    "type": "variability",
                    "column": col,
                    "message": f"{col} shows high variability (CV: {cv:.2f})",
                    "recommendation": "High variance may affect model performance"
                })
        
        return insights
    
    def _analyze_correlations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze correlations between numeric variables."""
        insights = []
        numeric_data = data.select_dtypes(include=['number'])
        
        if len(numeric_data.columns) < 2:
            return [{"message": "Need at least 2 numeric columns for correlation analysis"}]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > self.min_correlation_threshold:
                    strong_correlations.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_val,
                        "strength": self._correlation_strength(abs(corr_val))
                    })
        
        if strong_correlations:
            # Sort by absolute correlation value
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            for corr_info in strong_correlations[:5]:  # Top 5 correlations
                direction = "positive" if corr_info["correlation"] > 0 else "negative"
                insights.append({
                    "type": "correlation",
                    "message": f"Strong {direction} correlation between {corr_info['var1']} and {corr_info['var2']} ({corr_info['correlation']:.3f})",
                    "strength": corr_info["strength"],
                    "variables": [corr_info["var1"], corr_info["var2"]]
                })
        
        return insights
    
    def _detect_outliers(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect outliers in numeric columns."""
        insights = []
        numeric_data = data.select_dtypes(include=['number'])
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) < 10:  # Skip if too few data points
                continue
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_percentage = len(outliers) / len(col_data) * 100
            
            if outlier_percentage > 5:  # More than 5% outliers
                insights.append({
                    "type": "outliers",
                    "column": col,
                    "count": len(outliers),
                    "percentage": round(outlier_percentage, 1),
                    "message": f"{col} has {len(outliers)} outliers ({outlier_percentage:.1f}% of data)",
                    "recommendation": "Consider investigating or treating these outliers"
                })
        
        return insights
    
    def _analyze_distributions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze distributions of variables."""
        insights = []
        
        # Numeric distributions
        numeric_data = data.select_dtypes(include=['number'])
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) < 10:
                continue
            
            # Test for normality
            _, p_value = stats.normaltest(col_data)
            if p_value > 0.05:
                insights.append({
                    "type": "distribution",
                    "column": col,
                    "message": f"{col} appears to be normally distributed (p={p_value:.3f})",
                    "distribution_type": "normal"
                })
            else:
                insights.append({
                    "type": "distribution", 
                    "column": col,
                    "message": f"{col} is not normally distributed (p={p_value:.3f})",
                    "distribution_type": "non-normal"
                })
        
        # Categorical distributions
        categorical_data = data.select_dtypes(include=['object'])
        for col in categorical_data.columns:
            value_counts = data[col].value_counts()
            if len(value_counts) > 1:
                # Check for imbalance
                max_freq = value_counts.iloc[0]
                total_count = len(data[col].dropna())
                max_percentage = (max_freq / total_count) * 100
                
                if max_percentage > 80:
                    insights.append({
                        "type": "imbalance",
                        "column": col,
                        "message": f"{col} is highly imbalanced - {value_counts.index[0]} represents {max_percentage:.1f}% of data",
                        "recommendation": "Consider balancing techniques if using for modeling"
                    })
        
        return insights
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Data size recommendations
        if len(data) < 100:
            recommendations.append("Dataset is small (<100 rows). Consider collecting more data for reliable analysis.")
        elif len(data) > 100000:
            recommendations.append("Large dataset detected. Consider sampling for exploratory analysis.")
        
        # Column recommendations
        if len(data.columns) > 50:
            recommendations.append("High number of features. Consider feature selection or dimensionality reduction.")
        
        # Missing data recommendations
        missing_percentage = (data.isnull().sum().sum() / data.size) * 100
        if missing_percentage > 20:
            recommendations.append("Significant missing data (>20%). Implement comprehensive imputation strategy.")
        elif missing_percentage > 5:
            recommendations.append("Some missing data detected. Consider imputation methods.")
        
        # Data type recommendations
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            recommendations.append(f"Consider encoding categorical variables: {list(object_cols)[:3]}")
        
        return recommendations
    
    def _create_insight_summary(self, insights: Dict[str, Any]) -> str:
        """Create a natural language summary of insights."""
        summary_parts = []
        
        # Data overview
        overview = insights["data_overview"]
        summary_parts.append(f"Dataset contains {overview['shape'][0]} rows and {overview['shape'][1]} columns")
        
        # Data quality
        quality_issues = len(insights["data_quality"])
        if quality_issues > 0:
            summary_parts.append(f"Found {quality_issues} data quality issues requiring attention")
        
        # Correlations
        correlations = insights["correlations"]
        if correlations and len(correlations) > 0:
            summary_parts.append(f"Discovered {len(correlations)} significant correlations between variables")
        
        # Outliers
        outlier_cols = [insight for insight in insights["outliers"] if insight.get("type") == "outliers"]
        if outlier_cols:
            summary_parts.append(f"Detected outliers in {len(outlier_cols)} columns")
        
        return ". ".join(summary_parts) + "."
    
    def _correlation_strength(self, abs_corr: float) -> str:
        """Categorize correlation strength."""
        if abs_corr >= 0.7:
            return "very strong"
        elif abs_corr >= 0.5:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        else:
            return "weak"
    
    def suggest_next_analysis(self, insights: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """Suggest next analysis steps based on insights."""
        suggestions = []
        
        # Based on correlations
        if insights["correlations"]:
            suggestions.append("Create scatter plots to visualize strong correlations")
            suggestions.append("Investigate causal relationships between correlated variables")
        
        # Based on outliers
        outlier_insights = [i for i in insights["outliers"] if i.get("type") == "outliers"]
        if outlier_insights:
            suggestions.append("Investigate outlier patterns and consider outlier treatment")
            suggestions.append("Create box plots to visualize outlier distributions")
        
        # Based on distributions
        non_normal = [i for i in insights["distributions"] if i.get("distribution_type") == "non-normal"]
        if non_normal:
            suggestions.append("Consider data transformations for non-normal distributions")
            suggestions.append("Create histograms to visualize distribution shapes")
        
        # Based on data quality
        if insights["data_quality"]:
            suggestions.append("Address data quality issues before proceeding with analysis")
            suggestions.append("Implement data cleaning and preprocessing steps")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_automated_insights(self, data: pd.DataFrame, top_n: int = 10) -> List[str]:
        """Get top automated insights in natural language."""
        all_insights = self.generate_insights(data)
        
        # Collect all insight messages
        insight_messages = []
        
        # Add quality insights
        for insight in all_insights["data_quality"]:
            insight_messages.append(insight["message"])
        
        # Add statistical insights
        for insight in all_insights["statistical_insights"]:
            insight_messages.append(insight["message"])
        
        # Add correlation insights
        for insight in all_insights["correlations"]:
            insight_messages.append(insight["message"])
        
        # Add outlier insights
        for insight in all_insights["outliers"]:
            insight_messages.append(insight["message"])
        
        return insight_messages[:top_n] 