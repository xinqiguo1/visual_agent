#!/usr/bin/env python3
"""
Visual Analytics Agent - Direct Dashboard

A beautiful, user-friendly interface that connects directly to DataAnalyst,
bypassing the API for better performance and reliability.

Combines Pattern 2 (Progressive Disclosure) and Pattern 3 (Dashboard + Chat):
- Left: Auto-generated dashboard with instant insights
- Right: Conversational interface for specific questions  
- Bottom: Advanced custom analytics and visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from datetime import datetime
from pathlib import Path
import json
import os

# Direct imports from our agents
from src.agents.data_analyst import DataAnalyst
from src.agents.visualizer import Visualizer
from src.agents.web_visualizer import WebVisualizer
from src.agents.insight_generator import InsightGenerator

# Page config
st.set_page_config(
    page_title="Visual Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_analyst' not in st.session_state:
    # Initialize with API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    st.session_state.data_analyst = DataAnalyst(api_key=api_key)
    if api_key:
        st.session_state.using_llm = True
    else:
        st.session_state.using_llm = False
if 'web_visualizer' not in st.session_state:
    st.session_state.web_visualizer = WebVisualizer()
if 'insight_generator' not in st.session_state:
    st.session_state.insight_generator = InsightGenerator()
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'current_dataset_name' not in st.session_state:
    st.session_state.current_dataset_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'auto_insights' not in st.session_state:
    st.session_state.auto_insights = []

def load_dataset_from_file(uploaded_file):
    """Load dataset from uploaded file."""
    try:
        file_name = uploaded_file.name
        
        # Read file based on extension
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, JSON, or Excel files.")
            return None
        
        # Store dataset
        dataset_name = file_name.split('.')[0]
        st.session_state.datasets[dataset_name] = {
            'dataframe': df,
            'filename': file_name,
            'upload_time': datetime.now(),
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        # Set as current dataset
        st.session_state.current_dataset_name = dataset_name
        
        # Load into DataAnalyst
        load_result = st.session_state.data_analyst.load_dataset(df)
        
        return dataset_name, load_result
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def get_current_dataset():
    """Get the currently selected dataset."""
    if not st.session_state.current_dataset_name:
        return None
    return st.session_state.datasets.get(st.session_state.current_dataset_name)

def ask_question_direct(question):
    """Ask a question directly to DataAnalyst."""
    try:
        current_dataset = get_current_dataset()
        if not current_dataset:
            return {"error": "No dataset loaded"}
        
        # Make sure DataAnalyst has the current dataset
        st.session_state.data_analyst.load_dataset(current_dataset['dataframe'])
        
        # Process the question
        result = st.session_state.data_analyst.process_query(question)
        
        # Fix for general type responses - extract actual content from the result
        if result and result.get('type') == 'general':
            # For general type, we need to extract the specific answer
            if 'result' in result and isinstance(result['result'], dict):
                # Extract useful information from the result
                message = result['result'].get('message', '')
                suggestions = result['result'].get('suggestions', [])
                
                # Create a more specific answer instead of the generic message
                if "columns" in question.lower():
                    # Handle column questions
                    if current_dataset and 'dataframe' in current_dataset:
                        columns = list(current_dataset['dataframe'].columns)
                        result['explanation'] = f"The dataset has {len(columns)} columns: {', '.join(columns)}"
                elif "shape" in question.lower() or "size" in question.lower():
                    # Handle shape/size questions
                    if current_dataset and 'dataframe' in current_dataset:
                        df = current_dataset['dataframe']
                        result['explanation'] = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
                elif "summary" in question.lower() or "describe" in question.lower():
                    # Handle summary questions
                    if current_dataset and 'dataframe' in current_dataset:
                        df = current_dataset['dataframe']
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        cat_cols = df.select_dtypes(include=['object']).columns
                        result['explanation'] = f"The dataset has {df.shape[0]} rows, {df.shape[1]} columns ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)."
                else:
                    # For other general questions, provide a more helpful response
                    result['explanation'] = f"I can help you analyze this dataset with {current_dataset['dataframe'].shape[0]} rows and {current_dataset['dataframe'].shape[1]} columns. Try asking about specific columns, statistics, or patterns."
        
        return result
        
    except Exception as e:
        return {"error": f"Error processing question: {str(e)}"}

def generate_insights_direct():
    """Generate insights directly using InsightGenerator."""
    try:
        current_dataset = get_current_dataset()
        if not current_dataset:
            return []
        
        df = current_dataset['dataframe']
        insights = st.session_state.insight_generator.get_automated_insights(df, top_n=5)
        return insights
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return []

def create_visualization_direct(chart_type, x_column=None, y_column=None, color_column=None, title=None):
    """Create visualization directly using WebVisualizer."""
    try:
        current_dataset = get_current_dataset()
        if not current_dataset:
            return None
        
        df = current_dataset['dataframe']
        
        # Create chart using WebVisualizer
        chart_result = st.session_state.web_visualizer.create_web_chart(
            data=df,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title
        )
        
        return chart_result
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def render_auto_dashboard():
    """Render the auto-generated dashboard."""
    st.markdown("### 📊 Auto-Generated Dashboard")
    
    current_dataset = get_current_dataset()
    if not current_dataset:
        st.info("Upload a dataset to see the auto-generated dashboard.")
        return
    
    df = current_dataset['dataframe']
    
    # Dataset Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("💾 Size", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
    with col4:
        upload_time = current_dataset['upload_time'].strftime("%H:%M")
        st.metric("📅 Uploaded", upload_time)
    
    # Auto-generate insights
    if not st.session_state.auto_insights:
        with st.spinner("🤖 Generating automatic insights..."):
            st.session_state.auto_insights = generate_insights_direct()
    
    # Display insights
    if st.session_state.auto_insights:
        st.markdown("#### 🎯 Key Insights")
        for i, insight in enumerate(st.session_state.auto_insights[:3], 1):
            if isinstance(insight, dict):
                message = insight.get('message', insight.get('description', str(insight)))
                confidence = insight.get('confidence', 0.8)
            else:
                message = str(insight)
                confidence = 0.8
                
            st.markdown(f"""
            <div class="insight-card">
                <strong>💡 Insight {i}</strong><br>
                {message}
                <br><br>
                <small>Confidence: {confidence:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick Statistics
    st.markdown("#### 📈 Quick Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Quality:**")
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        st.progress(completeness, text=f"Complete: {completeness:.1%}")
        
    with col2:
        st.write("**Data Types:**")
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.write(f"📊 {numeric_cols} numeric columns")
        st.write(f"📝 {categorical_cols} categorical columns")

def render_chat_interface():
    """Render the chat interface for questions."""
    st.markdown("### 💬 Ask Questions")
    
    current_dataset = get_current_dataset()
    if not current_dataset:
        st.info("Upload a dataset to start asking questions.")
        return
    
    # Example questions
    st.markdown("**💡 Try these example questions:**")
    example_questions = [
        "What are the main patterns in this data?",
        "Show me correlations between variables",
        "Are there any outliers?",
        "What's the distribution of key variables?",
        "Generate code to calculate total sales by region",
        "Create code to find the highest values in each category",
        "Write code to calculate percentage distribution"
    ]
    
    # Display example questions as clickable buttons
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"❓ {question}", key=f"example_{i}"):
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'message': question,
                    'timestamp': datetime.now()
                })
                
                # Get answer directly from DataAnalyst
                with st.spinner("🤖 Analyzing..."):
                    result = ask_question_direct(question)
                    if result and not result.get('error'):
                        answer = result.get('explanation', result.get('answer', 'No answer available'))
                        confidence = 0.9 if result.get('type') else 0.8
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'message': answer,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        })
                        # Store in analysis results
                        st.session_state.analysis_results[question] = result
                    else:
                        error_msg = result.get('error', 'Sorry, I could not process that question.')
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'message': error_msg,
                            'confidence': 0.0,
                            'timestamp': datetime.now()
                        })
                st.rerun()
    
    # Custom question input
    st.markdown("**✏️ Or ask your own question:**")
    custom_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What's the correlation between sales and marketing spend?",
        key="custom_question_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍 Ask Question", type="primary") and custom_question:
            # Add to chat history
            st.session_state.chat_history.append({
                'type': 'user',
                'message': custom_question,
                'timestamp': datetime.now()
            })
            
            # Get answer directly from DataAnalyst
            with st.spinner("🤖 Thinking..."):
                result = ask_question_direct(custom_question)
                if result and not result.get('error'):
                    answer = result.get('explanation', result.get('answer', ''))
                    
                    # Check if this is a code execution result
                    is_code_execution = "Generated and Executed Python Code" in answer
                    
                    # If we still don't have an answer, try to extract it from other fields
                    if not answer:
                        if isinstance(result.get('result'), dict) and 'message' in result['result']:
                            answer = result['result']['message']
                        elif isinstance(result.get('result'), str):
                            answer = result['result']
                        else:
                            # Last resort - create a meaningful response based on result type
                            result_type = result.get('type', 'unknown')
                            if result_type == 'exploration':
                                answer = "Here's information about your dataset structure."
                            elif result_type == 'statistics':
                                answer = "I've calculated statistics for your numeric columns."
                            elif result_type == 'visualization':
                                answer = "I can create visualizations for your data. Try specifying a chart type."
                            else:
                                answer = "I've analyzed your data. Ask me specific questions about columns, statistics, or patterns."
                    
                    confidence = 0.9 if result.get('type') else 0.8
                    chat_entry = {
                        'type': 'assistant',
                        'message': answer,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    }
                    if is_code_execution:
                        chat_entry['is_code_execution'] = True
                    st.session_state.chat_history.append(chat_entry)
                    # Store in analysis results
                    st.session_state.analysis_results[custom_question] = result
                else:
                    error_msg = result.get('error', 'Sorry, I could not process that question.')
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'message': error_msg,
                        'confidence': 0.0,
                        'timestamp': datetime.now()
                    })
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 💭 Conversation History")
        
        # Create a scrollable chat container
        chat_container = st.container()
        with chat_container:
            for chat in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div style="text-align: right; margin: 0.5rem 0;">
                        <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 10px; display: inline-block; max-width: 80%;">
                            <strong>👤 You:</strong><br>
                            {chat['message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence_text = ""
                    if chat.get('confidence', 0) > 0:
                        confidence_text = f"<br><small>Confidence: {chat['confidence']:.1%}</small>"
                    
                    # Check if this is a code execution result
                    if chat.get('is_code_execution') or "Generated and Executed Python Code" in chat['message']:
                        # Parse and display code execution results specially
                        message = chat['message']
                        
                        # Extract components from the message
                        if "**Generated Code:**" in message and "**Execution Result:**" in message:
                            parts = message.split("**Generated Code:**")
                            intro = parts[0].strip()
                            
                            rest = parts[1]
                            code_and_result = rest.split("**Execution Result:**")
                            code_part = code_and_result[0].strip()
                            result_part = code_and_result[1].split("**Status:**")[0].strip() if "**Status:**" in code_and_result[1] else code_and_result[1].strip()
                            
                            # Extract status and error if present
                            status_success = "✅ Success" in message
                            status_color = "#4CAF50" if status_success else "#f44336"
                            status_icon = "✅" if status_success else "❌"
                            
                            # Clean up code (remove markdown)
                            if "```python" in code_part:
                                code_part = code_part.replace("```python", "").replace("```", "").strip()
                            
                            st.markdown(f"""
                            <div style="text-align: left; margin: 0.5rem 0;">
                                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; display: inline-block; max-width: 95%; border-left: 4px solid {status_color};">
                                    <strong>🤖 Code Generator:</strong><br>
                                    {intro}<br><br>
                                    
                                    <div style="background: #2d3748; color: #e2e8f0; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; font-family: 'Courier New', monospace; font-size: 0.9em;">
                                        <strong style="color: #68d391;">Generated Code:</strong><br>
                                        <pre style="margin: 0; white-space: pre-wrap;">{code_part}</pre>
                                    </div>
                                    
                                    <div style="background: #f7fafc; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 3px solid {status_color};">
                                        <strong style="color: {status_color};">{status_icon} Execution Result:</strong><br>
                                        <pre style="margin: 0.3rem 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em;">{result_part}</pre>
                                    </div>
                                    {confidence_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Fallback for malformed code execution results
                            st.markdown(f"""
                            <div style="text-align: left; margin: 0.5rem 0;">
                                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 10px; display: inline-block; max-width: 80%; border-left: 4px solid #1f77b4;">
                                    <strong>🤖 Assistant:</strong><br>
                                    {chat['message']}
                                    {confidence_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Regular assistant message
                        st.markdown(f"""
                        <div style="text-align: left; margin: 0.5rem 0;">
                            <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 10px; display: inline-block; max-width: 80%; border-left: 4px solid #1f77b4;">
                                <strong>🤖 Assistant:</strong><br>
                                {chat['message']}
                                {confidence_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

def render_detailed_analysis():
    """Render detailed analysis section at the bottom."""
    st.markdown("---")
    st.markdown("### 🔬 Detailed Analysis & Custom Analytics")
    
    current_dataset = get_current_dataset()
    if not current_dataset:
        st.info("Upload a dataset to see detailed analysis options.")
        return
    
    df = current_dataset['dataframe']
    
    # Custom Analysis Options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Custom Analysis")
        
        analysis_type = st.selectbox(
            "Choose analysis type:",
            [
                "comprehensive",
                "statistical", 
                "correlation",
                "outlier_detection",
                "data_quality",
                "summary"
            ],
            help="Select the type of analysis to perform"
        )
        
        if st.button("🚀 Run Custom Analysis", type="primary"):
            with st.spinner(f"Running {analysis_type} analysis..."):
                # Map analysis types to specific questions
                analysis_questions = {
                    "comprehensive": "Give me a comprehensive analysis of this dataset",
                    "statistical": "What are the key statistical insights?",
                    "correlation": "Show me correlations between variables",
                    "outlier_detection": "Are there any outliers in the data?",
                    "data_quality": "What's the data quality of this dataset?",
                    "summary": "Give me a summary of this dataset"
                }
                
                question = analysis_questions.get(analysis_type, "Analyze this dataset")
                result = ask_question_direct(question)
                
                if result and not result.get('error'):
                    st.session_state.analysis_results[f"custom_{analysis_type}"] = result
                    st.success(f"✅ {analysis_type.title()} analysis completed!")
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Please try again.")
    
    with col2:
        st.markdown("#### 📈 Custom Visualizations")
        
        chart_type = st.selectbox(
            "Choose chart type:",
            [
                "scatter",
                "line", 
                "bar",
                "histogram",
                "heatmap",
                "pie",
                "violin"
            ],
            help="Select the type of chart to create",
            key="chart_type_selector"
        )
        
        # Get column lists by type
        numeric_columns = list(df.select_dtypes(include=['number']).columns)
        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
        all_columns = list(df.columns)
        
        # Column selection based on chart type
        x_column = None
        y_column = None
        color_column = None
        
        # Configure column selection based on chart type
        if chart_type == "histogram":
            x_column = st.selectbox("Select column for histogram:", all_columns, 
                                   index=0 if numeric_columns else 0,
                                   help="Numeric columns work best for histograms")
            
            color_column = st.selectbox("Color by (optional):", 
                                      ["None"] + categorical_columns,
                                      help="Split histogram by category")
            if color_column == "None":
                color_column = None
                
        elif chart_type == "pie":
            x_column = st.selectbox("Select category for pie chart:", 
                                   categorical_columns if categorical_columns else all_columns,
                                   help="Categories to show in the pie chart")
            
            if numeric_columns:
                y_column_options = ["Count"] + numeric_columns
                y_column_selection = st.selectbox("Values to use:", y_column_options, 
                                                help="'Count' uses frequency, or select a numeric column for values")
                if y_column_selection != "Count":
                    y_column = y_column_selection
            
        elif chart_type == "heatmap":
            st.info("Heatmap will automatically use correlation matrix of numeric columns")
            
        else:
            # For scatter, line, bar, violin plots
            x_options = categorical_columns if chart_type == "bar" and categorical_columns else all_columns
            x_default_idx = 0
            
            x_column = st.selectbox("X-axis:", x_options, index=x_default_idx,
                                  help="Select column for X-axis")
            
            # For Y-axis, prefer numeric columns
            y_options = [col for col in all_columns if col != x_column]
            y_default_idx = 0
            for i, col in enumerate(y_options):
                if col in numeric_columns:
                    y_default_idx = i
                    break
                    
            y_column = st.selectbox("Y-axis:", y_options, index=min(y_default_idx, len(y_options)-1) if y_options else 0,
                                  help="Select column for Y-axis")
            
            # Optional color column
            if categorical_columns:
                color_options = ["None"] + [col for col in categorical_columns if col not in [x_column, y_column]]
                if color_options and len(color_options) > 1:
                    color_selection = st.selectbox("Color by (optional):", color_options,
                                                 help="Split data points by category")
                    if color_selection != "None":
                        color_column = color_selection
        
        # Chart title
        custom_title = st.text_input("Chart title (optional):", 
                                    placeholder=f"Enter a title for your {chart_type} chart")
        
        if st.button("📊 Create Visualization", type="primary"):
            with st.spinner(f"Creating {chart_type} chart..."):
                chart_result = create_visualization_direct(
                    chart_type=chart_type,
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    title=custom_title if custom_title else None
                )
                
                if chart_result and chart_result.get("success"):
                    st.session_state.analysis_results[f"chart_{chart_type}"] = chart_result
                    st.success(f"✅ {chart_type.title()} chart created!")
                    
                    # Display the chart immediately
                    if chart_result.get("html_content"):
                        st.components.v1.html(chart_result["html_content"], height=500)
                    
                    st.rerun()
                else:
                    error_msg = chart_result.get("error", "Chart creation failed") if chart_result else "Chart creation failed"
                    st.error(f"❌ {error_msg}")
    
    # Display Analysis Results
    if st.session_state.analysis_results:
        st.markdown("#### 📋 Analysis Results")
        
        # Create tabs for different analysis results
        result_keys = list(st.session_state.analysis_results.keys())
        if result_keys:
            # Limit tabs for better UI
            display_keys = result_keys[-5:]  # Show last 5 results
            tab_names = [key.replace('_', ' ').title() for key in display_keys]
            tabs = st.tabs(tab_names)
            
            for i, (tab, key) in enumerate(zip(tabs, display_keys)):
                with tab:
                    result = st.session_state.analysis_results[key]
                    
                    # Display different types of results
                    if 'explanation' in result:
                        # DataAnalyst result
                        st.write("**Analysis:**", result['explanation'])
                        
                        if result.get('result'):
                            if isinstance(result['result'], dict):
                                # Display structured data
                                for sub_key, sub_value in result['result'].items():
                                    if isinstance(sub_value, (list, dict)) and len(str(sub_value)) < 1000:
                                        st.write(f"**{sub_key}:**")
                                        if isinstance(sub_value, list):
                                            for item in sub_value:
                                                st.write(f"• {item}")
                                        elif isinstance(sub_value, dict):
                                            st.json(sub_value)
                                    elif not isinstance(sub_value, (list, dict)):
                                        st.write(f"**{sub_key}:** {sub_value}")
                            
                            # If result contains data that can be displayed as a dataframe
                            if 'sample_data' in result.get('result', {}):
                                st.write("**Sample Data:**")
                                sample_data = result['result']['sample_data']
                                if isinstance(sample_data, dict):
                                    try:
                                        df_sample = pd.DataFrame(sample_data)
                                        st.dataframe(df_sample, use_container_width=True)
                                    except:
                                        st.json(sample_data)
                    
                    elif result.get("success") and result.get("html_content"):
                        # Visualization result
                        st.write("**Visualization:**")
                        st.components.v1.html(result["html_content"], height=500)
                        st.write(f"**Chart Type:** {result.get('chart_type', 'Unknown')}")
                        if result.get('title'):
                            st.write(f"**Title:** {result['title']}")
                    
                    else:
                        # Generic result display
                        if isinstance(result, dict):
                            for key, value in result.items():
                                if key not in ['html_content', 'json_data']:  # Skip large content
                                    st.write(f"**{key}:** {value}")
                        else:
                            st.write(result)

def render_dataset_preview():
    """Render dataset preview in an expander."""
    current_dataset = get_current_dataset()
    if not current_dataset:
        return
    
    with st.expander("👁️ Dataset Preview", expanded=False):
        df = current_dataset['dataframe']
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{current_dataset['filename']}_full.csv",
            mime="text/csv"
        )

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Visual Analytics Agent (Direct)</h1>', unsafe_allow_html=True)
    
    # Show LLM status
    if st.session_state.get('using_llm', False):
        st.success("🧠 Using LLM for enhanced analysis")
    else:
        st.warning("⚠️ Using fallback mode - LLM not available")
    
    # File Upload Section (Top of page)
    with st.container():
        st.markdown("### 📁 Data Upload")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=['csv', 'json', 'xlsx', 'xls'],
                help="Supported formats: CSV, JSON, Excel (.xlsx, .xls)"
            )
        
        with col2:
            if uploaded_file is not None:
                if st.button("🚀 Upload Dataset", type="primary"):
                    with st.spinner("Loading dataset..."):
                        result = load_dataset_from_file(uploaded_file)
                        if result:
                            dataset_name, load_result = result
                            st.session_state.auto_insights = []  # Reset insights for new dataset
                            st.session_state.chat_history = []   # Reset chat for new dataset
                            st.session_state.analysis_results = {}  # Reset analysis results
                            st.success(f"✅ Successfully loaded {uploaded_file.name}")
                            st.rerun()
        
        with col3:
            # Dataset selector
            if st.session_state.datasets:
                dataset_names = list(st.session_state.datasets.keys())
                current_idx = dataset_names.index(st.session_state.current_dataset_name) if st.session_state.current_dataset_name in dataset_names else 0
                
                selected_name = st.selectbox(
                    "Or select existing:",
                    dataset_names,
                    index=current_idx,
                    key="dataset_selector"
                )
                
                if selected_name != st.session_state.current_dataset_name:
                    st.session_state.current_dataset_name = selected_name
                    # Load into DataAnalyst
                    df = st.session_state.datasets[selected_name]['dataframe']
                    st.session_state.data_analyst.load_dataset(df)
                    st.session_state.auto_insights = []  # Reset for different dataset
                    st.rerun()
    
    # Main Layout: Left Dashboard + Right Chat
    if st.session_state.current_dataset_name:
        current_dataset = get_current_dataset()
        
        # Display current dataset info
        st.markdown(f"**📊 Current Dataset:** {current_dataset['filename']} ({current_dataset['rows']:,} rows, {current_dataset['columns']} columns)")
        
        # Dataset preview
        render_dataset_preview()
        
        st.markdown("---")
        
        # Main Content: Left Dashboard + Right Chat
        left_col, right_col = st.columns([1.2, 0.8])
        
        with left_col:
            render_auto_dashboard()
        
        with right_col:
            render_chat_interface()
        
        # Bottom: Detailed Analysis
        render_detailed_analysis()
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Visual Analytics Agent (Direct Mode)!
        
        This version connects directly to the DataAnalyst for **faster and more reliable** analysis.
        
        ### 🚀 Quick Start Guide
        
        1. **📁 Upload Data**: Use the file uploader above to upload your CSV, JSON, or Excel file
        2. **📊 Auto-Dashboard**: Get an instant overview with key metrics and insights
        3. **💬 Ask Questions**: Use the chat interface to ask natural language questions
        4. **🔬 Deep Analysis**: Run custom analytics and create visualizations
        
        ### ✨ Advantages of Direct Mode
        
        - **🚀 Faster**: No API overhead
        - **🛡️ More Reliable**: No network issues
        - **🎯 Better Responses**: Direct access to AI capabilities
        - **📊 Instant Visualizations**: Charts appear immediately
        
        ### 📈 What You Can Do
        
        - **🤖 Automatic Insights**: Get instant patterns and trends
        - **💬 Natural Language Queries**: Ask questions like "What drives sales?"
        - **📊 Custom Visualizations**: Create charts tailored to your needs
        - **🔬 Advanced Analytics**: Statistical analysis, correlations, outlier detection
        - **📋 Export Results**: Download your analysis and visualizations
        
        **Ready to start?** Upload a dataset above! 👆
        """)
        
        # Sample data generation
        st.markdown("### 📂 No data? Try these sample datasets:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛒 Sales Data"):
                # Generate sample sales data
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', periods=100)
                sales_data = pd.DataFrame({
                    'date': dates,
                    'sales': np.random.normal(1000, 200, 100),
                    'marketing_spend': np.random.normal(500, 100, 100),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100)
                })
                
                csv_data = sales_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Sales Data",
                    csv_data,
                    "sample_sales_data.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("👥 Customer Data"):
                # Generate sample customer data
                np.random.seed(42)
                customer_data = pd.DataFrame({
                    'customer_id': range(1, 201),
                    'age': np.random.randint(18, 80, 200),
                    'income': np.random.normal(50000, 15000, 200),
                    'spending_score': np.random.randint(1, 100, 200),
                    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 200),
                    'satisfaction': np.random.randint(1, 11, 200)
                })
                
                csv_data = customer_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Customer Data",
                    csv_data,
                    "sample_customer_data.csv",
                    "text/csv"
                )
        
        with col3:
            if st.button("📊 Survey Data"):
                # Generate sample survey data
                np.random.seed(42)
                survey_data = pd.DataFrame({
                    'response_id': range(1, 301),
                    'satisfaction': np.random.randint(1, 11, 300),
                    'recommendation': np.random.randint(1, 11, 300),
                    'category': np.random.choice(['Product', 'Service', 'Support'], 300),
                    'tenure_months': np.random.randint(1, 60, 300),
                    'issue_resolved': np.random.choice([True, False], 300)
                })
                
                csv_data = survey_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Survey Data",
                    csv_data,
                    "sample_survey_data.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main() 