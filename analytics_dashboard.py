#!/usr/bin/env python3
"""
Visual Analytics Agent - Comprehensive Dashboard

A beautiful, user-friendly interface combining:
- Pattern 2: Progressive Disclosure (Upload → Auto-insights)  
- Pattern 3: Dashboard + Chat (Left: Dashboard, Right: Chat, Bottom: Detailed Analysis)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
import io
import numpy as np

# Page config
st.set_page_config(
    page_title="Visual Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # More space for main layout
)

# Custom CSS for better styling
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
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE = "http://localhost:8000/api"

# Session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"dashboard_{int(time.time())}"
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'auto_insights' not in st.session_state:
    st.session_state.auto_insights = []

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_dataset(file, file_name):
    """Upload a dataset to the API."""
    try:
        files = {'file': (file_name, file, 'application/octet-stream')}
        data = {'session_id': st.session_state.session_id}
        
        response = requests.post(f"{API_BASE}/data/upload", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def get_datasets():
    """Get list of uploaded datasets."""
    try:
        response = requests.get(f"{API_BASE}/data/datasets", 
                              params={'session_id': st.session_state.session_id})
        if response.status_code == 200:
            return response.json().get('datasets', [])
        return []
    except:
        return []

def get_dataset_preview(dataset_id, rows=20):
    """Get dataset preview."""
    try:
        response = requests.get(f"{API_BASE}/data/datasets/{dataset_id}/preview", 
                              params={'rows': rows})
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_dataset_stats(dataset_id):
    """Get dataset statistics."""
    try:
        response = requests.get(f"{API_BASE}/data/datasets/{dataset_id}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def ask_question(dataset_id, question):
    """Ask a natural language question about the dataset."""
    try:
        payload = {
            'dataset_id': dataset_id,
            'session_id': st.session_state.session_id,
            'query': question
        }
        response = requests.post(f"{API_BASE}/analysis/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def generate_insights(dataset_id, insight_type="automated"):
    """Generate insights for a dataset."""
    try:
        payload = {
            'dataset_id': dataset_id,
            'session_id': st.session_state.session_id,
            'insight_type': insight_type,
            'top_n': 5
        }
        response = requests.post(f"{API_BASE}/analysis/insights", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def perform_custom_analysis(dataset_id, analysis_type, options=None):
    """Perform custom analysis on the dataset."""
    try:
        payload = {
            'dataset_id': dataset_id,
            'session_id': st.session_state.session_id,
            'analysis_type': analysis_type,
            'use_ai_agent': True
        }
        if options:
            payload.update(options)
            
        response = requests.post(f"{API_BASE}/analysis/analyze", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_visualization(dataset_id, chart_type, options=None):
    """Create a visualization."""
    try:
        payload = {
            'dataset_id': dataset_id,
            'session_id': st.session_state.session_id,
            'chart_type': chart_type
        }
        if options:
            payload.update(options)
            
        response = requests.post(f"{API_BASE}/visualization/chart", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def render_auto_dashboard(dataset):
    """Render the auto-generated dashboard."""
    st.markdown("### 📊 Auto-Generated Dashboard")
    
    dataset_id = dataset['dataset_id']
    
    # Dataset Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{dataset['rows']:,}")
    with col2:
        st.metric("📋 Columns", dataset['columns'])
    with col3:
        st.metric("💾 Size", f"{dataset['file_size']/1024:.1f} KB")
    with col4:
        upload_time = dataset.get('upload_time', '')
        if upload_time:
            time_str = upload_time[:10]  # Just the date part
        else:
            time_str = 'Unknown'
        st.metric("📅 Uploaded", time_str)
    
    # Auto-generate insights if not already done
    if not st.session_state.auto_insights:
        with st.spinner("🤖 Generating automatic insights..."):
            insights_result = generate_insights(dataset_id, "automated")
            if insights_result:
                st.session_state.auto_insights = insights_result.get('insights', [])
    
    # Display insights
    if st.session_state.auto_insights:
        st.markdown("#### 🎯 Key Insights")
        for i, insight in enumerate(st.session_state.auto_insights[:3], 1):
            if isinstance(insight, dict):
                message = insight.get('message', str(insight))
                confidence = insight.get('confidence', 0.8)
            else:
                message = str(insight)
                confidence = 0.8
                
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <strong>💡 Insight {i}</strong><br>
                    {message}
                    <br><br>
                    <small>Confidence: {confidence:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick Statistics
    stats_result = get_dataset_stats(dataset_id)
    if stats_result:
        st.markdown("#### 📈 Quick Statistics")
        stats = stats_result.get('stats', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Quality:**")
            missing_pct = (stats.get('total_missing', 0) / (dataset['rows'] * dataset['columns']) * 100)
            st.progress(max(0, (100 - missing_pct) / 100), text=f"Complete: {100-missing_pct:.1f}%")
            
        with col2:
            st.write("**Memory Usage:**")
            memory_mb = stats.get('memory_mb', 0)
            st.write(f"💾 {memory_mb:.2f} MB")
            duplicates = stats.get('duplicates', 0)
            st.write(f"🔄 {duplicates} duplicates")

def render_chat_interface(dataset):
    """Render the chat interface for questions."""
    st.markdown("### 💬 Ask Questions")
    
    dataset_id = dataset['dataset_id']
    
    # Example questions
    st.markdown("**💡 Try these example questions:**")
    example_questions = [
        "What are the main patterns in this data?",
        "Show me correlations between variables",
        "Are there any outliers?",
        "What's the distribution of key variables?",
        "Generate a summary report"
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
                
                # Get answer
                with st.spinner("🤖 Analyzing..."):
                    result = ask_question(dataset_id, question)
                    if result:
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'message': result.get('answer', 'No answer provided'),
                            'confidence': result.get('confidence_score', 0.8),
                            'timestamp': datetime.now()
                        })
                        # Store in analysis results
                        st.session_state.analysis_results[question] = result
                    else:
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'message': 'Sorry, I could not process that question.',
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
            
            # Get answer
            with st.spinner("🤖 Thinking..."):
                result = ask_question(dataset_id, custom_question)
                if result:
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'message': result.get('answer', 'No answer provided'),
                        'confidence': result.get('confidence_score', 0.8),
                        'timestamp': datetime.now()
                    })
                    # Store in analysis results
                    st.session_state.analysis_results[custom_question] = result
                else:
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'message': 'Sorry, I could not process that question.',
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
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
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
    
    if not st.session_state.current_dataset:
        st.info("Upload a dataset to see detailed analysis options.")
        return
    
    dataset_id = st.session_state.current_dataset['dataset_id']
    
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
                "trend_analysis",
                "clustering"
            ],
            help="Select the type of analysis to perform"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            include_visualizations = st.checkbox("Include visualizations", value=True)
            confidence_threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.8)
            max_insights = st.slider("Maximum insights", 3, 20, 10)
        
        if st.button("🚀 Run Custom Analysis", type="primary"):
            options = {
                'include_visualizations': include_visualizations,
                'confidence_threshold': confidence_threshold,
                'max_insights': max_insights
            }
            
            with st.spinner(f"Running {analysis_type} analysis..."):
                result = perform_custom_analysis(dataset_id, analysis_type, options)
                if result:
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
                "violin",
                "area"
            ],
            help="Select the type of chart to create"
        )
        
        # Chart options
        with st.expander("🎨 Chart Options"):
            chart_title = st.text_input("Chart title", value="")
            chart_theme = st.selectbox("Chart theme", ["plotly", "plotly_white", "plotly_dark"])
            show_legend = st.checkbox("Show legend", value=True)
        
        if st.button("📊 Create Visualization", type="primary"):
            chart_options = {
                'title': chart_title,
                'theme': chart_theme,
                'show_legend': show_legend
            }
            
            with st.spinner(f"Creating {chart_type} chart..."):
                result = create_visualization(dataset_id, chart_type, chart_options)
                if result:
                    st.session_state.analysis_results[f"chart_{chart_type}"] = result
                    st.success(f"✅ {chart_type.title()} chart created!")
                    st.rerun()
                else:
                    st.error("❌ Chart creation failed. Please try again.")
    
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
                    if 'answer' in result:
                        # Natural language query result
                        st.write("**Answer:**", result['answer'])
                        if result.get('confidence_score'):
                            st.progress(result['confidence_score'], text=f"Confidence: {result['confidence_score']:.1%}")
                        
                        if result.get('data_results'):
                            st.write("**Data Results:**")
                            df_results = pd.DataFrame(result['data_results'])
                            st.dataframe(df_results, use_container_width=True)
                    
                    elif 'query_result' in result:
                        # Analysis result
                        query_result = result['query_result']
                        st.write("**Analysis Summary:**", result.get('message', 'Analysis completed'))
                        
                        if query_result.get('insights'):
                            st.write("**Insights:**")
                            for insight in query_result['insights']:
                                st.write(f"• {insight}")
                        
                        if query_result.get('statistical_results'):
                            st.write("**Statistical Results:**")
                            st.json(query_result['statistical_results'])
                    
                    elif 'chart_data' in result:
                        # Visualization result
                        st.write("**Visualization:**", result.get('message', 'Chart created'))
                        # In a real implementation, you'd display the actual chart here
                        st.info("Chart display would be implemented here with the chart data")
                    
                    else:
                        # Generic result
                        st.json(result)

def render_dataset_preview(dataset):
    """Render dataset preview in an expander."""
    with st.expander("👁️ Dataset Preview", expanded=False):
        preview_data = get_dataset_preview(dataset['dataset_id'], 20)
        if preview_data:
            df = pd.DataFrame(preview_data.get('preview_data', []))
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Preview as CSV",
                    data=csv,
                    file_name=f"{dataset['filename']}_preview.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No preview data available")
        else:
            st.error("Could not load dataset preview")

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Visual Analytics Agent</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🚨 **API Server is not running!**")
        st.markdown("""
        Please start the API server by running:
        ```bash
        python run_api.py
        ```
        """)
        st.stop()
    
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
                    with st.spinner("Uploading dataset..."):
                        result = upload_dataset(uploaded_file.getvalue(), uploaded_file.name)
                        if result:
                            st.session_state.current_dataset = result
                            st.session_state.auto_insights = []  # Reset insights for new dataset
                            st.session_state.chat_history = []   # Reset chat for new dataset
                            st.session_state.analysis_results = {}  # Reset analysis results
                            st.success(f"✅ Successfully uploaded {uploaded_file.name}")
                            st.rerun()
        
        with col3:
            # Dataset selector
            datasets = get_datasets()
            if datasets:
                dataset_names = [f"{d['filename']}" for d in datasets]
                selected_idx = st.selectbox(
                    "Or select existing:",
                    range(len(dataset_names)),
                    format_func=lambda x: dataset_names[x] if x < len(dataset_names) else "None",
                    key="dataset_selector"
                )
                
                if selected_idx is not None and selected_idx < len(datasets):
                    if st.session_state.current_dataset != datasets[selected_idx]:
                        st.session_state.current_dataset = datasets[selected_idx]
                        st.session_state.auto_insights = []  # Reset for different dataset
                        st.rerun()
    
    # Main Layout: Left Dashboard + Right Chat
    if st.session_state.current_dataset:
        dataset = st.session_state.current_dataset
        
        # Display current dataset info
        st.markdown(f"**📊 Current Dataset:** {dataset['filename']} ({dataset['rows']:,} rows, {dataset['columns']} columns)")
        
        # Dataset preview
        render_dataset_preview(dataset)
        
        st.markdown("---")
        
        # Main Content: Left Dashboard + Right Chat
        left_col, right_col = st.columns([1.2, 0.8])
        
        with left_col:
            render_auto_dashboard(dataset)
        
        with right_col:
            render_chat_interface(dataset)
        
        # Bottom: Detailed Analysis
        render_detailed_analysis()
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Visual Analytics Agent!
        
        This is your intelligent data analysis companion. Here's how to get started:
        
        ### 🚀 Quick Start Guide
        
        1. **📁 Upload Data**: Use the file uploader above to upload your CSV, JSON, or Excel file
        2. **📊 Auto-Dashboard**: Get an instant overview with key metrics and insights
        3. **💬 Ask Questions**: Use the chat interface to ask natural language questions
        4. **🔬 Deep Analysis**: Run custom analytics and create visualizations
        
        ### 📈 What You Can Do
        
        - **🤖 Automatic Insights**: Get instant patterns and trends
        - **💬 Natural Language Queries**: Ask questions like "What drives sales?"
        - **📊 Custom Visualizations**: Create charts tailored to your needs
        - **🔬 Advanced Analytics**: Statistical analysis, correlations, outlier detection
        - **📋 Export Results**: Download your analysis and visualizations
        
        ### 🎯 Example Use Cases
        
        - **📈 Sales Analysis**: "Which products are trending up?"
        - **👥 Customer Insights**: "What segments have the highest value?"
        - **🏆 Performance Tracking**: "How do our KPIs compare over time?"
        - **🔍 Anomaly Detection**: "Are there any unusual patterns?"
        
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