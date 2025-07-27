# 📊 Visual Analytics Agent Dashboard Guide

A comprehensive guide to using the Visual Analytics Agent's interactive dashboard interface.

## 🎯 **Overview**

The dashboard combines **Pattern 2 (Progressive Disclosure)** and **Pattern 3 (Dashboard + Chat)** to provide:
- **Left**: Auto-generated dashboard with instant insights
- **Right**: Conversational interface for specific questions  
- **Bottom**: Advanced custom analytics and visualizations

## 🚀 **Getting Started**

### **Prerequisites**
1. Python 3.8+ installed
2. Required packages installed (`pip install -r requirements.txt`)
3. OpenAI API key set (optional but recommended)

### **Quick Start**
```bash
# Terminal 1: Start API server
python run_api.py

# Terminal 2: Start dashboard  
python run_dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

## 📋 **Dashboard Layout**

### **🔝 Top Section: Data Upload**
- **File Uploader**: Supports CSV, JSON, Excel (.xlsx, .xls)
- **Upload Button**: Process and analyze your data
- **Dataset Selector**: Switch between previously uploaded datasets

### **📊 Left Column: Auto-Generated Dashboard**

#### **Key Metrics Row**
- **📊 Rows**: Total number of data rows
- **📋 Columns**: Number of variables/features
- **💾 Size**: File size in KB/MB
- **📅 Uploaded**: Upload timestamp

#### **🎯 Auto-Insights**
- AI-generated patterns and findings
- Key trends and relationships
- Data quality observations
- Confidence scores for each insight

#### **📈 Quick Statistics**
- **Data Quality**: Completeness percentage
- **Memory Usage**: RAM consumption
- **Duplicates**: Duplicate row count

### **💬 Right Column: Chat Interface**

#### **Example Questions**
Pre-built clickable queries:
- "What are the main patterns in this data?"
- "Show me correlations between variables"
- "Are there any outliers?"
- "What's the distribution of key variables?"
- "Generate a summary report"

#### **Custom Questions**
- Natural language input field
- Real-time processing with AI
- Confidence scoring for answers

#### **Conversation History**
- Last 10 messages displayed
- User questions and AI responses  
- Confidence indicators
- Clear chat functionality

### **🔬 Bottom Section: Detailed Analysis**

#### **📊 Custom Analysis (Left)**
Analysis types available:
- **Comprehensive**: Full dataset analysis
- **Statistical**: Descriptive statistics
- **Correlation**: Relationship analysis  
- **Outlier Detection**: Anomaly identification
- **Trend Analysis**: Time series patterns
- **Clustering**: Group discovery

**Advanced Options:**
- Include visualizations toggle
- Confidence threshold slider (0.5-1.0)
- Maximum insights slider (3-20)

#### **📈 Custom Visualizations (Right)**
Chart types available:
- **Scatter**: X-Y relationships
- **Line**: Time series trends
- **Bar**: Category comparisons
- **Histogram**: Distribution analysis
- **Heatmap**: Correlation matrices
- **Pie**: Composition analysis
- **Violin**: Distribution shapes
- **Area**: Filled time series

**Chart Options:**
- Custom titles
- Theme selection (plotly, white, dark)
- Legend toggle

#### **📋 Analysis Results Tabs**
- Multiple analysis results in tabs
- Different result types displayed appropriately
- JSON view for raw data
- Export functionality

## 🛠️ **How-To Guides**

### **📁 Uploading Your First Dataset**

1. **Prepare your data**:
   - Ensure CSV has headers
   - Clean obvious formatting issues
   - Save Excel files as .xlsx or .csv

2. **Upload process**:
   - Click "Browse files" or drag-and-drop
   - Select your file (CSV, JSON, Excel)
   - Click "🚀 Upload Dataset"
   - Wait for processing confirmation

3. **Review auto-dashboard**:
   - Check dataset metrics for accuracy
   - Read auto-generated insights
   - Note any data quality issues

### **💬 Asking Effective Questions**

#### **Start with Examples**
Begin with pre-built questions to understand your data:
- "What are the main patterns?" - Gets overall structure
- "Show me correlations" - Reveals relationships
- "Any outliers?" - Identifies anomalies

#### **Progress to Specific Questions**
- "What drives customer satisfaction scores?"
- "Which products have the highest profit margins?"
- "How does seasonality affect our sales?"
- "What factors predict customer churn?"

#### **Advanced Analytical Questions**
- "Segment customers based on purchasing behavior"
- "Identify the top 3 factors influencing revenue"
- "Create a predictive model for sales forecasting"
- "What's the statistical significance of regional differences?"

### **📊 Creating Custom Visualizations**

#### **Choosing the Right Chart Type**

| Data Type | Recommended Charts |
|-----------|-------------------|
| **Continuous vs Continuous** | Scatter, Line |
| **Category vs Continuous** | Bar, Violin, Box |
| **Time Series** | Line, Area |
| **Distributions** | Histogram, Violin |
| **Correlations** | Heatmap |
| **Compositions** | Pie, Stacked Bar |

#### **Chart Creation Process**
1. Select chart type from dropdown
2. Configure options (title, theme, legend)
3. Click "📊 Create Visualization"
4. Review result in analysis tabs
5. Export if needed

### **🔬 Running Custom Analysis**

#### **Analysis Type Selection Guide**

- **Comprehensive**: Use for initial exploration of new datasets
- **Statistical**: When you need descriptive statistics and distributions
- **Correlation**: To understand relationships between variables
- **Outlier Detection**: When data quality is a concern
- **Trend Analysis**: For time-based data patterns
- **Clustering**: To discover natural groupings in data

#### **Configuring Analysis Options**
- **Include Visualizations**: Toggle ON for richer results
- **Confidence Threshold**: Higher values (0.8+) for reliable insights
- **Max Insights**: 5-10 for focused results, 15+ for comprehensive

## 🎯 **Best Practices**

### **📊 Data Preparation**
- **Clean data first**: Remove obvious errors before upload
- **Check file format**: CSV with headers works best
- **Size considerations**: Large files (>100MB) may be slow
- **Column naming**: Use descriptive, readable column names

### **💬 Question Asking Strategy**
1. **Start broad**: "What patterns do you see?"
2. **Get specific**: "What affects customer retention?"
3. **Dig deeper**: "Why do premium customers have higher satisfaction?"
4. **Validate findings**: "Are these correlations statistically significant?"

### **📈 Analysis Workflow**
1. **Upload & Auto-insights**: Understand your data structure
2. **Chat exploration**: Ask specific business questions
3. **Custom analysis**: Run statistical tests and deeper analysis
4. **Visualization**: Create charts to communicate findings
5. **Export results**: Save insights and visualizations

### **🔧 Performance Optimization**
- **Use dataset selector**: Switch between datasets efficiently
- **Clear chat history**: Reset for new analytical directions
- **Sample large datasets**: Consider analyzing subsets first
- **Monitor confidence scores**: Trust higher-confidence insights

## 🚨 **Troubleshooting**

### **Common Issues**

#### **"API Server is not running"**
```bash
# Start the API server first
python run_api.py
```

#### **Upload fails with error 500**
- Check file format (CSV headers, valid JSON)
- Try smaller file sizes first
- Ensure no special characters in filename

#### **Chat responses are poor**
- Set OpenAI API key: `export OPENAI_API_KEY="your-key"`
- Try more specific questions
- Check if dataset has clear patterns

#### **Visualizations not appearing**
- Ensure dataset is selected
- Try different chart types
- Check browser console for errors

#### **Slow performance**
- Use smaller datasets for testing
- Clear chat history regularly
- Restart dashboard if memory issues

### **Error Messages**

| Message | Cause | Solution |
|---------|-------|---------|
| "Could not process question" | API/LLM error | Check API key, try simpler question |
| "No preview data available" | Dataset loading issue | Re-upload dataset |
| "Analysis failed" | Processing error | Try different analysis type |
| "Chart creation failed" | Visualization error | Select different chart type |

## 📚 **Advanced Features**

### **🔄 Multi-Dataset Analysis**
- Upload multiple datasets
- Use dataset selector to switch contexts
- Compare insights across datasets
- Maintain separate chat histories

### **📊 Dashboard Customization**
- Download sample datasets for testing
- Export analysis results as CSV
- Copy insights for external reporting
- Save visualizations locally

### **🤖 AI-Powered Insights**
- Automatic pattern detection
- Statistical significance testing
- Anomaly identification
- Trend forecasting capabilities

### **📈 Advanced Analytics**
- Custom statistical analysis
- Machine learning insights
- Predictive modeling suggestions
- Data quality assessment

## 💡 **Tips for Success**

### **🎯 Getting Better Insights**
- **Ask follow-up questions**: "Why is that correlation so strong?"
- **Request specific metrics**: "What's the correlation coefficient?"
- **Seek actionable insights**: "What should I do about these outliers?"
- **Validate findings**: "How confident are you in this pattern?"

### **📊 Creating Compelling Visualizations**
- **Start with auto-suggestions**: Let AI recommend chart types
- **Use meaningful titles**: Describe what the chart shows
- **Choose appropriate themes**: Dark for presentations, light for reports
- **Export high-quality**: Use built-in export features

### **🔬 Conducting Thorough Analysis**
- **Layer your analysis**: Start simple, add complexity
- **Cross-validate insights**: Ask questions in different ways
- **Document findings**: Use export features to save results
- **Share discoveries**: Copy insights for team discussions

## 🎊 **Success Stories**

### **Sales Analysis Example**
1. **Upload**: Monthly sales data (12 months, 5 products)
2. **Auto-insights**: "Product A shows 23% growth trend"
3. **Chat question**: "What drives Product A's success?"
4. **Custom analysis**: Correlation analysis reveals marketing spend impact
5. **Visualization**: Line chart showing sales vs marketing investment
6. **Outcome**: Identified optimal marketing spend ratio

### **Customer Segmentation Example**
1. **Upload**: Customer data (1000 customers, 8 attributes)
2. **Auto-insights**: "Three distinct customer groups identified"
3. **Chat question**: "How do I identify high-value customers?"
4. **Custom analysis**: Clustering analysis with 3 segments
5. **Visualization**: Scatter plot of income vs spending by segment
6. **Outcome**: Targeted marketing strategy for each segment

---

🎉 **Ready to explore your data?** Start with the upload section and let the AI guide your analytical journey! 