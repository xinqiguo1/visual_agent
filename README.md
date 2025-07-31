# 🚀 Visual Analytics Agent 3.0

An intelligent data analysis assistant powered by GPT-4o and LangChain that transforms how you interact with your data through natural language.

![Visual Analytics Agent](https://img.shields.io/badge/Visual%20Analytics-Agent%203.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![LangChain](https://img.shields.io/badge/LangChain-Integrated-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## ✨ What's New in 3.0

- **🧩 Code Generation & Execution**: Generate AND execute Python code directly in the dashboard
- **🎛️ Interactive Visualization Builder**: Select specific columns for each chart type
- **🚀 Direct Dashboard Mode**: Connect directly to agents without API server
- **🔍 Enhanced Insights**: More intelligent pattern detection and recommendations
- **🛠️ Improved Error Handling**: Robust fallback mechanisms for all operations

## 🎯 Key Features

### 🧠 Intelligent Data Analysis
- **Natural Language Interface**: Ask questions about your data in plain English
- **Automatic Insights**: AI-powered pattern detection and anomaly identification
- **Statistical Analysis**: Comprehensive summary statistics and correlations
- **Code Generation**: Create and execute custom Python code for specific analyses
- **Interactive Visualizations**: Dynamic, web-ready charts with zoom and hover capabilities

### 📊 Interactive Dashboard
- **Progressive Disclosure**: Upload → Auto-insights → Detailed analysis workflow
- **Dashboard + Chat Layout**: Auto-generated dashboard (left) + chat interface (right)
- **Custom Column Selection**: Choose specific columns for each visualization type
- **Executed Code Results**: See both generated code and its execution output
- **Multi-Dataset Support**: Upload and analyze multiple datasets in one session

### 🛡️ Reliability & Safety
- **Graceful Fallbacks**: Works even without API key using rule-based processing
- **Safe Code Execution**: Sandboxed environment for generated code
- **Error Recovery**: Helpful suggestions when things go wrong
- **Memory Management**: Efficient handling of large datasets

## 🚀 Quick Start

### Option 1: Direct Dashboard (Recommended)
```bash
# Install requirements
pip install -r requirements.txt

# Add your OpenAI API key to key.txt (optional but recommended)
echo "your-openai-api-key" > key.txt

# Launch the direct dashboard
python run_direct_dashboard.py
```
Open your browser to http://localhost:8503

### Option 2: API + Dashboard
```bash
# Terminal 1: Start API server
python run_api.py

# Terminal 2: Start dashboard
python run_dashboard.py
```
Open your browser to http://localhost:8501

## 💡 Usage Examples

### Data Exploration
```
"What columns do I have in this dataset?"
"Show me the first 10 rows"
"What are the data types of each column?"
```

### Statistical Analysis
```
"Calculate the correlation between income and spending"
"What's the average age grouped by customer segment?"
"Show me summary statistics for all numeric columns"
```

### Visualization Requests
```
"Create a scatter plot of age vs income colored by category"
"Make a bar chart showing sales by product category"
"Generate a histogram of customer ages"
```

### Code Generation & Execution
```
"Generate code to calculate total sales by region"
"Write code to find customers with above-average spending"
"Create code to identify the top 10 products by profit margin"
```

### Insight Discovery
```
"What patterns do you see in my data?"
"Are there any outliers in customer spending?"
"What factors seem to influence customer satisfaction?"
```

## 🏗️ Architecture

The Visual Analytics Agent uses a multi-agent architecture:

```
User Query → DataAnalyst → LangChain Agent → Tools → Results
                ↓
         [Visualizer, CodeGenerator, InsightGenerator]
```

### Agent Responsibilities

| Agent | Purpose | Key Methods |
|-------|---------|-------------|
| **DataAnalyst** | Main orchestrator, query understanding | `process_query()`, `analyze_query()` |
| **Visualizer** | Chart creation and recommendations | `create_web_chart()`, `suggest_visualizations()` |
| **CodeGenerator** | Python code generation for analysis | `generate_analysis_code()` |
| **InsightGenerator** | Pattern discovery and recommendations | `generate_insights()`, `detect_outliers()` |

## 📊 Dashboard Guide

### Layout
- **Left Side**: Auto-generated dashboard with upload, key metrics, and instant insights
- **Right Side**: Chat interface for natural language questions  
- **Bottom**: Detailed custom analysis and advanced visualizations

### Features
- **📁 Upload & Go**: Drag-and-drop CSV, JSON, or Excel files
- **🤖 Auto-Insights**: Instant pattern detection and key findings
- **💬 Chat Interface**: Ask questions in natural language
- **📊 Custom Analytics**: Advanced analysis types (correlation, outliers, clustering)
- **🔧 Code Generation & Execution**: Generate custom Python code and see live results
- **📈 Custom Charts**: Interactive Plotly visualizations with column selection
- **📋 Result Export**: Download analysis results and charts

### Visualization Column Selection
The dashboard now allows you to select specific columns for each chart type:
- **Scatter/Line/Bar**: Select X-axis, Y-axis, and optional color-by columns
- **Histogram**: Select column and optional color-by category
- **Pie Chart**: Select category column and optional values column
- **Heatmap**: Automatically uses correlation matrix of numeric columns

## 🔧 Configuration

### Environment Variables
```bash
# Required for enhanced features
export OPENAI_API_KEY="your-openai-api-key"

# Optional configurations
export AGENT_TEMPERATURE=0.1
export AGENT_MAX_TOKENS=2000
export AGENT_TIMEOUT=30
```

### API Key Setup
1. Create a file named `key.txt` in the project root
2. Add your OpenAI API key as the only content of the file
3. The dashboard will automatically load this key

## 🛠️ Development & Extension

### Adding New Tools
```python
from langchain.tools import Tool

def custom_analysis_tool(input_text: str) -> str:
    # Your custom analysis logic
    return "Analysis result"

# Add to DataAnalyst
tool = Tool(
    name="custom_analysis",
    description="Performs custom analysis",
    func=custom_analysis_tool
)
```

### Custom Prompts
```python
from src.config.settings import get_agent_prompts

prompts = get_agent_prompts()
prompts["custom_intent"] = "Your custom prompt template here"
```

## 🆘 Troubleshooting

### Common Issues

**"No OpenAI API key provided"**
- Create a `key.txt` file with your API key
- The agent will still work in fallback mode without an API key

**"Upload fails with error"**
- Check file format (CSV headers, valid JSON)
- Try smaller file sizes first
- Ensure no special characters in filename

**"Visualization doesn't show expected columns"**
- Use the new column selection feature to choose specific columns
- Different chart types require different column types (numeric vs categorical)

**"Code execution fails"**
- Check that your code references the correct column names
- Ensure operations are valid for the data types
- Look for syntax errors in the generated code

## 📚 Additional Resources

- **DASHBOARD_GUIDE.md**: Comprehensive guide to using the dashboard
- **ENHANCED_README.md**: Detailed technical documentation
- **project_design.md**: Architecture and design principles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using LangChain, GPT-4o, Streamlit, and Python**
