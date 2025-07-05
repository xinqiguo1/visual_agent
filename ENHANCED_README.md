# 🚀 Visual Analytics Agent v2.0 - LangChain Enhanced

An intelligent data analysis assistant powered by GPT-4o and LangChain, capable of understanding natural language queries and performing sophisticated data analysis through conversational interaction.

## ✨ Key Features

### 🧠 **LangChain Integration**
- **GPT-4o powered** natural language understanding
- **Multi-agent architecture** with specialized roles
- **Conversation memory** for context-aware interactions
- **Tool orchestration** for complex multi-step analysis

### 🔧 **Core Capabilities**
- **Data Exploration**: Automatic profiling and structure analysis
- **Statistical Analysis**: Correlations, distributions, summary statistics
- **Interactive Visualizations**: Plotly-powered web-ready charts with zoom, hover, and filtering
- **Multi-format Charts**: Scatter plots, bar charts, heatmaps, histograms, box plots, and more
- **Code Generation**: Python code creation with safety validation
- **Insight Discovery**: Automated pattern detection and recommendations
- **Data Quality Assessment**: Missing values, outliers, data type analysis
- **Dashboard Creation**: Multi-chart dashboards with professional layouts

### 🛡️ **Safety & Reliability**
- **Graceful fallback** to rule-based processing without API key
- **Code validation** and security checks
- **Error handling** with helpful suggestions
- **Memory management** for large datasets

## 🏗️ **Architecture Overview**

```
User Query → DataAnalyst → LangChain Agent → Tools → Results
                ↓
         [Visualizer, CodeGenerator, InsightGenerator]
```

### **Agent Responsibilities**

| Agent | Purpose | Key Methods |
|-------|---------|-------------|
| **DataAnalyst** | Main orchestrator, query understanding | `process_query()`, `analyze_query()` |
| **Visualizer** | Chart creation and recommendations | `create_chart()`, `suggest_visualizations()` |
| **CodeGenerator** | Python code generation for analysis | `generate_analysis_code()`, `validate_code()` |
| **InsightGenerator** | Pattern discovery and recommendations | `generate_insights()`, `detect_outliers()` |

## 📦 **Installation**

1. **Clone the repository**
```bash
git clone <repository-url>
cd visual_agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key** (for enhanced features)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🚀 **Quick Start**

### **Basic Usage (No API Key Required)**

```python
import pandas as pd
from src.agents import DataAnalyst

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize agent
analyst = DataAnalyst()

# Load dataset
summary = analyst.load_dataset(df)
print("Dataset loaded:", summary)

# Ask questions (works without API key!)
result = analyst.process_query("What columns do I have?")
print(result['explanation'])

result = analyst.process_query("Show me summary statistics")
print(result['explanation'])
```

### **Enhanced Mode (With API Key)**

```python
import os
from src.agents import DataAnalyst

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Initialize enhanced agent
analyst = DataAnalyst()

# Check status
print("Agent Status:", analyst.get_status())

# Advanced queries powered by GPT-4o
result = analyst.process_query("Analyze the correlation between age and income, and suggest visualizations")
print(result['explanation'])

result = analyst.process_query("Find interesting patterns in customer behavior")
print(result['explanation'])

result = analyst.process_query("Generate Python code to identify high-value customers")
print(result['explanation'])
```

## 🎯 **Example Queries**

### **Data Exploration**
- "What columns do I have in my dataset?"
- "Show me the shape and basic info about my data"
- "What are the data types of each column?"

### **Statistical Analysis**
- "Calculate the correlation between income and spending"
- "What's the average age of customers?"
- "Show me summary statistics for all numeric columns"

### **Interactive Visualization Requests**
- "Create an interactive scatter plot of age vs income colored by category"
- "Make a beautiful bar chart showing product categories"
- "Generate a correlation heatmap for all numeric variables"
- "Create a box plot comparing spending by customer segment"
- "Show me a histogram of spending scores with category breakdown"

### **Insight Discovery**
- "What patterns do you see in my data?"
- "Find outliers in customer spending"
- "Generate insights about data quality"

### **Code Generation**
- "Generate code to filter high-value customers"
- "Create Python code for correlation analysis"
- "Show me code to visualize sales trends"

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required for enhanced features
export OPENAI_API_KEY="your-openai-api-key"

# Optional configurations
export AGENT_TEMPERATURE=0.1
export AGENT_MAX_TOKENS=2000
export AGENT_TIMEOUT=30
```

### **Agent Configuration**
```python
from src.config.settings import AgentConfig

# Customize settings
AgentConfig.CORRELATION_THRESHOLD = 0.5
AgentConfig.MAX_CONVERSATION_HISTORY = 20
AgentConfig.ENABLE_SAFETY_CHECKS = True
```

## 🛠️ **Development & Extension**

### **Adding New Tools**
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

### **Custom Prompts**
```python
from src.config.settings import get_agent_prompts

prompts = get_agent_prompts()
prompts["custom_intent"] = "Your custom prompt template here"
```

## 📊 **Demo Scripts**

### **Main Demo** (with web visualizations)
```bash
python demo_langchain_agent.py
```

### **Web Visualization Demo** (comprehensive testing)
```bash
python demo_web_visualizations.py
```

### **What the demos include:**
1. **Sample Data Creation**: Realistic customer dataset with correlations
2. **Fallback Mode Testing**: Works without API key
3. **Enhanced Mode**: Full LangChain + GPT-4o capabilities
4. **Interactive Charts**: Plotly-powered web visualizations
5. **Dashboard Creation**: Multi-chart professional layouts
6. **File Output**: HTML files ready for web deployment

### **Output Files:**
- Interactive HTML charts saved to `outputs/` folder
- Dashboard files combining multiple visualizations
- All charts are web-ready and mobile-responsive

## 🔍 **Features Comparison**

| Feature | Without API Key | With GPT-4o API Key |
|---------|----------------|-------------------|
| Data Exploration | ✅ Rule-based | ✅ Enhanced with NLP |
| Basic Statistics | ✅ Standard calculations | ✅ Context-aware analysis |
| Visualizations | ✅ Suggestions only | ✅ Smart recommendations |
| Code Generation | ✅ Template-based | ✅ Custom code generation |
| Query Understanding | ✅ Keyword matching | ✅ Natural language parsing |
| Conversation Memory | ❌ Basic history | ✅ Context-aware memory |
| Multi-step Analysis | ❌ Single operations | ✅ Complex workflows |
| Insight Discovery | ✅ Automated patterns | ✅ AI-powered insights |

## 🚧 **Roadmap**

### **Phase 1: Core Enhancement** ✅
- [x] LangChain integration
- [x] GPT-4o support
- [x] Multi-agent architecture
- [x] Conversation memory
- [x] Tool orchestration

### **Phase 2: Advanced Features** 🚧
- [ ] Web interface (FastAPI)
- [ ] Real-time visualization updates
- [ ] Database connectivity
- [ ] Advanced statistical models
- [ ] Report generation

### **Phase 3: Enterprise Features** 🔮
- [ ] Multi-user support
- [ ] Authentication & authorization
- [ ] Scheduled analysis
- [ ] API endpoints
- [ ] Cloud deployment

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

### **Common Issues**

**"LangChain agent failed"**
- Check your OpenAI API key
- Verify internet connection
- Ensure sufficient API credits

**"No dataset loaded"**
- Call `analyst.load_dataset(df)` first
- Verify DataFrame is not empty

**"Module not found"**
- Install requirements: `pip install -r requirements.txt`
- Check Python path and virtual environment

### **Getting Help**

1. Check the demo script: `python demo_langchain_agent.py`
2. Review agent status: `analyst.get_status()`
3. Enable verbose mode for debugging
4. Check conversation history: `analyst.get_conversation_history()`

---

**Built with ❤️ using LangChain, GPT-4o, and Python** 