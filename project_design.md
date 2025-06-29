# Visual Analytics Agent - Project Design Document

## Technology Stack Recommendation

### Backend:
- **FastAPI** - REST API & WebSocket support for real-time interaction
- **LangChain** - Agent framework for LLM orchestration
- **Pandas/Polars** - Data manipulation and analysis
- **Matplotlib/Plotly** - Visualization libraries
- **SQLite/DuckDB** - Query engine for large datasets
- **Docker** - Sandboxed code execution environment

### Frontend:
- **React/Vue.js** - Web interface framework
- **File upload components** - Drag & drop data upload
- **Chat interface** - Conversational UI components
- **Chart rendering components** - Interactive visualization display

### LLM Integration:
- **OpenAI GPT-4** - Primary language model
- **Anthropic Claude** - Fallback option
- **Local models** - For privacy-focused deployments

## Agent Workflow Design

### Core Workflow: 
- User Query → Intent Classification → Tool Selection → Code Generation → Execution → Result Interpretation → Response

### Intent Categories:
- **Data exploration** - "What columns do I have?", "Show me the first 10 rows"
- **Statistical queries** - "What's the average?", "Calculate correlation"
- **Visualization requests** - "Show me a chart", "Plot sales over time"
- **Filtering/aggregation** - "Show me data where sales > 1000"
- **Comparative analysis** - "Compare A vs B", "Show trends"

### Agent Decision Flow:
1. **Parse Query** - Extract intent and parameters
2. **Validate Context** - Ensure data is available
3. **Generate Code** - Create Python analysis code
4. **Execute Safely** - Run in sandboxed environment
5. **Interpret Results** - Analyze outputs
6. **Format Response** - Present in natural language + visuals

## Key Features to Implement

### Phase 1: Core Functionality
- **File upload and parsing** - Support CSV, Excel, JSON formats
- **Basic data profiling** - Automatic schema inference, data types
- **Simple statistical queries** - Mean, median, count, sum operations
- **Standard chart types** - Bar, line, scatter, histogram plots
- **Error handling** - Graceful failure recovery and user feedback

### Phase 2: Advanced Analytics
- **Custom aggregations** - Group by operations, pivot tables
- **Time series analysis** - Trend detection, seasonal decomposition
- **Correlation analysis** - Feature relationships, heatmaps
- **Interactive visualizations** - Zoom, filter, hover capabilities
- **Export capabilities** - Download charts, data, generated code

### Phase 3: Intelligence Features
- **Automated insight generation** - Proactive data discoveries
- **Anomaly detection** - Outlier identification and highlighting
- **Trend identification** - Pattern recognition in time series
- **Recommendation system** - Suggest relevant analyses
- **Natural language explanations** - Interpret results in plain English

## Project Structure

visual_agent/
├── src/
│ ├── agents/
│ │ ├── init.py
│ │ ├── data_analyst.py # Main analyst agent
│ │ ├── visualizer.py # Chart generation agent
│ │ ├── code_generator.py # Python code generation
│ │ └── insight_generator.py # Automated insights
│ ├── tools/
│ │ ├── init.py
│ │ ├── data_processor.py # Data loading and cleaning
│ │ ├── stats_calculator.py # Statistical operations
│ │ ├── chart_generator.py # Visualization creation
│ │ └── query_executor.py # Safe code execution
│ ├── api/
│ │ ├── init.py
│ │ ├── main.py # FastAPI application
│ │ └── routes/
│ │ ├── init.py
│ │ ├── data.py # Data upload/management
│ │ ├── analysis.py # Analysis endpoints
│ │ └── visualization.py # Chart endpoints
│ ├── frontend/
│ │ ├── src/
│ │ │ ├── components/
│ │ │ │ ├── DataUpload.jsx
│ │ │ │ ├── ChatInterface.jsx
│ │ │ │ └── ChartDisplay.jsx
│ │ │ └── pages/
│ │ │ ├── Dashboard.jsx
│ │ │ └── Analysis.jsx
│ │ ├── public/
│ │ └── package.json
│ ├── models/
│ │ ├── init.py
│ │ ├── data_schema.py # Data models
│ │ └── response_models.py # API response models
│ └── utils/
│ ├── init.py
│ ├── file_handlers.py # File processing utilities
│ ├── security.py # Input validation, sanitization
│ └── logging.py # Logging configuration
├── data/
│ ├── uploads/ # User uploaded files
│ ├── cache/ # Processed data cache
│ └── examples/ # Sample datasets
├── tests/
│ ├── init.py
│ ├── test_agents/
│ ├── test_tools/
│ └── test_api/
├── docker/
│ ├── Dockerfile
│ ├── docker-compose.yml
│ └── sandbox/ # Code execution environment
├── docs/
│ ├── api_documentation.md
│ ├── user_guide.md
│ └── deployment_guide.md
├── config/
│ ├── settings.py # Application configuration
│ └── prompts/ # LLM prompt templates
├── requirements.txt
├── pyproject.toml
└── .env.example

### Key Architecture Principles:
- **Modular Design** - Separate concerns for maintainability
- **Scalable Structure** - Easy to add new features and tools
- **Security First** - Sandboxed execution and input validation
- **Test Coverage** - Comprehensive testing for reliability
- **Documentation** - Clear docs for users and developers


