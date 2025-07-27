"""
Data Analyst Agent

Main agent responsible for understanding user queries about data and 
coordinating analysis tasks with full LangChain integration.
"""

# Standard library imports
from typing import Dict, Any, List, Optional, Union
import json
import traceback
import sys
import io
import contextlib
import warnings
from datetime import datetime

# Data analysis imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Numpy conversion utility
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'name') and hasattr(obj, 'type'):  # pandas dtype objects
        return str(obj)
    elif str(type(obj)).startswith("<class 'pandas"):  # any pandas objects
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# LangChain imports
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ..config.settings import get_llm, get_agent_prompts, get_tool_descriptions, AgentConfig
from .visualizer import Visualizer
from .code_generator import CodeGenerator
from .insight_generator import InsightGenerator


class DataAnalyst:
    """
    Main data analyst agent that processes natural language queries
    and coordinates data analysis tasks using LangChain.
    """
    
    def __init__(self, llm=None, api_key=None):
        """
        Initialize the data analyst agent with LangChain integration.
        
        Args:
            llm: Optional pre-configured LLM instance
            api_key: OpenAI API key for LLM initialization
        """
        # Initialize LLM
        self.llm = llm or get_llm(api_key)
        
        # Initialize other agents
        self.visualizer = Visualizer()
        self.code_generator = CodeGenerator()
        self.insight_generator = InsightGenerator()
        
        # Dataset state
        self.current_dataset = None
        self.dataset_metadata = {}
        
        # LangChain memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=AgentConfig.MAX_CONVERSATION_HISTORY
        )
        
        # Initialize tools and agent
        self.tools = self._create_langchain_tools()
        self.agent_executor = self._create_langchain_agent() if self.llm else None
        
        # Prompts
        self.prompts = get_agent_prompts()
        
        # Fallback conversation history for when LLM is not available
        self.fallback_history = []
        
    def load_dataset(self, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a dataset for analysis.
        
        Args:
            data: Pandas DataFrame containing the dataset
            metadata: Optional metadata about the dataset
            
        Returns:
            Dictionary with dataset summary information
        """
        self.current_dataset = data
        self.dataset_metadata = metadata or {}
        
        # Clean the data to handle any remaining list objects
        cleaned_data = data.copy()
        for col in cleaned_data.columns:
            # Convert any remaining list objects to strings
            if cleaned_data[col].apply(lambda x: isinstance(x, (list, dict))).any():
                cleaned_data[col] = cleaned_data[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        
        self.current_dataset = cleaned_data
        
        # Generate comprehensive dataset summary
        summary = {
            "shape": cleaned_data.shape,
            "columns": list(cleaned_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in cleaned_data.dtypes.items()},
            "missing_values": convert_numpy_types(cleaned_data.isnull().sum().to_dict()),
            "memory_usage": convert_numpy_types(cleaned_data.memory_usage(deep=True).sum()),
            "numeric_columns": list(cleaned_data.select_dtypes(include=['number']).columns),
            "categorical_columns": list(cleaned_data.select_dtypes(include=['object']).columns),
            "date_columns": list(cleaned_data.select_dtypes(include=['datetime']).columns),
            "memory_mb": convert_numpy_types(round(cleaned_data.memory_usage(deep=True).sum() / 1024**2, 2)),
            "duplicates": convert_numpy_types(cleaned_data.duplicated().sum()),
            "total_missing": convert_numpy_types(cleaned_data.isnull().sum().sum())
        }
        
        # Clear conversation memory when new dataset is loaded
        if self.memory:
            self.memory.clear()
        self.fallback_history.clear()
        
        # Generate automatic insights for the new dataset
        if len(data) > 0:
            try:
                auto_insights = self.insight_generator.get_automated_insights(data, top_n=3)
                summary["auto_insights"] = auto_insights
            except Exception as e:
                summary["auto_insights"] = [f"Could not generate insights: {str(e)}"]
        
        return summary
    
    def _create_langchain_tools(self) -> List[Tool]:
        """Create LangChain tools for the agent."""
        tool_descriptions = get_tool_descriptions()
        
        return [
            Tool(
                name="data_exploration",
                description=tool_descriptions["data_exploration"],
                func=self._data_exploration_tool
            ),
            Tool(
                name="statistical_analysis",
                description=tool_descriptions["statistical_analysis"],
                func=self._statistical_analysis_tool
            ),
            Tool(
                name="visualization",
                description=tool_descriptions["visualization"],
                func=self._visualization_tool
            ),
            Tool(
                name="code_generation",
                description=tool_descriptions["code_generation"],
                func=self._code_generation_tool
            ),
            Tool(
                name="insight_generation",
                description=tool_descriptions["insight_generation"],
                func=self._insight_generation_tool
            ),
            Tool(
                name="data_filtering",
                description=tool_descriptions["data_filtering"],
                func=self._data_filtering_tool
            )
        ]
    
    def _create_langchain_agent(self) -> Optional[AgentExecutor]:
        """Create LangChain agent with tools."""
        if not self.llm:
            return None
        
        try:
            # Create the agent prompt with required ReAct template variables
            prompt = PromptTemplate.from_template("""You are an expert data analyst assistant. You help users analyze their datasets through natural language queries.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Available tools:
- data_exploration: Get dataset info, columns, shape, overview
- statistical_analysis: Calculate statistics, correlations, summaries  
- visualization: Create charts and plots
- code_generation: Generate Python code for analysis - IMPORTANT: When using this tool, be very specific about what code you need. Ask for code that solves the exact problem, such as "Generate code to calculate total sales by region" or "Generate code to find which region has the highest sales". The tool will create custom code for your specific needs.
- insight_generation: Find patterns and insights automatically
- data_filtering: Filter and subset data

When analyzing data:
1. Understand what the user wants to know
2. Use the appropriate tools to gather information or perform analysis
3. Provide clear, helpful responses
4. If you need to use multiple tools, explain your reasoning

When using the code_generation tool:
1. Be extremely specific about what code you need
2. Include all relevant details (columns, operations, etc.)
3. Ask for code that directly answers the user's question
4. Don't accept generic code - if the code doesn't solve the specific problem, ask again with more details

Always be helpful and provide actionable insights. If something cannot be done, explain why and suggest alternatives.

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
            
            # Create the agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=AgentConfig.TIMEOUT_SECONDS
            )
            
            return agent_executor
            
        except Exception as e:
            print(f"Warning: Could not create LangChain agent: {e}")
            return None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a natural language query using LLM or fallback to rule-based.
        
        Args:
            query: Natural language query from user
            
        Returns:
            Dictionary with query analysis results
        """
        if self.llm and self.current_dataset is not None:
            return self._analyze_query_with_llm(query)
        else:
            return self._analyze_query_fallback(query)
    
    def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Analyze query using LLM for enhanced understanding."""
        try:
            # Intent classification
            intent_prompt = self.prompts["intent_classification"].format(
                query=query,
                columns=list(self.current_dataset.columns),
                dtypes={col: str(dtype) for col, dtype in self.current_dataset.dtypes.items()}
            )
            
            intent = self.llm.predict(intent_prompt).strip().lower()
            
            # Parameter extraction
            param_prompt = self.prompts["parameter_extraction"].format(
                query=query,
                columns=list(self.current_dataset.columns),
                numeric_columns=list(self.current_dataset.select_dtypes(include=['number']).columns),
                categorical_columns=list(self.current_dataset.select_dtypes(include=['object']).columns)
            )
            
            try:
                params_response = self.llm.predict(param_prompt)
                # Try to parse JSON response
                parameters = json.loads(params_response.strip())
            except (json.JSONDecodeError, Exception):
                # Fallback to empty parameters
                parameters = {}
            
            return {
                "query": query,
                "intent": intent,
                "parameters": parameters,
                "requires_data": intent not in ["general", "help"],
                "complexity": "enhanced",
                "method": "llm"
            }
            
        except Exception as e:
            print(f"LLM analysis failed: {e}, falling back to rule-based")
            return self._analyze_query_fallback(query)
    
    def _analyze_query_fallback(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based query analysis."""
        intent = self._classify_intent(query.lower())
        
        return {
            "query": query,
            "intent": intent,
            "parameters": {},
            "requires_data": intent != "general",
            "complexity": "simple",
            "suggested_actions": self._suggest_actions(intent),
            "method": "rule_based"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using LangChain agent or fallback methods.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with analysis results
        """
        if self.current_dataset is None:
            return {
                "error": "No dataset loaded. Please upload a dataset first.",
                "type": "no_data",
                "suggestion": "Please upload a CSV, Excel, or other data file to begin analysis."
            }
        
        # Record start time
        start_time = datetime.now()
        
        try:
            # Try LangChain agent first if available
            if self.agent_executor and self.llm:
                result = self._process_with_langchain_agent(query)
            else:
                result = self._process_with_fallback(query)
            
            # Add metadata
            result.update({
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "dataset_shape": self.current_dataset.shape,
                "method": "langchain" if self.agent_executor else "fallback"
            })
            
            # Store in appropriate history
            if self.memory:
                self.memory.save_context(
                    {"input": query},
                    {"output": result.get("explanation", str(result))}
                )
            else:
                self.fallback_history.append({
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now()
                })
            
            return result
            
        except Exception as e:
            error_result = {
                "error": f"Error processing query: {str(e)}",
                "type": "processing_error",
                "query": query,
                "traceback": traceback.format_exc(),
                "suggestion": "Try rephrasing your question or check if your data is properly loaded."
            }
            
            print(f"Error processing query '{query}': {e}")
            return error_result
    
    def _process_with_langchain_agent(self, query: str) -> Dict[str, Any]:
        """Process query using LangChain agent."""
        try:
            # Get dataset context for the agent
            dataset_info = self._get_dataset_context()
            
            # Combine query with dataset info to provide context
            enhanced_query = f"""Dataset Context:
{dataset_info}

User Query: {query}"""
            
            # Run the agent with only the input parameter
            response = self.agent_executor.invoke({
                "input": enhanced_query
            })
            
            return {
                "type": "agent_response",
                "result": response.get("output", ""),
                "explanation": response.get("output", ""),
                "success": True,
                "agent_used": True
            }
            
        except Exception as e:
            print(f"LangChain agent failed: {e}, falling back to rule-based processing")
            return self._process_with_fallback(query)
    
    def _process_with_fallback(self, query: str) -> Dict[str, Any]:
        """Process query using fallback rule-based methods."""
        analysis = self.analyze_query(query)
        
        # Process based on intent
        if analysis["intent"] == "exploration":
            return self._handle_exploration_query(query, analysis.get("parameters", {}))
        elif analysis["intent"] == "statistics":
            return self._handle_statistics_query(query, analysis.get("parameters", {}))
        elif analysis["intent"] == "visualization":
            return self._handle_visualization_query(query, analysis.get("parameters", {}))
        elif analysis["intent"] == "filtering":
            return self._handle_filtering_query(query, analysis.get("parameters", {}))
        elif analysis["intent"] == "insights":
            return self._handle_insights_query(query, analysis.get("parameters", {}))
        else:
            return self._handle_general_query(query)
    
    def _get_dataset_context(self) -> str:
        """Get formatted dataset context for the agent."""
        if self.current_dataset is None:
            return "No dataset loaded"
        
        context = f"""
Dataset Summary:
- Shape: {self.current_dataset.shape[0]} rows, {self.current_dataset.shape[1]} columns
- Columns: {list(self.current_dataset.columns)}
- Numeric columns: {list(self.current_dataset.select_dtypes(include=['number']).columns)}
- Categorical columns: {list(self.current_dataset.select_dtypes(include=['object']).columns)}
- Missing values: {self.current_dataset.isnull().sum().sum()} total
- Memory usage: {round(self.current_dataset.memory_usage(deep=True).sum() / 1024**2, 2)} MB
"""
        return context.strip()
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of a user query."""
        if any(word in query for word in ["show", "display", "plot", "chart", "graph", "visualize"]):
            return "visualization"
        elif any(word in query for word in ["average", "mean", "median", "sum", "count", "min", "max", "std"]):
            return "statistics"
        elif any(word in query for word in ["filter", "where", "select", "subset", "rows"]):
            return "filtering"
        elif any(word in query for word in ["columns", "shape", "info", "describe", "overview", "summary"]):
            return "exploration"
        else:
            return "general"
    
    def _suggest_actions(self, intent: str) -> List[str]:
        """Suggest possible actions based on intent."""
        suggestions = {
            "exploration": ["Show dataset info", "Display column names", "Show data types"],
            "statistics": ["Calculate summary statistics", "Show correlations", "Compute aggregations"],
            "visualization": ["Create bar chart", "Generate scatter plot", "Show distribution"],
            "filtering": ["Filter by condition", "Select columns", "Sample data"],
            "general": ["Explore the data", "Show basic statistics", "Create visualizations"]
        }
        return suggestions.get(intent, [])
    
    # LangChain Tool Methods
    def _data_exploration_tool(self, input_text: str) -> str:
        """Tool for data exploration queries."""
        if self.current_dataset is None:
            return "No dataset loaded. Please upload data first."
        
        try:
            info = {
                "shape": self.current_dataset.shape,
                "columns": list(self.current_dataset.columns),
                "dtypes": dict(self.current_dataset.dtypes.astype(str)),
                "missing_values": convert_numpy_types(dict(self.current_dataset.isnull().sum())),
                "memory_usage_mb": convert_numpy_types(round(self.current_dataset.memory_usage(deep=True).sum() / 1024**2, 2))
            }
            
            return f"""Dataset Information:
- Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
- Columns: {', '.join(info['columns'])}
- Data types: {info['dtypes']}
- Missing values: {info['missing_values']}
- Memory usage: {info['memory_usage_mb']} MB

Sample data (first 3 rows):
{self.current_dataset.head(3).to_string()}"""
            
        except Exception as e:
            return f"Error exploring data: {str(e)}"
    
    def _statistical_analysis_tool(self, input_text: str) -> str:
        """Tool for statistical analysis."""
        if self.current_dataset is None:
            return "No dataset loaded."
        
        try:
            numeric_data = self.current_dataset.select_dtypes(include=['number'])
            if len(numeric_data.columns) == 0:
                return "No numeric columns found for statistical analysis."
            
            stats_summary = numeric_data.describe()
            correlations = numeric_data.corr()
            
            return f"""Statistical Analysis:

Summary Statistics:
{stats_summary.to_string()}

Correlation Matrix:
{correlations.to_string()}

Key Insights:
- Numeric columns: {len(numeric_data.columns)}
- Highest correlation: {correlations.abs().unstack().sort_values(ascending=False).drop_duplicates().iloc[1]:.3f}
"""
        except Exception as e:
            return f"Error in statistical analysis: {str(e)}"
    
    def _visualization_tool(self, input_text: str) -> str:
        """Enhanced tool for creating web-ready interactive visualizations."""
        if self.current_dataset is None:
            return "No dataset loaded."
        
        try:
            # Parse visualization request
            request_lower = input_text.lower()
            
            # Determine chart type
            chart_type = "scatter"  # default
            if "bar" in request_lower or "count" in request_lower:
                chart_type = "bar"
            elif "line" in request_lower or "trend" in request_lower:
                chart_type = "line"
            elif "histogram" in request_lower or "distribution" in request_lower:
                chart_type = "histogram"
            elif "box" in request_lower:
                chart_type = "box"
            elif "heatmap" in request_lower or "correlation" in request_lower:
                chart_type = "heatmap"
            elif "pie" in request_lower:
                chart_type = "pie"
            elif "violin" in request_lower:
                chart_type = "violin"
            elif "area" in request_lower:
                chart_type = "area"
            
            # Get column suggestions
            numeric_cols = list(self.current_dataset.select_dtypes(include=['number']).columns)
            categorical_cols = list(self.current_dataset.select_dtypes(include=['object']).columns)
            
            # Smart column selection
            x_col = None
            y_col = None
            color_col = None
            
            if chart_type == "heatmap":
                # Heatmap doesn't need specific columns
                pass
            else:
                # Determine columns based on chart type and available data
                if chart_type in ["bar", "pie"]:
                    if len(categorical_cols) >= 1:
                        x_col = categorical_cols[0]
                        if len(numeric_cols) >= 1 and chart_type == "bar":
                            y_col = numeric_cols[0]
                elif chart_type == "histogram":
                    if len(numeric_cols) >= 1:
                        x_col = numeric_cols[0]
                else:
                    # For scatter, line, box, etc.
                    if len(numeric_cols) >= 2:
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                    elif len(numeric_cols) >= 1:
                        x_col = numeric_cols[0]
                        if len(categorical_cols) >= 1:
                            y_col = x_col
                            x_col = categorical_cols[0]
                
                # Add color column for enhanced interactivity
                if len(categorical_cols) >= 1 and categorical_cols[0] not in [x_col, y_col]:
                    color_col = categorical_cols[0]
            
            # Create web-ready interactive chart
            result = self.visualizer.create_web_chart(
                self.current_dataset, 
                chart_type, 
                x_col, 
                y_col, 
                color_col
            )
            
            if result.get("success"):
                response = f"✅ Created interactive {chart_type} visualization!\n\n"
                response += f"📊 Chart details:\n"
                response += f"   • Type: {chart_type}\n"
                if x_col:
                    response += f"   • X-axis: {x_col}\n"
                if y_col:
                    response += f"   • Y-axis: {y_col}\n"
                if color_col:
                    response += f"   • Color: {color_col}\n"
                response += f"\n📁 Saved to: {result['html_path']}\n"
                response += f"🌐 Open the HTML file in your browser to view the interactive chart!\n"
                
                # Add suggestions for other charts
                suggestions = self.visualizer.suggest_web_visualizations(self.current_dataset)
                if suggestions:
                    suggestion_text = ", ".join([s["type"] for s in suggestions[:3]])
                    response += f"\n💡 Other suggestions: {suggestion_text}"
                
                return response
            else:
                return f"❌ Error creating visualization: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error in visualization tool: {str(e)}"
    
    def _code_generation_tool(self, input_text: str) -> str:
        """Tool for generating and executing Python code."""
        if self.current_dataset is None:
            return "No dataset loaded."
        
        try:
            # If LLM is available, use it to generate custom code
            if self.llm:
                # Create a prompt for the LLM to generate custom code
                columns_str = ", ".join(list(self.current_dataset.columns))
                numeric_cols = ", ".join(list(self.current_dataset.select_dtypes(include=['number']).columns))
                categorical_cols = ", ".join(list(self.current_dataset.select_dtypes(include=['object']).columns))
                
                # Sample data description
                sample_data = self.current_dataset.head(3).to_string()
                
                # Create a prompt for code generation
                code_prompt = f"""Generate Python code to solve this specific data analysis task:

Task: {input_text}

Dataset information:
- Shape: {self.current_dataset.shape}
- All columns: {columns_str}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

Sample data:
{sample_data}

Generate complete, executable Python code that:
1. Directly addresses the specific task requested
2. Uses pandas operations on the variable 'df' (the dataset is already loaded as 'df')
3. Includes helpful comments
4. Stores the final result in a variable called 'result'
5. Prints the result clearly

Important:
- The dataset is already available as 'df'
- Do not include data loading code
- Store your final answer in a variable called 'result'
- Use print() to display the result

Example format:
# Calculate total sales by region
result = df.groupby('region')['sales'].sum()
print("Total sales by region:")
print(result)

The code should be specific to this exact task, not a generic template.
"""
                
                # Get code from LLM
                try:
                    generated_code = self.llm.predict(code_prompt)
                    
                    # Clean up the code (remove markdown code blocks if present)
                    if "```python" in generated_code:
                        generated_code = generated_code.split("```python")[1]
                        if "```" in generated_code:
                            generated_code = generated_code.split("```")[0]
                    elif "```" in generated_code:
                        # Handle plain ``` blocks
                        code_parts = generated_code.split("```")
                        if len(code_parts) >= 3:
                            generated_code = code_parts[1]
                    
                    generated_code = generated_code.strip()
                    
                    # Execute the generated code safely
                    execution_result = self._execute_code_safely(generated_code)
                    
                    return f"""Generated and Executed Python Code for: {input_text}

**Generated Code:**
```python
{generated_code}
```

**Execution Result:**
{execution_result['output']}

**Status:** {'✅ Success' if execution_result['success'] else '❌ Error'}
{f"**Error Details:** {execution_result['error']}" if execution_result.get('error') else ""}
"""
                except Exception as e:
                    print(f"LLM code generation failed: {e}, falling back to template-based code")
                    # Fall through to template-based code generation
            
            # Fall back to template-based code generation with execution
            input_lower = input_text.lower()
            
            if "correlation" in input_lower:
                task_type = "statistics"
                params = {"operation": "correlation"}
            elif "mean" in input_lower or "average" in input_lower:
                task_type = "statistics"
                params = {"operation": "mean"}
            elif "plot" in input_lower or "chart" in input_lower:
                task_type = "visualization"
                params = {"chart_type": "scatter"}
            else:
                task_type = "statistics"
                params = {"operation": "describe"}
            
            # Generate code
            code_result = self.code_generator.generate_analysis_code(
                task_type=task_type,
                parameters=params,
                data_info={"columns": list(self.current_dataset.columns)}
            )
            
            if "error" in code_result:
                return f"Error generating code: {code_result['error']}"
            
            # Execute the template-generated code
            execution_result = self._execute_code_safely(code_result['code'])
            
            return f"""Generated and Executed Python Code:

**Generated Code:**
```python
{code_result['code']}
```

**Execution Result:**
{execution_result['output']}

**Status:** {'✅ Success' if execution_result['success'] else '❌ Error'}
{f"**Error Details:** {execution_result['error']}" if execution_result.get('error') else ""}

**Explanation:** {code_result.get('explanation', 'Performed the requested analysis')}
"""
        except Exception as e:
            return f"Error generating and executing code: {str(e)}"

    def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """
        Safely execute Python code and capture output.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dict containing execution results, output, and any errors
        """
        # Create a safe execution environment
        safe_globals = {
            'df': self.current_dataset.copy(),  # Provide the dataset
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'datetime': datetime,
            'print': print,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'round': round,
            'abs': abs,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'str': str,
            'int': int,
            'float': float,
            'result': None  # Initialize result variable
        }
        
        # Capture stdout to get print outputs
        captured_output = io.StringIO()
        
        try:
            # Suppress warnings during execution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Redirect stdout to capture print statements
                with contextlib.redirect_stdout(captured_output):
                    # Execute the code
                    exec(code, safe_globals)
            
            # Get the captured output
            output = captured_output.getvalue()
            
            # If there's a result variable, include it in the output
            if safe_globals.get('result') is not None:
                result_value = safe_globals['result']
                
                # Format the result nicely
                if hasattr(result_value, 'to_string'):  # Pandas DataFrame/Series
                    if not output or output.strip() == '':
                        output = str(result_value)
                elif isinstance(result_value, (dict, list)):
                    if not output or output.strip() == '':
                        output = str(result_value)
                elif isinstance(result_value, (int, float, str)):
                    if not output or output.strip() == '':
                        output = str(result_value)
            
            # If no output was captured, provide a default message
            if not output or output.strip() == '':
                output = "Code executed successfully. Result stored in 'result' variable."
                if safe_globals.get('result') is not None:
                    output += f"\nResult: {safe_globals['result']}"
            
            return {
                'success': True,
                'output': output.strip(),
                'result': safe_globals.get('result'),
                'error': None
            }
            
        except Exception as e:
            # Handle execution errors
            error_msg = str(e)
            
            # Provide more helpful error messages for common issues
            if "name" in error_msg and "is not defined" in error_msg:
                error_msg += "\nTip: Make sure you're using available variables (df, pd, np) and functions."
            elif "KeyError" in error_msg:
                error_msg += f"\nTip: Check column names. Available columns: {list(self.current_dataset.columns)}"
            
            return {
                'success': False,
                'output': captured_output.getvalue() or "No output generated before error.",
                'result': None,
                'error': error_msg
            }
            
        finally:
            captured_output.close()
    
    def _insight_generation_tool(self, input_text: str) -> str:
        """Tool for generating automatic insights."""
        if self.current_dataset is None:
            return "No dataset loaded."
        
        try:
            insights = self.insight_generator.generate_insights(self.current_dataset)
            
            summary = insights.get("summary", "No summary available")
            recommendations = insights.get("recommendations", [])
            
            key_findings = []
            for category in ["correlations", "outliers", "data_quality"]:
                if category in insights and insights[category]:
                    key_findings.extend([
                        item.get("message", str(item)) 
                        for item in insights[category][:2]
                    ])
            
            findings_text = "\n".join([f"- {finding}" for finding in key_findings[:5]])
            recs_text = "\n".join([f"- {rec}" for rec in recommendations[:3]])
            
            return f"""Automated Data Insights:

Summary: {summary}

Key Findings:
{findings_text}

Recommendations:
{recs_text}
"""
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def _data_filtering_tool(self, input_text: str) -> str:
        """Tool for data filtering operations."""
        if self.current_dataset is None:
            return "No dataset loaded."
        
        try:
            # Basic filtering suggestions
            numeric_cols = list(self.current_dataset.select_dtypes(include=['number']).columns)
            categorical_cols = list(self.current_dataset.select_dtypes(include=['object']).columns)
            
            return f"""Data Filtering Options:

Numeric Columns (can filter by value ranges):
{', '.join(numeric_cols) if numeric_cols else 'None'}

Categorical Columns (can filter by categories):
{', '.join(categorical_cols) if categorical_cols else 'None'}

Example filters:
- For numeric: "Show rows where {numeric_cols[0] if numeric_cols else 'column'} > 100"
- For categorical: "Filter by {categorical_cols[0] if categorical_cols else 'category'} = 'value'"

Current dataset size: {len(self.current_dataset)} rows
"""
        except Exception as e:
            return f"Error with filtering: {str(e)}"
    
    def _handle_exploration_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle data exploration queries."""
        parameters = parameters or {}
        
        if "columns" in query.lower():
            return {
                "type": "exploration",
                "result": {
                    "columns": list(self.current_dataset.columns),
                    "count": len(self.current_dataset.columns),
                    "dtypes": {col: str(dtype) for col, dtype in self.current_dataset.dtypes.items()}
                },
                "explanation": f"The dataset has {len(self.current_dataset.columns)} columns: {', '.join(list(self.current_dataset.columns))}"
            }
        elif "shape" in query.lower():
            return {
                "type": "exploration", 
                "result": {
                    "rows": self.current_dataset.shape[0],
                    "columns": self.current_dataset.shape[1],
                    "memory_mb": convert_numpy_types(round(self.current_dataset.memory_usage(deep=True).sum() / 1024**2, 2))
                },
                "explanation": f"The dataset has {self.current_dataset.shape[0]} rows and {self.current_dataset.shape[1]} columns, using {round(self.current_dataset.memory_usage(deep=True).sum() / 1024**2, 2)} MB of memory."
            }
        else:
            sample_data = convert_numpy_types(self.current_dataset.head().to_dict())
            return {
                "type": "exploration",
                "result": {
                    "sample_data": sample_data,
                    "shape": self.current_dataset.shape,
                    "dtypes": {col: str(dtype) for col, dtype in self.current_dataset.dtypes.items()}
                },
                "explanation": "Here are the first 5 rows of your dataset with data type information."
            }
    
    def _handle_statistics_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle statistical analysis queries."""
        parameters = parameters or {}
        numeric_data = self.current_dataset.select_dtypes(include=['number'])
        
        if len(numeric_data.columns) == 0:
            return {
                "type": "statistics",
                "error": "No numeric columns found for statistical analysis.",
                "suggestion": "Try exploring categorical data or checking data types."
            }
        
        operation = parameters.get("operation", "")
        if "average" in query.lower() or "mean" in query.lower() or operation == "mean":
            result = convert_numpy_types(numeric_data.mean().to_dict())
            return {
                "type": "statistics",
                "result": result,
                "explanation": f"Average values for numeric columns: {', '.join(numeric_data.columns)}"
            }
        elif "correlation" in query.lower() or operation == "correlation":
            if len(numeric_data.columns) < 2:
                return {
                    "type": "statistics",
                    "error": "Need at least 2 numeric columns for correlation analysis.",
                    "available_columns": list(numeric_data.columns)
                }
            corr_matrix = numeric_data.corr()
            return {
                "type": "statistics",
                "result": convert_numpy_types(corr_matrix.to_dict()),
                "explanation": "Correlation matrix showing relationships between numeric variables."
            }
        else:
            result = convert_numpy_types(numeric_data.describe().to_dict())
            return {
                "type": "statistics", 
                "result": result,
                "explanation": f"Summary statistics for {len(numeric_data.columns)} numeric columns."
            }
    
    def _handle_visualization_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle visualization requests."""
        parameters = parameters or {}
        
        # Get chart suggestions
        suggestions = self.visualizer.suggest_visualizations(self.current_dataset)
        
        chart_type = parameters.get("chart_type", "auto")
        x_column = parameters.get("x_column")
        y_column = parameters.get("y_column")
        
        return {
            "type": "visualization",
            "result": {
                "chart_type": chart_type,
                "x_column": x_column,
                "y_column": y_column,
                "suggestions": suggestions[:3],
                "available_columns": {
                    "numeric": list(self.current_dataset.select_dtypes(include=['number']).columns),
                    "categorical": list(self.current_dataset.select_dtypes(include=['object']).columns)
                }
            },
            "explanation": f"Visualization suggestions based on your data. {len(suggestions)} chart types recommended."
        }
    
    def _handle_filtering_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle data filtering queries."""
        parameters = parameters or {}
        
        filters = parameters.get("filters", "")
        columns = parameters.get("columns", [])
        
        return {
            "type": "filtering",
            "result": {
                "original_size": len(self.current_dataset),
                "available_columns": list(self.current_dataset.columns),
                "numeric_columns": list(self.current_dataset.select_dtypes(include=['number']).columns),
                "categorical_columns": list(self.current_dataset.select_dtypes(include=['object']).columns),
                "suggested_filters": self._get_filter_suggestions()
            },
            "explanation": "Data filtering options. Specify conditions to filter your dataset."
        }
    
    def _handle_insights_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle automated insights requests."""
        try:
            insights = self.insight_generator.generate_insights(self.current_dataset)
            auto_insights = self.insight_generator.get_automated_insights(self.current_dataset, top_n=5)
            
            return {
                "type": "insights",
                "result": {
                    "summary": insights.get("summary", ""),
                    "key_insights": auto_insights,
                    "recommendations": insights.get("recommendations", []),
                    "data_quality_score": self._calculate_quality_score(insights)
                },
                "explanation": "Automated insights and patterns discovered in your data."
            }
        except Exception as e:
            return {
                "type": "insights",
                "error": f"Could not generate insights: {str(e)}",
                "suggestion": "Try with a smaller dataset or check data quality."
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries."""
        if self.current_dataset is not None:
            dataset_info = f"Current dataset: {self.current_dataset.shape[0]} rows, {self.current_dataset.shape[1]} columns"
        else:
            dataset_info = "No dataset loaded"
        
        return {
            "type": "general",
            "result": {
                "message": "I can help you analyze your data! Try asking about:",
                "suggestions": [
                    "Show me the columns in my dataset",
                    "What are the summary statistics?",
                    "Create a visualization",
                    "Find patterns in my data",
                    "Generate insights automatically"
                ],
                "dataset_status": dataset_info
            },
            "explanation": "General assistance with data analysis. Ask me anything about your data!"
        }
    
    def _get_filter_suggestions(self) -> List[str]:
        """Get filtering suggestions based on data characteristics."""
        suggestions = []
        
        numeric_cols = self.current_dataset.select_dtypes(include=['number']).columns
        categorical_cols = self.current_dataset.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            suggestions.append(f"Filter {col} > median value")
            suggestions.append(f"Show top 10% of {col} values")
        
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            unique_values = self.current_dataset[col].value_counts().head(3).index.tolist()
            if unique_values:
                suggestions.append(f"Filter by {col} = '{unique_values[0]}'")
        
        return suggestions
    
    def _calculate_quality_score(self, insights: Dict[str, Any]) -> float:
        """Calculate a simple data quality score."""
        score = 100.0
        
        # Deduct points for data quality issues
        quality_issues = insights.get("data_quality", [])
        for issue in quality_issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                score -= 20
            elif severity == "medium":
                score -= 10
            else:
                score -= 5
        
        # Deduct points for missing data
        overview = insights.get("data_overview", {})
        missing_pct = overview.get("missing_percentage", 0)
        score -= missing_pct * 0.5
        
        return max(0.0, min(100.0, score))
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if self.memory:
            return [{"role": msg.type, "content": msg.content} for msg in self.memory.chat_memory.messages]
        else:
            return self.fallback_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        if self.memory:
            self.memory.clear()
        self.fallback_history.clear()
    
    def set_api_key(self, api_key: str):
        """Set OpenAI API key and reinitialize LLM."""
        self.llm = get_llm(api_key)
        if self.llm:
            self.agent_executor = self._create_langchain_agent()
            print("✅ LangChain agent initialized successfully!")
        else:
            print("❌ Failed to initialize LLM with provided API key")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "llm_available": self.llm is not None,
            "agent_available": self.agent_executor is not None,
            "dataset_loaded": self.current_dataset is not None,
            "dataset_shape": self.current_dataset.shape if self.current_dataset is not None else None,
            "conversation_length": len(self.get_conversation_history()),
            "tools_available": len(self.tools) if self.tools else 0
        } 