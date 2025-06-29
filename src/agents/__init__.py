"""
Visual Analytics Agent - Agents Package

This package contains the core AI agents responsible for data analysis,
visualization, code generation, and insight discovery.

Enhanced with LangChain integration for sophisticated natural language
understanding and multi-agent orchestration.
"""

from .data_analyst import DataAnalyst
from .visualizer import Visualizer
from .code_generator import CodeGenerator
from .insight_generator import InsightGenerator

__all__ = [
    "DataAnalyst",
    "Visualizer", 
    "CodeGenerator",
    "InsightGenerator"
]

# Version and capabilities info
__version__ = "2.0.0"
__features__ = [
    "LangChain Integration",
    "GPT-4o Support", 
    "Multi-Agent Architecture",
    "Conversation Memory",
    "Tool Orchestration",
    "Enhanced NLP Understanding"
] 