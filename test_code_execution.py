#!/usr/bin/env python3
"""
Test script for code generation and execution functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.data_analyst import DataAnalyst

def set_openai_api_key():
    """Load API key from key.txt if it exists."""
    key_file = 'key.txt'
    if os.path.exists(key_file):
        try:
            with open(key_file, 'r') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ OpenAI API key loaded from key.txt")
            return True
        except Exception as e:
            print(f"❌ Error loading API key: {e}")
            return False
    else:
        print("❌ key.txt not found. Please create it with your OpenAI API key.")
        return False

def create_test_dataset():
    """Create a test dataset for code execution testing."""
    np.random.seed(42)
    
    # Create sample sales data
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C']
    
    data = []
    for _ in range(100):
        data.append({
            'region': np.random.choice(regions),
            'product': np.random.choice(products),
            'sales': np.random.randint(100, 1000),
            'marketing_spend': np.random.randint(10, 100),
            'customer_count': np.random.randint(50, 200),
            'date': pd.date_range('2023-01-01', periods=100)[_]
        })
    
    df = pd.DataFrame(data)
    print(f"📊 Created test dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df

def test_code_generation_queries():
    """Test various code generation queries."""
    return [
        "Generate code to calculate total sales by region",
        "Create code to find the average marketing spend for each product",
        "Write code to calculate the correlation between sales and marketing spend",
        "Generate code to find the top 3 regions by total sales",
        "Create code to calculate what percentage of total sales each region represents",
        "Write code to find the product with highest average sales",
        "Generate code to create a summary of sales statistics by region"
    ]

def main():
    print("🧪 Testing Code Generation and Execution Functionality")
    print("=" * 60)
    
    # Set up API key
    if not set_openai_api_key():
        print("⚠️  Will test with fallback code generation only.")
    
    # Create test dataset
    df = create_test_dataset()
    
    # Initialize DataAnalyst
    print("\n🤖 Initializing DataAnalyst...")
    analyst = DataAnalyst()
    
    # Load dataset
    print("📥 Loading dataset into analyst...")
    load_result = analyst.load_dataset(df)
    print(f"✅ Dataset loaded: {load_result.get('message', 'Success')}")
    
    # Test code generation queries
    queries = test_code_generation_queries()
    
    print(f"\n🔍 Testing {len(queries)} code generation queries...")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Test the code generation tool directly
            result = analyst._code_generation_tool(query)
            print(result)
            
            # Check if execution was successful
            if "✅ Success" in result:
                print("✅ Code generation and execution: SUCCESS")
            elif "❌ Error" in result:
                print("❌ Code generation and execution: ERROR")
            else:
                print("ℹ️  Code generation completed (status unclear)")
                
        except Exception as e:
            print(f"❌ Exception during test: {str(e)}")
        
        print("-" * 40)
    
    print("\n🏁 Code execution testing completed!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run the dashboard: python run_direct_dashboard.py")
    print("2. Try the code generation questions in the chat interface")
    print("3. Look for 'Generate code' or 'Create code' or 'Write code' prompts")

if __name__ == "__main__":
    main() 