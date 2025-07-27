#!/usr/bin/env python3
"""
Quick test for code execution functionality.
Run from the src directory.
"""

import os
import sys
import pandas as pd
import numpy as np

# Import our modules
from agents.data_analyst import DataAnalyst

def main():
    print("🧪 Quick Code Execution Test")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    df = pd.DataFrame({
        'sales': np.random.randint(100, 1000, 20),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 20),
        'product': np.random.choice(['A', 'B', 'C'], 20)
    })
    print(f"📊 Created test dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize DataAnalyst
    print("\n🤖 Initializing DataAnalyst...")
    analyst = DataAnalyst()
    
    # Load dataset
    print("📥 Loading dataset...")
    load_result = analyst.load_dataset(df)
    print(f"✅ Dataset loaded: {load_result.get('message', 'Success')}")
    
    # Test code generation queries
    queries = [
        "Generate code to calculate total sales by region",
        "Create code to find the average sales for each product"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 40)
        
        try:
            result = analyst._code_generation_tool(query)
            print("Result:")
            print(result)
            
            if "✅ Success" in result:
                print("\n✅ Code execution: SUCCESS!")
            elif "❌ Error" in result:
                print("\n❌ Code execution: ERROR!")
            else:
                print("\nℹ️ Code generation completed")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 40)
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main() 