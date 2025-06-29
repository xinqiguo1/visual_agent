"""
Demo Script for LangChain-Enhanced Visual Analytics Agent

This script demonstrates the capabilities of the enhanced DataAnalyst
with full LangChain integration and GPT-4o.
"""

import os
import pandas as pd
import numpy as np
from src.agents import DataAnalyst
from src.config.settings import setup_environment

def create_sample_data():
    """Create sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples),
        'spending_score': np.random.normal(50, 20, n_samples),
        'purchase_amount': np.random.exponential(100, n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
        'is_premium': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    # Add some correlations and patterns
    data['spending_score'] = data['spending_score'] + 0.3 * (data['income'] / 1000) + np.random.normal(0, 5, n_samples)
    data['purchase_amount'] = data['purchase_amount'] + 0.5 * data['spending_score'] + np.random.normal(0, 20, n_samples)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in missing_indices:
        data['satisfaction'][idx] = None
    
    df = pd.DataFrame(data)
    
    # Ensure realistic constraints
    df['age'] = df['age'].clip(18, 80)
    df['income'] = df['income'].clip(20000, 200000)
    df['spending_score'] = df['spending_score'].clip(0, 100)
    df['purchase_amount'] = df['purchase_amount'].clip(10, 1000)
    
    return df

def demonstrate_capabilities():
    """Demonstrate the enhanced agent capabilities."""
    print("🚀 Visual Analytics Agent - LangChain Demo")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    df = create_sample_data()
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize enhanced agent
    print("\n🤖 Initializing Enhanced DataAnalyst...")
    analyst = DataAnalyst()
    
    # Check status
    status = analyst.get_status()
    print(f"Agent Status: {status}")
    
    # Load dataset
    print("\n📁 Loading dataset...")
    summary = analyst.load_dataset(df)
    print("Dataset Summary:")
    for key, value in summary.items():
        if key != "auto_insights":
            print(f"  {key}: {value}")
    
    if "auto_insights" in summary:
        print("\n🔍 Automatic Insights:")
        for insight in summary["auto_insights"][:3]:
            print(f"  • {insight}")
    
    # Test queries without LLM (fallback mode)
    print("\n" + "=" * 50)
    print("TESTING FALLBACK MODE (No API Key)")
    print("=" * 50)
    
    test_queries = [
        "What columns do I have?",
        "Show me summary statistics",
        "Create a visualization",
        "Find insights in my data",
        "What's the average income?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        result = analyst.process_query(query)
        print(f"📋 Result: {result.get('explanation', 'No explanation')}")
        if result.get('error'):
            print(f"⚠️  Error: {result['error']}")
    
    # Instructions for API key setup
    print("\n" + "=" * 50)
    print("ENHANCED MODE SETUP (With API Key)")
    print("=" * 50)
    print("\n🔑 To enable enhanced LangChain capabilities:")
    print("1. Get your OpenAI API key from https://platform.openai.com/api-keys")
    print("2. Set it using one of these methods:")
    print("   • Environment variable: export OPENAI_API_KEY='your-key-here'")
    print("   • In Python: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
    print("   • Using agent method: analyst.set_api_key('your-key-here')")
    print("\n3. Then run enhanced queries like:")
    print("   • 'Analyze the correlation between income and spending'")
    print("   • 'Create a scatter plot showing age vs purchase amount'") 
    print("   • 'What patterns do you see in customer behavior?'")
    print("   • 'Generate code to find high-value customers'")
    
    # Check if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("\n🎉 API Key detected! Testing enhanced mode...")
        
        enhanced_queries = [
            "Analyze the relationship between income and spending score",
            "What insights can you find about customer satisfaction?",
            "Create a visualization showing purchase patterns by category",
            "Generate Python code to identify premium customers"
        ]
        
        for query in enhanced_queries:
            print(f"\n❓ Enhanced Query: '{query}'")
            try:
                result = analyst.process_query(query)
                print(f"📋 Result: {result.get('explanation', 'No explanation')}")
                if result.get('agent_used'):
                    print("✨ Powered by LangChain Agent")
            except Exception as e:
                print(f"⚠️  Error in enhanced mode: {e}")
    
    print("\n✅ Demo completed!")
    print("\nNext steps:")
    print("• Add your OpenAI API key for full functionality")
    print("• Explore the visualization capabilities") 
    print("• Try complex multi-step analysis queries")
    print("• Build a web interface using FastAPI")

if __name__ == "__main__":
    demonstrate_capabilities() 