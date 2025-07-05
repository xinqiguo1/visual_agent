"""
Demo Script for Enhanced Web Visualizations

This script demonstrates the new Plotly-integrated web visualization capabilities
of the Visual Analytics Agent.
"""

import os
import pandas as pd
import numpy as np
from src.agents import DataAnalyst, WebVisualizer
from src.config.settings import setup_environment
import webbrowser
import time

def create_sample_data():
    """Create a rich sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic customer data
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Income correlated with age
    incomes = 30000 + (ages - 18) * 1000 + np.random.normal(0, 10000, n_samples)
    incomes = np.clip(incomes, 20000, 120000)
    
    # Spending score correlated with income
    spending_scores = 20 + (incomes - 20000) / 2000 + np.random.normal(0, 15, n_samples)
    spending_scores = np.clip(spending_scores, 0, 100)
    
    # Purchase amounts
    purchase_amounts = 50 + spending_scores * 2 + np.random.exponential(30, n_samples)
    
    # Categories
    categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples)
    
    # Customer segments
    segments = []
    for i in range(n_samples):
        if incomes[i] > 80000 and spending_scores[i] > 70:
            segments.append('Premium')
        elif incomes[i] > 50000 and spending_scores[i] > 50:
            segments.append('Standard')
        else:
            segments.append('Basic')
    
    # Satisfaction based on spending and income
    satisfaction_scores = []
    for i in range(n_samples):
        if spending_scores[i] > 70 and incomes[i] > 60000:
            satisfaction_scores.append(np.random.choice(['High', 'Very High'], p=[0.3, 0.7]))
        elif spending_scores[i] > 40:
            satisfaction_scores.append(np.random.choice(['Medium', 'High'], p=[0.6, 0.4]))
        else:
            satisfaction_scores.append(np.random.choice(['Low', 'Medium'], p=[0.7, 0.3]))
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': ages,
        'income': incomes,
        'spending_score': spending_scores,
        'purchase_amount': purchase_amounts,
        'category': categories,
        'segment': segments,
        'satisfaction': satisfaction_scores,
        'is_premium': [seg == 'Premium' for seg in segments],
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    return pd.DataFrame(data)

def demo_web_visualizer():
    """Demonstrate the standalone WebVisualizer capabilities."""
    print("🎨 Testing WebVisualizer Standalone...")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize WebVisualizer
    web_viz = WebVisualizer(output_dir="outputs")
    
    # Test different chart types
    test_charts = [
        {
            'type': 'scatter',
            'x': 'age',
            'y': 'income',
            'color': 'segment',
            'title': 'Age vs Income by Segment'
        },
        {
            'type': 'bar',
            'x': 'category',
            'title': 'Product Categories'
        },
        {
            'type': 'histogram',
            'x': 'spending_score',
            'color': 'segment',
            'title': 'Spending Score Distribution'
        },
        {
            'type': 'box',
            'x': 'segment',
            'y': 'purchase_amount',
            'title': 'Purchase Amount by Segment'
        },
        {
            'type': 'heatmap',
            'title': 'Correlation Matrix'
        }
    ]
    
    created_charts = []
    
    for chart_config in test_charts:
        print(f"  • Creating {chart_config['type']} chart...")
        
        result = web_viz.create_web_chart(
            data=df,
            chart_type=chart_config['type'],
            x_column=chart_config.get('x'),
            y_column=chart_config.get('y'),
            color_column=chart_config.get('color'),
            title=chart_config.get('title')
        )
        
        if result.get('success'):
            created_charts.append(result)
            print(f"    ✅ Created: {result['html_path']}")
        else:
            print(f"    ❌ Error: {result.get('error', 'Unknown error')}")
    
    # Create dashboard
    if created_charts:
        print(f"\n📊 Creating dashboard with {len(created_charts)} charts...")
        dashboard_html = web_viz.create_dashboard(
            created_charts, 
            "Customer Analytics Dashboard"
        )
        print("✅ Dashboard created successfully!")
        
        # Try to open the dashboard in browser
        try:
            dashboard_path = "outputs/dashboard_*.html"
            import glob
            dashboard_files = glob.glob(dashboard_path)
            if dashboard_files:
                latest_dashboard = max(dashboard_files, key=os.path.getctime)
                print(f"🌐 Opening dashboard: {latest_dashboard}")
                webbrowser.open(f"file://{os.path.abspath(latest_dashboard)}")
        except Exception as e:
            print(f"Could not open browser: {e}")
    
    return created_charts

def demo_enhanced_analyst():
    """Demonstrate the enhanced DataAnalyst with web visualization."""
    print("\n🤖 Testing Enhanced DataAnalyst...")
    
    # Setup environment
    setup_environment()
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize enhanced analyst
    analyst = DataAnalyst()
    
    # Load dataset
    print("📁 Loading dataset...")
    summary = analyst.load_dataset(df)
    print(f"Dataset loaded: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
    
    # Test visualization queries
    visualization_queries = [
        "Create a scatter plot showing the relationship between age and income",
        "Show me a bar chart of product categories",
        "Create a histogram of spending scores",
        "Make a heatmap showing correlations",
        "Generate a box plot comparing segments",
        "Create a pie chart of customer satisfaction levels"
    ]
    
    created_files = []
    
    for query in visualization_queries:
        print(f"\n❓ Query: '{query}'")
        
        try:
            result = analyst.process_query(query)
            print(f"📋 Response: {result.get('explanation', 'No explanation')}")
            
            # Track created files
            if '📁 Saved to:' in result.get('explanation', ''):
                file_path = result['explanation'].split('📁 Saved to: ')[1].split('\n')[0]
                created_files.append(file_path)
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    return created_files

def show_results_summary(created_files):
    """Show a summary of created files."""
    print("\n" + "="*60)
    print("📊 VISUALIZATION DEMO COMPLETE!")
    print("="*60)
    
    if created_files:
        print(f"\n🎉 Created {len(created_files)} interactive visualizations!")
        print("\n📁 Files created:")
        for file_path in created_files:
            print(f"   • {file_path}")
        
        print("\n🌐 To view your visualizations:")
        print("   1. Open any HTML file in your web browser")
        print("   2. Look for the 'outputs' folder in your project")
        print("   3. Interactive charts support zooming, hovering, and filtering!")
        
        print("\n💡 Key Features:")
        print("   ✅ Interactive charts with Plotly")
        print("   ✅ Automatic column selection")
        print("   ✅ Color-coded by categories")
        print("   ✅ Hover information")
        print("   ✅ Zoom and pan capabilities")
        print("   ✅ Export to PNG/PDF")
        print("   ✅ Mobile-responsive design")
        
    else:
        print("\n⚠️ No files were created. Check for errors above.")
    
    # Show output directory contents
    output_dir = "outputs"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            print(f"\n📂 Output directory contains {len(files)} files:")
            for file in files[:10]:  # Show first 10 files
                print(f"   • {file}")
            if len(files) > 10:
                print(f"   • ... and {len(files) - 10} more files")

def main():
    """Main demo function."""
    print("🚀 Visual Analytics Agent - Enhanced Web Visualizations Demo")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Demo 1: Standalone WebVisualizer
    chart_results = demo_web_visualizer()
    
    # Demo 2: Enhanced DataAnalyst
    analyst_files = demo_enhanced_analyst()
    
    # Show results
    all_files = []
    
    # Add chart file paths
    for chart in chart_results:
        if chart.get('html_path'):
            all_files.append(chart['html_path'])
    
    # Add analyst files
    if analyst_files:
        all_files.extend(analyst_files)
    
    show_results_summary(all_files)
    
    print("\n🎯 Next Steps:")
    print("   • Try your own queries with the enhanced agent")
    print("   • Explore different chart types")
    print("   • Create dashboards with multiple charts")
    print("   • Build a web interface using FastAPI")

if __name__ == "__main__":
    main() 