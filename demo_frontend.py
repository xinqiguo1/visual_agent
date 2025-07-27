#!/usr/bin/env python3
"""
Frontend Demo Script

This script demonstrates how to test the frontend with demo data.
It uploads demo datasets and provides sample queries to test with.
"""

import requests
import json
import os
from pathlib import Path

# API Configuration
API_BASE = "http://localhost:8000/api"

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def upload_demo_data():
    """Upload demo datasets to test the frontend."""
    data_dir = Path("data")
    demo_files = []
    
    if data_dir.exists():
        # Find demo CSV files
        csv_files = list(data_dir.glob("*.csv"))
        json_files = list(data_dir.glob("*.json"))
        
        print(f"\n📁 Found {len(csv_files)} CSV files and {len(json_files)} JSON files in data/")
        
        for file_path in csv_files + json_files:
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (file_path.name, f, 'text/csv' if file_path.suffix == '.csv' else 'application/json')}
                    response = requests.post(f"{API_BASE}/data/upload", files=files)
                    
                if response.status_code == 200:
                    result = response.json()
                    dataset_id = result.get('dataset_id')
                    filename = result.get('filename')
                    rows = result.get('rows', 0)
                    columns = result.get('columns', 0)
                    
                    print(f"✅ Uploaded {filename}")
                    print(f"   Dataset ID: {dataset_id}")
                    print(f"   Size: {rows} rows × {columns} columns")
                    
                    demo_files.append({
                        'dataset_id': dataset_id,
                        'filename': filename,
                        'rows': rows,
                        'columns': columns
                    })
                else:
                    print(f"❌ Failed to upload {file_path.name}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error uploading {file_path.name}: {e}")
    
    return demo_files

def provide_sample_queries(demo_files):
    """Provide sample queries based on uploaded demo data."""
    if not demo_files:
        print("\n⚠️  No demo files uploaded. Please upload data first.")
        return
    
    print("\n🔍 Sample Queries to Test in Frontend:")
    print("=" * 50)
    
    for file_info in demo_files:
        filename = file_info['filename']
        dataset_id = file_info['dataset_id']
        
        print(f"\n📊 {filename} (ID: {dataset_id})")
        
        if 'sales' in filename.lower():
            queries = [
                "What are the total sales by region?",
                "Which product category has the highest sales?",
                "Show me the sales trend over time",
                "What is the average sales per customer segment?",
                "Which sales representative has the best performance?"
            ]
        elif 'stock' in filename.lower():
            queries = [
                "Which stock has the highest price?",
                "Show me the trading volume trends",
                "What are the price correlations between different stocks?",
                "Which company has the most volatile stock price?",
                "What is the average daily price change?"
            ]
        elif 'user' in filename.lower():
            queries = [
                "How many users are from each country?",
                "What is the average session duration?",
                "Which subscription tier is most popular?",
                "How many users logged in today?",
                "What are the most common user actions?"
            ]
        elif 'demographic' in filename.lower():
            queries = [
                "What is the average satisfaction score by age group?",
                "How does income correlate with education level?",
                "Which cities have the highest satisfaction scores?",
                "What is the gender distribution in the dataset?",
                "How does product usage vary by employment status?"
            ]
        elif 'weather' in filename.lower():
            queries = [
                "Which city has the highest average temperature?",
                "How does humidity correlate with precipitation?",
                "What are the weather patterns by city?",
                "Which city has the most rainfall?",
                "Show me the temperature trends over time"
            ]
        else:
            queries = [
                "What are the main patterns in this data?",
                "Show me summary statistics",
                "What are the correlations between numeric columns?",
                "Are there any outliers in the data?",
                "What insights can you find?"
            ]
        
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")

def demonstrate_visualization_suggestions(demo_files):
    """Show visualization suggestions for demo datasets."""
    if not demo_files:
        return
    
    print("\n📈 Recommended Visualizations:")
    print("=" * 50)
    
    for file_info in demo_files:
        filename = file_info['filename']
        dataset_id = file_info['dataset_id']
        
        print(f"\n📊 {filename} (ID: {dataset_id})")
        
        if 'sales' in filename.lower():
            suggestions = [
                ("Bar Chart", "Sales by Region", "Region", "Sales"),
                ("Line Chart", "Sales Trend", "Date", "Sales"),
                ("Pie Chart", "Sales by Category", "Category", None),
                ("Scatter Plot", "Sales vs Units", "Units", "Sales"),
                ("Heatmap", "Sales Correlation Matrix", None, None)
            ]
        elif 'stock' in filename.lower():
            suggestions = [
                ("Line Chart", "Stock Price Trends", "Date", "Close"),
                ("Bar Chart", "Trading Volume", "Symbol", "Volume"),
                ("Box Plot", "Price Distribution", "Symbol", "Close"),
                ("Scatter Plot", "Volume vs Price", "Volume", "Close"),
                ("Area Chart", "Price Range", "Date", "High")
            ]
        elif 'demographic' in filename.lower():
            suggestions = [
                ("Histogram", "Age Distribution", "Age", None),
                ("Bar Chart", "Satisfaction by Education", "Education", "Satisfaction_Score"),
                ("Scatter Plot", "Income vs Satisfaction", "Income", "Satisfaction_Score"),
                ("Pie Chart", "Gender Distribution", "Gender", None),
                ("Box Plot", "Income by State", "State", "Income")
            ]
        elif 'weather' in filename.lower():
            suggestions = [
                ("Line Chart", "Temperature Trends", "Date", "Temperature_C"),
                ("Bar Chart", "Rainfall by City", "City", "Precipitation_mm"),
                ("Scatter Plot", "Temperature vs Humidity", "Temperature_C", "Humidity_%"),
                ("Box Plot", "Temperature by City", "City", "Temperature_C"),
                ("Heatmap", "Weather Correlations", None, None)
            ]
        else:
            suggestions = [
                ("Bar Chart", "Column Distribution", "column1", "column2"),
                ("Line Chart", "Trend Analysis", "x_column", "y_column"),
                ("Scatter Plot", "Correlation Plot", "x_column", "y_column"),
                ("Histogram", "Data Distribution", "numeric_column", None)
            ]
        
        for chart_type, title, x_col, y_col in suggestions:
            print(f"   • {chart_type}: {title}")
            if x_col and y_col:
                print(f"     X: {x_col}, Y: {y_col}")
            elif x_col:
                print(f"     Column: {x_col}")

def main():
    """Main demo function."""
    print("🚀 Visual Analytics Agent - Frontend Demo")
    print("=" * 50)
    
    # Check API status
    if not check_api_status():
        print("\n❌ Please start the API server first:")
        print("   python run_api.py")
        return
    
    # Upload demo data
    print("\n📤 Uploading Demo Data...")
    demo_files = upload_demo_data()
    
    if demo_files:
        print(f"\n✅ Successfully uploaded {len(demo_files)} demo datasets!")
        
        # Provide sample queries
        provide_sample_queries(demo_files)
        
        # Show visualization suggestions
        demonstrate_visualization_suggestions(demo_files)
        
        print("\n🌐 Frontend Usage Instructions:")
        print("=" * 50)
        print("1. Open frontend.html in your web browser")
        print("2. The page will automatically check API status")
        print("3. Use the Dataset IDs above to test different routes")
        print("4. Try the sample queries in the Analysis section")
        print("5. Create visualizations using the suggested chart types")
        print("6. Test all 31 API endpoints interactively!")
        
        print(f"\n📂 Demo Dataset IDs:")
        for file_info in demo_files:
            print(f"   {file_info['filename']}: {file_info['dataset_id']}")
    
    else:
        print("\n⚠️  No demo data uploaded. You can still test the frontend")
        print("   by uploading your own CSV/JSON files through the interface.")
    
    print("\n🔗 Quick Links:")
    print("   Frontend: file:///[path-to-project]/frontend.html")
    print("   API Docs: http://localhost:8000/docs")
    print("   Health Check: http://localhost:8000/api/health")

if __name__ == "__main__":
    main() 