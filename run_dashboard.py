#!/usr/bin/env python3
"""
Dashboard Runner

Simple script to run the Visual Analytics Agent dashboard.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import plotly
        print("✅ Required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install streamlit plotly")
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    if not check_requirements():
        return
    
    dashboard_file = "analytics_dashboard.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file '{dashboard_file}' not found!")
        return
    
    print("🚀 Starting Visual Analytics Agent Dashboard...")
    print("📊 Dashboard will open in your browser automatically")
    print("🔗 Manual URL: http://localhost:8501")
    print("⚠️  Make sure your API server is running: python run_api.py")
    print("\n" + "="*50)
    
    try:
        # Run streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

if __name__ == "__main__":
    run_dashboard() 