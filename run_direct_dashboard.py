#!/usr/bin/env python3
"""
Direct Dashboard Runner

Launch the Visual Analytics Agent dashboard that connects directly to DataAnalyst.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✅ Required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install streamlit plotly pandas numpy")
        return False

def set_openai_api_key():
    """Set OpenAI API key from key.txt."""
    key_file = "key.txt"
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                api_key = f.read().strip()
                os.environ["OPENAI_API_KEY"] = api_key
                print("✅ OpenAI API key loaded from key.txt")
                return True
        except Exception as e:
            print(f"❌ Error loading API key: {e}")
    else:
        print("⚠️ No key.txt file found. LangChain will use fallback methods.")
    return False

def run_direct_dashboard():
    """Run the direct Streamlit dashboard."""
    if not check_requirements():
        return
    
    # Set OpenAI API key
    set_openai_api_key()
    
    dashboard_file = "analytics_dashboard_direct.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file '{dashboard_file}' not found!")
        return
    
    print("🚀 Starting Visual Analytics Agent Dashboard (Direct Mode)")
    print("📊 Dashboard will open in your browser automatically")
    print("🔗 URL: http://localhost:8504")
    print("⚡ Direct mode - No API server needed!")
    print("\n" + "="*60)
    
    try:
        # Run streamlit with different port to avoid conflicts
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8503",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

if __name__ == "__main__":
    run_direct_dashboard() 