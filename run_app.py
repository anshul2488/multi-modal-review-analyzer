#!/usr/bin/env python3
"""
Multimodal Review Analyzer - Application Runner
==============================================

This script provides an easy way to run the multimodal review analyzer application
with proper configuration and error handling.

Usage:
    python run_app.py [--port PORT] [--host HOST] [--debug]

Examples:
    python run_app.py                    # Run with default settings
    python run_app.py --port 8502       # Run on port 8502
    python run_app.py --host 0.0.0.0    # Run on all interfaces
    python run_app.py --debug           # Run in debug mode
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'torch', 'transformers', 'pandas', 'numpy', 
        'plotly', 'scikit-learn', 'textblob', 'sentence-transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("All required dependencies are installed")
    return True

def check_data_files():
    """Check if data files exist"""
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print("WARNING: Data directory not found. Creating sample structure...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample data file
        sample_data = [
            '{"reviewText": "Great product, highly recommended!", "overall": 5, "helpful": [2, 3], "asin": "sample1"}',
            '{"reviewText": "Average quality, could be better.", "overall": 3, "helpful": [1, 2], "asin": "sample2"}',
            '{"reviewText": "Terrible product, waste of money.", "overall": 1, "helpful": [5, 8], "asin": "sample3"}'
        ]
        
        sample_file = data_dir / "sample_reviews.jsonl"
        with open(sample_file, 'w') as f:
            f.write('\n'.join(sample_data))
        
        print(f"Created sample data file: {sample_file}")
    
    return True

def run_streamlit_app(port=8501, host="localhost", debug=False):
    """Run the Streamlit application"""
    
    print("Starting Multimodal Review Analyzer...")
    print(f"Application will be available at: http://{host}:{port}")
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
        print("Debug mode enabled")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error running application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run the Multimodal Review Analyzer application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run with default settings
  python run_app.py --port 8502       # Run on port 8502
  python run_app.py --host 0.0.0.0    # Run on all interfaces
  python run_app.py --debug           # Run in debug mode
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the application on (default: 8501)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to run the application on (default: localhost)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run in debug mode"
    )
    
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip dependency and data checks"
    )
    
    args = parser.parse_args()
    
    print("Multimodal Review Analyzer - Application Runner")
    print("=" * 50)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        # Check data files
        if not check_data_files():
            sys.exit(1)
    
    # Run the application
    success = run_streamlit_app(args.port, args.host, args.debug)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
