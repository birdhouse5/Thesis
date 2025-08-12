#!/usr/bin/env python3
"""
Standalone script to diagnose data quality issues.
Usage: python diagnose_data.py
"""

import logging
from environments.data_preparation import download_stock_data, diagnose_data_quality

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run data quality diagnosis"""
    print("🔍 Starting Data Quality Diagnosis...")
    
    # Download fresh data
    print("\n📥 Downloading stock data...")
    raw_data = download_stock_data()
    
    # Run diagnosis
    print("\n🔍 Running diagnosis...")
    diagnose_data_quality(raw_data)
    
    print("\n✅ Diagnosis complete!")

if __name__ == "__main__":
    main()