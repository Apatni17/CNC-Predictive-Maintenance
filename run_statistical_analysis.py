#!/usr/bin/env python3

# Simple script to run the statistical analysis
import sys
import os
sys.path.append(os.getcwd())

try:
    from statistical_wear_analyzer import main
    print("Running statistical wear analysis...")
    analyzer, ranges = main()
    print("\n🎯 Statistical Analysis Complete!")
    print(f"Recommended ranges: {ranges}")
except Exception as e:
    print(f"Error running analysis: {e}")
    import traceback
    traceback.print_exc()