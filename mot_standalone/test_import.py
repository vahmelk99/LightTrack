"""
Simple test to verify imports work correctly
"""

import sys
import os

# Test imports
try:
    print("Testing imports...")
    
    print("  - Importing mot_tracker...")
    import mot_tracker
    print("    ✓ mot_tracker imported successfully")
    
    print("  - Importing mot_wrapper...")
    from mot_wrapper import MOT
    print("    ✓ MOT class imported successfully")
    
    print("\nAll imports successful!")
    print("\nMOT class methods:")
    methods = [m for m in dir(MOT) if not m.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
