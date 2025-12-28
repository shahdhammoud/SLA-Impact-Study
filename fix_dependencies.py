#!/usr/bin/env python3

import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    steps = [
        {"cmd": "python -m pip uninstall -y pgmpy numpy networkx", "desc": "Step 1/3: Uninstalling conflicting packages"},
        {"cmd": 'python -m pip install pgmpy==0.1.19 "numpy<2.0" "networkx<2.6.0"', "desc": "Step 2/3: Installing correct package versions"},
        {"cmd": "python -m pip show pgmpy numpy networkx", "desc": "Step 3/3: Verifying installations"}
    ]
    success = True
    for step in steps:
        if not run_command(step["cmd"], step["desc"]):
            print(f"\nFailed: {step['desc']}")
            success = False
            break
    if success:
        print(f"\n{'='*70}")
        print("All dependencies fixed successfully!")
        print(f"{'='*70}")
        print("\nTesting imports...")
        test_result = run_command(
            'python -c "from pgmpy.models import BayesianNetwork; print(\\"pgmpy imports successfully\\")"',
            "Testing pgmpy import"
        )
        if test_result:
            print("\n" + "="*70)
            print("SUCCESS! Your code is now ready to run!")
            print("="*70)
            print("\nNext steps:")
            print("  1. Try running the demo: python demo_features.py")
            print("  2. Or preprocess data: python scripts/01_preprocess_data.py --dataset <name>")
            print()
        else:
            print("\nInstallation succeeded but import test failed.")
            print("Please check the error message above.")
    else:
        print("\nDependency fix failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
