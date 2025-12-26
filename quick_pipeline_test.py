#!/usr/bin/env python3
"""
Quick pipeline test - runs the entire workflow end-to-end with minimal settings.
This tests: preprocess -> train -> generate -> learn structure -> evaluate
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import subprocess
import time

def run_command(cmd, description):
    """Run a command and report results."""
    print("\n" + "="*80)
    print(f"▶ {description}")
    print("="*80)
    print(f"Command: {cmd}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Check result
    if result.returncode == 0:
        print(f"✓ SUCCESS (took {elapsed:.1f}s)")
        return True
    else:
        print(f"✗ FAILED (exit code {result.returncode})")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║             QUICK PIPELINE TEST - Full Workflow Check                ║
║  Tests all steps with minimal settings for fastest verification      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Use smallest dataset (asia is typically small)
    dataset = "asia"
    model = "gmm"  # Fastest model

    steps = [
        {
            "cmd": f"python scripts/01_preprocess_data.py --dataset {dataset}",
            "desc": "Step 1/7: Preprocess data",
            "required": True
        },
        {
            "cmd": f"python scripts/03_tune_model.py --dataset {dataset} --model {model} --trials 10",
            "desc": "Step 2/7: Tune GMM hyperparameters (10 trials)",
            "required": True
        },
        {
            "cmd": f"python scripts/02_train_model.py --dataset {dataset} --model {model}",
            "desc": "Step 3/7: Train GMM model (fastest)",
            "required": True
        },
        {
            "cmd": f"python scripts/04_generate_synthetic.py --dataset {dataset} --model {model} --n-samples 1000",
            "desc": "Step 4/7: Generate synthetic data",
            "required": True
        },
        {
            "cmd": f"python scripts/05_learn_structure.py --dataset {dataset} --algorithm pc --data-type real",
            "desc": "Step 5/7: Learn causal structure with PC algorithm",
            "required": True
        },
        {
            "cmd": f"python scripts/06_evaluate.py --dataset {dataset} --model {model} --structure ground_truth",
            "desc": "Step 6/7: Evaluate model with ground truth structure",
            "required": True
        },
        {
            "cmd": f"python scripts/07_compare_rankings.py --dataset {dataset}",
            "desc": "Step 7/7: Compare rankings",
            "required": False  # May fail if not enough models trained
        }
    ]

    results = []
    for i, step in enumerate(steps):
        success = run_command(step["cmd"], step["desc"])
        results.append(success)

        if not success and step["required"]:
            print(f"\n{'='*80}")
            print(f"❌ Pipeline test FAILED at step {i+1}")
            print(f"{'='*80}")
            print("\nPlease fix the error above before continuing.")
            return False

    # Summary
    print("\n" + "="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)

    for i, (step, success) in enumerate(zip(steps, results)):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Step {i+1}: {status} - {step['desc']}")

    all_passed = all(results[:6])  # First 6 are required

    if all_passed:
        print("\n" + "="*80)
        print("🎉 SUCCESS! The entire pipeline is working correctly!")
        print("="*80)
        print(f"\nYou can now run the full workflow with any dataset:")
        print(f"  - Tuning study: outputs/models/{dataset}_{model}_study.pkl")
        print(f"  - Preprocessed data: data/preprocessed/{dataset}/")
        print(f"  - Trained model: outputs/models/{dataset}_{model}.pkl")
        print(f"  - Synthetic data: outputs/synthetic/{dataset}_{model}_synthetic.csv")
        print(f"  - Learned structure: outputs/structures/{dataset}_pc_real.json")
        print(f"  - Evaluation results: outputs/evaluations/{dataset}_{model}_*.json")
        print("\nReady to run experiments! 🚀")
        return True
    else:
        print("\n" + "="*80)
        print("❌ Pipeline test had failures. Please review errors above.")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

