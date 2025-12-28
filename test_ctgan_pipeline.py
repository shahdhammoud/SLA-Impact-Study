#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import subprocess
import time

def run_command(cmd, description):
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Command: {cmd}\n")
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0:
        print(f"SUCCESS (took {elapsed:.1f}s)")
        return True
    else:
        print(f"FAILED (exit code {result.returncode})")
        return False

def main():
    dataset = "asia"
    model = "ctgan"
    steps = [
        {"cmd": f"python scripts/01_preprocess_data.py --dataset {dataset}", "desc": "Step 1/6: Preprocess data", "required": True},
        {"cmd": f"python scripts/03_tune_model.py --dataset {dataset} --model {model} --trials 10", "desc": "Step 2/6: Tune CTGAN hyperparameters (10 trials for quick test)", "required": True},
        {"cmd": f"python scripts/02_train_model.py --dataset {dataset} --model {model} --epochs 50", "desc": "Step 3/6: Train CTGAN model (50 epochs for quick test)", "required": True},
        {"cmd": f"python scripts/04_generate_synthetic.py --dataset {dataset} --model {model} --n-samples 1000", "desc": "Step 4/6: Generate synthetic data", "required": True},
        {"cmd": f"python scripts/05_learn_structure.py --dataset {dataset} --algorithm pc --data-type synthetic --model {model}", "desc": "Step 5/6: Learn structure from synthetic data", "required": True},
        {"cmd": f"python scripts/06_evaluate.py --dataset {dataset} --model {model} --structure ground_truth", "desc": "Step 6/6: Evaluate CTGAN with ground truth structure", "required": True}
    ]
    results = []
    for i, step in enumerate(steps):
        success = run_command(step["cmd"], step["desc"])
        results.append(success)
        if not success and step["required"]:
            print(f"\n{'='*80}")
            print(f"CTGAN pipeline test FAILED at step {i+1}")
            print(f"{'='*80}")
            print("\nPlease fix the error above before continuing.")
            return False

    for i, (step, success) in enumerate(zip(steps, results)):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Step {i+1}: {status} - {step['desc']}")

    all_passed = all(results)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 SUCCESS! CTGAN pipeline is working correctly!")
        print("="*80)
        print(f"\nGenerated outputs:")
        print(f"  - Tuning study: outputs/models/{dataset}_{model}_study.pkl")
        print(f"  - Trained CTGAN model: outputs/models/{dataset}_{model}.pkl")
        print(f"  - Synthetic data: outputs/synthetic/{dataset}_{model}_synthetic.csv")
        print(f"  - Learned structure: outputs/structures/{dataset}_pc_synthetic_{model}.json")
        print(f"  - Evaluation results: outputs/evaluations/{dataset}_{model}_ground_truth_eval.json")
        print("\n✨ CTGAN is ready for full experiments! You can now:")
        print(f"  1. Train with more epochs: python scripts/02_train_model.py --dataset {dataset} --model ctgan --epochs 300")
        print(f"  2. Tune hyperparameters: python scripts/03_tune_model.py --dataset {dataset} --model ctgan --trials 50")
        print(f"  3. Test on other datasets: alarm, cancer, earthquake, etc.")
        print("\n🚀 Ready to go!")
        return True
    else:
        print("\n" + "="*80)
        print("❌ CTGAN pipeline test had failures. Please review errors above.")
        print("="*80)
        return False

if __name__ == "__main__":
    main()
