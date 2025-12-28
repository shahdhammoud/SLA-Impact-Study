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
    model = "tabddpm"
    steps = [
        {"cmd": f"python scripts/01_preprocess_data.py --dataset {dataset}", "desc": "Step 1/5: Preprocess data", "required": True},
        {"cmd": f"python scripts/02_train_model.py --dataset {dataset} --model {model} --epochs 10 --batch-size 256", "desc": "Step 2/5: Train TabDDPM model (10 epochs for ultra-quick test)", "required": True},
        {"cmd": f"python scripts/04_generate_synthetic.py --dataset {dataset} --model {model} --n-samples 500", "desc": "Step 3/5: Generate synthetic data (500 samples)", "required": True},
        {"cmd": f"python scripts/05_learn_structure.py --dataset {dataset} --algorithm pc --data-type synthetic --model {model}", "desc": "Step 4/5: Learn structure from synthetic data", "required": True},
        {"cmd": f"python scripts/06_evaluate.py --dataset {dataset} --model {model} --structure ground_truth", "desc": "Step 5/5: Evaluate TabDDPM with ground truth structure", "required": True}
    ]
    results = []
    for i, step in enumerate(steps):
        success = run_command(step["cmd"], step["desc"])
        results.append(success)
        if not success and step["required"]:
            print(f"\n{'='*80}")
            print(f"TabDDPM pipeline test FAILED at step {i+1}")
            print(f"{'='*80}")
            print("\nPlease fix the error above before continuing.")
            return False
    print("\n" + "="*80)
    print("TabDDPM PIPELINE TEST SUMMARY")
    print("="*80)

    for i, (step, success) in enumerate(zip(steps, results)):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Step {i+1}: {status} - {step['desc']}")

    all_passed = all(results)

    if all_passed:
        print("\n" + "="*80)
        print("SUCCESS! TabDDPM pipeline is working correctly!")
        print("="*80)
        print(f"\nGenerated outputs:")
        print(f"  - Trained TabDDPM model: outputs/models/{dataset}_{model}.pkl")
        print(f"  - Synthetic data: outputs/synthetic/{dataset}_{model}_synthetic.csv")
        print(f"  - Learned structure: outputs/structures/{dataset}_pc_synthetic_{model}.json")
        print(f"  - Evaluation results: outputs/evaluations/{dataset}_{model}_ground_truth_eval.json")
        print("\nTabDDPM is ready for full experiments!")
        print("\nIMPORTANT: For production use:")
        print(f"  - Train with 500-1000 epochs for good quality")
        print(f"  - Use larger batch sizes (1024-2048)")
        print(f"  - Consider GPU acceleration (CUDA)")
        print(f"  - Run tuning separately: python scripts/03_tune_model.py --dataset {dataset} --model tabddpm --trials 20")
        print("\nExample full training command:")
        print(f"  python scripts/02_train_model.py --dataset {dataset} --model tabddpm --epochs 500")
        print("\nReady to go!")
        return True
    else:
        print("\n" + "="*80)
        print("TabDDPM pipeline test had failures. Please review errors above.")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
