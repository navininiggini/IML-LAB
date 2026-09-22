#!/usr/bin/env python3
"""
IML-LAB: Automated Test Runner
Executes each experiment notebook in its respective folder context to verify
that all data loading, preprocessing, model training, and visualization run without errors.
"""

import os
import sys
import time
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    {
        "name": "Experiment 1: Exploratory Data Analysis & Preprocessing",
        "path": "Experiment-1/Experiment_1_EDA.ipynb",
        "timeout": 120
    },
    {
        "name": "Experiment 2: Email Spam/Ham Classification (Naïve Bayes & KNN)",
        "path": "Experiment-2/Experiment_2_Spam_Classification.ipynb",
        "timeout": 180
    },
    {
        "name": "Experiment 3: Regression Analysis (Linear, Ridge, Lasso, Elastic Net)",
        "path": "Experiment-3/Experiment_3_Regression_Analysis.ipynb",
        "timeout": 180
    },
    {
        "name": "Experiment 4: Binary Classification (Logistic Regression & SVM)",
        "path": "Experiment-4/Experiment_4_LR_SVM.ipynb",
        "timeout": 240
    },
    {
        "name": "Experiment 5: Comparative Classification (Decision Tree & Random Forest)",
        "path": "Experiment-5/Experiment_5_DT_RF.ipynb",
        "timeout": 300
    },
    {
        "name": "Experiment 6: Dimensionality Reduction and Model Evaluation (With and Without PCA)",
        "path": "Experiment-6/Experiment_6_PCA_Model_Evaluation.ipynb",
        "timeout": 300
    },
    {
        "name": "Experiment 7: Bagging, Boosting, and Stacked Ensemble Models",
        "path": "Experiment-7/Experiment_7_Ensemble_Learning.ipynb",
        "timeout": 300
    },
    {
        "name": "Experiment 8: Clustering Human Activity Recognition Data (K-Means, DBSCAN, HAC)",
        "path": "Experiment-8/Experiment_8_Clustering_HAR.ipynb",
        "timeout": 300
    },
    {
        "name": "Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment)",
        "path": "Experiment-9/Experiment_9_Perceptron_vs_MLP.ipynb",
        "timeout": 300
    }
]

def run_tests():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    total = len(NOTEBOOKS)
    passed = 0
    failed = 0
    results = []

    print("=" * 70)
    print("           ICS1512 - Machine Learning Algorithms Laboratory")
    print("                 Automated Notebook Verification Suite")
    print("=" * 70)

    overall_start = time.time()

    for idx, item in enumerate(NOTEBOOKS, 1):
        nb_rel_path = item["path"]
        nb_abs_path = os.path.join(root_dir, nb_rel_path)
        nb_dir = os.path.dirname(nb_abs_path)
        name = item["name"]
        timeout = item["timeout"]

        print(f"\n[{idx}/{total}] Testing: {name}")
        print(f"      File:    {nb_rel_path}")
        print(f"      Timeout: {timeout}s")

        if not os.path.exists(nb_abs_path):
            print("      Result:  FAILED (File not found!)")
            failed += 1
            results.append((name, False, "File not found", 0))
            continue

        start_time = time.time()
        try:
            nb = nbformat.read(nb_abs_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": nb_dir}}
            )
            client.execute()
            elapsed = time.time() - start_time
            print(f"      Result:  PASSED ({elapsed:.1f}s)")
            passed += 1
            results.append((name, True, "Success", elapsed))
        except Exception as e:
            elapsed = time.time() - start_time
            err_msg = str(e).split("\n")[0]
            print(f"      Result:  FAILED ({elapsed:.1f}s)")
            print(f"      Details: {err_msg}")
            failed += 1
            results.append((name, False, err_msg, elapsed))

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 70)
    print("                       VERIFICATION SUMMARY")
    print("=" * 70)
    for name, success, msg, elapsed in results:
        status = "PASSED" if success else "FAILED"
        print(f" - [{status:6s}] {name} ({elapsed:.1f}s)")
    print("-" * 70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Time: {overall_elapsed:.1f}s")
    print("=" * 70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(run_tests())
