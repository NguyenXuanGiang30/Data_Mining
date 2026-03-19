"""
run_papermill.py - Chạy tất cả notebook bằng papermill

Usage:
    python scripts/run_papermill.py
"""

import os
import papermill as pm
import time


NOTEBOOK_DIR = "notebooks"
OUTPUT_DIR = "notebooks"  # overwrite notebook gốc hoặc dùng folder riêng

NOTEBOOKS = [
    "01_eda.ipynb",
    "02_preprocess_feature.ipynb",
    "03_mining_or_clustering.ipynb",
    "04_modeling.ipynb",
    "04b_semi_supervised.ipynb",
    "05_evaluation_report.ipynb",
]


def run_all_notebooks():
    """Chạy tất cả notebook theo thứ tự."""
    print("=" * 60)
    print("📓 RUNNING ALL NOTEBOOKS (Papermill)")
    print("=" * 60)
    
    for nb in NOTEBOOKS:
        input_path = os.path.join(NOTEBOOK_DIR, nb)
        output_path = os.path.join(OUTPUT_DIR, nb)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {nb} (file not found)")
            continue
        
        print(f"\n--- Running: {nb} ---")
        start = time.time()
        
        try:
            pm.execute_notebook(
                input_path,
                output_path,
                kernel_name="python3",
            )
            elapsed = time.time() - start
            print(f"✅ {nb} completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"❌ {nb} failed: {e}")
    
    print(f"\n{'=' * 60}")
    print("🎉 All notebooks completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_all_notebooks()
