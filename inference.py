"""
inference.py — OpenEnv required inference script
Alias for baseline.py — runs greedy agent across all 3 tasks.

Usage:
    python inference.py
    python inference.py --task task_1_easy
    python inference.py --policy llm --api https://rishabh55122-agronexus.hf.space
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline import main, get_args

if __name__ == "__main__":
    main()
