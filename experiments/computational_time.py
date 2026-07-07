"""
Run pipeline_check N times and store the mean/average computational time for the entire pipeline.
"""

from pathlib import Path
import time
from pipeline_check import main as run_pipeline_check

OUTPUT_PATH = Path(__file__).parent.parent / "results" / "computational_time.txt"

if __name__ == "__main__":
    total_time = 0
    n_runs = 20
    all_times = []

    for _ in range(n_runs):
        start_time = time.time()
        run_pipeline_check()
        end_time = time.time()
        total_time += end_time - start_time
        all_times.append(end_time - start_time)

    average_time = total_time / n_runs
    time_std = (sum((x - average_time) ** 2 for x in all_times) / n_runs) ** 0.5

    with open(OUTPUT_PATH, "w") as f:
        f.write(f"Average computational time: {average_time:.6f} seconds\n")
        f.write(f"Standard deviation: {time_std:.6f} seconds")
