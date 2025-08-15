import os
import shlex
import subprocess
from typing import List


def dataset_to_version(dataset_name: str) -> str:
    if dataset_name == "robo_net":
        return "1.0.0"
    if dataset_name == "language_table":
        return "0.0.1"
    return "0.1.0"


def download_datasets(out_dir: str, dataset_names: List[str]) -> int:
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Datasets: {dataset_names}")

    for dataset_name in dataset_names:
        version = dataset_to_version(dataset_name)
        src = f"gs://gresearch/robotics/{dataset_name}/{version}"
        dst = os.path.join(out_dir, dataset_name)
        os.makedirs(dst, exist_ok=True)
        print(f"[INFO] Copying {src} -> {dst}")
        cmd = ["gsutil", "-m", "cp", "-r", src, dst]
        print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[ERROR] Failed to copy {src}")
            return rc

    print("[DONE] Download complete")
    return 0


