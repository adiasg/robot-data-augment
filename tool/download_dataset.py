import json
import os
import shlex
import subprocess
from typing import List, Optional


def dataset_to_version(dataset_name: str) -> str:
    if dataset_name == "droid":
        return "1.0.1"
    if dataset_name == "robo_net":
        return "1.0.0"
    if dataset_name == "cmu_playing_with_food":
        return "1.0.0"
    if dataset_name == "language_table":
        return "0.0.1"
    return "0.1.0"



def _get_dataset_prefix(dataset_name: str) -> str:
    """Get the prefix used in tfrecord filenames for different datasets."""
    if dataset_name == "droid":
        return "droid_101"  # droid uses droid_101-train.tfrecord-XXXXX-of-YYYYY
    return dataset_name  # Most datasets use {dataset_name}-train.tfrecord-XXXXX-of-YYYYY


def _download_selective(src_base: str, dst: str, dataset_name: str, version: str, max_episodes: int) -> int:
    """Download only the required shards to get max_episodes episodes."""
    # First, download metadata files
    metadata_files = ["dataset_info.json", "features.json"]
    version_dst = os.path.join(dst, version)
    os.makedirs(version_dst, exist_ok=True)
    
    for metadata_file in metadata_files:
        src_file = f"{src_base}/{metadata_file}"
        dst_file = os.path.join(version_dst, metadata_file)
        print(f"[INFO] Downloading metadata: {src_file} -> {dst_file}")
        cmd = ["gsutil", "cp", src_file, dst_file]
        print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[ERROR] Failed to download metadata {src_file}")
            return rc
    
    # Parse dataset_info.json to get shard information
    dataset_info_path = os.path.join(version_dst, "dataset_info.json")
    try:
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse dataset_info.json: {e}")
        return 1
    
    # Find the train split and get shard lengths
    train_split = None
    for split in dataset_info.get("splits", []):
        if split.get("name") == "train":
            train_split = split
            break
    
    if not train_split:
        print(f"[ERROR] No train split found in dataset {dataset_name}")
        return 1
    
    shard_lengths = train_split.get("shardLengths", [])
    if not shard_lengths:
        print(f"[ERROR] No shard lengths found for dataset {dataset_name}")
        return 1
    
    # Calculate which shards we need
    episodes_so_far = 0
    shards_needed = []
    
    for shard_idx, shard_length in enumerate(shard_lengths):
        if episodes_so_far >= max_episodes:
            break
        shards_needed.append(shard_idx)
        episodes_so_far += int(shard_length)  # Convert to int since JSON loads as string
    
    print(f"[INFO] Need {len(shards_needed)} shard(s) for {max_episodes} episodes. {episodes_so_far} total episodes in {len(shards_needed)} shard(s).")
    
    # Download the required shards
    filepath_template = train_split.get("filepathTemplate", "")
    if not filepath_template:
        print(f"[ERROR] No filepath template found for dataset {dataset_name}")
        return 1
    
    total_shards = len(shard_lengths)
    new_total_shards = len(shards_needed)
    dataset_prefix = _get_dataset_prefix(dataset_name)
    
    for new_shard_idx, original_shard_idx in enumerate(shards_needed):
        # Build original filename from template
        # Template format: "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"
        # But some datasets like droid use custom prefixes
        original_filename = filepath_template.format(
            DATASET=dataset_prefix,
            SPLIT="train",
            FILEFORMAT="tfrecord",
            SHARD_X_OF_Y=f"{original_shard_idx:05d}-of-{total_shards:05d}"
        )
        
        # Build new filename with updated shard count
        new_filename = filepath_template.format(
            DATASET=dataset_prefix,
            SPLIT="train",
            FILEFORMAT="tfrecord",
            SHARD_X_OF_Y=f"{new_shard_idx:05d}-of-{new_total_shards:05d}"
        )
        
        src_file = f"{src_base}/{original_filename}"
        temp_dst_file = os.path.join(version_dst, original_filename)
        final_dst_file = os.path.join(version_dst, new_filename)
        
        print(f"[INFO] Downloading shard {new_shard_idx+1}/{len(shards_needed)}: {src_file}")
        cmd = ["gsutil", "cp", src_file, temp_dst_file]
        print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[ERROR] Failed to download shard {src_file}")
            return rc
        
        # Rename the file to match the new shard numbering
        if temp_dst_file != final_dst_file:
            os.rename(temp_dst_file, final_dst_file)
            print(f"[INFO] Renamed {original_filename} -> {new_filename}")
    
    # Update dataset_info.json to only reference the downloaded shards
    # This is crucial so TensorFlow Datasets doesn't try to read missing files
    train_split["shardLengths"] = [shard_lengths[i] for i in shards_needed]
    
    # Write the updated dataset_info.json
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"[INFO] Updated dataset_info.json to reference only {len(shards_needed)} downloaded shards")
    print(f"[DONE] Downloaded {len(shards_needed)} shards containing ~{episodes_so_far} episodes")
    return 0


def download_datasets(out_dir: str, dataset_names: List[str], max_episodes: Optional[int] = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Datasets: {dataset_names}")
    if max_episodes is not None:
        print(f"[INFO] Max episodes per dataset: {max_episodes}")

    for dataset_name in dataset_names:
        version = dataset_to_version(dataset_name)
        src_base = f"gs://gresearch/robotics/{dataset_name}/{version}"
        dst = os.path.join(out_dir, dataset_name)
        os.makedirs(dst, exist_ok=True)
        
        if max_episodes is None:
            # Download entire dataset
            print(f"[INFO] Copying entire dataset {src_base} -> {dst}")
            cmd = ["gsutil", "-m", "cp", "-r", src_base, dst]
            print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[ERROR] Failed to copy {src_base}")
                return rc
        else:
            # Download selectively based on max_episodes
            rc = _download_selective(src_base, dst, dataset_name, version, max_episodes)
            if rc != 0:
                return rc

    print("[DONE] Download complete")
    return 0


