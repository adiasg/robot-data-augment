import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import List

import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from download_dataset import dataset_to_version

# Removed inquirer dependency - using simple text-based selection only

# Reduce TensorFlow Python logger verbosity as well
tf.get_logger().setLevel("ERROR")


def dataset2path(dataset_name: str) -> str:
    version = dataset_to_version(dataset_name)
    return f"{dataset_name}/{version}"


def _select_image_key_interactively(dataset_name: str, observation_features, pre_selected_choice: int = None) -> str:
    """Interactively select the appropriate image key for the dataset."""
    # Filter for image-like keys
    image_keys = []
    for key in observation_features.keys():
        if "image" in key.lower() or "rgb" in key.lower():
            image_keys.append(key)
    
    if not image_keys:
        print(f"[WARN] No image-like keys found in {dataset_name}")
        print(f"Available keys: {list(observation_features.keys())}")
        return None
    
    if len(image_keys) == 1:
        print(f"[INFO] Only one image key available: {image_keys[0]}")
        return image_keys[0]
    
    # Multiple image keys available
    print(f"\n[INFO] Multiple image keys found for dataset '{dataset_name}':")
    for i, key in enumerate(image_keys):
        print(f"  {i + 1}. {key}")
    
    # Use pre-selected choice if provided
    if pre_selected_choice is not None:
        if 1 <= pre_selected_choice <= len(image_keys):
            selected_key = image_keys[pre_selected_choice - 1]
            print(f"[INFO] Using pre-selected choice {pre_selected_choice}: {selected_key}")
            return selected_key
        else:
            print(f"[WARN] Invalid pre-selected choice {pre_selected_choice}, must be 1-{len(image_keys)}. Falling back to interactive selection.")
    
    # Check if we're in a truly interactive environment first
    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    
    if not is_interactive:
        # Not in interactive mode, use default immediately
        print(f"[INFO] Non-interactive mode detected. Using default: {image_keys[0]}")
        print(f"[TIP] Use --image_key_choice N to pre-select, or run with 'docker run -it' for interactive mode")
        return image_keys[0]
    
    # Interactive mode - simple text-based selection
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(image_keys)}) [default: 1]: ").strip()
            if not choice:
                return image_keys[0]
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(image_keys):
                return image_keys[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(image_keys)}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"[INFO] Using default: {image_keys[0]}")
            return image_keys[0]


def write_video(frames: List[Image.Image], out_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            writer.append_data(arr)


def export_videos(
    data_dir: str,
    out_dir: str,
    dataset_names: List[str],
    split: str = "train",
    max_episodes: int = 5,
    fps: int = 24,
    display_key: str = "image",
    image_key_choice: int = None,
    info: bool = False,
) -> None:
    for ds_name in dataset_names:
        builder_dir = os.path.join(data_dir, dataset2path(ds_name))
        if not os.path.isdir(builder_dir):
            print(f"[WARN] Skipping. Dataset not found: {builder_dir}")
            continue

        builder = tfds.builder_from_directory(builder_dir=builder_dir)
        observation_features = builder.info.features["steps"]["observation"]
        
        # Auto-detect the appropriate image key if the default doesn't exist
        if display_key not in observation_features.keys():
            selected_key = _select_image_key_interactively(ds_name, observation_features, image_key_choice)
            if selected_key:
                print(f"[INFO] Using '{selected_key}' instead of '{display_key}' for {ds_name}")
                display_key = selected_key
            else:
                print(f"[WARN] Skipping. No image key selected for {ds_name}")
                continue

        if info:
            try:
                ds_probe = builder.as_dataset(split=split)
                rgb_shape = None
                for episode in ds_probe:
                    for step in episode["steps"]:
                        t = step["observation"][display_key]
                        rgb_shape = tuple(int(x) for x in t.shape)
                        break
                    break
                shape_str = "unknown" if not rgb_shape else "x".join(str(x) for x in rgb_shape)
                print(f"[INFO] {ds_name}: rgb_shape={shape_str} fps={fps}")
            except Exception:
                print(f"[INFO] {ds_name}: rgb_shape=unknown fps={fps}")

        ds = builder.as_dataset(split=split)
        # Iterate as NumPy to avoid TF rendezvous end-of-sequence noise
        ds = tfds.as_numpy(ds)

        exported = 0
        for ep_idx, episode in enumerate(ds):
            if exported >= max_episodes:
                break
            frames: List[Image.Image] = []
            for step in episode["steps"]:
                frames.append(Image.fromarray(step["observation"][display_key]))
            # Create dataset-specific subdirectory
            dataset_video_dir = os.path.join(out_dir, ds_name)
            os.makedirs(dataset_video_dir, exist_ok=True)
            out_path = os.path.join(dataset_video_dir, f"ep{ep_idx:05d}.mp4")
            print(f"[INFO] Writing {len(frames)} frames to {out_path} at {fps} fps")
            write_video(frames, out_path, fps=fps)
            exported += 1

        print(f"[DONE] Exported {exported} episode(s) for {ds_name} to {dataset_video_dir}")
    


