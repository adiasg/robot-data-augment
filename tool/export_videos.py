import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import List

import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from download_dataset import dataset_to_version

# Reduce TensorFlow Python logger verbosity as well
tf.get_logger().setLevel("ERROR")


def dataset2path(dataset_name: str) -> str:
    version = dataset_to_version(dataset_name)
    return f"{dataset_name}/{version}"


def _get_image_key(dataset_name: str, observation_features) -> str:
    """Get the appropriate image key for different datasets."""
    # Standard key used by most datasets
    if "image" in observation_features:
        return "image"
    
    # Dataset-specific mappings
    if dataset_name == "droid":
        # Droid has multiple camera views, prefer exterior view
        if "exterior_image_1_left" in observation_features:
            return "exterior_image_1_left"
        elif "exterior_image_2_left" in observation_features:
            return "exterior_image_2_left"
        elif "wrist_image_left" in observation_features:
            return "wrist_image_left"
    
    # Fallback: try to find any image-like key
    for key in observation_features.keys():
        if "image" in key.lower():
            return key
    
    return None


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
            detected_key = _get_image_key(ds_name, observation_features)
            if detected_key:
                print(f"[INFO] Using '{detected_key}' instead of '{display_key}' for {ds_name}")
                display_key = detected_key
            else:
                print(f"[WARN] Skipping. No image key found in {ds_name} (available keys: {list(observation_features.keys())})")
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
    


