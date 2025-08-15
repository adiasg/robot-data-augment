import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import List

import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Reduce TensorFlow Python logger verbosity as well
tf.get_logger().setLevel("ERROR")


def dataset2path(dataset_name: str) -> str:
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"{dataset_name}/{version}"


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
        if display_key not in builder.info.features["steps"]["observation"].keys():
            print(f"[WARN] Skipping. Display key '{display_key}' missing in {ds_name}")
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
    


