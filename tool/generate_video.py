import json
import math
import os
import shlex
import subprocess
from typing import Optional

import replicate
import mimetypes


SUPPORTED_ASPECT_RATIOS = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1 / 1,
    "21:9": 21 / 9,
}


def _run_ffprobe(video_path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return json.loads(proc.stdout.decode("utf-8"))


def _require_24_fps(avg_frame_rate: str) -> None:
    # avg_frame_rate is like "24/1" or "24000/1001"
    try:
        num_str, den_str = avg_frame_rate.split("/")
        num, den = int(num_str), int(den_str)
    except Exception:
        raise ValueError(f"Unexpected frame rate format: {avg_frame_rate}")
    g = math.gcd(num, den)
    num //= g
    den //= g
    if not (num == 24 and den == 1):
        raise ValueError(f"Video must be 24fps exactly. Found {num}/{den} fps")


def _select_supported_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions")
    ratio = width / height
    # Choose the aspect ratio whose value is within a tight tolerance
    best: Optional[str] = None
    min_delta = 1e9
    for key, target in SUPPORTED_ASPECT_RATIOS.items():
        delta = abs(ratio - target)
        if delta < min_delta:
            min_delta = delta
            best = key
    # require close match
    if best is None or min_delta > 0.01:
        raise ValueError(
            f"Unsupported aspect ratio {width}:{height} (~{ratio:.3f}). "
            f"Supported: {', '.join(SUPPORTED_ASPECT_RATIOS.keys())}"
        )
    return best


def _require_max_duration_seconds(meta: dict, max_seconds: float = 5.0) -> None:
    duration_str = None
    if isinstance(meta, dict):
        fmt = meta.get("format", {})
        duration_str = fmt.get("duration")
    if not duration_str:
        raise ValueError("Unable to determine video duration from ffprobe metadata")
    try:
        duration = float(duration_str)
    except Exception:
        raise ValueError(f"Unexpected duration value: {duration_str}")

    if duration > max_seconds + 1e-6:
        raise ValueError(f"Video must be <= {max_seconds:.0f}s. Found {duration:.3f}s")


def _require_max_file_size(file_path: str, max_size_mb: float = 1.0) -> None:
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValueError(f"Video file must be <= {max_size_mb}MB for data URI. Found {file_size/1024/1024:.2f}MB")


# Removed unused upload functions since we're using data URI approach


def generate_video(
    video_dir_path: str,
    dataset_name: str,
    video_name: str,
    prompt: str,
    seed: Optional[int] = None,
) -> str:
    # Construct input video path
    video_path = os.path.join(video_dir_path, dataset_name, video_name)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Determine output path with generated suffix
    output_dir = os.path.join(video_dir_path, dataset_name, "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find next available generated number
    base_name = os.path.splitext(video_name)[0]  # Remove extension
    extension = os.path.splitext(video_name)[1] or ".mp4"  # Keep original extension or default to .mp4
    
    existing_numbers = []
    for filename in os.listdir(output_dir):
        if filename.startswith(f"{base_name}_generated-") and filename.endswith(extension):
            try:
                # Extract number from filename like "ep00001_generated-042.mp4"
                number_part = filename[len(f"{base_name}_generated-"):-len(extension)]
                existing_numbers.append(int(number_part))
            except ValueError:
                continue  # Skip malformed filenames
    
    next_number = max(existing_numbers, default=0) + 1
    output_filename = f"{base_name}_generated-{next_number:03d}{extension}"
    output_path = os.path.join(output_dir, output_filename)

    meta = _run_ffprobe(video_path)
    streams = meta.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found")
    stream = streams[0]
    avg_frame_rate = stream.get("avg_frame_rate", "")
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    _require_24_fps(avg_frame_rate)
    aspect_ratio = _select_supported_aspect_ratio(width, height)
    _require_max_duration_seconds(meta, max_seconds=5.0)
    _require_max_file_size(video_path, max_size_mb=1.0)
    print(f"[INFO] Using aspect_ratio={aspect_ratio} (width={width}, height={height})")

    # Prepare inputs for Replicate
    guessed_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    filename = os.path.basename(video_path) or "input.mp4"
    
    # Use data URI for video input
    import base64
    
    file_size = os.path.getsize(video_path)
    print(f"[INFO] Video file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print("[INFO] Creating data URI for video file...")
    
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        video_b64 = base64.b64encode(video_data).decode('utf-8')
        data_uri = f"data:{guessed_type};base64,{video_b64}"
        
    print(f"[INFO] Created data URI (length: {len(data_uri):,} chars)")
    inputs = {
        "video": data_uri,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    if seed is not None:
        inputs["seed"] = int(seed)
    
    print(f"[INFO] Creating Replicate prediction (runwayml/gen4-aleph) with prompt: '{prompt}'")
    output = replicate.run("runwayml/gen4-aleph", input=inputs)

    # Normalize to FileOutput
    file_output = output[0] if isinstance(output, (list, tuple)) else output

    # Save to disk
    with open(output_path, "wb") as f:
        f.write(file_output.read())
    print(f"[DONE] Wrote generated video to: {output_path}")

    # Return the local output path
    return output_path


