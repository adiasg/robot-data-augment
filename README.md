## Robot Training Data Augmentation

This repo contains a plug-and-play tool to massively multiply robot training datasets by augmenting training episodes with new scenes. Currently [RLDS](https://github.com/google-research/rlds) formatted datasets are supported, and [Open-X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) is used as an example in this repo.

<table>
  <tr>
    <td align="center">
      <strong>Original Input</strong><br>
      <img src="assets/input.gif" width="300" alt="Original robot training video">
    </td>
    <td align="center">
      <strong>Outdoor Scene</strong><br>
      <img src="assets/output-park.gif" width="300" alt="Robot video transformed to park setting">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Factory Scene</strong><br>
      <img src="assets/output-factory.gif" width="300" alt="Robot video transformed to factory setting">
    </td>
    <td align="center">
      <strong>Flashing Lights Scene</strong><br>
      <img src="assets/output-flash.gif" width="300" alt="Robot video with flash lighting effects">
    </td>
  </tr>
</table>

Images from training episodes are transformed into new scenes while leaving unchanged critical visual aspect such as trajectories and object interaction. Currently, this is powered by [RunwayML](https://runwayml.com)'s [Gen4-Alpeh](https://runwayml.com/research/introducing-runway-aleph).

Tools are provided for:
- Downloading the dataset.
- Extracting videos from the dataset.
- Generating new videos.
- (coming soon) Writing back new episodes to the dataset.

### Quickstart

#### Prerequisites

- Docker is required - this tool is packaged as a container.
- Load your [Replicate](https://replicate.com) API key in `.env` - video-to-video generative model APIs from Replicate are used:
```bash
REPLICATE_API_TOKEN=XXXXXXXXXXX
```

#### Build

```bash
docker build -f tool/Dockerfile -t oxe-tool .
```

#### Run

- Make directories for inputs and outputs:
```bash
mkdir oxe-datasets videos
```

- Download a sample dataset from Open-X-Embodiment:
```bash
docker run --rm \
  -v "$(pwd)/oxe-datasets:/datasets" \
  oxe-tool download_dataset \
  --dataset dlr_sara_grid_clamp_converted_externally_to_rlds
```

- Download only a few episodes (saves time and storage):
```bash
docker run --rm \
  -v "$(pwd)/oxe-datasets:/datasets" \
  oxe-tool download_dataset \
  --dataset dlr_sara_grid_clamp_converted_externally_to_rlds \
  --max_episodes 10
```

- Export videos into `./videos`:
```bash
docker run --rm \
  -v "$(pwd)/oxe-datasets:/datasets:ro" \
  -v "$(pwd)/videos:/videos" \
  oxe-tool export_video \
  --dataset dlr_sara_grid_clamp_converted_externally_to_rlds \
  --max_episodes 5 --fps 24 --info
```

- Generate a transformed video:
```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/videos:/videos" \
  oxe-tool generate_video \
  --dataset dlr_sara_grid_clamp_converted_externally_to_rlds \
  --video-name ep00001.mp4 \
  --prompt "Place the robot & table in a factory" \
  --seed 1234
```

### CLI Overview

Run help:

```bash
docker run --rm oxe-tool --help
docker run --rm oxe-tool download_dataset --help
docker run --rm oxe-tool export_video --help
docker run --rm oxe-tool generate_video --help
```

Subcommands and key options:

- `download_dataset`
  - Downloads to `/datasets` (mounted via Docker `-v`)
  - `--dataset` (repeatable), or `--datasets` (comma/space-separated). If not provided, defaults to `dlr_sara_grid_clamp_converted_externally_to_rlds`.
  - `--max_episodes`: optional integer to limit episodes downloaded per dataset (default: download all episodes).

- `export_video`
  - Reads from `/datasets`, writes to `/videos` (both mounted via Docker `-v`)
  - `--dataset` (repeatable), or `--datasets` (comma/space-separated). If not provided, defaults to `dlr_sara_grid_clamp_converted_externally_to_rlds`.
  - `--split` (default `train`), `--max_episodes` (default `5`), `--fps` (default `24`), `--display_key` (default `image`), `--info`

- `generate_video`
  - Reads/writes from `/videos` (mounted via Docker `-v`)
  - `--dataset`: dataset name (matches directory name in video structure)
  - `--video-name`: video filename (e.g., `ep00001.mp4`)
  - `--prompt`: text prompt for the model
  - `--seed`: optional integer for reproducible generations
  - Input video must be 24fps, ≤5s, ≤1MB; aspect ratio must be one of `16:9`, `9:16`, `4:3`, `3:4`, `1:1`, `21:9`
  - Output saved to `/videos/{dataset}/generated/{video-name}_generated-{number}.mp4`
  - Requires `REPLICATE_API_TOKEN` in the environment (e.g., `--env-file .env`).

- Future commands (coming soon): `augment_dataset`.

Dataset selection can also be provided via the `DATASETS` env var (comma or space separated). If no datasets are specified, the default is `dlr_sara_grid_clamp_converted_externally_to_rlds`.

### Notes

- The tool downloads the Open-X-Embodiment dataset from the public mirror `gs://gresearch/robotics/`.
- **File Organization**: 
  - `export_video` creates dataset-specific subdirectories: `{video-dir}/{dataset}/ep{N}.mp4`
  - `generate_video` creates organized output: `{video-dir}/{dataset}/generated/{video}_generated-{N}.mp4`
  - Generated videos use automatic numbering to prevent overwrites
- **Video Requirements for AI Generation**: 24fps, ≤5 seconds duration, ≤1MB file size, supported aspect ratios only
- **Video-to-Video AI Model**: Uses RunwayML's Gen4-Aleph
