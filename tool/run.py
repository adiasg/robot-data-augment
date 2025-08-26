import argparse
import os
from typing import List, Sequence

from download_dataset import download_datasets
from export_videos import export_videos
from generate_video import generate_video


DEFAULT_DATASETS: Sequence[str] = (
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
)


def parse_dataset_args(arg_list: List[str] | None, csv: str | None, env: str | None) -> List[str]:
    datasets: List[str] = []
    if arg_list:
        datasets.extend(arg_list)
    if csv:
        datasets.extend(csv.replace(",", " ").split())
    if env:
        datasets.extend(env.replace(",", " ").split())
    if not datasets:
        datasets = list(DEFAULT_DATASETS)
    # De-duplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for name in datasets:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


# No shell orchestration needed; all actions are Python functions


def subcommand_download(args: argparse.Namespace) -> int:
    datasets = parse_dataset_args(args.dataset, args.datasets, os.getenv("DATASETS"))
    max_episodes = getattr(args, 'max_episodes', None)
    return download_datasets(out_dir="/datasets", dataset_names=datasets, max_episodes=max_episodes)


def subcommand_export(args: argparse.Namespace) -> int:
    datasets = parse_dataset_args(args.dataset, args.datasets, os.getenv("DATASETS"))

    return export_videos(
        data_dir="/datasets",
        out_dir="/videos",
        dataset_names=datasets,
        split=args.split,
        max_episodes=args.max_episodes,
        fps=args.fps,
        display_key=args.display_key,
        image_key_choice=args.image_key_choice,
        info=args.info,
    ) or 0


# (intentionally left without a combined command; future commands like generate_video/augment_dataset can be added here)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified OXE dataset tool: download datasets and export videos")
    sub = p.add_subparsers(dest="command", required=True)

    # Common dataset options (added separately to each subparser)
    def add_dataset_opts(sp: argparse._SubParsersAction | argparse.ArgumentParser):
        sp.add_argument(
            "--dataset",
            action="append",
            default=None,
            help="Dataset name (repeatable). Example: --dataset austin_buds_dataset_converted_externally_to_rlds",
        )
        sp.add_argument(
            "--datasets",
            type=str,
            default=None,
            help="Comma or space separated list of datasets (alternative to repeating --dataset)",
        )

    # download
    pd = sub.add_parser("download_dataset", help="Download one or more RLDS datasets from the public OXE GCS mirror")
    pd.add_argument("--max_episodes", type=int, default=None, help="Max episodes to download per dataset (default: download all)")
    add_dataset_opts(pd)
    pd.set_defaults(func=subcommand_download)

    # export
    pe = sub.add_parser("export_video", help="Export MP4 videos from local RLDS datasets")
    pe.add_argument("--split", type=str, default="train", help="Split to read (default: train)")
    pe.add_argument("--max_episodes", type=int, default=5, help="Max episodes per dataset (default: 5)")
    pe.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24)")
    pe.add_argument("--display_key", type=str, default="image", help="Observation key for frames (default: image)")
    pe.add_argument("--image_key_choice", type=int, default=None, help="Pre-select image key choice (1-based index) to avoid interactive prompts")
    pe.add_argument("--info", action="store_true", help="Print per-dataset table with RGB shape and FPS")
    add_dataset_opts(pe)
    pe.set_defaults(func=subcommand_export)

    # generate_video
    pg = sub.add_parser("generate_video", help="Transform a video using an AI model on Replicate")
    pg.add_argument("--dataset", type=str, required=True, help="Dataset name")
    pg.add_argument("--video-name", type=str, required=True, help="Video filename (e.g., ep00001.mp4)")
    pg.add_argument("--prompt", type=str, required=True, help="Prompt string for the model")
    pg.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")

    def subcommand_generate(args: argparse.Namespace) -> int:
        try:
            output_path = generate_video(
                video_dir_path="/videos",
                dataset_name=args.dataset,
                video_name=args.video_name,
                prompt=args.prompt,
                seed=args.seed,
            )
            print(f"[SUCCESS] Generated video saved to: {output_path}")
            return 0
        except Exception as e:
            print(f"[ERROR] {e}")
            return 1

    pg.set_defaults(func=subcommand_generate)

    # (no combined command; run the two commands separately)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


