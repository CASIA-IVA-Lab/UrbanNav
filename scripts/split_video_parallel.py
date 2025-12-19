# -----------------------------------------------------------------------------
# Parallel Video Segmenter - Split MP4 videos into fixed-duration chunks using FFmpeg.
#
# - Processes multiple videos concurrently via multiprocessing (one process per video).
# - Uses stream copy (`-c copy`) for fast, lossless splitting (no re-encoding).
# - Outputs segments as: {output_dir}/{video_name}_0000.mp4, etc.
#
# Author: UrbanNav Project Contributors
# -----------------------------------------------------------------------------

import os
import ffmpeg
import argparse
from multiprocessing import Pool
import sys


def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"[ERROR] Failed to probe {os.path.basename(video_path)}: {e.stderr.decode()}", file=sys.stderr)
        return None


def split_video_task(args):
    """
    Single video processing task. No tqdm inside.
    Returns a status message (str) for logging.
    """
    input_file, segment_time, output_dir = args
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(input_file)
    if duration is None:
        return f"[ERROR] Could not get duration for {base_name}"

    num_segments = int(duration // segment_time) + (1 if duration % segment_time > 0 else 0)

    for i in range(num_segments):
        start_time = i * segment_time
        output_filename = os.path.join(output_dir, f"{base_name}_{i:04d}.mp4")

        if os.path.exists(output_filename):
            continue

        try:
            (
                ffmpeg
                .input(input_file, ss=start_time, t=segment_time)
                .output(output_filename, c="copy", map="0", reset_timestamps=1)
                .run(overwrite_output=True, quiet=True)
            )
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            print(f"[ERROR] Segment {i} of {base_name}: {err_msg[:200]}...", file=sys.stderr)
            continue

    print(f"[DONE] {base_name} ({num_segments} segments)")
    return f"[DONE] {base_name} ({num_segments} segments)"


def process_all_videos(input_dir, output_dir, segment_time, num_workers):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist!")

    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if not video_files:
        raise ValueError(f"No MP4 files found in '{input_dir}'.")

    print(f"Found {len(video_files)} videos. Starting parallel processing with {num_workers} workers...\n")

    tasks = [(os.path.join(input_dir, vf), segment_time, output_dir) for vf in video_files]

    results = []
    with Pool(processes=num_workers) as pool:
        # Use imap to get results as they complete
        for result in pool.imap(split_video_task, tasks):
            results.append(result)

    print(f"\nAll {len(results)} videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Video Segmenter")
    parser.add_argument("--video-dir", type=str, required=True, help="Input video folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder")
    parser.add_argument("--duration", type=int, default=120, help="Segment duration (seconds)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()
    workers = args.workers

    try:
        process_all_videos(args.input_folder, args.output_folder, args.duration, workers)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        exit(1)

