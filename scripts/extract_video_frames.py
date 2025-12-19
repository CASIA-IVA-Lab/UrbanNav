# -----------------------------------------------------------------------------
# Frame Extraction from MP4 Videos (Multiprocessing Version)
#
# This script extracts frames from all MP4 video files in a given input directory
# at a fixed interval (stride) and saves them as JPEG images in corresponding
# subdirectories under the specified output directory. It leverages multiprocessing
# to process multiple videos in parallel, significantly speeding up large-scale
# frame extraction tasks.
#
# Each video file `example.mp4` will result in a folder `example/` inside the output
# directory, containing frames named as `0000.jpg`, `0001.jpg`, etc., sampled every
# `stride` frames from the original video.
#
# Author: UrbanNav Project Contributors
# -----------------------------------------------------------------------------

import os
import cv2
import argparse
import pickle as pkl
from tqdm import tqdm
from multiprocessing import Process, Queue, Lock


def extract_frames(video_path, output_folder, stride=6, lock=None):
    """
    Extracts frames from a video file at regular intervals defined by `stride`.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where extracted frames will be saved.
        stride (int): Interval (in frames) at which to extract frames. 
                      For example, stride=6 means saving every 6th frame.
        lock (multiprocessing.Lock, optional): A lock to ensure thread-safe file writing 
                                              when used in multiprocessing environments.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Extract frame only if current frame index is divisible by stride
        if frame_count % stride == 0:
            frame_filename = os.path.join(output_folder, f"{extracted_count:04d}.jpg")
            if lock:
                with lock:
                    cv2.imwrite(frame_filename, frame)  # Thread-safe write
            else:
                cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1

    cap.release()


def process_video(video_rel_path, input_dir, output_dir, stride, lock, progress_queue):
    """
    Processes a single video: creates an output subdirectory and extracts frames.

    Args:
        video_rel_path (str): Relative path (filename) of the video within `input_dir`.
        input_dir (str): Root directory containing input video files.
        output_dir (str): Root directory where extracted frames will be stored.
        stride (int): Frame extraction interval.
        lock (multiprocessing.Lock): Lock for synchronized file I/O.
        progress_queue (multiprocessing.Queue): Queue to signal completion of this video to the main process.
    """
    video_path = os.path.join(input_dir, video_rel_path)
    # Derive output subdirectory name by removing the video extension
    images_dir = os.path.join(output_dir, os.path.splitext(video_rel_path)[0])

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        extract_frames(video_path, images_dir, stride, lock)
    else:
        print(f"Skip video {video_path} (output directory already exists)")

    # Signal that this video has been processed
    progress_queue.put(1)


def worker(queue, input_dir, output_dir, stride, lock, progress_queue):
    """
    Worker function run by each subprocess. Consumes video filenames from a shared queue 
    and processes them until the queue is empty.

    Args:
        queue (multiprocessing.Queue): Shared queue containing relative paths of videos to process.
        input_dir (str): Input directory containing videos.
        output_dir (str): Output root directory for frame extraction.
        stride (int): Frame sampling interval.
        lock (multiprocessing.Lock): Lock for safe concurrent file writing.
        progress_queue (multiprocessing.Queue): Queue to report processing progress back to main process.
    """
    while not queue.empty():
        try:
            video_rel_path = queue.get_nowait()  # Non-blocking get
        except:
            break  # Queue is empty or inaccessible; exit gracefully
        process_video(video_rel_path, input_dir, output_dir, stride, lock, progress_queue)


if __name__ == '__main__':
    """
    Main entry point. Parses command-line arguments, discovers MP4 videos in the input directory,
    and spawns multiple worker processes to extract frames in parallel.
    """

    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 videos in a directory using multiprocessing."
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help="Path to the directory containing input MP4 video files."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help="Path to the root output directory where extracted frames will be saved. "
             "Each video will have its own subdirectory named after the video (without extension)."
    )
    parser.add_argument(
        '--stride',
        type=int, 
        default=6,
        help="Frame extraction interval. For example, stride=6 saves every 6th frame (i.e., ~5 FPS if source is 30 FPS). Default: 6."
    )
    parser.add_argument(
        '--workers',
        type=int, 
        default=4,
        help="Number of parallel worker processes to use. Default: 4."
    )
    args = parser.parse_args()

    # Discover all MP4 video files in the input directory
    video_list = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.mp4'):
            video_list.append(file)

    if not video_list:
        print(f"No MP4 files found in {args.input_dir}. Exiting.")
        exit(0)

    # Initialize a queue to distribute video filenames among workers
    video_queue = Queue()
    for video_rel_path in video_list:
        video_queue.put(video_rel_path)

    # Queue for workers to report completion of individual videos
    progress_queue = Queue()

    # Lock to synchronize file writing operations across processes
    lock = Lock()

    # Launch worker processes
    processes = []
    for _ in range(args.workers):
        p = Process(
            target=worker,
            args=(video_queue, args.input_dir, args.output_dir, args.stride, lock, progress_queue)
        )
        p.start()
        processes.append(p)

    # Monitor progress using tqdm
    with tqdm(total=len(video_list), desc="Processing videos", unit="videos", ncols=120) as pbar:
        completed = 0
        while completed < len(video_list):
            progress_queue.get()  # Block until a worker reports completion
            completed += 1
            pbar.update(1)

    # Ensure all worker processes terminate cleanly
    for p in processes:
        p.join()

    print("All videos processed successfully.")

