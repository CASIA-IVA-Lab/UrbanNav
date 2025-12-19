# -----------------------------------------------------------------------------
# Merge label (.json) and pose (.txt) files into corresponding trajectory data directories.
#
# Given a root directory containing per-trajectory subfolders (e.g., /data/traj_001/),
# this script looks for matching files in separate label and pose directories:
#   - {traj_name}.json in the label directory
#   - {traj_name}.txt in the pose directory
#
# If both files exist, they are copied into the trajectory's folder as:
#   - label.json
#   - traj_data.txt
#
# Trajectories missing either file are recorded in 'filtered_trajs.txt' (or a user-specified output).
#
# Author: UrbanNav Project Contributors
# -----------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def get_traj_names(data_dir: Path):
    """Get all subdirectory names (i.e., trajectory names) in the data directory."""
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    return [p.name for p in data_dir.iterdir() if p.is_dir()]


def merge_data(data_dir: Path, anno_dir: Path, traj_name: str) -> bool:
    """
    Copy the corresponding label.json and traj_data.txt into the trajectory data folder.
    Returns False if either source file is missing.
    """
    data_traj_path = data_dir / traj_name
    label_file = anno_dir / traj_name / f"{traj_name}.json"
    pose_file = anno_dir / traj_name / f"{traj_name}.txt"

    # Check if source files exist
    if label_file.is_file() and pose_file.is_file():
        # Ensure the target directory exists (safer even if it should already exist)
        data_traj_path.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy(label_file, data_traj_path / "label.json")
            shutil.copy(pose_file, data_traj_path / "traj_data.txt")
            return True
        except Exception as e:
            print(f"[Error] Failed to copy files for trajectory '{traj_name}': {e}")
            return False
    
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge label and pose files into trajectory data folders.")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Directory containing trajectory subfolders"
        )
    parser.add_argument(
        "--anno-dir", 
        type=str, 
        required=True, 
        help="Directory containing .json label files"
        )
    parser.add_argument(
        "--output-filtered", 
        type=str, 
        default="filtered_trajs.txt",
        help="Output file listing trajectories with missing files (default: filtered_trajs.txt)"
        )
    args = parser.parse_args()

    # Convert to Path objects and resolve to absolute paths
    data_dir = Path(args.data_dir).resolve()
    anno_dir = Path(args.anno_dir).resolve()
    output_filtered = Path(args.output_filtered)

    # Validate that input directories exist
    for name, path in [("data", data_dir), ("anno", anno_dir)]:
        if not path.is_dir():
            print(f"Error: The {name} directory does not exist: {path}")
            exit(1)

    all_trajs = get_traj_names(data_dir)
    filtered_trajs = []

    for traj in tqdm(all_trajs, desc="Merging data"):
        success = merge_data(data_dir, anno_dir, traj)
        if not success:
            filtered_trajs.append(traj)

    # Write the list of filtered-out trajectories to a file
    with open(output_filtered, "w") as f:
        for traj in filtered_trajs:
            f.write(traj + "\n")

    print(f"Merging completed. {len(filtered_trajs)} out of {len(all_trajs)} trajectories were filtered out.")

