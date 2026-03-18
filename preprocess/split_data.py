import os
import random
import shutil
from pathlib import Path

# ===================== Configuration Parameters =====================
seed_value = 2  

random.seed(seed_value) 
# Root directory of original dataset
src_root = "/root/autodl-tmp/wads"
# Directory to save split dataset (train/val)
dst_root = f"/root/autodl-tmp/wads_{seed_value}"  # Customizable path
# Sequences to exclude from classification
exclude_seqs = {"11", "14", "16"}
# Ratio of training set (80%)
train_ratio = 0.8
# Data copy mode: True=copy files (takes more space), False=create symlinks (space-saving)
copy_files = False

# ===================== Core Logic =====================
def main():
    # 1. Get all valid sequences: 
    #    - Only keep numeric directories
    #    - Exclude specified sequences
    all_seqs = []
    for d in os.listdir(src_root):
        # Check if the item is a directory AND its name is a pure number
        dir_path = os.path.join(src_root, d)
        if os.path.isdir(dir_path) and d.isdigit():
            if d not in exclude_seqs:
                all_seqs.append(d)
    
    # Sort sequences numerically (for traceability)
    all_seqs = sorted(all_seqs, key=lambda x: int(x))
    print(f"List of valid sequences: {all_seqs}")
    print(f"Total number of valid sequences: {len(all_seqs)}")

    # 2. Random shuffle and split into train/validation sets
    random.shuffle(all_seqs)
    train_size = int(len(all_seqs) * train_ratio)
    train_seqs = all_seqs[:train_size]
    val_seqs = all_seqs[train_size:]

    print(f"\nTraining set sequences ({len(train_seqs)}): {sorted(train_seqs, key=lambda x: int(x))}")
    print(f"Validation set sequences ({len(val_seqs)}): {sorted(val_seqs, key=lambda x: int(x))}")

    # 3. Create target directory structure
    for split in ["train", "val"]:
        target_seqs = train_seqs if split == "train" else val_seqs
        for seq in target_seqs:
            # Create labels and velodyne subdirectories
            for sub_dir in ["labels", "velodyne"]:
                dst_path = Path(dst_root) / split / seq / sub_dir
                dst_path.mkdir(parents=True, exist_ok=True)

                # 4. Link/copy original data to target directory
                src_path = Path(src_root) / seq / sub_dir
                # Process all files in the original subdirectory
                if os.path.exists(src_path):
                    for file in os.listdir(src_path):
                        src_file = src_path / file
                        dst_file = dst_path / file

                        if copy_files:
                            # Copy files (preserve metadata)
                            shutil.copy2(src_file, dst_file)
                        else:
                            # Create symbolic link (skip if link already exists)
                            if not os.path.exists(dst_file):
                                os.symlink(src_file, dst_file)

    print(f"\nDataset splitting completed!")
    print(f"Split dataset saved to: {dst_root}")
    print(f"Training set path: {dst_root}/train")
    print(f"Validation set path: {dst_root}/val")

if __name__ == "__main__":
    main()