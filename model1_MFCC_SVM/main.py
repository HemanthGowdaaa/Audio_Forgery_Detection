import os
import shutil
import numpy as np

from utils import get_all_wav_files, create_directory
from forgery_generation import create_copy_move_forgery
from feature_extraction import extract_mfcc_features
from train_model import train_svm_model


# ==============================
# TIMIT DATASET PATH
# ==============================

timit_path = "/Users/hemanth/Projects/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN"

# Local dataset folders (resolves dynamically based on execution folder)
base_dir = ".." if not os.path.exists("dataset") and os.path.exists("../dataset") else ""
ORIGINAL_FOLDER = os.path.join(base_dir, "dataset/original")
FORGED_FOLDER = os.path.join(base_dir, "dataset/forged")


def copy_timit_files():
    """
    Copy all TIMIT wav files into dataset/original
    """

    create_directory(ORIGINAL_FOLDER)

    wav_files = get_all_wav_files(timit_path)

    copied = 0

    print("\nCopying TIMIT audio files...\n")

    for file_path in wav_files:

        try:
            filename = os.path.basename(file_path)

            destination = os.path.join(
                ORIGINAL_FOLDER,
                filename
            )

            shutil.copy(file_path, destination)

            copied += 1

        except Exception as e:
            print(f"Error copying {file_path}: {e}")

    print(f"Total copied files: {copied}")


def generate_forged_dataset():
    """
    Generate forged audio samples
    """

    create_directory(FORGED_FOLDER)

    original_files = get_all_wav_files(ORIGINAL_FOLDER)

    forged_count = 0

    print("\nGenerating forged audio samples...\n")

    for file_path in original_files:

        filename = os.path.basename(file_path)

        forged_output = os.path.join(
            FORGED_FOLDER,
            f"forged_{filename}"
        )

        success = create_copy_move_forgery(
            file_path,
            forged_output
        )

        if success:
            forged_count += 1

    print(f"Forged files created: {forged_count}")


def prepare_dataset():

    X = []
    y = []

    # Original audio
    original_files = get_all_wav_files(ORIGINAL_FOLDER)

    print("\nExtracting original audio features...\n")

    for file_path in original_files:

        features = extract_mfcc_features(file_path)

        if features is not None:
            X.append(features)
            y.append(0)

    # Forged audio
    forged_files = get_all_wav_files(FORGED_FOLDER)

    print("\nExtracting forged audio features...\n")

    for file_path in forged_files:

        features = extract_mfcc_features(file_path)

        if features is not None:
            X.append(features)
            y.append(1)

    print("\n========== DATASET INFO ==========")
    print(f"Original Samples : {len(original_files)}")
    print(f"Forged Samples  : {len(forged_files)}")

    return np.array(X), np.array(y)


def main():

    # Step 1: Copy TIMIT files
    copy_timit_files()

    # Step 2: Generate forged audio
    generate_forged_dataset()

    # Step 3: Extract features
    X, y = prepare_dataset()

    # Step 4: Train model
    train_svm_model(X, y)


if __name__ == "__main__":
    main()
