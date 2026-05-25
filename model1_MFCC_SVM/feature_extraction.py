import librosa
import numpy as np


def extract_mfcc_features(file_path, n_mfcc=13):
    """
    Extract MFCC features from audio.

    Returns:
        Fixed-length feature vector
    """

    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)

        # Ignore very short files
        if len(audio) < sr:
            print(f"Short audio skipped: {file_path}")
            return None

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc
        )

        # Compute statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Combine mean + std
        features = np.concatenate((mfcc_mean, mfcc_std))

        return features

    except Exception as e:
        print(f"Feature extraction error in {file_path}: {e}")
        return None
