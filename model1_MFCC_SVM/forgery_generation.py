import os
import random
import librosa
import soundfile as sf
import numpy as np


def create_copy_move_forgery(input_file, output_file):
    """
    Create copy-move forged audio by:
    1. Selecting a random segment
    2. Copying it
    3. Pasting it into another location
    """

    try:
        # Load audio
        audio, sr = librosa.load(input_file, sr=None)

        # Ignore very short audio
        if len(audio) < sr * 2:
            print(f"Skipped short audio: {input_file}")
            return False

        duration = len(audio)

        # Random segment length (0.3s to 1s)
        segment_length = random.randint(int(sr * 0.3), int(sr * 1.0))

        # Random start position
        start = random.randint(0, duration - segment_length - 1)

        # Extract segment
        copied_segment = audio[start:start + segment_length]

        # Random paste location
        paste_position = random.randint(0, duration - segment_length - 1)

        # Create forged audio copy
        forged_audio = np.copy(audio)

        # Paste copied segment
        forged_audio[paste_position:paste_position + segment_length] = copied_segment

        # Save forged audio
        sf.write(output_file, forged_audio, sr)

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False
