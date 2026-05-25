import os

def get_all_wav_files(folder_path):
    """
    Returns all .wav files recursively.
    Supports both .wav and .WAV
    """

    wav_files = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            # Supports .wav and .WAV
            if file.lower().endswith(".wav"):

                full_path = os.path.join(root, file)

                wav_files.append(full_path)

    return wav_files


def create_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)
