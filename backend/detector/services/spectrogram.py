# Log-Mel Spectrogram Service Wrapper
from audio_forgery.features import build_resnet_tensor

def generate_log_mel_spectrogram_tensor(file_path, config):
    """
    Generates and returns the Mel-spectrogram tensor ready for ResNet++ input.
    """
    return build_resnet_tensor(file_path, config)
