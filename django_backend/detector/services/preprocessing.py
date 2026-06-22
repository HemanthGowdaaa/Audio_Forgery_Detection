# Audio Preprocessing Service Wrapper
from audio_forgery.features import extract_feature_bundle, aggregate_svm_features

def preprocess_audio_for_svm(file_path, config):
    """
    Extracts and aggregates features for SVM baseline input.
    """
    raw_features = extract_feature_bundle(file_path, config)
    aggregated = aggregate_svm_features(raw_features)
    return aggregated.reshape(1, -1)
