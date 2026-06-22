import os
from rest_framework import serializers

class AudioFileSerializer(serializers.Serializer):
    file = serializers.FileField(required=True)

    def validate_file(self, value):
        # Extract and validate file extension
        ext = os.path.splitext(value.name)[1].lower()
        valid_extensions = [".wav", ".mp3", ".flac"]
        if ext not in valid_extensions:
            raise serializers.ValidationError(
                f"Unsupported file format: {ext}. Supported formats: .wav, .mp3, .flac"
            )
        return value
