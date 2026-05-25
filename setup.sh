#!/bin/bash

echo "=== Restructuring and Git Setup ==="

# 1. Clean up duplicate files from the root directory
echo "Cleaning up duplicate files from the root..."
rm -f feature_extraction.py forgery_generation.py main.py train_model.py utils.py requirements.txt

# 2. Initialize Git repository
echo "Initializing git repository..."
git init

# 3. Rename branch to main
git branch -M main

# 4. Add remote origin
echo "Adding remote origin..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/HemanthGowdaaa/Audio_Forgery_Detection.git

# 5. Add files (respects .gitignore, excluding dataset WAV files)
echo "Staging files (excluding datasets and cache)..."
git add .gitignore README.md dataset/ model1_MFCC_SVM/ model2_Spectrogram_Fusion_CNN/ setup.sh

# 6. Commit
echo "Committing..."
git commit -m "first commit"

# 7. Push to GitHub
echo "Pushing to main branch..."
git push -u origin main

echo "=== Done! ==="
