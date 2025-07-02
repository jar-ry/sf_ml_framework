#!/bin/bash

# Git Project Setup Script
# This script helps set up a new Git project and connects it to GitHub
# Before running this script:
# 1. Create a new repository on GitHub (https://github.com/new)
# 2. DO NOT initialize it with README, license, or .gitignore
# 3. Copy the repository URL from GitHub

# =====================================================
# CONFIGURATION - Update these variables for your project
# =====================================================

# Replace with your new repository name
REPO_NAME="sf_ml_framework"

# Replace with your GitHub username
GITHUB_USERNAME="jar-ry"

# Construct the repository URL
REPO_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# =====================================================
# Git Setup and Configuration
# =====================================================

# Print current Git configuration
echo "Current Git Configuration:"
echo "------------------------"
git config --get user.name
git config --get user.email

# Initialize Git repository (if not already initialized)
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    echo "Git repository initialized."
else
    echo "Git repository already initialized."
fi

# =====================================================
# Repository Setup
# =====================================================

# Add all files in the current directory
echo "Adding files to Git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit"

# Add remote repository
echo "Adding remote repository..."
git remote add origin $REPO_URL

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

# =====================================================
# Verification
# =====================================================

# Verify remote configuration
echo "Verifying remote configuration:"
git remote -v

echo "Setup complete! Your repository is now connected to GitHub."
echo "Repository URL: $REPO_URL"

# =====================================================
# Additional Information
# =====================================================
# Note: This script uses the GitHub credentials stored in your macOS keychain
# If you need to update your credentials:
# 1. Create a new Personal Access Token on GitHub
# 2. Use it once with git push, and it will be stored in your keychain
# 3. Future operations will use the stored credentials automatically 