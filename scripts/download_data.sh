#!/bin/bash

# --- Download and Prepare KuaiRec Dataset ---
# This script should be run from the root directory of the project

# --- Configuration ---
TARGET_DIR="raw_data"
DOWNLOAD_URL="https://nas.chongminggao.top:4430/datasets/KuaiRec.zip"
ZIP_FILE="KuaiRec.zip"
EXPECTED_EXTRACT_FOLDER="KuaiRec 2.0" # The folder name inside the zip based on user output
FINAL_FOLDER_NAME="KuaiRec"           # The desired folder name after extraction

echo "Ensuring target directory '$TARGET_DIR' exists..."
mkdir -p "$TARGET_DIR"

echo "Changing directory to '$TARGET_DIR'..."
cd "$TARGET_DIR" || { echo "Error: Could not change to directory '$TARGET_DIR'"; exit 1; }

# --- Download Data ---
if [ -d "$FINAL_FOLDER_NAME/data" ]; then
    echo "Data directory '$FINAL_FOLDER_NAME/data' already exists. Skipping download and unzip."
else
    # Check if zip file exists, download if not
    if [ ! -f "$ZIP_FILE" ]; then
        echo "Downloading dataset from $DOWNLOAD_URL..."
        wget "$DOWNLOAD_URL" --no-check-certificate -O "$ZIP_FILE"

        # Check if wget was successful
        if [ $? -ne 0 ]; then
            echo "Error: Download failed. Please check the URL and your internet connection."
            rm -f "$ZIP_FILE"
            cd .. # Go back to project root before exiting
            exit 1
        fi
        echo "Download complete."
    else
        echo "Zip file '$ZIP_FILE' already exists. Skipping download."
    fi

    echo "Unzipping '$ZIP_FILE'..."
    unzip -q "$ZIP_FILE"

    # Check if unzip was successful
    if [ ! -d "$EXPECTED_EXTRACT_FOLDER" ]; then
        echo "Error: Unzipping failed or the expected folder '$EXPECTED_EXTRACT_FOLDER' was not found."
        cd .. # Go back to project root before exiting
        exit 1
    fi
    echo "Unzipping complete."

    echo "Renaming '$EXPECTED_EXTRACT_FOLDER' to '$FINAL_FOLDER_NAME'..."
    mv "$EXPECTED_EXTRACT_FOLDER" "$FINAL_FOLDER_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Renaming failed."
        cd .. # Go back to project root before exiting
        exit 1
    fi
    echo "Renaming successful."

    echo "Removing zip file '$ZIP_FILE'..."
    rm "$ZIP_FILE"
    echo "Cleanup complete."

fi

echo "Returning to project root directory..."
cd ..

echo "--- Data download and preparation script finished. ---"
echo "Raw data should now be available in '$TARGET_DIR/$FINAL_FOLDER_NAME/data/'"

exit 0
