import os
import requests
import re

# Define base directory for datasets
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hard_dataset"))

# Define subfolders
DATASETS = ["all", "paired", "cv", "vox"]
FOLDERS = {dataset: {"noisy": os.path.join(BASE_DIR, dataset, "noisy"), 
                     "clean": os.path.join(BASE_DIR, dataset, "clean")} 
           for dataset in DATASETS}

# Ensure all directories exist
for dataset in DATASETS:
    os.makedirs(FOLDERS[dataset]["noisy"], exist_ok=True)
    os.makedirs(FOLDERS[dataset]["clean"], exist_ok=True)

# Define files
NOISY_FILES = [
    "https://google.github.io/df-conformer/miipher/data/obj1_noisy.wav",
    "https://google.github.io/df-conformer/miipher/data/obj0_noisy.wav",
    "https://google.github.io/df-conformer/miipher/data/force_align/force_align_ex1_noisy.wav",
]

CLEAN_FILES = [
    "https://google.github.io/df-conformer/miipher/data/obj1_clean.wav",
    "https://google.github.io/df-conformer/miipher/data/obj0_clean.wav",
    "https://google.github.io/df-conformer/miipher/data/force_align/force_align_ex1_clean.wav",
    "https://google.github.io/df-conformer/miipher/data/cv2.wav",
    "https://google.github.io/df-conformer/miipher/data/cv1.wav",
    "https://google.github.io/df-conformer/miipher/data/cv3.wav",
    "https://google.github.io/df-conformer/miipher/data/cv4.wav",
    "https://google.github.io/df-conformer/miipher/data/vox1.wav",
    "https://google.github.io/df-conformer/miipher/data/vox4.wav",
    "https://google.github.io/df-conformer/miipher/data/vox2.wav",
    "https://google.github.io/df-conformer/miipher/data/vox3.wav",
]

PAIRED_CLEAN = [
    "https://google.github.io/df-conformer/miipher/data/obj1_clean.wav",
    "https://google.github.io/df-conformer/miipher/data/obj0_clean.wav",
    "https://google.github.io/df-conformer/miipher/data/force_align/force_align_ex1_clean.wav",
]

CV_NOISY = [
    "https://google.github.io/df-conformer/miipher/data/cv2.wav",
    "https://google.github.io/df-conformer/miipher/data/cv1.wav",
    "https://google.github.io/df-conformer/miipher/data/cv3.wav",
    "https://google.github.io/df-conformer/miipher/data/cv4.wav",
]

VOX_NOISY = [
    "https://google.github.io/df-conformer/miipher/data/vox1.wav",
    "https://google.github.io/df-conformer/miipher/data/vox4.wav",
    "https://google.github.io/df-conformer/miipher/data/vox2.wav",
    "https://google.github.io/df-conformer/miipher/data/vox3.wav",
]

def download_file(url, folder):
    """Downloads a file and saves it in the specified folder."""
    filename = os.path.join(folder, os.path.basename(url))
    if os.path.exists(filename):
        print(f"File already exists: {filename}, skipping download.")
        return

    print(f"Downloading: {url} -> {filename}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        print(f"Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def rename_files_in_folder(folder):
    """Removes '_clean' and '_noisy' from filenames in the given folder."""
    for filename in os.listdir(folder):
        old_path = os.path.join(folder, filename)
        if os.path.isfile(old_path):
            new_filename = re.sub(r"_(clean|noisy)\.wav$", ".wav", filename)  # Remove suffix
            new_path = os.path.join(folder, new_filename)
            
            if old_path != new_path:  # Only rename if different
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

# Download files into "all" (everything goes here)
for url in NOISY_FILES + CLEAN_FILES:
    download_file(url, FOLDERS["all"]["clean"] if "clean" in url else FOLDERS["all"]["noisy"])

# Download files into "paired"
for url in NOISY_FILES:
    download_file(url, FOLDERS["paired"]["noisy"])
for url in PAIRED_CLEAN:
    download_file(url, FOLDERS["paired"]["clean"])

# Rename files in paired/noisy and paired/clean (remove _clean and _noisy)
rename_files_in_folder(FOLDERS["paired"]["noisy"])
rename_files_in_folder(FOLDERS["paired"]["clean"])

# Download files into "cv"
for url in CV_NOISY:
    download_file(url, FOLDERS["cv"]["noisy"])

# Download files into "vox"
for url in VOX_NOISY:
    download_file(url, FOLDERS["vox"]["noisy"])

print("All downloads complete and files renamed in paired dataset!")
