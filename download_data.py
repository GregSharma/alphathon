import requests
import zipfile
import os

# Dropbox shared link
DROPBOX_URL = "https://www.dropbox.com/scl/fo/3uewkaxi3boeebmuj0y4z/APsOovGyE4VnQFAhBukaM5U?rlkey=okt4vad6sik9dekcj0ku41dk9&st=xnhnpf06&dl=1"
ZIP_PATH = "data.zip"
EXTRACT_DIR = "./data"

def download_file(url, output_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {output_path}")

def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted into {extract_dir}")

if __name__ == "__main__":
    download_file(DROPBOX_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, EXTRACT_DIR)
