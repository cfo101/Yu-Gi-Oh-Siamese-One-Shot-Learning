import requests
import os
import re

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
SAVE_DIR = "yugioh_images"
#NUM_CARDS = 10  # Only download images for the first 10 cards

os.makedirs(SAVE_DIR, exist_ok=True)


def sanitize_filename(filename):
    # Remove or replace potentially unsafe characters for Windows
    unsafe_chars = r'[\\\/:*?"<>|\s☆★]'
    # Replace with underscore
    filename = re.sub(unsafe_chars, '_', filename)
    return filename

def download_image(url, save_path):
    if os.path.exists(save_path):
        print(f"Already exists, skipping: {save_path}")
        return
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    print("Fetching card data...")
    resp = requests.get(API_URL)
    resp.raise_for_status()
    data = resp.json()
    cards = data.get("data", [])

    print(f"Found {len(cards)} cards. ")

    for card in (cards):
        card_name = sanitize_filename(card["name"])
        card_id = card.get("id", "unknown")
        images = card.get("card_images", [])
        for j, img_info in enumerate(images):
            img_url = img_info.get("image_url")
            if img_url:
                ext = img_url.split(".")[-1]
                filename = f"{card_id}_{card_name}_img{j+1}.{ext}"
                save_path = os.path.join(SAVE_DIR, filename)
                download_image(img_url, save_path)

if __name__ == "__main__":
    main()