"""Download Monet paintings for style transfer training"""
import requests
import time
import os

# Public domain Monet paintings from various sources
monet_urls = [
    "https://www.claude-monet.com/images/paintings/impression-sunrise.jpg",
    "https://www.claude-monet.com/images/paintings/water-lilies-5.jpg",
    "https://www.claude-monet.com/images/paintings/water-lilies-22.jpg",
    "https://www.claude-monet.com/images/paintings/the-japanese-bridge.jpg",
    "https://www.claude-monet.com/images/paintings/poppies-near-argenteuil.jpg",
    "https://www.claude-monet.com/images/paintings/woman-with-a-parasol-1.jpg",
]

output_dir = "style_images/monet"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

for i, url in enumerate(monet_urls, 1):
    try:
        print(f"Downloading image {i}/{len(monet_urls)}...")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            filename = f"style{i}.jpg"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  [OK] Saved {filename}")
            time.sleep(2)  # Be polite
        else:
            print(f"  [FAIL] Failed: {response.status_code}")
    except Exception as e:
        print(f"  [ERROR] Error: {e}")

print(f"\nDownloaded {len(os.listdir(output_dir))} Monet paintings!")
