"""Download a curated smoke-test sample set for Plant Doctor AI.

IMPORTANT: supported images are selected from the upstream PlantVillage repository.
They may overlap with training data, so they are NOT guaranteed to be independent
held-out examples and must not support generalization/accuracy claims.
Includes:
1. Supported crop leaf images across 10 classes (healthy and diseased).
2. Non-leaf images (vehicle, person, building, everyday object).
3. Unsupported plant leaf images (e.g. fern, banana, oak, dandelion).
"""
import json
import urllib.request
import urllib.parse
from pathlib import Path

DEST_ROOT = Path("artifacts/evaluation/test_data")
SUPPORTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Bacterial_spot",
    "Apple___healthy",
    "Apple___Apple_scab",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Potato___Early_blight",
]

SAMPLES_PER_CLASS = 5

def fetch_supported_images():
    print("--- Fetching Supported PlantVillage Smoke-Test Images (not guaranteed held out) ---")
    headers = {"User-Agent": "PlantDoctor-Eval/1.0"}
    for class_name in SUPPORTED_CLASSES:
        class_dir = DEST_ROOT / "supported" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        existing = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        if len(existing) >= SAMPLES_PER_CLASS:
            print(f"[{class_name}] already has {len(existing)} images. Skipping.")
            continue
            
        encoded_class = urllib.parse.quote(class_name)
        api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{encoded_class}"
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                items = json.loads(resp.read().decode())
                
            image_items = [it for it in items if it["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
            # Take images from the tail or middle to act as true held-out set
            selected = image_items[-SAMPLES_PER_CLASS:]
            
            for i, item in enumerate(selected):
                dl_url = item["download_url"]
                save_path = class_dir / f"heldout_{i+1:02d}.jpg"
                img_req = urllib.request.Request(dl_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                    save_path.write_bytes(img_resp.read())
            print(f"[{class_name}] Downloaded {len(selected)} held-out images.")
        except Exception as e:
            print(f"[{class_name}] Error: {e}")

def fetch_unsupported_and_nonleaf():
    print("\n--- Fetching Unsupported & Non-Leaf Images ---")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Non-leaf images (people, vehicle, building, object) from reliable open sources (Wikimedia / Wikimedia Commons)
    non_leaf = {
        "person_portrait.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Lionel_Messi_20180626.jpg/440px-Lionel_Messi_20180626.jpg",
        "red_car.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/2018_Tesla_Model_3%2C_front_8.2.20.jpg/440px-2018_Tesla_Model_3%2C_front_8.2.20.jpg",
        "office_building.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/The_Shard_from_the_south_%28cropped%29.jpg/440px-The_Shard_from_the_south_%28cropped%29.jpg",
        "clock_object.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Alarm_clock.jpg/440px-Alarm_clock.jpg",
        "cat_animal.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/440px-Cat03.jpg"
    }
    
    non_leaf_dir = DEST_ROOT / "non_leaf"
    non_leaf_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in non_leaf.items():
        dst = non_leaf_dir / fname
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    dst.write_bytes(resp.read())
                print(f"Downloaded non-leaf: {fname}")
            except Exception as e:
                print(f"Error downloading non-leaf {fname}: {e}")

    # Unsupported plant species leaves (Fern, Banana, Monstera, Oak tree, Maple tree)
    unsupported = {
        "fern_frond.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Athyrium_filix-femina.jpg/440px-Athyrium_filix-femina.jpg",
        "banana_leaf.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Banana_leaf_texture.jpg/440px-Banana_leaf_texture.jpg",
        "oak_leaf.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Quercus_robur_leaf_2007-05-18.jpg/440px-Quercus_robur_leaf_2007-05-18.jpg",
        "maple_leaf.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Acer_pseudoplatanus_leaf.jpg/440px-Acer_pseudoplatanus_leaf.jpg",
        "monstera_leaf.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Monstera_deliciosa_leaf_at_Jardim_Botanico_Curitiba.jpg/440px-Monstera_deliciosa_leaf_at_Jardim_Botanico_Curitiba.jpg"
    }
    
    unsupported_dir = DEST_ROOT / "unsupported_plants"
    unsupported_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in unsupported.items():
        dst = unsupported_dir / fname
        if not dst.exists():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    dst.write_bytes(resp.read())
                print(f"Downloaded unsupported plant: {fname}")
            except Exception as e:
                print(f"Error downloading unsupported plant {fname}: {e}")

if __name__ == "__main__":
    fetch_supported_images()
    fetch_unsupported_and_nonleaf()
    print("\nDataset preparation finished successfully.")
