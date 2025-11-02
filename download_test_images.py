"""
Download Sample Plant Disease Images for Testing
Downloads sample images from various sources for disease detection testing
"""
import os
import urllib.request
from pathlib import Path
import ssl

# Disable SSL verification for downloads (for testing only)
ssl._create_default_https_context = ssl._create_unverified_context

print("🌿 Downloading Sample Plant Disease Images")
print("=" * 60)

# Create test images directory
test_dir = Path("test_images")
test_dir.mkdir(exist_ok=True)

# Sample plant disease images from PlantVillage dataset (public domain)
# These are real disease images you can use for testing

sample_images = {
    "apple_scab_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab%203417.JPG",
    "apple_healthy_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___healthy/0a64cb9e-f018-4c26-abfe-4bad37b83b14___RS_HL%207848.JPG",
    "tomato_early_blight_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Early_blight/0a1e5e8a-8c72-4e99-8452-76a07fdf587d___RS_Early.B%206952.JPG",
    "tomato_healthy_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___healthy/0a3c72e6-9fd3-45d2-a39c-4e6f62bc88ff___GH_HL%20Leaf%20407.JPG",
    "potato_late_blight_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Potato___Late_blight/0a5b6c42-0ae4-4f4c-8652-f6856d4e11f5___RS_LB%204028.JPG",
    "potato_healthy_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Potato___healthy/0a7f8a85-a144-4b96-a926-3c8d9fbb7c70___RS_HL%201899.JPG",
    "grape_black_rot_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Grape___Black_rot/0a0b0cc0-5e8a-4074-bd32-8f5f8394f7c6___FAM_B.Rot%203653.JPG",
    "grape_healthy_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Grape___healthy/0a0e89c3-f5c4-4311-a75b-0a7efa5bb913___FAM_H.S%207821.JPG",
    "corn_rust_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Corn_(maize)___Common_rust_/0a0e77b2-22c4-4299-a19e-ec6ca1e53c0e___RS_Rust%201808.JPG",
    "pepper_bacterial_spot_1.jpg": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Pepper,_bell___Bacterial_spot/0a3b1b1c-d9df-4e90-aaca-c46da218ae43___JR_B.Spot%203905.JPG",
}

downloaded = 0
failed = 0

for filename, url in sample_images.items():
    try:
        output_path = test_dir / filename
        
        if output_path.exists():
            print(f"  ⏭️  Skipping {filename} (already exists)")
            downloaded += 1
            continue
        
        print(f"  📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✅ Saved to: {output_path}")
        downloaded += 1
        
    except Exception as e:
        print(f"  ❌ Failed to download {filename}: {str(e)[:50]}...")
        failed += 1

print("\n" + "=" * 60)
print(f"✅ Download Complete!")
print(f"   Downloaded: {downloaded}/{len(sample_images)} images")
print(f"   Failed: {failed}")
print(f"   Location: {test_dir.absolute()}")
print("\n💡 You can now upload these images in the Disease Detection tab!")
print("=" * 60)

# Also create a README
readme_content = """# Test Images for Plant Disease Detection

This folder contains sample plant disease images for testing the disease detection system.

## Available Images:

### Apple:
- `apple_scab_1.jpg` - Apple with scab disease
- `apple_healthy_1.jpg` - Healthy apple leaf

### Tomato:
- `tomato_early_blight_1.jpg` - Tomato with early blight
- `tomato_healthy_1.jpg` - Healthy tomato leaf

### Potato:
- `potato_late_blight_1.jpg` - Potato with late blight
- `potato_healthy_1.jpg` - Healthy potato leaf

### Grape:
- `grape_black_rot_1.jpg` - Grape with black rot
- `grape_healthy_1.jpg` - Healthy grape leaf

### Corn:
- `corn_rust_1.jpg` - Corn with common rust

### Pepper:
- `pepper_bacterial_spot_1.jpg` - Pepper with bacterial spot

## How to Use:

1. Go to the Disease Detection tab in the application
2. Select the crop type (e.g., "apple", "tomato", "potato")
3. Click "Browse files" and select one of these images
4. Click "Analyze Disease" to get the prediction

## Image Sources:

All images are from the PlantVillage Dataset, which is publicly available for research and educational purposes.

Repository: https://github.com/spMohanty/PlantVillage-Dataset
"""

with open(test_dir / "README.md", "w") as f:
    f.write(readme_content)

print(f"\n📄 Created README.md in {test_dir}")
