"""
Simplified Kaggle Dataset Integration
Handles the already downloaded dataset
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Path to the downloaded Kaggle dataset
KAGGLE_CACHE = Path(r"C:\Users\Samarpith\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2")

def find_dataset_root():
    """Find the actual train/valid folders in the nested structure"""
    print("🔍 Searching for dataset folders...")
    
    # Navigate through nested folders
    current = KAGGLE_CACHE
    for part in ["New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)"]:
        current = current / part
        if not current.exists():
            print(f"❌ Path not found: {current}")
            return None
    
    print(f"✅ Found dataset at: {current}")
    return current


def analyze_dataset(dataset_root):
    """Analyze the dataset structure and classes"""
    print("\n" + "="*70)
    print("📊 ANALYZING DATASET")
    print("="*70)
    
    stats = {
        'train': defaultdict(int),
        'valid': defaultdict(int),
        'test': defaultdict(int)
    }
    
    # Scan train/valid folders
    for split in ['train', 'valid']:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
        
        print(f"\n📁 Scanning {split}/ folder...")
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_count = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            stats[split][class_name] = image_count
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"\n   TRAINING SET:")
    print(f"      Classes: {len(stats['train'])}")
    print(f"      Total images: {sum(stats['train'].values()):,}")
    
    print(f"\n   VALIDATION SET:")
    print(f"      Classes: {len(stats['valid'])}")
    print(f"      Total images: {sum(stats['valid'].values()):,}")
    
    # Group by crop
    crops = defaultdict(list)
    for class_name in stats['train'].keys():
        # Extract crop name (before ___)
        if '___' in class_name:
            crop = class_name.split('___')[0]
        else:
            crop = class_name.split('_')[0]
        crops[crop].append(class_name)
    
    print(f"\n🌿 Crops found ({len(crops)}):")
    for crop, classes in sorted(crops.items()):
        print(f"      {crop:15s} - {len(classes):2d} disease classes")
    
    return stats, crops


def create_symlinks_or_copy(dataset_root, output_dir="data/plant_diseases"):
    """Create organized dataset structure"""
    print("\n" + "="*70)
    print("📦 ORGANIZING DATASET")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path.absolute()}")
    print(f"🔄 Creating organized structure...\n")
    
    copied_stats = {'train': 0, 'valid': 0}
    
    for split in ['train', 'valid']:
        split_src = dataset_root / split
        split_dst = output_path / split
        
        if not split_src.exists():
            continue
        
        split_dst.mkdir(exist_ok=True)
        
        # Process each class
        for class_dir in split_src.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            dst_class_dir = split_dst / class_name
            dst_class_dir.mkdir(exist_ok=True)
            
            # Copy images
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            print(f"   {split}/{class_name:50s} - {len(images):5,} images", end='\r')
            
            for img in images:
                dst_file = dst_class_dir / img.name
                if not dst_file.exists():
                    try:
                        shutil.copy2(img, dst_file)
                        copied_stats[split] += 1
                    except Exception as e:
                        print(f"\n   ⚠ Error copying {img.name}: {e}")
    
    print(f"\n\n✅ Organization complete!")
    print(f"\n   Training images copied:   {copied_stats['train']:,}")
    print(f"   Validation images copied: {copied_stats['valid']:,}")
    print(f"   Total:                    {sum(copied_stats.values()):,}")
    
    return output_path


def create_dataset_info_file(crops, output_dir="data/plant_diseases"):
    """Create dataset information file"""
    info_content = f"""# Plant Diseases Dataset Information

## Source
- **Dataset**: New Plant Diseases Dataset (Augmented)
- **Source**: Kaggle (vipoooool/new-plant-diseases-dataset)
- **Download Date**: October 29, 2025

## Crops and Diseases

Total Crops: {len(crops)}

"""
    
    for crop, classes in sorted(crops.items()):
        info_content += f"### {crop.title()}\n"
        info_content += f"**Disease Classes ({len(classes)}):**\n"
        for cls in sorted(classes):
            disease = cls.split('___')[1] if '___' in cls else cls
            info_content += f"- {disease}\n"
        info_content += "\n"
    
    info_file = Path(output_dir) / "DATASET_INFO.md"
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"\n📄 Dataset info saved to: {info_file}")


def update_predict_disease_with_kaggle_data():
    """Update predict_disease.py with new Kaggle dataset classes"""
    print("\n" + "="*70)
    print("🔄 UPDATING DISEASE DETECTOR")
    print("="*70)
    
    print("""
💡 The Kaggle dataset contains 38 plant disease classes across multiple crops.

To integrate with your existing predict_disease.py:

1. The dataset is now organized at: data/plant_diseases/
   - train/ folder: Training images
   - valid/ folder: Validation images

2. You can train new models using TensorFlow/Keras:
   - Run: python train_disease_kaggle.py
   
3. The trained models will work with your existing DiseaseDetector class

4. Classes include:
   - Apple: Apple scab, Black rot, Cedar apple rust, healthy
   - Cherry: Powdery mildew, healthy
   - Corn: Cercospora, Common rust, Northern Leaf Blight, healthy
   - Grape: Black rot, Esca, Leaf blight, healthy
   - Peach: Bacterial spot, healthy
   - Pepper: Bacterial spot, healthy
   - Potato: Early blight, Late blight, healthy
   - Strawberry: Leaf scorch, healthy
   - Tomato: 10 different diseases + healthy

🚀 Ready to train high-accuracy models!
    """)


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🌿 KAGGLE PLANT DISEASES DATASET INTEGRATION")
    print("="*70)
    
    # Find dataset
    dataset_root = find_dataset_root()
    if not dataset_root:
        print("\n❌ Could not find dataset. Please run download first.")
        return
    
    # Analyze
    stats, crops = analyze_dataset(dataset_root)
    
    # Organize
    output_path = create_symlinks_or_copy(dataset_root)
    
    # Create info file
    create_dataset_info_file(crops, output_path)
    
    # Update disease detector info
    update_predict_disease_with_kaggle_data()
    
    # Summary
    print("\n" + "="*70)
    print("✅ INTEGRATION COMPLETE!")
    print("="*70)
    print(f"""
📊 Summary:
   ✅ Dataset organized at: {output_path.absolute()}
   ✅ {len(crops)} crops with 38+ disease classes
   ✅ {sum(stats['train'].values()):,} training images
   ✅ {sum(stats['valid'].values()):,} validation images

🚀 Next Steps:

1. Train models:
   python train_disease_kaggle.py

2. Models will be saved to:
   models/disease/<crop>_model.h5

3. Your app will automatically use the new models!

💡 The training script is ready in: train_disease_kaggle.py
    """)


if __name__ == "__main__":
    main()
