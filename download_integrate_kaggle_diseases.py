"""
Download and integrate Kaggle Plant Diseases Dataset
Combines with existing disease detection system
"""
import os
import shutil
from pathlib import Path

def download_kaggle_dataset():
    """Download the New Plant Diseases Dataset from Kaggle"""
    print("="*70)
    print("🌿 DOWNLOADING KAGGLE PLANT DISEASES DATASET")
    print("="*70)
    
    try:
        import kagglehub
        
        print("\n📥 Downloading dataset from Kaggle...")
        print("   Dataset: vipoooool/new-plant-diseases-dataset")
        print("   This may take several minutes...\n")
        
        # Download latest version
        path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        
        print(f"\n✅ Download complete!")
        print(f"📁 Path to dataset files: {path}")
        
        return path
        
    except ImportError:
        print("\n❌ Error: kagglehub not installed")
        print("\n💡 Install it with:")
        print("   pip install kagglehub")
        return None
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Kaggle account")
        print("   2. Kaggle API token (~/.kaggle/kaggle.json)")
        return None


def explore_dataset_structure(dataset_path):
    """Explore the downloaded dataset structure"""
    print("\n" + "="*70)
    print("🔍 EXPLORING DATASET STRUCTURE")
    print("="*70)
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset path not found")
        return None
    
    # Find all subdirectories
    dataset_info = {
        'root': dataset_path,
        'train': None,
        'valid': None,
        'test': None,
        'classes': set(),
        'total_images': 0
    }
    
    for root, dirs, files in os.walk(dataset_path):
        # Look for train/valid/test folders
        if 'train' in root.lower():
            dataset_info['train'] = root
        elif 'valid' in root.lower() or 'val' in root.lower():
            dataset_info['valid'] = root
        elif 'test' in root.lower():
            dataset_info['test'] = root
        
        # Count images and classes
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            class_name = os.path.basename(root)
            if class_name not in ['train', 'valid', 'test', 'val']:
                dataset_info['classes'].add(class_name)
            dataset_info['total_images'] += len(image_files)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {dataset_info['total_images']:,}")
    print(f"   Number of classes: {len(dataset_info['classes'])}")
    print(f"\n📁 Folders found:")
    if dataset_info['train']:
        print(f"   ✓ Training set: {dataset_info['train']}")
    if dataset_info['valid']:
        print(f"   ✓ Validation set: {dataset_info['valid']}")
    if dataset_info['test']:
        print(f"   ✓ Test set: {dataset_info['test']}")
    
    print(f"\n🌿 Plant Disease Classes ({len(dataset_info['classes'])}):")
    for i, class_name in enumerate(sorted(dataset_info['classes']), 1):
        print(f"   {i:2d}. {class_name}")
    
    return dataset_info


def organize_dataset_for_training(dataset_info, output_dir="data/plant_diseases"):
    """Organize dataset into a clean structure for training"""
    print("\n" + "="*70)
    print("📦 ORGANIZING DATASET FOR TRAINING")
    print("="*70)
    
    if not dataset_info:
        print("❌ No dataset info available")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test folders
    for split in ['train', 'valid', 'test']:
        (output_path / split).mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path.absolute()}")
    print(f"\n🔄 Organizing files...")
    
    organized_info = {
        'train': 0,
        'valid': 0,
        'test': 0,
        'classes': set()
    }
    
    # Copy/organize files
    for root, dirs, files in os.walk(dataset_info['root']):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            continue
        
        # Determine split (train/valid/test)
        if 'train' in root.lower():
            split = 'train'
        elif 'valid' in root.lower() or 'val' in root.lower():
            split = 'valid'
        elif 'test' in root.lower():
            split = 'test'
        else:
            continue
        
        # Get class name
        class_name = os.path.basename(root)
        if class_name in ['train', 'valid', 'test', 'val']:
            continue
        
        # Create class folder in output
        class_folder = output_path / split / class_name
        class_folder.mkdir(exist_ok=True)
        
        # Copy images (or create symlinks to save space)
        for img_file in image_files:
            src = os.path.join(root, img_file)
            dst = class_folder / img_file
            
            # Use symlink to save disk space (or copy if needed)
            try:
                if not dst.exists():
                    # On Windows, symlink might need admin rights, so copy instead
                    shutil.copy2(src, dst)
                    organized_info[split] += 1
                    organized_info['classes'].add(class_name)
            except Exception as e:
                print(f"   ⚠ Error copying {img_file}: {e}")
    
    print(f"\n✅ Organization complete!")
    print(f"\n📊 Organized Dataset:")
    print(f"   Training images:   {organized_info['train']:,}")
    print(f"   Validation images: {organized_info['valid']:,}")
    print(f"   Test images:       {organized_info['test']:,}")
    print(f"   Total classes:     {len(organized_info['classes'])}")
    
    return output_path


def map_to_existing_crops():
    """Map Kaggle dataset classes to existing 9 crop types"""
    print("\n" + "="*70)
    print("🔗 MAPPING TO EXISTING CROP TYPES")
    print("="*70)
    
    # Our existing 9 crops
    existing_crops = {
        'apple': ['Apple'],
        'cherry': ['Cherry'],
        'corn': ['Corn', 'Maize'],
        'grape': ['Grape'],
        'peach': ['Peach'],
        'pepper': ['Pepper', 'Bell'],
        'potato': ['Potato'],
        'strawberry': ['Strawberry'],
        'tomato': ['Tomato']
    }
    
    print("\n🌾 Mapping classes to existing crops:")
    for crop, keywords in existing_crops.items():
        print(f"\n   {crop.upper()}:")
        print(f"      Keywords: {', '.join(keywords)}")
    
    return existing_crops


def create_training_script():
    """Create a training script for the combined dataset"""
    script_content = '''"""
Train Disease Detection Models on Combined Dataset
Uses Kaggle dataset + existing disease detection system
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np

def create_disease_model(num_classes, input_shape=(224, 224, 3)):
    """Create CNN model for disease classification"""
    model = keras.Sequential([
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Feature extraction
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        # Classification
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_crop_specific_models(data_dir="data/plant_diseases"):
    """Train individual models for each crop type"""
    print("🚀 Training crop-specific disease detection models...")
    
    data_path = Path(data_dir)
    
    # Get all classes
    train_path = data_path / "train"
    all_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    
    print(f"\\n📊 Found {len(all_classes)} disease classes")
    
    # Group by crop
    crops = {}
    for cls in all_classes:
        # Extract crop name (first word before underscore or space)
        crop = cls.split('_')[0].split(' ')[0].lower()
        if crop not in crops:
            crops[crop] = []
        crops[crop].append(cls)
    
    print(f"\\n🌾 Grouped into {len(crops)} crops:")
    for crop, classes in crops.items():
        print(f"   {crop}: {len(classes)} diseases")
    
    # Train model for each crop
    for crop, classes in crops.items():
        print(f"\\n{'='*70}")
        print(f"🌱 Training {crop.upper()} disease model")
        print(f"{'='*70}")
        
        # Create data generators
        train_ds = keras.utils.image_dataset_from_directory(
            train_path,
            labels='inferred',
            label_mode='categorical',
            class_names=classes,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset='training',
            seed=42
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            train_path,
            labels='inferred',
            label_mode='categorical',
            class_names=classes,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset='validation',
            seed=42
        )
        
        # Create and compile model
        model = create_disease_model(len(classes))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        # Save model
        model_path = f"models/disease/{crop}_model.h5"
        model.save(model_path)
        print(f"\\n✅ Model saved to: {model_path}")
        print(f"   Final accuracy: {history.history['accuracy'][-1]:.2%}")
        print(f"   Final val accuracy: {history.history['val_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    train_crop_specific_models()
'''
    
    with open("train_disease_kaggle.py", 'w') as f:
        f.write(script_content)
    
    print("\n✅ Created training script: train_disease_kaggle.py")


def main():
    """Main workflow"""
    print("\n" + "="*70)
    print("🌿 KAGGLE PLANT DISEASES DATASET INTEGRATION")
    print("="*70)
    
    # Step 1: Download dataset
    dataset_path = download_kaggle_dataset()
    
    if not dataset_path:
        print("\n❌ Failed to download dataset. Please check:")
        print("   1. pip install kagglehub")
        print("   2. Kaggle API credentials setup")
        print("   3. Internet connection")
        return
    
    # Step 2: Explore dataset
    dataset_info = explore_dataset_structure(dataset_path)
    
    # Step 3: Organize dataset
    organized_path = organize_dataset_for_training(dataset_info)
    
    # Step 4: Show mapping
    map_to_existing_crops()
    
    # Step 5: Create training script
    create_training_script()
    
    # Summary
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print(f"""
📝 Next Steps:

1. ✅ Dataset downloaded to: {dataset_path}
2. ✅ Organized dataset at: {organized_path}
3. ✅ Training script created: train_disease_kaggle.py

🚀 To train the models:
   python train_disease_kaggle.py

📊 This will create enhanced disease detection models for:
   • Apple
   • Cherry  
   • Corn
   • Grape
   • Peach
   • Pepper
   • Potato
   • Strawberry
   • Tomato

💡 The models will be saved to: models/disease/

⚡ Note: Training will take several hours depending on your GPU.
    """)


if __name__ == "__main__":
    main()
