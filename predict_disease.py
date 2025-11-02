"""
Unified Plant Disease Detection System
Handles all 9 crops with complete disease classification
"""
import numpy as np
from PIL import Image
import os


class DiseaseDetector:
    """Complete disease detection for all supported crops"""
    
    # Supported crops and their diseases
    DISEASE_INFO = {
        'apple': {
            'classes': ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy'],
            'model_file': 'models/disease/apple_model.h5'
        },
        'cherry': {
            'classes': ['Powdery_mildew', 'healthy'],
            'model_file': 'models/disease/cherry_model.h5'
        },
        'corn': {
            'classes': ['Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust', 
                       'Northern_Leaf_Blight', 'healthy'],
            'model_file': 'models/disease/corn_model.h5'
        },
        'grape': {
            'classes': ['Black_rot', 'Esca_(Black_Measles)', 
                       'Leaf_blight_(Isariopsis_Leaf_Spot)', 'healthy'],
            'model_file': 'models/disease/grape_model.h5'
        },
        'peach': {
            'classes': ['Bacterial_spot', 'healthy'],
            'model_file': 'models/disease/peach_model.h5'
        },
        'pepper': {
            'classes': ['Bacterial_spot', 'healthy'],
            'model_file': 'models/disease/pepper_model.h5'
        },
        'potato': {
            'classes': ['Early_blight', 'Late_blight', 'healthy'],
            'model_file': 'models/disease/potato_model.h5'
        },
        'strawberry': {
            'classes': ['Leaf_scorch', 'healthy'],
            'model_file': 'models/disease/strawberry_model.h5'
        },
        'tomato': {
            'classes': ['Bacterial_spot', 'Early_blight', 'Late_blight', 
                       'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
                       'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy'],
            'model_file': 'models/disease/tomato_model.h5'
        }
    }
    
    # Treatment recommendations
    TREATMENTS = {
        'Apple_scab': {
            'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit',
            'treatment': [
                'Remove and destroy infected leaves',
                'Apply fungicide (Captan, Mancozeb) early in season',
                'Prune trees for better air circulation',
                'Plant resistant varieties'
            ],
            'prevention': [
                'Rake and destroy fallen leaves',
                'Avoid overhead watering',
                'Space trees properly for air flow'
            ]
        },
        'Black_rot': {
            'description': 'Fungal disease causing leaf spots and fruit rot',
            'treatment': [
                'Remove infected plant parts immediately',
                'Apply copper-based fungicide',
                'Improve air circulation around plants',
                'Avoid wetting foliage when watering'
            ],
            'prevention': [
                'Use drip irrigation',
                'Mulch to prevent soil splash',
                'Rotate crops annually'
            ]
        },
        'Cedar_apple_rust': {
            'description': 'Fungal disease requiring both apple and cedar trees',
            'treatment': [
                'Remove nearby cedar trees if possible',
                'Apply fungicide during wet spring weather',
                'Use sulfur-based sprays',
                'Remove galls from cedar trees'
            ],
            'prevention': [
                'Plant resistant apple varieties',
                'Avoid planting apples near cedars',
                'Monitor for early symptoms'
            ]
        },
        'Powdery_mildew': {
            'description': 'White powdery fungal growth on leaves and shoots',
            'treatment': [
                'Apply sulfur or potassium bicarbonate spray',
                'Use neem oil as organic option',
                'Prune infected areas',
                'Increase air circulation'
            ],
            'prevention': [
                'Water at base of plants',
                'Avoid over-fertilizing with nitrogen',
                'Plant in full sun'
            ]
        },
        'Common_rust': {
            'description': 'Fungal disease with orange-brown pustules on leaves',
            'treatment': [
                'Apply fungicide at first sign',
                'Remove severely infected leaves',
                'Use resistant hybrids',
                'Ensure proper plant spacing'
            ],
            'prevention': [
                'Plant early to avoid peak infection time',
                'Choose resistant varieties',
                'Maintain field sanitation'
            ]
        },
        'Northern_Leaf_Blight': {
            'description': 'Fungal disease causing large, cigar-shaped lesions',
            'treatment': [
                'Apply fungicides (Azoxystrobin, Pyraclostrobin)',
                'Remove crop debris after harvest',
                'Use resistant hybrids',
                'Practice crop rotation'
            ],
            'prevention': [
                'Rotate with non-host crops',
                'Bury crop residue',
                'Plant resistant varieties'
            ]
        },
        'Bacterial_spot': {
            'description': 'Bacterial disease causing dark spots with yellow halos',
            'treatment': [
                'Apply copper-based bactericide',
                'Remove and destroy infected plants',
                'Use drip irrigation',
                'Avoid working with wet plants'
            ],
            'prevention': [
                'Use disease-free seeds/transplants',
                'Practice 3-year crop rotation',
                'Disinfect tools regularly'
            ]
        },
        'Early_blight': {
            'description': 'Fungal disease with concentric ring patterns on leaves',
            'treatment': [
                'Apply fungicide (Chlorothalonil, Mancozeb)',
                'Remove infected lower leaves',
                'Mulch to prevent soil splash',
                'Ensure adequate plant spacing'
            ],
            'prevention': [
                'Rotate crops for 3 years',
                'Stake plants for air flow',
                'Water at soil level only'
            ]
        },
        'Late_blight': {
            'description': 'Devastating fungal disease affecting leaves and tubers',
            'treatment': [
                'Apply fungicide immediately (Copper, Mancozeb)',
                'Remove and destroy infected plants',
                'Hill up soil around plants',
                'Harvest potatoes during dry weather'
            ],
            'prevention': [
                'Use certified disease-free seed',
                'Avoid overhead irrigation',
                'Monitor weather for favorable conditions'
            ]
        },
        'Leaf_Mold': {
            'description': 'Fungal disease with olive-green mold on leaf undersides',
            'treatment': [
                'Improve greenhouse ventilation',
                'Lower humidity below 85%',
                'Apply fungicide',
                'Remove infected leaves'
            ],
            'prevention': [
                'Use resistant varieties',
                'Maintain good air circulation',
                'Avoid overhead watering'
            ]
        },
        'Septoria_leaf_spot': {
            'description': 'Fungal disease with small spots with dark borders',
            'treatment': [
                'Apply fungicide (Chlorothalonil)',
                'Remove infected leaves immediately',
                'Mulch around plants',
                'Improve air circulation'
            ],
            'prevention': [
                'Rotate crops for 3 years',
                'Use drip irrigation',
                'Stake and prune plants'
            ]
        },
        'Spider_mites Two-spotted_spider_mite': {
            'description': 'Tiny pests causing stippled, yellowing leaves',
            'treatment': [
                'Spray with insecticidal soap or neem oil',
                'Use miticide if severe',
                'Increase humidity',
                'Release predatory mites'
            ],
            'prevention': [
                'Keep plants well-watered',
                'Avoid dusty conditions',
                'Monitor regularly'
            ]
        },
        'Target_Spot': {
            'description': 'Fungal disease with concentric ring lesions',
            'treatment': [
                'Apply fungicide (Azoxystrobin)',
                'Remove infected plant debris',
                'Improve air flow',
                'Reduce leaf wetness'
            ],
            'prevention': [
                'Rotate crops',
                'Use resistant varieties',
                'Mulch to prevent splash'
            ]
        },
        'Tomato_Yellow_Leaf_Curl_Virus': {
            'description': 'Viral disease transmitted by whiteflies',
            'treatment': [
                'Remove and destroy infected plants',
                'Control whiteflies with insecticides',
                'Use reflective mulches',
                'No cure - prevention critical'
            ],
            'prevention': [
                'Use virus-resistant varieties',
                'Control whitefly populations',
                'Use insect netting'
            ]
        },
        'Tomato_mosaic_virus': {
            'description': 'Viral disease causing mottled, distorted leaves',
            'treatment': [
                'Remove infected plants immediately',
                'Disinfect tools with bleach',
                'No chemical treatment available',
                'Control aphid vectors'
            ],
            'prevention': [
                'Use certified disease-free seeds',
                'Wash hands before handling plants',
                'Avoid tobacco use near plants'
            ]
        },
        'Leaf_scorch': {
            'description': 'Fungal disease causing brown, scorched leaf edges',
            'treatment': [
                'Remove infected leaves',
                'Apply fungicide',
                'Improve soil drainage',
                'Ensure proper spacing'
            ],
            'prevention': [
                'Use drip irrigation',
                'Mulch plants',
                'Plant in well-drained soil'
            ]
        },
        'Esca_(Black_Measles)': {
            'description': 'Complex fungal disease of grape vines',
            'treatment': [
                'Prune out infected wood',
                'Apply trunk protectants',
                'No fully effective treatment',
                'Maintain vine health'
            ],
            'prevention': [
                'Use clean pruning tools',
                'Avoid large pruning wounds',
                'Maintain vine vigor'
            ]
        },
        'Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'description': 'Fungal disease causing brown leaf spots',
            'treatment': [
                'Apply copper-based fungicide',
                'Remove infected leaves',
                'Improve air circulation',
                'Reduce humidity'
            ],
            'prevention': [
                'Space vines properly',
                'Prune for air flow',
                'Avoid overhead irrigation'
            ]
        },
        'Cercospora_leaf_spot Gray_leaf_spot': {
            'description': 'Fungal disease with gray-brown rectangular lesions',
            'treatment': [
                'Apply fungicide (Strobilurin)',
                'Remove crop debris',
                'Use resistant hybrids',
                'Practice crop rotation'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Bury crop residue',
                'Rotate crops'
            ]
        },
        'healthy': {
            'description': 'No disease detected - plant appears healthy',
            'treatment': [
                'Continue regular monitoring',
                'Maintain good cultural practices',
                'No treatment needed'
            ],
            'prevention': [
                'Keep monitoring regularly',
                'Maintain proper nutrition and watering',
                'Practice crop rotation'
            ]
        }
    }
    
    def __init__(self):
        """Initialize disease detector"""
        print("🍃 Initializing Disease Detection System...")
        
        self.models = {}
        self.class_mappings = {}
        self.loaded_crops = []
        
        # Try to load TensorFlow/Keras
        try:
            import tensorflow as tf
            import json
            self.tf = tf
            self.has_tf = True
            
            # Load models for all crops
            for crop, info in self.DISEASE_INFO.items():
                try:
                    if os.path.exists(info['model_file']):
                        self.models[crop] = tf.keras.models.load_model(info['model_file'])
                        
                        # Load class mapping if available
                        mapping_file = info['model_file'].replace('.h5', '_classes.json')
                        if os.path.exists(mapping_file):
                            with open(mapping_file, 'r') as f:
                                mapping = json.load(f)
                                # Convert string keys to int
                                self.class_mappings[crop] = {int(k): v for k, v in mapping.items()}
                        else:
                            # Use default classes from DISEASE_INFO
                            self.class_mappings[crop] = {i: cls for i, cls in enumerate(info['classes'])}
                        
                        self.loaded_crops.append(crop)
                except Exception as e:
                    print(f"  ⚠ Failed to load {crop}: {e}")
                    pass
            
            if self.loaded_crops:
                print(f"  ✓ Loaded models for: {', '.join(self.loaded_crops)}")
            else:
                print("  ⚠ No models loaded - using fallback mode")
                
        except ImportError:
            self.has_tf = False
            print("  ⚠ TensorFlow not available - using fallback mode")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path, crop_type):
        """
        Predict disease from image
        
        Args:
            image_path: Path to leaf image
            crop_type: Type of crop (apple, tomato, etc.)
        
        Returns:
            Dictionary with prediction results
        """
        crop_type = crop_type.lower()
        
        if crop_type not in self.DISEASE_INFO:
            return {
                'error': f"Crop '{crop_type}' not supported",
                'supported_crops': list(self.DISEASE_INFO.keys())
            }
        
        print(f"\n🔍 Analyzing {crop_type} leaf image...")
        
        # Check if model is loaded
        if crop_type not in self.models:
            # Fallback prediction
            print("  ⚠ Using fallback prediction (model not loaded)")
            return {
                'crop': crop_type,
                'disease': 'healthy',
                'confidence': 0.75,
                'is_healthy': True,
                'fallback': True,
                'message': 'Using fallback prediction - download models for accurate results'
            }
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.models[crop_type].predict(img_array, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Get disease name from mapping
        disease = self.class_mappings[crop_type].get(predicted_idx, f"Unknown_{predicted_idx}")
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'disease': self.class_mappings[crop_type].get(idx, f"Unknown_{idx}"),
                'confidence': float(predictions[idx])
            }
            for idx in top_3_idx
        ]
        
        # Get treatment info
        treatment_info = self.TREATMENTS.get(disease, {})
        
        result = {
            'crop': crop_type,
            'disease': disease,
            'confidence': float(confidence),
            'is_healthy': disease == 'healthy',
            'top_3': top_predictions,
            'treatment': treatment_info,
            'fallback': False
        }
        
        print(f"  ✅ Disease: {disease}")
        print(f"  📊 Confidence: {confidence:.1%}")
        
        return result
    
    def get_treatment(self, disease):
        """Get treatment recommendations for a disease"""
        return self.TREATMENTS.get(disease, {
            'description': 'Unknown disease',
            'treatment': ['Consult agricultural expert'],
            'prevention': ['Practice good crop management']
        })


def main():
    """Example usage"""
    print("="*60)
    print("🍃 PLANT DISEASE DETECTION SYSTEM")
    print("="*60)
    
    detector = DiseaseDetector()
    
    print("\n📋 Supported crops:")
    for i, crop in enumerate(detector.DISEASE_INFO.keys(), 1):
        diseases = detector.DISEASE_INFO[crop]['classes']
        print(f"  {i}. {crop.capitalize():12s} - {len(diseases)} diseases")
    
    # Example usage (requires actual image file)
    # result = detector.predict('path/to/leaf_image.jpg', 'tomato')
    # 
    # if not result.get('error'):
    #     print(f"\n🔬 Analysis Result:")
    #     print(f"   Crop: {result['crop'].capitalize()}")
    #     print(f"   Disease: {result['disease']}")
    #     print(f"   Confidence: {result['confidence']:.1%}")
    #     print(f"   Status: {'Healthy ✅' if result['is_healthy'] else 'Diseased ⚠️'}")
    #     
    #     if result['treatment']:
    #         print(f"\n📖 Treatment:")
    #         for step in result['treatment'].get('treatment', []):
    #             print(f"   • {step}")
    
    print("\n" + "="*60)
    print("✅ System ready!")
    print("="*60)


if __name__ == "__main__":
    main()
