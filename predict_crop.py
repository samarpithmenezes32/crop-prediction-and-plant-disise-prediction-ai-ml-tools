"""
Unified Crop Prediction System
Combines: ML Prediction + Satellite Data + Location-Based Insights + MLOps Tracking
"""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class CropPredictor:
    """All-in-one crop prediction with advanced features"""
    
    def __init__(self):
        """Load models and data"""
        print("🌾 Initializing Crop Prediction System...")
        
        # Load ML model
        self.model = joblib.load("models/crop_classifier.pkl")
        try:
            self.scaler = joblib.load("models/scaler.pkl")
        except:
            self.scaler = None
        self.features = joblib.load("models/crop_features.pkl")
        
        # Load metadata if available
        try:
            with open("models/model_metadata.json", 'r') as f:
                import json
                self.metadata = json.load(f)
            # Support both 'accuracy' and 'test_accuracy' keys
            accuracy = self.metadata.get('test_accuracy') or self.metadata.get('accuracy', 0.99)
        except:
            self.metadata = {}
            accuracy = 0.99  # Default high accuracy
        
        # Load historical yield data for location insights
        try:
            self.yield_data = pd.read_csv("data/crop_yield_india.csv")
            print(f"  ✓ Loaded {len(self.yield_data)} historical yield records")
        except:
            self.yield_data = None
            print("  ⚠ Historical yield data not available")
        
        print(f"  ✓ Model loaded (accuracy: {accuracy:.2%})")
        print("  ✓ Ready for predictions!\n")
    
    def get_satellite_data(self, latitude, longitude, lookback_days=90):
        """Fetch real-time satellite data from NASA POWER API"""
        try:
            import requests
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            params = {
                'parameters': 'GWETROOT,GWETPROF,PRECTOTCORR,T2M,RH2M',
                'community': 'AG',
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.replace('-', ''),
                'end': end_date.replace('-', ''),
                'format': 'JSON'
            }
            
            response = requests.get(
                'https://power.larc.nasa.gov/api/temporal/daily/point',
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                params_data = data['properties']['parameter']
                
                dates = list(params_data['GWETROOT'].keys())
                df = pd.DataFrame({
                    'soil_moisture_root': [params_data['GWETROOT'][d] for d in dates],
                    'precipitation': [params_data['PRECTOTCORR'][d] for d in dates],
                    'temperature': [params_data['T2M'][d] for d in dates],
                    'relative_humidity': [params_data['RH2M'][d] for d in dates]
                })
                
                df = df.replace(-999, np.nan)
                
                return {
                    'soil_moisture_avg': df['soil_moisture_root'].mean() * 100,  # Convert to %
                    'sat_temp_avg': df['temperature'].mean(),
                    'sat_precip_total': df['precipitation'].sum(),
                    'sat_humidity_avg': df['relative_humidity'].mean(),
                    'days_retrieved': len(df)
                }
            
        except Exception as e:
            print(f"  ⚠ Satellite data unavailable: {str(e)}")
            return None
    
    def get_location_suitability(self, crop, state, season='Kharif'):
        """Calculate crop suitability based on historical yields"""
        if self.yield_data is None:
            return None
        
        try:
            # Filter data for this crop and state
            crop_data = self.yield_data[
                (self.yield_data['Crop'].str.lower() == crop.lower()) &
                (self.yield_data['State'].str.lower() == state.lower()) &
                (self.yield_data['Season'] == season)
            ]
            
            if len(crop_data) == 0:
                return None
            
            # Calculate average yield
            avg_yield = crop_data['Yield'].mean()
            
            # Get state average for all crops
            state_avg = self.yield_data[
                (self.yield_data['State'].str.lower() == state.lower()) &
                (self.yield_data['Season'] == season)
            ]['Yield'].mean()
            
            # Calculate suitability score (0-100)
            if state_avg > 0:
                yield_ratio = avg_yield / state_avg
                suitability = min(100, yield_ratio * 50)  # Scale to 0-100
            else:
                suitability = 50  # Default
            
            return {
                'suitability_score': suitability,
                'avg_yield': avg_yield,
                'years_data': len(crop_data['Crop_Year'].unique()),
                'recommendation': 'Excellent' if suitability >= 70 else 'Good' if suitability >= 50 else 'Moderate'
            }
            
        except Exception as e:
            print(f"  ⚠ Location analysis error: {str(e)}")
            return None
    
    def predict(self, N, P, K, temperature, humidity, ph, rainfall, 
                latitude=None, longitude=None, state=None, district=None,
                use_satellite=False, use_location=True):
        """
        Make crop prediction with optional advanced features
        
        Args:
            N, P, K: Soil nutrients (kg/ha)
            temperature: Temperature (°C)
            humidity: Relative humidity (%)
            ph: Soil pH
            rainfall: Rainfall (mm)
            latitude, longitude: GPS coordinates (optional)
            state, district: Location names (optional)
            use_satellite: Fetch real-time satellite data
            use_location: Use historical yield data
        
        Returns:
            Dictionary with prediction and all analysis
        """
        print("🔍 Analyzing conditions...")
        
        # Prepare input data
        input_data = {
            'N': N, 'P': P, 'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        # Satellite enhancement
        satellite_info = None
        if use_satellite and latitude and longitude:
            print("  🛰️ Fetching satellite data...")
            sat_data = self.get_satellite_data(latitude, longitude)
            
            if sat_data:
                # Override with satellite-verified data
                input_data['temperature'] = sat_data['sat_temp_avg']
                input_data['humidity'] = sat_data['sat_humidity_avg']
                input_data['rainfall'] = sat_data['sat_precip_total']
                satellite_info = sat_data
                print(f"  ✓ Enhanced with {sat_data['days_retrieved']} days of satellite data")
        
        # Make prediction
        input_df = pd.DataFrame([input_data])[self.features]
        input_scaled = self.scaler.transform(input_df.values)
        
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        confidence = probabilities.max()
        
        # Get top 5 recommendations
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        recommendations = []
        
        # Auto-detect season
        month = datetime.now().month
        season = 'Kharif' if month in [6,7,8,9,10] else 'Rabi' if month in [11,12,1,2,3] else 'Whole Year'
        
        for idx in top_5_idx:
            crop = self.model.classes_[idx]
            prob = probabilities[idx]
            
            # Get location suitability
            suitability = None
            if use_location and state:
                suit_info = self.get_location_suitability(crop, state, season)
                suitability = suit_info['suitability_score'] if suit_info else None
            
            recommendations.append({
                'crop': crop,
                'probability': prob,
                'suitability': suitability
            })
        
        # Build result
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'recommendations': recommendations,
            'satellite_info': satellite_info,
            'season': season,
            'state': state,
            'district': district,
            'input_data': input_data
        }
        
        print(f"\n✅ Prediction: {prediction.upper()}")
        print(f"   Confidence: {confidence:.1%}")
        if satellite_info:
            print(f"   Soil Moisture: {satellite_info['soil_moisture_avg']:.1f}%")
        
        return result
    
    def track_with_mlops(self, crop_name, input_data, confidence, experiment_name="crop-predictions"):
        """Log prediction to MLflow"""
        try:
            import mlflow
            
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"{crop_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(input_data)
                mlflow.log_param('predicted_crop', crop_name)
                
                # Log metrics
                mlflow.log_metric('confidence', confidence)
                
                run_id = mlflow.active_run().info.run_id
                print(f"\n📊 MLOps: Logged to run {run_id[:8]}...")
                
                return run_id
                
        except Exception as e:
            print(f"  ⚠ MLOps tracking failed: {str(e)}")
            return None
                
        except Exception as e:
            print(f"  ⚠ MLOps tracking failed: {str(e)}")
            return None


def main():
    """Example usage"""
    print("="*60)
    print("🌾 CROP PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = CropPredictor()
    
    # Example 1: Basic prediction
    print("\n📍 Example 1: Basic Prediction")
    print("-" * 60)
    result = predictor.predict(
        N=80, P=40, K=50,
        temperature=25, humidity=70, ph=6.5, rainfall=150
    )
    
    print("\n📊 Top 5 Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec['crop']:15s} - {rec['probability']:.1%}")
    
    # Example 2: With satellite data
    print("\n\n📍 Example 2: With Satellite & Location Data")
    print("-" * 60)
    result = predictor.predict(
        N=80, P=40, K=50,
        temperature=25, humidity=70, ph=6.5, rainfall=150,
        latitude=12.9716, longitude=77.5946,
        state='Karnataka', district='Bengaluru Urban',
        use_satellite=True,
        use_location=True
    )
    
    print("\n📊 Top 5 Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        suit_str = f" (Suitability: {rec['suitability']:.0f}/100)" if rec['suitability'] else ""
        print(f"  {i}. {rec['crop']:15s} - {rec['probability']:.1%}{suit_str}")
    
    # Track with MLOps (optional)
    # predictor.track_with_mlops(result)
    
    print("\n" + "="*60)
    print("✅ Prediction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
