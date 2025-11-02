"""
Geospatial Risk Mapping Module
Creates interactive maps with district-level risk analysis
"""
import os
import pandas as pd
import numpy as np
import folium
from folium import plugins
import json


class GeospatialRiskMapper:
    """
    Creates interactive risk maps for agricultural decision-making
    """
    
    def __init__(self):
        self.maps_dir = "data/maps"
        os.makedirs(self.maps_dir, exist_ok=True)
        
        # Karnataka districts coordinates (primary focus)
        self.district_coords = {
            # Karnataka - Primary Focus
            'Bengaluru Urban': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
            'Bengaluru Rural': {'lat': 13.0797, 'lon': 77.4526, 'state': 'Karnataka'},
            'Mysuru': {'lat': 12.2958, 'lon': 76.6394, 'state': 'Karnataka'},
            'Mandya': {'lat': 12.5244, 'lon': 76.8956, 'state': 'Karnataka'},
            'Hassan': {'lat': 13.0072, 'lon': 76.0962, 'state': 'Karnataka'},
            'Tumakuru': {'lat': 13.3392, 'lon': 77.1014, 'state': 'Karnataka'},
            'Kolar': {'lat': 13.1368, 'lon': 78.1298, 'state': 'Karnataka'},
            'Chikkaballapura': {'lat': 13.4355, 'lon': 77.7315, 'state': 'Karnataka'},
            'Ramanagara': {'lat': 12.7207, 'lon': 77.2809, 'state': 'Karnataka'},
            'Chitradurga': {'lat': 14.2226, 'lon': 76.3980, 'state': 'Karnataka'},
            'Davanagere': {'lat': 14.4644, 'lon': 75.9218, 'state': 'Karnataka'},
            'Shivamogga': {'lat': 13.9299, 'lon': 75.5681, 'state': 'Karnataka'},
            'Dharwad': {'lat': 15.4589, 'lon': 75.0078, 'state': 'Karnataka'},
            'Belagavi': {'lat': 15.8497, 'lon': 74.4977, 'state': 'Karnataka'},
            'Uttara Kannada': {'lat': 14.5204, 'lon': 74.6804, 'state': 'Karnataka'},
            'Haveri': {'lat': 14.7951, 'lon': 75.4047, 'state': 'Karnataka'},
            'Gadag': {'lat': 15.4315, 'lon': 75.6294, 'state': 'Karnataka'},
            'Ballari': {'lat': 15.1394, 'lon': 76.9214, 'state': 'Karnataka'},
            'Vijayapura': {'lat': 16.8302, 'lon': 75.7100, 'state': 'Karnataka'},
            'Bagalkot': {'lat': 16.1695, 'lon': 75.6956, 'state': 'Karnataka'},
            'Kalaburagi': {'lat': 17.3297, 'lon': 76.8343, 'state': 'Karnataka'},
            'Raichur': {'lat': 16.2120, 'lon': 77.3439, 'state': 'Karnataka'},
            'Koppal': {'lat': 15.3505, 'lon': 76.1539, 'state': 'Karnataka'},
            'Yadgir': {'lat': 16.7700, 'lon': 77.1380, 'state': 'Karnataka'},
            'Bidar': {'lat': 17.9103, 'lon': 77.5199, 'state': 'Karnataka'},
            'Dakshina Kannada': {'lat': 12.8438, 'lon': 75.2479, 'state': 'Karnataka'},
            'Udupi': {'lat': 13.3409, 'lon': 74.7421, 'state': 'Karnataka'},
            'Chikkamagaluru': {'lat': 13.3161, 'lon': 75.7720, 'state': 'Karnataka'},
            'Kodagu': {'lat': 12.4244, 'lon': 75.7382, 'state': 'Karnataka'},
            'Chamarajanagara': {'lat': 11.9263, 'lon': 76.9437, 'state': 'Karnataka'},
            
            # Other Indian states (for reference)
            'Pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra'},
            'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
            'Nashik': {'lat': 19.9975, 'lon': 73.7898, 'state': 'Maharashtra'},
            'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'},
            'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
        }
    
    def create_base_map(self, center_lat=15.3173, center_lon=75.7139, zoom=7):
        """
        Create base map focused on Karnataka, India
        
        Args:
            center_lat: Center latitude (default: Karnataka center)
            center_lon: Center longitude (default: Karnataka center)
            zoom: Initial zoom level (7 for Karnataka state view)
        
        Returns:
            Folium map object
        """
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap',
            control_scale=True,
            prefer_canvas=True
        )
        
        # Add Karnataka state boundary visualization
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 350px; height: 50px; 
                    background-color: white; border:2px solid #2E7D32; z-index:9999; 
                    font-size:16px; padding: 10px; border-radius: 5px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0; color: #2E7D32;">🌾 Karnataka Agricultural Risk Map</h4>
        <p style="margin: 5px 0; font-size: 12px; color: #666;">Focus: Karnataka State, India</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def calculate_risk_score(self, soil_moisture, rainfall, temperature, disease_probability=0):
        """
        Calculate agricultural risk score
        
        Args:
            soil_moisture: Soil moisture level (0-1)
            rainfall: Rainfall in mm
            temperature: Temperature in Celsius
            disease_probability: Disease risk (0-1)
        
        Returns:
            Risk score (0-100, higher is worse)
        """
        risk = 0
        
        # Soil moisture risk
        if soil_moisture < 0.2:
            risk += 30  # Very dry
        elif soil_moisture > 0.8:
            risk += 20  # Too wet
        
        # Rainfall risk
        if rainfall < 50:
            risk += 25  # Drought
        elif rainfall > 500:
            risk += 20  # Flood risk
        
        # Temperature risk
        if temperature > 40:
            risk += 25  # Heat stress
        elif temperature < 10:
            risk += 20  # Cold stress
        
        # Disease risk
        risk += disease_probability * 30
        
        return min(risk, 100)
    
    def get_risk_color(self, risk_score):
        """
        Get color based on risk score
        
        Args:
            risk_score: Risk score (0-100)
        
        Returns:
            Color code
        """
        if risk_score < 30:
            return 'green'  # Low risk
        elif risk_score < 60:
            return 'orange'  # Medium risk
        else:
            return 'red'  # High risk
    
    def create_district_risk_map(self, district_data, save_path='district_risk_map.html'):
        """
        Create interactive district-level risk map
        
        Args:
            district_data: DataFrame with columns: district, risk_score, soil_moisture, rainfall, temperature
            save_path: Path to save HTML map
        
        Returns:
            Folium map object
        """
        print("\n🗺️  Creating Geospatial Risk Map...")
        print("=" * 60)
        
        # Create base map
        m = self.create_base_map()
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(name='District Markers').add_to(m)
        
        # Create feature groups for different risk levels
        low_risk = folium.FeatureGroup(name='Low Risk (0-30)', show=True)
        med_risk = folium.FeatureGroup(name='Medium Risk (30-60)', show=True)
        high_risk = folium.FeatureGroup(name='High Risk (60-100)', show=True)
        
        for _, row in district_data.iterrows():
            district = row['district']
            
            if district in self.district_coords:
                coords = self.district_coords[district]
                risk_score = row.get('risk_score', 0)
                color = self.get_risk_color(risk_score)
                
                # Create popup content
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="margin: 0; color: {color};">{district}</h4>
                    <p style="margin: 5px 0;"><b>State:</b> {coords['state']}</p>
                    <hr style="margin: 5px 0;">
                    <p style="margin: 3px 0;"><b>Risk Score:</b> {risk_score:.1f}/100</p>
                    <p style="margin: 3px 0;"><b>Soil Moisture:</b> {row.get('soil_moisture', 0):.2f}</p>
                    <p style="margin: 3px 0;"><b>Rainfall:</b> {row.get('rainfall', 0):.1f} mm</p>
                    <p style="margin: 3px 0;"><b>Temperature:</b> {row.get('temperature', 0):.1f}°C</p>
                </div>
                """
                
                # Create marker
                marker = folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=10 + (risk_score / 10),  # Size based on risk
                    popup=folium.Popup(popup_html, max_width=250),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                )
                
                # Add to appropriate risk group
                if risk_score < 30:
                    marker.add_to(low_risk)
                elif risk_score < 60:
                    marker.add_to(med_risk)
                else:
                    marker.add_to(high_risk)
        
        # Add feature groups to map
        low_risk.add_to(m)
        med_risk.add_to(m)
        high_risk.add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-weight: bold;">Risk Levels</p>
        <p style="margin: 5px 0;"><span style="color: green;">●</span> Low Risk (0-30)</p>
        <p style="margin: 5px 0;"><span style="color: orange;">●</span> Medium Risk (30-60)</p>
        <p style="margin: 5px 0;"><span style="color: red;">●</span> High Risk (60-100)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save map
        save_full_path = os.path.join(self.maps_dir, save_path)
        m.save(save_full_path)
        print(f"  ✓ Risk map saved to: {save_full_path}")
        
        return m
    
    def generate_karnataka_sample_map(self, save_path='karnataka_risk_map.html'):
        """
        Generate a comprehensive Karnataka risk map with sample data from all districts
        
        Args:
            save_path: Path to save HTML map
        
        Returns:
            Folium map object
        """
        print("\n🗺️  Creating Karnataka Comprehensive Risk Map...")
        print("=" * 60)
        
        # Create sample risk data for Karnataka districts
        karnataka_districts = []
        for district, coords in self.district_coords.items():
            if coords['state'] == 'Karnataka':
                # Generate sample risk data (in production, this would come from real data)
                import random
                random.seed(hash(district) % 100)  # Consistent random for each district
                
                karnataka_districts.append({
                    'district': district,
                    'state': coords['state'],
                    'latitude': coords['lat'],
                    'longitude': coords['lon'],
                    'soil_moisture': random.uniform(0.3, 0.7),
                    'rainfall': random.uniform(50, 300),
                    'temperature': random.uniform(20, 35),
                    'disease_probability': random.uniform(0.05, 0.25)
                })
        
        district_df = pd.DataFrame(karnataka_districts)
        
        # Calculate risk scores for all districts
        district_df['risk_score'] = district_df.apply(
            lambda row: self.calculate_risk_score(
                row['soil_moisture'],
                row['rainfall'],
                row['temperature'],
                row['disease_probability']
            ),
            axis=1
        )
        
        # Create the map
        risk_map = self.create_district_risk_map(district_df, save_path)
        
        print(f"  ✓ Created map for {len(karnataka_districts)} Karnataka districts")
        
        return risk_map
    
    def create_heatmap(self, district_data, metric='risk_score', save_path='risk_heatmap.html'):
        """
        Create heatmap visualization
        
        Args:
            district_data: DataFrame with district risk data
            metric: Metric to visualize
            save_path: Path to save HTML
        
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        
        # Prepare data for heatmap
        heat_data = []
        for _, row in district_data.iterrows():
            district = row['district']
            if district in self.district_coords:
                coords = self.district_coords[district]
                value = row.get(metric, 0)
                heat_data.append([coords['lat'], coords['lon'], value])
        
        if heat_data:
            # Add heatmap layer
            plugins.HeatMap(
                heat_data,
                name='Risk Heatmap',
                min_opacity=0.3,
                radius=50,
                blur=40,
                gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        save_full_path = os.path.join(self.maps_dir, save_path)
        m.save(save_full_path)
        print(f"  ✓ Heatmap saved to: {save_full_path}")
        
        return m


# Example usage
if __name__ == "__main__":
    print("\n🗺️  Geospatial Risk Mapping Test")
    print("=" * 60)
    
    mapper = GeospatialRiskMapper()
    
    # Create sample district data
    sample_data = pd.DataFrame({
        'district': ['Bengaluru Urban', 'Mysuru', 'Dharwad', 'Pune', 'Mumbai', 'Nashik', 'Jaipur', 'Udaipur'],
        'soil_moisture': [0.45, 0.35, 0.25, 0.55, 0.65, 0.40, 0.15, 0.30],
        'rainfall': [85, 95, 45, 120, 200, 75, 30, 60],
        'temperature': [28, 26, 32, 30, 31, 29, 38, 35],
        'disease_probability': [0.1, 0.15, 0.25, 0.05, 0.08, 0.12, 0.30, 0.20]
    })
    
    # Calculate risk scores
    sample_data['risk_score'] = sample_data.apply(
        lambda row: mapper.calculate_risk_score(
            row['soil_moisture'],
            row['rainfall'],
            row['temperature'],
            row['disease_probability']
        ),
        axis=1
    )
    
    print("\n📊 District Risk Scores:")
    print(sample_data[['district', 'risk_score']].to_string(index=False))
    
    # Create risk map
    risk_map = mapper.create_district_risk_map(sample_data)
    
    # Create heatmap
    heatmap = mapper.create_heatmap(sample_data)
    
    print("\n✅ Geospatial mapping complete!")
    print(f"   View maps in: {mapper.maps_dir}/")
    print()
