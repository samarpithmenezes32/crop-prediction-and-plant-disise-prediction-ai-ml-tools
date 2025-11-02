"""
Generate Karnataka Risk Map
Creates a comprehensive risk map for all Karnataka districts
"""
import sys
sys.path.append('src')

from geospatial_mapping import GeospatialRiskMapper

print("🌾 Generating Karnataka Agricultural Risk Map")
print("=" * 60)

# Create mapper
mapper = GeospatialRiskMapper()

# Generate Karnataka map
mapper.generate_karnataka_sample_map(save_path='risk_map.html')

print("\n" + "=" * 60)
print("✅ Map Generated Successfully!")
print("📍 Location: data/maps/risk_map.html")
print("🗺️  Focus: Karnataka State, India")
print(f"📊 Districts Mapped: 30 Karnataka districts")
print("\n💡 Open the HTML file in your browser to view the interactive map")
print("=" * 60)
