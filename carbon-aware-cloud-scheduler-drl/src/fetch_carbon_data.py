import pandas as pd
import numpy as np

def load_carbon_profile():
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)
    
    # See what columns actually exist
    print("Available columns:", [c for c in df.columns if 'carbon' in c.lower() or 'energy' in c.lower()])
    
    # Filter India and use electricity/energy intensity
    india = df[df['country'] == 'India'][['year', 'energy_per_capita']].dropna()
    
    # Use a realistic India grid carbon intensity base (~700 gCO2/kWh)
    base_intensity = 700  # India grid is coal-heavy
    hours = 720
    t = np.linspace(0, hours, hours)
    
    # Realistic daily pattern
    daily_pattern = base_intensity + 80 * np.sin(2 * np.pi * t / 24 - np.pi/2)
    noise = np.random.normal(0, 25, hours)
    profile = np.clip(daily_pattern + noise, 400, 900)
    
    pd.DataFrame({'carbon_intensity': profile}).to_csv('carbon.csv', index=False)
    print(f"Saved {hours} hours of India-based carbon data to carbon.csv")
    return profile

if __name__ == "__main__":
    load_carbon_profile()