"""Real geographic coordinates for the 18 Indian cities PRISM covers, plus a
deterministic jitter helper so localities within a city spread out visibly on
a map instead of stacking on one point (locality-level coordinates aren't
part of the dataset, so this is an approximate visual spread around the real
city center, not actual locality geolocation)."""

import numpy as np

CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Delhi NCR": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Indore": (22.7196, 75.8577),
    "Kochi": (9.9312, 76.2673),
    "Surat": (21.1702, 72.8311),
    "Bhubaneswar": (20.2961, 85.8245),
    "Raipur": (21.2514, 81.6296),
    "Ranchi": (23.3441, 85.3096),
    "Dehradun": (30.3165, 78.0322),
}

INDIA_CENTER = {"lat": 22.5, "lon": 79.0}


def add_city_coords(df, city_col="city"):
    df = df.copy()
    df["lat"] = df[city_col].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    df["lon"] = df[city_col].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    return df


def add_jittered_coords(df, city_col="city", locality_col="locality", spread=0.12):
    """Deterministic per-locality offset around the city center — same
    locality always lands at the same point across reruns, so the map is
    stable rather than reshuffling on every refresh."""
    df = df.copy()
    lats, lons = [], []
    for _, row in df.iterrows():
        base_lat, base_lon = CITY_COORDS.get(row[city_col], (None, None))
        if base_lat is None:
            lats.append(None); lons.append(None)
            continue
        seed = abs(hash(str(row[locality_col]))) % (2**32)
        rng = np.random.default_rng(seed)
        lats.append(base_lat + rng.uniform(-spread, spread))
        lons.append(base_lon + rng.uniform(-spread, spread))
    df["lat"] = lats
    df["lon"] = lons
    return df
