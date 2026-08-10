"""
PRISM Fraud Detection — Listings Dataset Generator
=====================================================
Builds a synthetic classifieds-style listing dataset derived from the
properties base, then injects realistic fraud patterns seen on Indian
real estate classifieds (99acres/MagicBricks/OLX-style platforms):

  1. Duplicate listings (same property re-posted with a different price
     or broker contact, to game search ranking or run "bait" pricing)
  2. Price-manipulation outliers (priced far below/above the
     locality+specs-implied fair value to lure inbound calls)
  3. Fake/ghost listings (property details that don't correspond to any
     real inventory pattern — e.g. impossible carpet-area-to-BHK ratios)
  4. Broker phone-number reuse across many unrelated listings (a known
     scam-ring signal)
  5. Recycled/templated descriptions (near-identical text reused across
     supposedly distinct listings)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

DESC_TEMPLATES = [
    "Spacious {bhk}BHK in {locality}, close to metro & schools. Ready to move. Floor {floor}, {facing} facing.",
    "Premium {bhk}BHK apartment at {locality}, {builder} project. RERA approved. {age} yrs old, {facing} facing.",
    "Well-ventilated {bhk}BHK for sale in {locality}. Great connectivity. Floor {floor} of the tower.",
    "Newly renovated {bhk}BHK, {locality}. Family-friendly society with amenities. {age} yrs old.",
    "{bhk}BHK flat in prime {locality} location. Negotiable price, urgent sale. {facing} facing, floor {floor}.",
    "Beautiful {bhk}BHK near {locality} main road, built by {builder}. Vastu compliant, {facing} facing.",
    "{bhk}BHK independent unit in {locality} society, floor {floor}. Immediate possession available.",
]

PHONE_POOL = [f"+91-9{RNG.integers(100000000, 999999999)}" for _ in range(400)]
SCAM_RING_NUMBERS = [f"+91-9{RNG.integers(100000000, 999999999)}" for _ in range(4)]


def build_listings(df_properties: pd.DataFrame, fraud_rate=0.12) -> pd.DataFrame:
    listings = []
    listing_id = 500000

    for _, prop in df_properties.iterrows():
        listing_id += 1
        desc_t = DESC_TEMPLATES[RNG.integers(0, len(DESC_TEMPLATES))]
        desc = desc_t.format(
            bhk=prop["bhk"], locality=prop["locality"], builder=prop["builder"],
            floor=RNG.integers(1, 25), facing=RNG.choice(["North", "South", "East", "West"]),
            age=prop["age_years"],
        )
        listed_price = prop["total_price"] * RNG.normal(1.0, 0.03)
        phone = PHONE_POOL[RNG.integers(0, len(PHONE_POOL))]

        listings.append({
            "listing_id": listing_id,
            "property_id": prop["property_id"],
            "city": prop["city"],
            "locality": prop["locality"],
            "bhk": prop["bhk"],
            "carpet_area_sqft": prop["carpet_area_sqft"],
            "listed_price": round(listed_price, 0),
            "fair_value_est": prop["total_price"],
            "price_deviation_pct": round((listed_price - prop["total_price"]) / prop["total_price"] * 100, 2),
            "broker_phone": phone,
            "description": desc,
            "rera_registered": prop["rera_registered"],
            "days_on_market": int(RNG.integers(1, 180)),
            "num_images": int(RNG.integers(2, 15)),
            "is_fraud": 0,
            "fraud_type": "none",
        })

    df = pd.DataFrame(listings)

    n_fraud = int(len(df) * fraud_rate)
    fraud_idx = RNG.choice(df.index, size=n_fraud, replace=False)
    chunks = np.array_split(fraud_idx, 5)

    # 1. Duplicate listing (re-post same property, different price/phone)
    for idx in chunks[0]:
        orig = df.loc[idx]
        dup_price = orig["listed_price"] * RNG.uniform(0.85, 1.15)
        listing_id += 1
        df.loc[len(df)] = {
            **orig.to_dict(),
            "listing_id": listing_id,
            "listed_price": round(dup_price, 0),
            "price_deviation_pct": round((dup_price - orig["fair_value_est"]) / orig["fair_value_est"] * 100, 2),
            "broker_phone": PHONE_POOL[RNG.integers(0, len(PHONE_POOL))],
            "days_on_market": int(RNG.integers(1, 30)),
            "is_fraud": 1, "fraud_type": "duplicate_listing",
        }
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "duplicate_listing"

    # 2. Price manipulation outlier (bait pricing, far off fair value)
    for idx in chunks[1]:
        direction = RNG.choice([-1, 1])
        deviation = RNG.uniform(0.35, 0.6) * direction
        new_price = df.loc[idx, "fair_value_est"] * (1 + deviation)
        df.loc[idx, "listed_price"] = round(new_price, 0)
        df.loc[idx, "price_deviation_pct"] = round(deviation * 100, 2)
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "price_manipulation"

    # 3. Fake/ghost listing (impossible carpet-area-to-BHK ratio)
    for idx in chunks[2]:
        bhk = df.loc[idx, "bhk"]
        fake_area = bhk * RNG.uniform(120, 180)  # unrealistically small for the BHK count
        df.loc[idx, "carpet_area_sqft"] = round(fake_area, 1)
        df.loc[idx, "num_images"] = int(RNG.integers(0, 2))
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "fake_listing"

    # 4. Broker scam-ring reuse (same handful of numbers across many listings)
    for idx in chunks[3]:
        df.loc[idx, "broker_phone"] = SCAM_RING_NUMBERS[RNG.integers(0, len(SCAM_RING_NUMBERS))]
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "scam_ring_broker"

    # 5. Recycled/templated description reused verbatim across unrelated listings
    recycled_desc = DESC_TEMPLATES[0].format(
        bhk=2, locality="Prime Location", builder="Reputed Builder", floor=5, facing="North", age=3
    )
    for idx in chunks[4]:
        df.loc[idx, "description"] = recycled_desc
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "recycled_description"

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_per_sqft_listed"] = df["listed_price"] / df["carpet_area_sqft"]
    df["area_per_bhk"] = df["carpet_area_sqft"] / df["bhk"]
    df["abs_price_deviation_pct"] = df["price_deviation_pct"].abs()
    phone_counts = df["broker_phone"].value_counts()
    df["broker_listing_count"] = df["broker_phone"].map(phone_counts)
    desc_counts = df["description"].value_counts()
    df["description_reuse_count"] = df["description"].map(desc_counts)
    return df


if __name__ == "__main__":
    props = pd.read_csv("/home/claude/prism/data/properties.csv")
    listings = build_listings(props, fraud_rate=0.12)
    listings = add_engineered_features(listings)
    listings.to_csv("/home/claude/prism/data/listings.csv", index=False)

    print(f"Generated {len(listings)} listings")
    print(f"Fraud rate: {listings['is_fraud'].mean():.2%}")
    print(listings["fraud_type"].value_counts())
