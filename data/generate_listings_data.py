"""
PRISM Fraud Detection — Listings Dataset Generator (v2)
===========================================================
Builds a synthetic classifieds-style listing dataset from the properties
base, now split across SALE and RENT listings (rentable properties get a
rental listing too), then injects realistic fraud patterns seen on Indian
real estate classifieds:

  1. Duplicate listings (same property re-posted with a different price
     or broker contact)
  2. Price-manipulation outliers (priced far below/above the
     locality+specs-implied fair value)
  3. Fake/ghost listings (impossible carpet-area-to-BHK ratios)
  4. Broker phone-number reuse across many unrelated listings
  5. Recycled/templated descriptions reused across listings

Every listing carries a unified `ask_value` / `fair_value` pair (sale
price or monthly rent, whichever applies) so the fraud model can treat
both listing types with the same feature set.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

DESC_TEMPLATES_SALE = [
    "Spacious {bhk}BHK in {locality}, close to metro & schools. Ready to move. Floor {floor}, {facing} facing.",
    "Premium {bhk}BHK apartment at {locality}, {builder} project. RERA approved. {age} yrs old, {facing} facing.",
    "Well-ventilated {bhk}BHK for sale in {locality}. Great connectivity. Floor {floor} of the tower.",
    "Newly renovated {bhk}BHK, {locality}. Family-friendly society with amenities. {age} yrs old.",
    "{bhk}BHK flat in prime {locality} location. Negotiable price, urgent sale. {facing} facing, floor {floor}.",
    "Beautiful {bhk}BHK near {locality} main road, built by {builder}. Vastu compliant, {facing} facing.",
    "{bhk}BHK independent unit in {locality} society, floor {floor}. Immediate possession available.",
]

DESC_TEMPLATES_RENT = [
    "{bhk}BHK available for rent in {locality}. Semi-furnished, floor {floor}, {facing} facing. Family preferred.",
    "Well-maintained {bhk}BHK for rent at {locality}, {builder} project. No brokerage. Bachelors allowed.",
    "Spacious {bhk}BHK to let in {locality}. Immediate possession, {facing} facing, floor {floor}.",
    "Furnished {bhk}BHK for rent near {locality}. Society with amenities, {age} yrs old building.",
    "{bhk}BHK rental in prime {locality}. Negotiable rent, floor {floor}. Ideal for working professionals.",
]

PHONE_POOL = [f"+91-9{RNG.integers(100000000, 999999999)}" for _ in range(400)]
SCAM_RING_NUMBERS = [f"+91-9{RNG.integers(100000000, 999999999)}" for _ in range(4)]


def _make_description(is_rent, bhk, locality, builder, area_sqft):
    templates = DESC_TEMPLATES_RENT if is_rent else DESC_TEMPLATES_SALE
    t = templates[RNG.integers(0, len(templates))]
    base = t.format(
        bhk=bhk, locality=locality, builder=builder,
        floor=RNG.integers(1, 25), facing=RNG.choice(["North", "South", "East", "West"]),
        age=RNG.integers(0, 15),
    )
    # area is near-continuous per listing — mentioning it adds the entropy needed
    # so independent listings don't collide into identical text purely by chance
    return f"{base} Carpet area approx {area_sqft:.0f} sqft."


def build_listings(df_properties: pd.DataFrame, fraud_rate=0.12) -> pd.DataFrame:
    listings = []
    listing_id = 500000

    for _, prop in df_properties.iterrows():
        is_plot = prop["property_type"] == "Plot/Land"

        # every property gets a sale listing; rentable, non-plot properties
        # also get an independent rental listing ~55% of the time
        listing_id += 1
        ask_sale = prop["asking_total_price"] * RNG.normal(1.0, 0.02)
        listings.append({
            "listing_id": listing_id, "property_id": prop["property_id"],
            "listing_type": "sale",
            "city": prop["city"], "locality": prop["locality"], "property_type": prop["property_type"],
            "bhk": prop["bhk"], "carpet_area_sqft": prop["carpet_area_sqft"],
            "ask_value": round(ask_sale, 0), "fair_value": prop["total_price"],
            "broker_phone": PHONE_POOL[RNG.integers(0, len(PHONE_POOL))],
            "description": _make_description(False, prop["bhk"], prop["locality"], prop["builder"], prop["carpet_area_sqft"]),
            "rera_registered": prop["rera_registered"],
            "days_on_market": int(RNG.integers(1, 180)),
            "num_images": int(RNG.integers(2, 15)),
            "is_fraud": 0, "fraud_type": "none",
        })

        if (not is_plot) and prop["rentable"] and RNG.random() < 0.55:
            listing_id += 1
            ask_rent = prop["monthly_rent_asking"] if prop["monthly_rent_asking"] > 0 else prop["monthly_rent_est"]
            ask_rent = ask_rent * RNG.normal(1.0, 0.03)
            listings.append({
                "listing_id": listing_id, "property_id": prop["property_id"],
                "listing_type": "rent",
                "city": prop["city"], "locality": prop["locality"], "property_type": prop["property_type"],
                "bhk": prop["bhk"], "carpet_area_sqft": prop["carpet_area_sqft"],
                "ask_value": round(ask_rent, 0), "fair_value": prop["monthly_rent_est"],
                "broker_phone": PHONE_POOL[RNG.integers(0, len(PHONE_POOL))],
                "description": _make_description(True, prop["bhk"], prop["locality"], prop["builder"], prop["carpet_area_sqft"]),
                "rera_registered": prop["rera_registered"],
                "days_on_market": int(RNG.integers(1, 90)),
                "num_images": int(RNG.integers(2, 12)),
                "is_fraud": 0, "fraud_type": "none",
            })

    df = pd.DataFrame(listings)
    df["price_deviation_pct"] = round((df["ask_value"] - df["fair_value"]) / df["fair_value"] * 100, 2)

    n_fraud = int(len(df) * fraud_rate)
    fraud_idx = RNG.choice(df.index, size=n_fraud, replace=False)
    chunks = np.array_split(fraud_idx, 5)

    # 1. Duplicate listing (re-post same property, different price/phone)
    for idx in chunks[0]:
        orig = df.loc[idx]
        dup_ask = orig["ask_value"] * RNG.uniform(0.85, 1.15)
        listing_id += 1
        new_row = orig.to_dict()
        new_row.update({
            "listing_id": listing_id, "ask_value": round(dup_ask, 0),
            "price_deviation_pct": round((dup_ask - orig["fair_value"]) / orig["fair_value"] * 100, 2),
            "broker_phone": PHONE_POOL[RNG.integers(0, len(PHONE_POOL))],
            "days_on_market": int(RNG.integers(1, 30)),
            "is_fraud": 1, "fraud_type": "duplicate_listing",
        })
        df.loc[len(df)] = new_row
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "duplicate_listing"

    # 2. Price manipulation outlier
    for idx in chunks[1]:
        direction = RNG.choice([-1, 1])
        deviation = RNG.uniform(0.35, 0.6) * direction
        new_ask = df.loc[idx, "fair_value"] * (1 + deviation)
        df.loc[idx, "ask_value"] = round(new_ask, 0)
        df.loc[idx, "price_deviation_pct"] = round(deviation * 100, 2)
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "price_manipulation"

    # 3. Fake/ghost listing (impossible carpet-area-to-BHK ratio)
    for idx in chunks[2]:
        bhk = max(1, df.loc[idx, "bhk"])
        fake_area = bhk * RNG.uniform(120, 180)
        df.loc[idx, "carpet_area_sqft"] = round(fake_area, 1)
        df.loc[idx, "num_images"] = int(RNG.integers(0, 2))
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "fake_listing"

    # 4. Broker scam-ring reuse
    for idx in chunks[3]:
        df.loc[idx, "broker_phone"] = SCAM_RING_NUMBERS[RNG.integers(0, len(SCAM_RING_NUMBERS))]
        df.loc[idx, "is_fraud"] = 1
        df.loc[idx, "fraud_type"] = "scam_ring_broker"

    # 5. Recycled/templated description
    recycled_desc = DESC_TEMPLATES_SALE[0].format(
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
    df["ask_per_sqft"] = df["ask_value"] / df["carpet_area_sqft"]
    df["area_per_bhk"] = df["carpet_area_sqft"] / df["bhk"].replace(0, 1)
    df["abs_price_deviation_pct"] = df["price_deviation_pct"].abs()
    phone_counts = df["broker_phone"].value_counts()
    df["broker_listing_count"] = df["broker_phone"].map(phone_counts)
    desc_counts = df["description"].value_counts()
    df["description_reuse_count"] = df["description"].map(desc_counts)
    df["is_rent"] = (df["listing_type"] == "rent").astype(int)
    return df


if __name__ == "__main__":
    props = pd.read_csv("/home/claude/prism/data/properties.csv")
    listings = build_listings(props, fraud_rate=0.12)
    listings = add_engineered_features(listings)
    listings.to_csv("/home/claude/prism/data/listings.csv", index=False)

    print(f"Generated {len(listings)} listings")
    print(f"Sale vs rent: {listings['listing_type'].value_counts().to_dict()}")
    print(f"Fraud rate: {listings['is_fraud'].mean():.2%}")
    print(listings["fraud_type"].value_counts())
