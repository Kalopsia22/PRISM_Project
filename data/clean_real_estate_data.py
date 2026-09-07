"""
PRISM Real-Data Layer — Real Listings Cleaner
==================================================
Cleans and parses an uploaded, real scraped-listings dataset (7 Indian
cities, ~14.5K rows) into a canonical schema PRISM can use for two things:

  1. Recalibrating the circle-rate bands of existing PRISM localities that
     have enough real matching listings (n >= 10) — replacing an
     illustrative estimate with an actual observed price range.
  2. Genuine out-of-sample validation of the trained price model — these
     rows were never used to generate PRISM's synthetic training data, so
     running the model against them is real external validation, not a
     circular test against its own synthetic assumptions.

Raw schema (as uploaded): Name, Property Title, Price, Location, Total_Area,
Price_per_SQFT, Description, Baths, Balcony. Price is a string like
"₹1.99 Cr" / "₹48.0 L" / "₹95.45 Lacs" / occasionally "₹55.0k" (a handful of
rows are evidently mislabeled rental listings that leaked into the sale
data — filtered out as part of cleaning, see `_parse_price`).
"""

import re
import numpy as np
import pandas as pd

RAW_PATH = "/home/claude/prism/data/raw_real_estate_listings.csv"
OUT_PATH = "/home/claude/prism/data/real_listings.csv"
CALIBRATION_OUT_PATH = "/home/claude/prism/data/real_locality_calibration.csv"

CITY_RENAME = {"New Delhi": "Delhi NCR"}

PROPERTY_TYPE_MAP = {
    "Flat": "Apartment",
    "Independent House": "Independent House",
    "Villa": "Villa",
}


def _parse_price(raw: str):
    """Returns price in INR, or None for unparseable/out-of-scope values
    (e.g. the 'k' suffix rows, which are ~1000x too small to be sale prices
    and are almost certainly mislabeled rental listings)."""
    if not isinstance(raw, str):
        return None
    s = raw.replace("₹", "").strip()
    if s.endswith("Cr"):
        try:
            return float(s.replace("Cr", "").strip()) * 1e7
        except ValueError:
            return None
    if s.endswith("Lacs"):
        try:
            return float(s.replace("Lacs", "").strip()) * 1e5
        except ValueError:
            return None
    if s.endswith("L"):
        try:
            return float(s.replace("L", "").strip()) * 1e5
        except ValueError:
            return None
    return None  # 'k' suffix and blanks fall through here, filtered out


def clean_real_listings(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df["city"] = df["Location"].str.split(",").str[-1].str.strip()
    df["city"] = df["city"].replace(CITY_RENAME)

    locality_full = df["Location"].str.rsplit(",", n=1).str[0].str.strip().str.rstrip(",")
    df["locality"] = locality_full.str.split(",").str[-1].str.strip()
    df.loc[df["locality"] == "", "locality"] = locality_full

    df["bhk"] = df["Property Title"].str.extract(r"(\d+)\s*BHK").astype(float)
    df["property_type_raw"] = df["Property Title"].str.extract(r"BHK\s+([A-Za-z\s]+?)\s+for sale")
    df["property_type"] = df["property_type_raw"].map(PROPERTY_TYPE_MAP)

    df["price_inr"] = df["Price"].apply(_parse_price)
    df["price_per_sqft_calc"] = df["price_inr"] / df["Total_Area"]

    before = len(df)
    df = df[
        df["price_inr"].notna()
        & df["property_type"].notna()
        & df["bhk"].notna()
        & df["Total_Area"].between(200, 10000)
        & df["price_per_sqft_calc"].between(500, 60000)
        & df["city"].isin(list(CITY_RENAME.values()) + [
            "Bangalore", "Pune", "Chennai", "Kolkata", "Mumbai", "Hyderabad",
        ])
    ].copy()
    dropped = before - len(df)

    df = df.rename(columns={
        "Total_Area": "carpet_area_sqft", "Baths": "baths", "Balcony": "balcony", "Name": "listing_name",
    })
    df["has_balcony"] = (df["balcony"] == "Yes").astype(int)

    keep_cols = ["listing_name", "city", "locality", "property_type", "bhk",
                  "carpet_area_sqft", "price_inr", "price_per_sqft_calc", "baths", "has_balcony"]
    df = df[keep_cols].reset_index(drop=True)

    print(f"Cleaned {len(df)} / {before} rows ({dropped} dropped — unparseable price, "
          f"missing BHK/type, or outside sane area/price-per-sqft bounds)")
    return df


def build_locality_calibration(cleaned: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Per (city, locality) real price/sqft stats, for any PRISM locality
    with enough matching real listings to recalibrate against — used to
    replace an illustrative circle-rate band with an actual observed one."""
    cleaned = cleaned.copy()
    cleaned["locality_lower"] = cleaned["locality"].str.lower()
    agg = cleaned.groupby(["city", "locality_lower"]).agg(
        n=("price_per_sqft_calc", "count"),
        real_p25=("price_per_sqft_calc", lambda x: x.quantile(0.25)),
        real_p75=("price_per_sqft_calc", lambda x: x.quantile(0.75)),
        real_median=("price_per_sqft_calc", "median"),
    ).reset_index()
    return agg[agg["n"] >= min_n].sort_values("n", ascending=False)


if __name__ == "__main__":
    cleaned = clean_real_listings()
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    print(f"\nRows per city:\n{cleaned['city'].value_counts()}")
    print(f"\nRows per property type:\n{cleaned['property_type'].value_counts()}")

    calibration = build_locality_calibration(cleaned)
    calibration.to_csv(CALIBRATION_OUT_PATH, index=False)
    print(f"\nSaved locality calibration ({len(calibration)} localities with n>=10) to {CALIBRATION_OUT_PATH}")
