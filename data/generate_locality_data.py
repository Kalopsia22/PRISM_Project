"""
PRISM Data Layer — Synthetic Locality & Property Dataset Generator
=====================================================================
Generates a realistic, India-calibrated property dataset at pincode /
micro-market granularity for Mumbai and Bangalore.

IMPORTANT — DATA DISCLOSURE:
Bulk registered-transaction-price data in India is not available via a
single public API — it is fragmented across state sub-registrar (IGRS)
and RERA portals with inconsistent formats. This generator produces
SYNTHETIC data that is CALIBRATED to real, publicly-available anchors:
  - State/city Ready Reckoner (circle rate) bands per micro-market
  - Realistic builder/RERA registration patterns (RERA mandatory
    post-2017 for new residential projects)
  - Realistic amenity premium structures (15-25% for gated/full-amenity
    stock, per industry data)
  - Realistic stamp duty + registration charge ranges by state

This is the same "synthetic-but-regulation-calibrated" approach used in
the AA Financial Health Scoring project — the intent is to demonstrate
modeling methodology, not to claim real transaction-level ground truth.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Micro-market definitions (Mumbai + Bangalore), with a calibration anchor
#    for Ready Reckoner / Guidance Value band (INR per sqft) as of ~2025-26.
#    These bands are directionally realistic (relative ordering of localities)
#    rather than official government figures.
# ---------------------------------------------------------------------------

MICRO_MARKETS = [
    # city, locality, pincode, rr_rate_min, rr_rate_max, metro_proximity_km, tier
    ("Mumbai", "Bandra West", "400050", 28000, 42000, 0.6, "premium"),
    ("Mumbai", "Andheri West", "400058", 18000, 26000, 0.4, "mid-premium"),
    ("Mumbai", "Powai", "400076", 16000, 23000, 1.2, "mid-premium"),
    ("Mumbai", "Malad West", "400064", 12000, 17000, 0.8, "mid"),
    ("Mumbai", "Thane West", "400601", 9500, 14500, 1.5, "mid"),
    ("Mumbai", "Mulund West", "400080", 13000, 18500, 0.9, "mid"),
    ("Mumbai", "Chembur", "400071", 15000, 21000, 1.0, "mid-premium"),
    ("Mumbai", "Kandivali East", "400101", 11500, 16000, 1.3, "mid"),
    ("Mumbai", "Goregaon West", "400062", 15500, 21500, 0.7, "mid-premium"),
    ("Mumbai", "Dombivli", "421201", 6500, 9500, 2.0, "affordable"),
    ("Bangalore", "Indiranagar", "560038", 15000, 22000, 0.5, "premium"),
    ("Bangalore", "Whitefield", "560066", 7500, 11500, 2.5, "mid"),
    ("Bangalore", "Koramangala", "560034", 13500, 19500, 0.8, "premium"),
    ("Bangalore", "HSR Layout", "560102", 9500, 14000, 1.4, "mid-premium"),
    ("Bangalore", "Electronic City", "560100", 5500, 8500, 3.0, "affordable"),
    ("Bangalore", "Hebbal", "560024", 8000, 12000, 1.0, "mid"),
    ("Bangalore", "JP Nagar", "560078", 8500, 12500, 1.6, "mid"),
    ("Bangalore", "Sarjapur Road", "560035", 6000, 9000, 3.5, "affordable"),
    ("Bangalore", "Yelahanka", "560064", 5000, 7500, 4.0, "affordable"),
    ("Bangalore", "Malleshwaram", "560003", 14000, 20000, 0.6, "premium"),
]

BUILDERS = [
    ("Lodha Group", 0.95, "Tier1"), ("Godrej Properties", 0.93, "Tier1"),
    ("Prestige Group", 0.92, "Tier1"), ("Sobha Ltd", 0.91, "Tier1"),
    ("Brigade Group", 0.88, "Tier1"), ("Oberoi Realty", 0.90, "Tier1"),
    ("Kolte-Patil", 0.82, "Tier2"), ("Puravankara", 0.80, "Tier2"),
    ("Runwal Group", 0.78, "Tier2"), ("Local Developer Co", 0.55, "Tier3"),
    ("Regional Builders Ltd", 0.50, "Tier3"),
]

AMENITIES_POOL = [
    "Clubhouse", "Swimming Pool", "Gymnasium", "Children's Play Area",
    "Jogging Track", "24/7 Security", "Power Backup", "Lift",
    "Visitor Parking", "CCTV", "Landscaped Garden", "Indoor Games Room",
]

STAMP_DUTY_BY_STATE = {"Mumbai": 0.06, "Bangalore": 0.055}  # Maharashtra ~6%, Karnataka ~5.5%
REGISTRATION_CHARGE_BY_STATE = {"Mumbai": 0.01, "Bangalore": 0.01}


def _sample_amenities():
    n = RNG.integers(3, len(AMENITIES_POOL) + 1)
    chosen = RNG.choice(AMENITIES_POOL, size=n, replace=False)
    return list(chosen)


def generate_properties(n_per_market=120) -> pd.DataFrame:
    rows = []
    prop_id = 100000
    for city, locality, pincode, rr_min, rr_max, metro_km, tier in MICRO_MARKETS:
        for _ in range(n_per_market):
            prop_id += 1
            rr_rate = RNG.uniform(rr_min, rr_max)

            builder_name, builder_score, builder_tier = BUILDERS[RNG.integers(0, len(BUILDERS))]
            rera_registered = 1 if (builder_tier != "Tier3" or RNG.random() > 0.35) else 0
            rera_number = f"P{'MHA' if city=='Mumbai' else 'KAR'}{RNG.integers(10000,99999)}" if rera_registered else None

            amenities = _sample_amenities()
            amenity_premium = min(0.25, 0.02 * len(amenities))  # up to ~25% premium

            bhk = RNG.choice([1, 2, 2, 3, 3, 4], p=[0.15, 0.30, 0.05, 0.30, 0.05, 0.15])
            carpet_area = {1: RNG.uniform(380, 550), 2: RNG.uniform(650, 950),
                            3: RNG.uniform(950, 1400), 4: RNG.uniform(1500, 2200)}[bhk]

            age_years = RNG.integers(0, 25)
            age_discount = max(0, 1 - 0.006 * age_years)  # gentle depreciation

            metro_premium = max(0, 1 - 0.03 * metro_km)  # closer = premium

            builder_premium = 0.85 + 0.3 * builder_score  # 0.85x - 1.15x

            noise = RNG.normal(1.0, 0.05)

            price_per_sqft = (
                rr_rate
                * (1 + amenity_premium)
                * (1 + metro_premium * 0.15)
                * builder_premium
                * age_discount
                * noise
            )
            total_price = price_per_sqft * carpet_area

            circle_rate_value = rr_rate * carpet_area
            stamp_duty_pct = STAMP_DUTY_BY_STATE[city]
            reg_charge_pct = REGISTRATION_CHARGE_BY_STATE[city]
            stamp_duty_amt = total_price * stamp_duty_pct
            reg_charge_amt = min(total_price * reg_charge_pct, 30000)  # KA caps reg charge

            # rental yield: affordable/mid tiers tend to show higher yield %, premium lower —
            # but affordable/emerging markets also carry more yield volatility (less mature
            # rental demand, more supply-driven swings), which is what makes risk-averse
            # investors genuinely trade off yield for stability rather than always picking
            # the highest-yield tier.
            base_yield = {"affordable": 0.032, "mid": 0.028, "mid-premium": 0.024, "premium": 0.019}[tier]
            yield_volatility = {"affordable": 0.009, "mid": 0.006, "mid-premium": 0.004, "premium": 0.0025}[tier]
            rental_yield_pct = max(0.010, RNG.normal(base_yield, yield_volatility))
            monthly_rent = (total_price * rental_yield_pct) / 12

            rows.append({
                "property_id": prop_id,
                "city": city,
                "locality": locality,
                "pincode": pincode,
                "tier": tier,
                "metro_distance_km": round(metro_km + RNG.normal(0, 0.15), 2),
                "bhk": bhk,
                "carpet_area_sqft": round(carpet_area, 1),
                "age_years": age_years,
                "builder": builder_name,
                "builder_score": builder_score,
                "builder_tier": builder_tier,
                "rera_registered": rera_registered,
                "rera_number": rera_number,
                "num_amenities": len(amenities),
                "amenities": ", ".join(amenities),
                "circle_rate_per_sqft": round(rr_rate, 0),
                "price_per_sqft": round(price_per_sqft, 0),
                "total_price": round(total_price, 0),
                "stamp_duty_pct": stamp_duty_pct,
                "stamp_duty_amt": round(stamp_duty_amt, 0),
                "registration_charge_amt": round(reg_charge_amt, 0),
                "monthly_rent_est": round(monthly_rent, 0),
                "rental_yield_pct": round(rental_yield_pct * 100, 2),
                "vastu_compliant": int(RNG.random() > 0.4),
                "gated_community": int(len(amenities) >= 6),
            })

    df = pd.DataFrame(rows)
    return df


def generate_appreciation_trend(df_localities: pd.DataFrame, years=5) -> pd.DataFrame:
    """Synthetic YoY appreciation trend per locality for the yield/recommendation module."""
    trend_rows = []
    localities = df_localities[["city", "locality", "pincode", "tier"]].drop_duplicates()
    base_growth = {"affordable": 0.09, "mid": 0.075, "mid-premium": 0.06, "premium": 0.045}
    # emerging/affordable markets swing more year-to-year (less established demand base,
    # more sensitive to new-supply announcements); premium markets are the steady compounders
    growth_volatility = {"affordable": 0.045, "mid": 0.03, "mid-premium": 0.02, "premium": 0.012}
    for _, row in localities.iterrows():
        g = base_growth[row["tier"]]
        vol = growth_volatility[row["tier"]]
        cumulative = 1.0
        for yr in range(years):
            yoy = max(0.005, RNG.normal(g, vol))
            cumulative *= (1 + yoy)
            trend_rows.append({
                "city": row["city"], "locality": row["locality"], "pincode": row["pincode"],
                "year_offset": yr + 1, "yoy_appreciation_pct": round(yoy * 100, 2),
                "cumulative_index": round(cumulative, 3),
            })
    return pd.DataFrame(trend_rows)


if __name__ == "__main__":
    props = generate_properties(n_per_market=120)
    trend = generate_appreciation_trend(props, years=5)

    props.to_csv("/home/claude/prism/data/properties.csv", index=False)
    trend.to_csv("/home/claude/prism/data/appreciation_trend.csv", index=False)

    print(f"Generated {len(props)} properties across {props['locality'].nunique()} micro-markets")
    print(f"Generated {len(trend)} appreciation trend rows")
    print("\nSample:")
    print(props.head(3).T)
