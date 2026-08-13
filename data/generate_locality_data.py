"""
PRISM Data Layer — Synthetic Locality & Property Dataset Generator (v2)
==========================================================================
Generates a realistic, India-calibrated property dataset spanning Tier 1,
Tier 2, and Tier 3 cities, across multiple property categories, with both
sale and rental pricing.

Bulk registered-transaction-price data is not available via a single
public API in India — it is fragmented across state sub-registrar (IGRS)
and RERA portals with inconsistent formats. This generator produces
data CALIBRATED to real, publicly-available anchors (Ready Reckoner /
guidance-value bands, RERA registration patterns, stamp duty rates by
state) rather than claiming real transaction-level ground truth.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Micro-market definitions across city tiers.
# (city, locality, pincode, rr_rate_min, rr_rate_max, metro_proximity_km,
#  locality_quality_tier, city_tier)
# ---------------------------------------------------------------------------

MICRO_MARKETS = [
    # ---- Tier 1 ----
    ("Mumbai", "Bandra West", "400050", 28000, 42000, 0.6, "premium", "Tier1"),
    ("Mumbai", "Andheri West", "400058", 18000, 26000, 0.4, "mid-premium", "Tier1"),
    ("Mumbai", "Powai", "400076", 16000, 23000, 1.2, "mid-premium", "Tier1"),
    ("Mumbai", "Malad West", "400064", 12000, 17000, 0.8, "mid", "Tier1"),
    ("Mumbai", "Thane West", "400601", 9500, 14500, 1.5, "mid", "Tier1"),
    ("Mumbai", "Mulund West", "400080", 13000, 18500, 0.9, "mid", "Tier1"),
    ("Mumbai", "Chembur", "400071", 15000, 21000, 1.0, "mid-premium", "Tier1"),
    ("Mumbai", "Kandivali East", "400101", 11500, 16000, 1.3, "mid", "Tier1"),
    ("Mumbai", "Goregaon West", "400062", 15500, 21500, 0.7, "mid-premium", "Tier1"),
    ("Mumbai", "Dombivli", "421201", 6500, 9500, 2.0, "affordable", "Tier1"),
    ("Bangalore", "Indiranagar", "560038", 15000, 22000, 0.5, "premium", "Tier1"),
    ("Bangalore", "Whitefield", "560066", 7500, 11500, 2.5, "mid", "Tier1"),
    ("Bangalore", "Koramangala", "560034", 13500, 19500, 0.8, "premium", "Tier1"),
    ("Bangalore", "HSR Layout", "560102", 9500, 14000, 1.4, "mid-premium", "Tier1"),
    ("Bangalore", "Electronic City", "560100", 5500, 8500, 3.0, "affordable", "Tier1"),
    ("Bangalore", "Hebbal", "560024", 8000, 12000, 1.0, "mid", "Tier1"),
    ("Bangalore", "JP Nagar", "560078", 8500, 12500, 1.6, "mid", "Tier1"),
    ("Bangalore", "Sarjapur Road", "560035", 6000, 9000, 3.5, "affordable", "Tier1"),
    ("Bangalore", "Yelahanka", "560064", 5000, 7500, 4.0, "affordable", "Tier1"),
    ("Bangalore", "Malleshwaram", "560003", 14000, 20000, 0.6, "premium", "Tier1"),
    ("Delhi NCR", "Gurgaon Sector 54", "122002", 14000, 20000, 1.0, "premium", "Tier1"),
    ("Delhi NCR", "Noida Sector 62", "201301", 8000, 12000, 1.5, "mid-premium", "Tier1"),
    ("Delhi NCR", "Dwarka Expressway", "122018", 6500, 9500, 2.5, "mid", "Tier1"),
    ("Chennai", "Adyar", "600020", 12000, 17000, 0.8, "premium", "Tier1"),
    ("Chennai", "OMR Sholinganallur", "600119", 6000, 9000, 2.0, "mid", "Tier1"),
    ("Chennai", "Anna Nagar", "600040", 10000, 14000, 1.0, "mid-premium", "Tier1"),
    ("Hyderabad", "Jubilee Hills", "500033", 13000, 19000, 0.7, "premium", "Tier1"),
    ("Hyderabad", "Gachibowli", "500032", 7500, 11000, 1.8, "mid-premium", "Tier1"),
    ("Hyderabad", "Kukatpally", "500072", 5500, 8000, 2.2, "mid", "Tier1"),
    ("Pune", "Koregaon Park", "411001", 11000, 16000, 0.9, "premium", "Tier1"),
    ("Pune", "Hinjewadi", "411057", 6000, 9000, 3.0, "mid", "Tier1"),
    ("Pune", "Viman Nagar", "411014", 8500, 12000, 1.4, "mid-premium", "Tier1"),
    ("Kolkata", "Ballygunge", "700019", 10000, 15000, 0.6, "premium", "Tier1"),
    ("Kolkata", "Salt Lake", "700064", 6500, 9500, 1.5, "mid", "Tier1"),
    ("Kolkata", "New Town", "700156", 5000, 7500, 2.5, "affordable", "Tier1"),
    ("Ahmedabad", "Satellite", "380015", 6500, 9500, 1.2, "mid-premium", "Tier1"),
    ("Ahmedabad", "Bopal", "380058", 4500, 6500, 2.5, "mid", "Tier1"),
    ("Ahmedabad", "SG Highway", "380054", 5500, 8000, 1.8, "mid", "Tier1"),
    # ---- Tier 2 ----
    ("Jaipur", "Vaishali Nagar", "302021", 4500, 6500, 2.0, "mid", "Tier2"),
    ("Jaipur", "Malviya Nagar", "302017", 5000, 7000, 1.5, "mid", "Tier2"),
    ("Jaipur", "Mansarovar", "302020", 3800, 5500, 2.8, "affordable", "Tier2"),
    ("Lucknow", "Gomti Nagar", "226010", 4200, 6000, 2.0, "mid", "Tier2"),
    ("Lucknow", "Hazratganj", "226001", 5500, 7500, 1.0, "mid-premium", "Tier2"),
    ("Lucknow", "Indira Nagar", "226016", 3500, 5000, 2.5, "affordable", "Tier2"),
    ("Chandigarh", "Sector 22", "160022", 7000, 10000, 1.0, "mid-premium", "Tier2"),
    ("Chandigarh", "Sector 43", "160043", 5500, 8000, 1.8, "mid", "Tier2"),
    ("Chandigarh", "Zirakpur", "140603", 4000, 5800, 3.0, "affordable", "Tier2"),
    ("Indore", "Vijay Nagar", "452010", 4200, 6000, 1.5, "mid", "Tier2"),
    ("Indore", "Palasia", "452001", 5000, 7200, 1.0, "mid-premium", "Tier2"),
    ("Indore", "Rau", "453331", 3200, 4500, 3.5, "affordable", "Tier2"),
    ("Kochi", "Kakkanad", "682030", 4500, 6500, 2.0, "mid", "Tier2"),
    ("Kochi", "Marine Drive", "682031", 6500, 9500, 0.8, "mid-premium", "Tier2"),
    ("Kochi", "Edappally", "682024", 4000, 5800, 2.5, "affordable", "Tier2"),
    ("Surat", "Vesu", "395007", 4800, 7000, 1.8, "mid", "Tier2"),
    ("Surat", "Adajan", "395009", 4200, 6000, 2.0, "mid", "Tier2"),
    ("Surat", "City Light", "395007", 5500, 7800, 1.2, "mid-premium", "Tier2"),
    # ---- Tier 3 ----
    ("Bhubaneswar", "Patia", "751024", 3200, 4500, 2.5, "mid", "Tier3"),
    ("Bhubaneswar", "Saheed Nagar", "751007", 3800, 5200, 1.5, "mid", "Tier3"),
    ("Bhubaneswar", "Chandrasekharpur", "751016", 2800, 4000, 3.0, "affordable", "Tier3"),
    ("Raipur", "Shankar Nagar", "492007", 3000, 4200, 2.0, "mid", "Tier3"),
    ("Raipur", "Telibandha", "492006", 2600, 3800, 2.8, "affordable", "Tier3"),
    ("Raipur", "VIP Road", "492001", 3400, 4800, 1.5, "mid", "Tier3"),
    ("Ranchi", "Lalpur", "834001", 2800, 4000, 2.0, "mid", "Tier3"),
    ("Ranchi", "Harmu", "834002", 2400, 3500, 2.5, "affordable", "Tier3"),
    ("Ranchi", "Kanke Road", "834006", 2200, 3200, 3.0, "affordable", "Tier3"),
    ("Dehradun", "Rajpur Road", "248001", 3800, 5500, 1.5, "mid-premium", "Tier3"),
    ("Dehradun", "Sahastradhara Road", "248013", 2800, 4000, 2.5, "mid", "Tier3"),
    ("Dehradun", "Clement Town", "248002", 2200, 3200, 3.5, "affordable", "Tier3"),
]

BUILDERS = [
    ("Lodha Group", 0.95, "Tier1"), ("Godrej Properties", 0.93, "Tier1"),
    ("Prestige Group", 0.92, "Tier1"), ("Sobha Ltd", 0.91, "Tier1"),
    ("Brigade Group", 0.88, "Tier1"), ("Oberoi Realty", 0.90, "Tier1"),
    ("DLF Ltd", 0.92, "Tier1"), ("Kolte-Patil", 0.82, "Tier2"),
    ("Puravankara", 0.80, "Tier2"), ("Runwal Group", 0.78, "Tier2"),
    ("Local Developer Co", 0.55, "Tier3"), ("Regional Builders Ltd", 0.50, "Tier3"),
]

AMENITIES_POOL = [
    "Clubhouse", "Swimming Pool", "Gymnasium", "Children's Play Area",
    "Jogging Track", "24/7 Security", "Power Backup", "Lift",
    "Visitor Parking", "CCTV", "Landscaped Garden", "Indoor Games Room",
]

STAMP_DUTY_BY_STATE = {
    "Mumbai": 0.06, "Bangalore": 0.055, "Delhi NCR": 0.06, "Chennai": 0.07,
    "Hyderabad": 0.055, "Pune": 0.06, "Kolkata": 0.06, "Ahmedabad": 0.049,
    "Jaipur": 0.06, "Lucknow": 0.07, "Chandigarh": 0.06, "Indore": 0.075,
    "Kochi": 0.08, "Surat": 0.049, "Bhubaneswar": 0.05, "Raipur": 0.05,
    "Ranchi": 0.04, "Dehradun": 0.05,
}
REGISTRATION_CHARGE_BY_STATE = {c: 0.01 for c in STAMP_DUTY_BY_STATE}

# ---------------------------------------------------------------------------
# Property categories — each with its own area range, BHK pattern, and price
# multiplier relative to the standard apartment price/sqft for that locality.
# ---------------------------------------------------------------------------

PROPERTY_TYPES = {
    "Apartment":         {"weight": 0.50, "area": (550, 1500), "bhk": [1, 2, 2, 3, 3], "price_mult": 1.00, "amenities": True},
    "Villa":              {"weight": 0.08, "area": (2200, 4500), "bhk": [3, 4, 4, 5], "price_mult": 1.35, "amenities": True},
    "Independent House":  {"weight": 0.10, "area": (1400, 3000), "bhk": [2, 3, 3, 4], "price_mult": 1.15, "amenities": False},
    "Penthouse":          {"weight": 0.05, "area": (1800, 3500), "bhk": [3, 3, 4], "price_mult": 1.50, "amenities": True},
    "Studio/1RK":         {"weight": 0.12, "area": (250, 450), "bhk": [1], "price_mult": 1.08, "amenities": True},
    "Row House":          {"weight": 0.08, "area": (1200, 2200), "bhk": [2, 3], "price_mult": 1.20, "amenities": False},
    "Plot/Land":          {"weight": 0.07, "area": (800, 3000), "bhk": [0], "price_mult": 0.65, "amenities": False},
}
TYPE_NAMES = list(PROPERTY_TYPES.keys())
TYPE_WEIGHTS = [PROPERTY_TYPES[t]["weight"] for t in TYPE_NAMES]


def _sample_amenities(has_amenities: bool):
    if not has_amenities:
        n = RNG.integers(0, 3)
    else:
        n = RNG.integers(3, len(AMENITIES_POOL) + 1)
    if n == 0:
        return []
    n = min(n, len(AMENITIES_POOL))
    chosen = RNG.choice(AMENITIES_POOL, size=n, replace=False)
    return list(chosen)


def generate_properties(n_per_market=80) -> pd.DataFrame:
    rows = []
    prop_id = 100000
    for city, locality, pincode, rr_min, rr_max, metro_km, tier, city_tier in MICRO_MARKETS:
        for _ in range(n_per_market):
            prop_id += 1
            rr_rate = RNG.uniform(rr_min, rr_max)

            property_type = RNG.choice(TYPE_NAMES, p=TYPE_WEIGHTS)
            type_spec = PROPERTY_TYPES[property_type]

            is_plot = property_type == "Plot/Land"

            if is_plot:
                builder_name, builder_score, builder_tier = "Individual/Land Owner", 0.60, "Tier3"
                rera_registered = 0
                rera_number = None
            else:
                builder_name, builder_score, builder_tier = BUILDERS[RNG.integers(0, len(BUILDERS))]
                rera_registered = 1 if (builder_tier != "Tier3" or RNG.random() > 0.35) else 0
                rera_number = f"P{'MHA' if city=='Mumbai' else 'REG'}{RNG.integers(10000,99999)}" if rera_registered else None

            amenities = [] if is_plot else _sample_amenities(type_spec["amenities"])
            amenity_premium = min(0.25, 0.02 * len(amenities))

            bhk = int(RNG.choice(type_spec["bhk"]))
            area_min, area_max = type_spec["area"]
            carpet_area = RNG.uniform(area_min, area_max)

            age_years = 0 if is_plot else RNG.integers(0, 25)
            age_discount = 1.0 if is_plot else max(0, 1 - 0.006 * age_years)

            metro_premium = max(0, 1 - 0.03 * metro_km)
            builder_premium = 0.85 + 0.3 * builder_score

            # FAIR value: deterministic function of features + small residual
            # noise only (no seller markup). This is what the price model
            # trains to predict, and what buyer-fairness checks compare against.
            noise = RNG.normal(1.0, 0.05)
            price_per_sqft = (
                rr_rate
                * type_spec["price_mult"]
                * (1 + amenity_premium)
                * (1 + metro_premium * 0.15)
                * (builder_premium if not is_plot else 1.0)
                * age_discount
                * noise
            )
            total_price = price_per_sqft * carpet_area

            # ASKING value: what a seller actually lists at — typically a markup
            # over fair value (negotiation room, urgency premium, or genuine
            # overpricing), which is exactly the gap a price-fairness check
            # should be able to detect.
            markup = RNG.uniform(-0.03, 0.12)
            asking_price_per_sqft = price_per_sqft * (1 + markup)
            asking_total_price = asking_price_per_sqft * carpet_area

            stamp_duty_pct = STAMP_DUTY_BY_STATE.get(city, 0.06)
            reg_charge_pct = REGISTRATION_CHARGE_BY_STATE.get(city, 0.01)
            stamp_duty_amt = total_price * stamp_duty_pct
            reg_charge_amt = min(total_price * reg_charge_pct, 30000)

            base_yield = {"affordable": 0.032, "mid": 0.028, "mid-premium": 0.024, "premium": 0.019}[tier]
            yield_volatility = {"affordable": 0.009, "mid": 0.006, "mid-premium": 0.004, "premium": 0.0025}[tier]
            rental_yield_pct = 0.0 if is_plot else max(0.010, RNG.normal(base_yield, yield_volatility))
            monthly_rent = 0.0 if is_plot else (total_price * rental_yield_pct) / 12
            rent_markup = 0.0 if is_plot else RNG.uniform(-0.04, 0.10)
            monthly_rent_asking = monthly_rent * (1 + rent_markup)

            rows.append({
                "property_id": prop_id,
                "city": city,
                "locality": locality,
                "pincode": pincode,
                "city_tier": city_tier,
                "tier": tier,
                "property_type": property_type,
                "metro_distance_km": round(metro_km + RNG.normal(0, 0.15), 2),
                "bhk": bhk,
                "carpet_area_sqft": round(carpet_area, 1),
                "age_years": int(age_years),
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
                "asking_price_per_sqft": round(asking_price_per_sqft, 0),
                "asking_total_price": round(asking_total_price, 0),
                "stamp_duty_pct": stamp_duty_pct,
                "stamp_duty_amt": round(stamp_duty_amt, 0),
                "registration_charge_amt": round(reg_charge_amt, 0),
                "monthly_rent_est": round(monthly_rent, 0),
                "monthly_rent_asking": round(monthly_rent_asking, 0),
                "rental_yield_pct": round(rental_yield_pct * 100, 2),
                "vastu_compliant": int(RNG.random() > 0.4),
                "gated_community": int(len(amenities) >= 6),
                "rentable": int(not is_plot),
            })

    df = pd.DataFrame(rows)
    return df


def generate_appreciation_trend(df_localities: pd.DataFrame, years=5) -> pd.DataFrame:
    trend_rows = []
    localities = df_localities[["city", "locality", "pincode", "tier"]].drop_duplicates()
    base_growth = {"affordable": 0.09, "mid": 0.075, "mid-premium": 0.06, "premium": 0.045}
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
    props = generate_properties(n_per_market=80)
    trend = generate_appreciation_trend(props, years=5)

    props.to_csv("/home/claude/prism/data/properties.csv", index=False)
    trend.to_csv("/home/claude/prism/data/appreciation_trend.csv", index=False)

    print(f"Generated {len(props)} properties across {props['locality'].nunique()} micro-markets, {props['city'].nunique()} cities")
    print(f"City tier breakdown:\n{props.groupby('city_tier')['city'].nunique()}")
    print(f"Property type breakdown:\n{props['property_type'].value_counts()}")
    print(f"Generated {len(trend)} appreciation trend rows")
