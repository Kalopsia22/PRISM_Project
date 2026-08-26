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

Rental yield is calibrated to the Magicbricks Rental Index (Jan-Mar 2026
report), the one real external source used in this project — it publishes
trailing rental yields for Chennai (4.87%), Kolkata (4.81%), Bengaluru
(4.19%), and Hyderabad (4.06%), which are fit against those cities' average
circle rates and applied at the locality level. See the citation in
generate_properties() for the fitted formula.
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
    ("Delhi NCR", "Vasant Kunj", "110070", 15000, 22000, 0.8, "premium", "Tier1"),
    ("Delhi NCR", "Rohini", "110085", 7000, 10000, 2.5, "mid", "Tier1"),
    ("Delhi NCR", "Greater Noida West", "201308", 5000, 7500, 3.5, "affordable", "Tier1"),
    ("Delhi NCR", "Indirapuram", "201014", 6000, 8500, 2.8, "mid", "Tier1"),
    ("Delhi NCR", "Saket", "110017", 16000, 23000, 0.9, "premium", "Tier1"),
    ("Delhi NCR", "Faridabad Sector 21", "121001", 5500, 8000, 3.0, "affordable", "Tier1"),
    ("Delhi NCR", "Golf Course Road Gurgaon", "122002", 18000, 26000, 0.6, "premium", "Tier1"),
    ("Chennai", "Adyar", "600020", 12000, 17000, 0.8, "premium", "Tier1"),
    ("Chennai", "OMR Sholinganallur", "600119", 6000, 9000, 2.0, "mid", "Tier1"),
    ("Chennai", "Anna Nagar", "600040", 10000, 14000, 1.0, "mid-premium", "Tier1"),
    ("Chennai", "Velachery", "600042", 7000, 10000, 1.8, "mid", "Tier1"),
    ("Chennai", "T Nagar", "600017", 13000, 18000, 0.7, "premium", "Tier1"),
    ("Chennai", "Porur", "600116", 6000, 8500, 2.5, "mid", "Tier1"),
    ("Chennai", "Tambaram", "600045", 4500, 6500, 3.5, "affordable", "Tier1"),
    ("Chennai", "Perungudi", "600096", 6500, 9000, 2.0, "mid", "Tier1"),
    ("Chennai", "Adambakkam", "600088", 6000, 8500, 2.2, "mid", "Tier1"),
    ("Chennai", "ECR", "600041", 9000, 13000, 3.0, "mid-premium", "Tier1"),
    ("Hyderabad", "Jubilee Hills", "500033", 13000, 19000, 0.7, "premium", "Tier1"),
    ("Hyderabad", "Gachibowli", "500032", 7500, 11000, 1.8, "mid-premium", "Tier1"),
    ("Hyderabad", "Kukatpally", "500072", 5500, 8000, 2.2, "mid", "Tier1"),
    ("Hyderabad", "Banjara Hills", "500034", 14000, 20000, 0.6, "premium", "Tier1"),
    ("Hyderabad", "Madhapur", "500081", 8500, 12000, 1.5, "mid-premium", "Tier1"),
    ("Hyderabad", "Miyapur", "500049", 5000, 7500, 2.8, "affordable", "Tier1"),
    ("Hyderabad", "Kondapur", "500084", 7500, 10500, 1.8, "mid-premium", "Tier1"),
    ("Hyderabad", "Uppal", "500039", 4500, 6500, 3.2, "affordable", "Tier1"),
    ("Hyderabad", "Manikonda", "500089", 6000, 8500, 2.2, "mid", "Tier1"),
    ("Hyderabad", "Secunderabad", "500003", 6500, 9500, 1.5, "mid", "Tier1"),
    ("Pune", "Koregaon Park", "411001", 11000, 16000, 0.9, "premium", "Tier1"),
    ("Pune", "Hinjewadi", "411057", 6000, 9000, 3.0, "mid", "Tier1"),
    ("Pune", "Viman Nagar", "411014", 8500, 12000, 1.4, "mid-premium", "Tier1"),
    ("Pune", "Baner", "411045", 8000, 11500, 1.8, "mid-premium", "Tier1"),
    ("Pune", "Wakad", "411057", 6500, 9500, 2.2, "mid", "Tier1"),
    ("Pune", "Kothrud", "411038", 8500, 12000, 1.5, "mid-premium", "Tier1"),
    ("Pune", "Kharadi", "411014", 7000, 10000, 2.0, "mid", "Tier1"),
    ("Pune", "Aundh", "411007", 9000, 13000, 1.2, "mid-premium", "Tier1"),
    ("Pune", "Hadapsar", "411028", 5500, 8000, 2.8, "affordable", "Tier1"),
    ("Pune", "Wagholi", "412207", 4500, 6500, 3.5, "affordable", "Tier1"),
    ("Kolkata", "Ballygunge", "700019", 10000, 15000, 0.6, "premium", "Tier1"),
    ("Kolkata", "Salt Lake", "700064", 6500, 9500, 1.5, "mid", "Tier1"),
    ("Kolkata", "New Town", "700156", 5000, 7500, 2.5, "affordable", "Tier1"),
    ("Kolkata", "Behala", "700034", 4500, 6500, 2.8, "affordable", "Tier1"),
    ("Kolkata", "Rajarhat", "700135", 5000, 7500, 2.5, "mid", "Tier1"),
    ("Kolkata", "Alipore", "700027", 12000, 17000, 0.7, "premium", "Tier1"),
    ("Kolkata", "Garia", "700084", 4200, 6000, 3.0, "affordable", "Tier1"),
    ("Kolkata", "Dum Dum", "700028", 4500, 6500, 2.5, "affordable", "Tier1"),
    ("Kolkata", "Tollygunge", "700033", 5500, 8000, 1.8, "mid", "Tier1"),
    ("Kolkata", "Park Street", "700016", 13000, 18500, 0.6, "premium", "Tier1"),
    ("Ahmedabad", "Satellite", "380015", 6500, 9500, 1.2, "mid-premium", "Tier1"),
    ("Ahmedabad", "Bopal", "380058", 4500, 6500, 2.5, "mid", "Tier1"),
    ("Ahmedabad", "SG Highway", "380054", 5500, 8000, 1.8, "mid", "Tier1"),
    ("Ahmedabad", "Navrangpura", "380009", 6500, 9500, 1.0, "mid-premium", "Tier1"),
    ("Ahmedabad", "Prahlad Nagar", "380015", 7000, 10000, 1.3, "mid-premium", "Tier1"),
    ("Ahmedabad", "Maninagar", "380008", 4200, 6000, 2.5, "affordable", "Tier1"),
    ("Ahmedabad", "Vastrapur", "380015", 6800, 9800, 1.2, "mid-premium", "Tier1"),
    ("Ahmedabad", "Chandkheda", "382424", 4000, 5800, 3.0, "affordable", "Tier1"),
    ("Ahmedabad", "Thaltej", "380059", 6500, 9200, 1.5, "mid", "Tier1"),
    ("Ahmedabad", "Gota", "382481", 4200, 6200, 2.8, "affordable", "Tier1"),
    # ---- Tier 2 ----
    ("Jaipur", "Vaishali Nagar", "302021", 4500, 6500, 2.0, "mid", "Tier2"),
    ("Jaipur", "Malviya Nagar", "302017", 5000, 7000, 1.5, "mid", "Tier2"),
    ("Jaipur", "Mansarovar", "302020", 3800, 5500, 2.8, "affordable", "Tier2"),
    ("Jaipur", "C-Scheme", "302001", 6500, 9200, 0.8, "mid-premium", "Tier2"),
    ("Jaipur", "Jagatpura", "302017", 3800, 5500, 2.8, "affordable", "Tier2"),
    ("Jaipur", "Tonk Road", "302015", 4200, 6200, 2.0, "mid", "Tier2"),
    ("Jaipur", "Ajmer Road", "302006", 3600, 5200, 3.2, "affordable", "Tier2"),
    ("Jaipur", "Raja Park", "302004", 4500, 6500, 1.5, "mid", "Tier2"),
    ("Jaipur", "Jhotwara", "302012", 3500, 5000, 3.0, "affordable", "Tier2"),
    ("Lucknow", "Gomti Nagar", "226010", 4200, 6000, 2.0, "mid", "Tier2"),
    ("Lucknow", "Hazratganj", "226001", 5500, 7500, 1.0, "mid-premium", "Tier2"),
    ("Lucknow", "Indira Nagar", "226016", 3500, 5000, 2.5, "affordable", "Tier2"),
    ("Lucknow", "Aliganj", "226024", 3800, 5500, 2.2, "mid", "Tier2"),
    ("Lucknow", "Alambagh", "226005", 3200, 4800, 2.8, "affordable", "Tier2"),
    ("Lucknow", "Vibhuti Khand", "226010", 4500, 6500, 1.5, "mid", "Tier2"),
    ("Lucknow", "Mahanagar", "226006", 4000, 5800, 1.8, "mid", "Tier2"),
    ("Lucknow", "Chinhat", "226028", 2800, 4200, 3.5, "affordable", "Tier2"),
    ("Lucknow", "Vikas Nagar", "226022", 3500, 5000, 2.5, "affordable", "Tier2"),
    ("Chandigarh", "Sector 22", "160022", 7000, 10000, 1.0, "mid-premium", "Tier2"),
    ("Chandigarh", "Sector 43", "160043", 5500, 8000, 1.8, "mid", "Tier2"),
    ("Chandigarh", "Zirakpur", "140603", 4000, 5800, 3.0, "affordable", "Tier2"),
    ("Chandigarh", "Sector 17", "160017", 8000, 11500, 0.5, "premium", "Tier2"),
    ("Chandigarh", "Mohali Phase 7", "160062", 5500, 8000, 2.0, "mid", "Tier2"),
    ("Chandigarh", "Panchkula Sector 8", "134109", 6000, 8800, 1.8, "mid", "Tier2"),
    ("Chandigarh", "Sector 35", "160035", 6500, 9200, 1.2, "mid-premium", "Tier2"),
    ("Chandigarh", "Kharar", "140301", 3800, 5500, 3.5, "affordable", "Tier2"),
    ("Chandigarh", "Sector 9", "160009", 7500, 10500, 0.9, "mid-premium", "Tier2"),
    ("Indore", "Vijay Nagar", "452010", 4200, 6000, 1.5, "mid", "Tier2"),
    ("Indore", "Palasia", "452001", 5000, 7200, 1.0, "mid-premium", "Tier2"),
    ("Indore", "Rau", "453331", 3200, 4500, 3.5, "affordable", "Tier2"),
    ("Indore", "Bhawarkuan", "452001", 4000, 5800, 2.0, "mid", "Tier2"),
    ("Indore", "Sudama Nagar", "452009", 3500, 5000, 2.5, "affordable", "Tier2"),
    ("Indore", "AB Road", "452008", 4800, 6800, 1.5, "mid", "Tier2"),
    ("Indore", "Bengali Square", "452010", 3800, 5500, 2.2, "mid", "Tier2"),
    ("Indore", "Rajendra Nagar", "452012", 3200, 4600, 3.0, "affordable", "Tier2"),
    ("Indore", "Scheme 78", "452010", 4500, 6500, 1.8, "mid", "Tier2"),
    ("Kochi", "Kakkanad", "682030", 4500, 6500, 2.0, "mid", "Tier2"),
    ("Kochi", "Marine Drive", "682031", 6500, 9500, 0.8, "mid-premium", "Tier2"),
    ("Kochi", "Edappally", "682024", 4000, 5800, 2.5, "affordable", "Tier2"),
    ("Kochi", "Panampilly Nagar", "682036", 8000, 11500, 0.9, "premium", "Tier2"),
    ("Kochi", "Vyttila", "682019", 5500, 8000, 1.5, "mid", "Tier2"),
    ("Kochi", "Kaloor", "682017", 5800, 8200, 1.3, "mid", "Tier2"),
    ("Kochi", "Fort Kochi", "682001", 6500, 9500, 1.8, "mid-premium", "Tier2"),
    ("Kochi", "Aluva", "683101", 3800, 5500, 3.0, "affordable", "Tier2"),
    ("Kochi", "Thrikkakara", "682021", 4500, 6500, 2.2, "mid", "Tier2"),
    ("Surat", "Vesu", "395007", 4800, 7000, 1.8, "mid", "Tier2"),
    ("Surat", "Adajan", "395009", 4200, 6000, 2.0, "mid", "Tier2"),
    ("Surat", "City Light", "395007", 5500, 7800, 1.2, "mid-premium", "Tier2"),
    ("Surat", "Piplod", "395007", 5500, 7800, 1.5, "mid", "Tier2"),
    ("Surat", "Althan", "395017", 4800, 6800, 2.0, "mid", "Tier2"),
    ("Surat", "Ghod Dod Road", "395001", 5800, 8200, 1.2, "mid-premium", "Tier2"),
    ("Surat", "Katargam", "395004", 3500, 5000, 3.0, "affordable", "Tier2"),
    ("Surat", "Pal", "395009", 4500, 6500, 2.2, "mid", "Tier2"),
    ("Surat", "Bhatar", "395017", 4200, 6000, 2.5, "affordable", "Tier2"),
    # ---- Tier 3 ----
    ("Bhubaneswar", "Patia", "751024", 3200, 4500, 2.5, "mid", "Tier3"),
    ("Bhubaneswar", "Saheed Nagar", "751007", 3800, 5200, 1.5, "mid", "Tier3"),
    ("Bhubaneswar", "Chandrasekharpur", "751016", 2800, 4000, 3.0, "affordable", "Tier3"),
    ("Bhubaneswar", "Jaydev Vihar", "751013", 4200, 6000, 1.8, "mid", "Tier3"),
    ("Bhubaneswar", "Nayapalli", "751012", 3800, 5500, 2.2, "mid", "Tier3"),
    ("Bhubaneswar", "Khandagiri", "751030", 3000, 4400, 3.0, "affordable", "Tier3"),
    ("Bhubaneswar", "Rasulgarh", "751010", 2800, 4000, 3.2, "affordable", "Tier3"),
    ("Bhubaneswar", "Kalinga Nagar", "751003", 3500, 5000, 2.5, "affordable", "Tier3"),
    ("Bhubaneswar", "Old Town", "751002", 2600, 3800, 3.5, "affordable", "Tier3"),
    ("Raipur", "Shankar Nagar", "492007", 3000, 4200, 2.0, "mid", "Tier3"),
    ("Raipur", "Telibandha", "492006", 2600, 3800, 2.8, "affordable", "Tier3"),
    ("Raipur", "VIP Road", "492001", 3400, 4800, 1.5, "mid", "Tier3"),
    ("Raipur", "Civil Lines", "492001", 3800, 5200, 1.5, "mid", "Tier3"),
    ("Raipur", "Pandri", "492004", 2600, 3800, 2.8, "affordable", "Tier3"),
    ("Raipur", "Devendra Nagar", "492009", 2800, 4000, 2.5, "affordable", "Tier3"),
    ("Raipur", "Amlidih", "492001", 2400, 3600, 3.2, "affordable", "Tier3"),
    ("Raipur", "Samta Colony", "492009", 3000, 4400, 2.0, "mid", "Tier3"),
    ("Raipur", "GE Road", "492001", 3200, 4600, 1.8, "mid", "Tier3"),
    ("Ranchi", "Lalpur", "834001", 2800, 4000, 2.0, "mid", "Tier3"),
    ("Ranchi", "Harmu", "834002", 2400, 3500, 2.5, "affordable", "Tier3"),
    ("Ranchi", "Kanke Road", "834006", 2200, 3200, 3.0, "affordable", "Tier3"),
    ("Ranchi", "Doranda", "834002", 2600, 3800, 2.5, "affordable", "Tier3"),
    ("Ranchi", "Bariatu", "834009", 2800, 4000, 2.0, "mid", "Tier3"),
    ("Ranchi", "Ashok Nagar", "834002", 3000, 4400, 1.8, "mid", "Tier3"),
    ("Ranchi", "Hinoo", "834002", 2400, 3600, 3.0, "affordable", "Tier3"),
    ("Ranchi", "Circular Road", "834001", 3200, 4600, 1.5, "mid", "Tier3"),
    ("Ranchi", "Kokar", "834001", 2200, 3200, 3.2, "affordable", "Tier3"),
    ("Dehradun", "Rajpur Road", "248001", 3800, 5500, 1.5, "mid-premium", "Tier3"),
    ("Dehradun", "Sahastradhara Road", "248013", 2800, 4000, 2.5, "mid", "Tier3"),
    ("Dehradun", "Clement Town", "248002", 2200, 3200, 3.5, "affordable", "Tier3"),
    ("Dehradun", "Chakrata Road", "248001", 3200, 4600, 2.0, "mid", "Tier3"),
    ("Dehradun", "GMS Road", "248001", 2800, 4000, 2.5, "affordable", "Tier3"),
    ("Dehradun", "Race Course", "248001", 4200, 6000, 1.2, "mid-premium", "Tier3"),
    ("Dehradun", "Vasant Vihar", "248006", 3600, 5200, 1.8, "mid", "Tier3"),
    ("Dehradun", "Ballupur", "248001", 2600, 3800, 2.8, "affordable", "Tier3"),
    ("Dehradun", "Prem Nagar", "248007", 2400, 3400, 3.2, "affordable", "Tier3"),
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


def generate_properties(n_per_market=50) -> pd.DataFrame:
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

            # ---------------------------------------------------------------
            # Rental yield, calibrated to the Magicbricks Rental Index
            # (Jan-Mar 2026 report) — the only cited external market-data
            # source in this project, chosen because it's the one place a
            # rent/price yield ratio is actually reported for Indian cities.
            # The report gives explicit trailing rental yields for four
            # cities: Chennai 4.87%, Kolkata 4.81%, Bengaluru 4.19%,
            # Hyderabad 4.06%. Yield falls as price rises (higher-priced
            # markets rent for a smaller fraction of their sale value), so
            # those four (avg circle rate, yield%) points were fit with a
            # simple linear regression — yield_pct = 6.35 - 0.00019 * price —
            # and applied at the locality level using each locality's own
            # circle rate, preserving the intra-city premium/affordable
            # spread the fitted anchors themselves show (Bengaluru's higher
            # average price maps to its lower 4.19% yield, Kolkata's lower
            # average price to its higher 4.81%).
            fitted_yield_pct = 6.35 - 0.00019 * rr_rate
            base_yield = max(0.015, min(0.06, fitted_yield_pct / 100))

            # The same report flags Bengaluru/Hyderabad as running "strong
            # momentum" (Hyderabad +15% YoY rent growth, Bengaluru +8.6% QoQ)
            # and NCR/Mumbai as past a demand surge into a softer, more
            # volatile phase (Gurugram rents -1.1% QoQ even as supply rose
            # +10.4% QoQ) — reflected here as a yield-volatility adjustment,
            # not a sale-price effect (rent growth and capital appreciation
            # are different metrics; this project doesn't conflate them).
            city_yield_vol_mult = {"Delhi NCR": 1.3, "Mumbai": 1.15, "Bangalore": 1.1, "Hyderabad": 1.1, "Kolkata": 0.85}.get(city, 1.0)
            yield_volatility_base = {"affordable": 0.009, "mid": 0.006, "mid-premium": 0.004, "premium": 0.0025}[tier]
            yield_volatility = yield_volatility_base * city_yield_vol_mult

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
    props = generate_properties(n_per_market=50)
    trend = generate_appreciation_trend(props, years=5)

    props.to_csv("/home/claude/prism/data/properties.csv", index=False)
    trend.to_csv("/home/claude/prism/data/appreciation_trend.csv", index=False)

    print(f"Generated {len(props)} properties across {props['locality'].nunique()} micro-markets, {props['city'].nunique()} cities")
    print(f"City tier breakdown:\n{props.groupby('city_tier')['city'].nunique()}")
    print(f"Property type breakdown:\n{props['property_type'].value_counts()}")
    print(f"Generated {len(trend)} appreciation trend rows")
