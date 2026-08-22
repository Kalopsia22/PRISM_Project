# PRISM — Property Risk, Intelligence, Score & Monitoring

A decision-support platform for Indian residential real estate, spanning Tier 1, Tier 2, and
Tier 3 cities, built around a **unified property graph** and a single **PRISM Score**
(300–900, bureau-style) that four ML modules feed into and two audience-specific surfaces consume.

## Architecture

```
Inputs                         Unified Layer                  ML Modules              Score & Surfaces
─────────────────────          ──────────────────────          ──────────────           ─────────────────
Registry & RERA data    ─┐
Listing photos/text     ─┼──▶  Unified Property Graph   ──▶   Price/Rent Predictor ─┐
Circle-rate / infra data─┤     (geo-indexed by pincode/         (XGBoost + SHAP)     │
Rental yield data       ─┤      micro-market)                                       │
Transaction/entity data ─┘                                   Fraud Detector         │
                                                                (XGBoost + TF-IDF)   ├─▶ Unified PRISM Score ──▶ Buyer App (Buy/Rent)
                                                              AML/Compliance Engine  │        (300–900)          Investor Dashboard
                                                                (XGBoost + Isolation │                           (Rental / Purchase)
                                                                 Forest + graph)     │
                                                              Yield Recommender      │
                                                                (Collaborative       │
                                                                 filtering)         ─┘
```

## Coverage

- **18 cities** across Tier 1 (Mumbai, Bangalore, Delhi NCR, Chennai, Hyderabad, Pune, Kolkata,
  Ahmedabad), Tier 2 (Jaipur, Lucknow, Chandigarh, Indore, Kochi, Surat), and Tier 3
  (Bhubaneswar, Raipur, Ranchi, Dehradun)
- **68 micro-markets**, **7 property types** (Apartment, Villa, Independent House, Penthouse,
  Studio/1RK, Row House, Plot/Land)
- **5,440 properties**, **~8,400 listings** split across sale and rental markets, each with an
  associated transaction/entity record for AML analysis

## Modules

| Module | Approach | Key metric |
|---|---|---|
| Price Prediction | XGBoost regressor, pincode/micro-market/property-type features, SHAP explainability | MAPE 4.9%, R² 0.99 |
| Rent Prediction | Same feature schema, separate model trained on rentable inventory | MAPE 20.7%, R² 0.85 |
| Rental Yield & Investment Recommender | Risk-profile scoring + collaborative filtering (SVD) | 4 investor personas |
| **AML & Transaction Structuring Risk** | XGBoost classifier + Isolation Forest anomaly detection + graph-based ring detection (networkx) | AUC 0.996, 8 rings detected |
| Fraud Detection in Listings | XGBoost classifier + TF-IDF/structural duplicate detection | AUC ~0.99, sale + rent listings |

Rent prediction carries a meaningfully higher MAPE than sale price — this mirrors real rental
markets, where landlord-level idiosyncrasy adds noise that locality/property features alone
don't fully explain, unlike sale prices which track circle-rate anchors more tightly.

### Why AML/Compliance instead of construction monitoring

The original module concept (satellite/drone construction progress monitoring) was scoped down
to a classical-CV proof of concept because this build environment has no GPU and ran out of disk
space installing a deep-learning framework — real satellite imagery is also too coarse (~10m
resolution) to detect construction stage in the first place. Rather than ship a weak CV demo,
this module was replaced with something more differentiated and more directly relevant to
banking/fintech risk roles: **money-laundering typology detection in real estate transactions**.

Real estate is a globally recognized laundering channel (FATF flags it as high-risk), and Indian
law gives this real regulatory teeth — PMLA 2002 requires reporting entities to flag suspicious
transactions to FIU-IND, Income Tax Act Sections 269SS/269ST effectively cap cash consideration
for property deals, and RBI/NHB KYC master directions require beneficial-ownership verification
for non-individual buyers. The module runs three complementary layers mirroring how a real bank
AML function actually works:

1. **Supervised classifier** (XGBoost) trained on 5 known typologies — undervaluation, rapid
   flips, high cash components, shell-entity buyers, and structuring/smurfing
2. **Unsupervised anomaly detection** (Isolation Forest), run independently of any label, to
   catch novel deviations the labeled typologies don't cover (correlates ~0.70 with the label
   despite never seeing it)
3. **Graph-based ring detection** (networkx strongly-connected-components) over the buyer→seller
   transaction network — finds circular trading rings, a pattern that only exists in network
   structure and can't be seen from any single transaction's features

## Unified PRISM Score

Every property gets one score (300–900) combining:
- **Price Fairness** (30%) — asking price/rent vs. model-fair value, computed separately for
  the sale-side and rent-side listing when both exist
- **Trust** (30%) — inverse listing-fraud probability + RERA/builder reputation
- **Compliance Risk** (15%) — inverse AML/transaction-structuring risk probability
- **Investment Value** (25%) — locality yield, appreciation, stability

Bands: Excellent (750+) · Good (650–749) · Fair (550–649) · Needs Review (<550)

Trust and Compliance are deliberately separate, complementary checks: Trust asks "is this
listing genuine" (fraud detection on the ad itself), while Compliance asks "is the money behind
this transaction clean" (AML risk on the underlying registered sale) — a listing can pass one
check and fail the other.

The score is deliberately calibrated to discriminate rather than cluster everyone at the top:
fair value is a clean function of features with no seller markup baked in, while listings
carry a realistic asking-price markup (roughly -3% to +12%) — so price fairness actually
varies across listings instead of trivially matching by construction.

## Surfaces

- **Buyer App** — single-property lookup with a Buy/Rent toggle, property-type filter, and a
  plain-language score breakdown for whichever mode is selected
- **Investor Dashboard** — two dedicated views: a Rental Income dashboard (yield/rent-trust
  weighted) and a Purchase/Appreciation dashboard (appreciation/price-fairness/compliance weighted)
- **Module deep-dives** (sidebar, pages 3–6) — each underlying model explorable on its own,
  including a force-directed network graph for the AML ring-detection layer

## Data methodology

All data in this project is synthetic, generated to be calibrated to real, publicly available
regulatory anchors rather than claimed as real transaction records — bulk registered-transaction
data is not available via a single public API in India (fragmented across state sub-registrar
and RERA portals), so prices are calibrated to realistic Ready Reckoner/circle-rate bands per
locality instead. RERA registration patterns, builder tiers, stamp duty rates by state, and
amenity premium structures are modeled on real, publicly documented norms. Listing-fraud labels
and AML-transaction labels are both synthetic, built by injecting realistic typologies onto the
synthetic base rather than sourced from real cases.

Treat this project as a demonstration of modeling methodology and system architecture, not a
real property valuation, fraud detection, AML screening, or investment tool.

## Running locally

```bash
pip install -r requirements.txt

# regenerate data (optional — CSVs are already included)
python data/generate_locality_data.py
python data/generate_listings_data.py
python data/generate_transactions_data.py

# retrain models (optional — .joblib files are already included)
python models/price_model.py
python models/fraud_model.py
python models/aml_risk.py
python models/yield_recommender.py
python models/unified_graph.py

# launch the app
streamlit run app.py
```

## Project structure

```
prism/
├── app.py                          # Home dashboard
├── pages/
│   ├── 1_Buyer_App.py              # Buy/Rent property lookup
│   ├── 2_Investor_Dashboard.py     # Rental income + purchase/appreciation views
│   ├── 3_Price_Prediction.py       # Price/rent model deep-dive
│   ├── 4_Rental_Yield.py           # Yield recommender deep-dive
│   ├── 5_AML_Compliance.py         # AML/transaction structuring risk deep-dive
│   └── 6_Fraud_Detection.py        # Listing fraud model deep-dive
├── data/                           # Generators + generated CSVs
├── models/                         # Training scripts + saved .joblib models
├── utils/
│   ├── styling.py                  # Shared visual identity
│   ├── charts.py                   # Reusable chart helpers (gauge, radar, map, network graph)
│   └── geo.py                      # Real city coordinates for map visuals
└── README.md
```
