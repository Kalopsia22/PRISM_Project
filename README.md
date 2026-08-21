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
Rental yield data       ─┘      micro-market)                                       │
                                                              Fraud Detector         │
                                                                (XGBoost + TF-IDF)   ├─▶ Unified PRISM Score ──▶ Buyer App (Buy/Rent)
                                                              Construction Monitor   │        (300–900)          Investor Dashboard
                                                                (CV pipeline)        │                           (Rental / Purchase)
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
- **5,440 properties**, **~8,400 listings** split across sale and rental markets

## Modules

| Module | Approach | Key metric |
|---|---|---|
| Price Prediction | XGBoost regressor, pincode/micro-market/property-type features, SHAP explainability | MAPE 4.9%, R² 0.99 |
| Rent Prediction | Same feature schema, separate model trained on rentable inventory | MAPE 20.7%, R² 0.85 |
| Rental Yield & Investment Recommender | Risk-profile scoring + collaborative filtering (SVD) | 4 investor personas |
| Construction Progress Monitoring | Classical CV feature extraction + Random Forest classifier | 5-stage classification |
| Fraud Detection in Listings | XGBoost classifier + TF-IDF/structural duplicate detection | AUC ~0.99, sale + rent listings |

Rent prediction carries a meaningfully higher MAPE than sale price — this mirrors real rental
markets, where landlord-level idiosyncrasy adds noise that locality/property features alone
don't fully explain, unlike sale prices which track circle-rate anchors more tightly.

## Unified PRISM Score

Every property gets one score (300–900) combining:
- **Price Fairness** (30%) — asking price/rent vs. model-fair value, computed separately for
  the sale-side and rent-side listing when both exist
- **Trust** (30%) — inverse fraud probability + RERA/builder reputation
- **Delivery Risk** (15%) — construction schedule adherence
- **Investment Value** (25%) — locality yield, appreciation, stability

Bands: Excellent (750+) · Good (650–749) · Fair (550–649) · Needs Review (<550)

The score is deliberately calibrated to discriminate rather than cluster everyone at the top:
fair value is a clean function of features with no seller markup baked in, while listings
carry a realistic asking-price markup (roughly -3% to +12%) — so price fairness actually
varies across listings instead of trivially matching by construction.

## Surfaces

- **Buyer App** — single-property lookup with a Buy/Rent toggle, property-type filter, and a
  plain-language score breakdown for whichever mode is selected
- **Investor Dashboard** — two dedicated views: a Rental Income dashboard (yield/rent-trust
  weighted) and a Purchase/Appreciation dashboard (appreciation/price-fairness/delivery weighted)
- **Module deep-dives** (sidebar, pages 3–6) — each underlying model explorable on its own

## Data methodology

All data in this project is synthetic, generated to be calibrated to real, publicly available
regulatory anchors rather than claimed as real transaction records — bulk registered-transaction
data is not available via a single public API in India (fragmented across state sub-registrar
and RERA portals), so prices are calibrated to realistic Ready Reckoner/circle-rate bands per
locality instead. RERA registration patterns, builder tiers, stamp duty rates by state, and
amenity premium structures are modeled on real, publicly documented norms. Fraud labels are
synthetic, built by injecting realistic fraud patterns onto the synthetic listing base.
Construction monitoring uses procedurally generated stage imagery (not real drone/satellite
photos) with classical CV feature extraction standing in for a transfer-learned CNN — this
build environment had no GPU and ran out of disk space installing a deep-learning framework,
so the module was scoped down rather than faked.

Treat this project as a demonstration of modeling methodology and system architecture, not a
real property valuation, fraud detection, or investment tool.

## Running locally

```bash
pip install -r requirements.txt

# regenerate data (optional — CSVs are already included)
python data/generate_locality_data.py
python data/generate_listings_data.py

# retrain models (optional — .joblib files are already included)
python models/price_model.py
python models/fraud_model.py
python models/construction_monitor.py
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
│   ├── 5_Construction_Monitoring.py# Construction CV deep-dive
│   └── 6_Fraud_Detection.py        # Fraud model deep-dive
├── data/                           # Generators + generated CSVs
├── models/                         # Training scripts + saved .joblib models
├── utils/styling.py                # Shared visual identity
└── README.md
```
