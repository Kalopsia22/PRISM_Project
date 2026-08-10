# PRISM — Property Risk, Intelligence, Score & Monitoring

A decision-support platform for Indian residential real estate, built around a **unified
property graph** and a single **PRISM Score** (300–900, bureau-style) that four ML modules
feed into and three audience-specific surfaces consume.

## Architecture

```
Inputs                         Unified Layer                  ML Modules              Score & Surfaces
─────────────────────          ──────────────────────          ──────────────           ─────────────────
Registry & RERA data    ─┐
Listing photos/text     ─┼──▶  Unified Property Graph   ──▶   Price Predictor    ─┐
Circle-rate / infra data─┤     (geo-indexed by pincode/         (XGBoost + SHAP)   │
Rental yield data       ─┘      micro-market)                                     │
                                                              Fraud Detector       │
                                                                (XGBoost + TF-IDF) ├─▶ Unified PRISM Score ──▶ Buyer App
                                                              Construction Monitor │        (300–900)          Investor Dashboard
                                                                (CV pipeline)      │                           Lender API
                                                              Yield Recommender    │
                                                                (Collaborative     │
                                                                 filtering)       ─┘
```

## Modules

| Module | Approach | Key metric |
|---|---|---|
| Price Prediction | XGBoost regressor, pincode/micro-market features, SHAP explainability | MAPE 4.5%, R² 0.98 |
| Rental Yield & Investment Recommender | Risk-profile scoring + collaborative filtering (SVD) | 4 investor personas |
| Construction Progress Monitoring | Classical CV feature extraction + Random Forest classifier | 5-stage classification |
| Fraud Detection in Listings | XGBoost classifier + TF-IDF/cosine duplicate detection | AUC 0.98 |

## Unified PRISM Score

Every property gets one score (300–900) combining:
- **Price Fairness** (30%) — asking price vs. model-fair value
- **Trust** (30%) — inverse fraud probability + RERA/builder reputation
- **Delivery Risk** (15%) — construction schedule adherence
- **Investment Value** (25%) — locality yield, appreciation, stability

Bands: Excellent (750+) · Good (650–749) · Fair (550–649) · Needs Review (<550)

## Surfaces

- **Buyer App** — single-property lookup with plain-language score breakdown
- **Investor Dashboard** — portfolio-style filtering by risk profile, blending yield with trust
- **Lender API** — simulated `GET /v1/properties/{id}/disbursement-check` endpoint for
  construction-linked loan tranche decisions (APPROVE / CONDITIONAL_APPROVE / HOLD)
- **Module deep-dives** (sidebar, pages 4–7) — each underlying model explorable on its own

## ⚠️ Data disclosure — read this before citing numbers from this project

**All data in this project is synthetic**, generated to be *calibrated to real, publicly
available regulatory anchors* rather than claimed as real transaction records:

- Bulk registered-transaction-price data is **not available via a single public API** in
  India — it's fragmented across state sub-registrar (IGRS) and RERA portals with
  inconsistent formats. This project uses synthetic prices calibrated to realistic
  Ready Reckoner / circle-rate bands per locality instead.
- RERA registration patterns, builder tiers, stamp duty (~6% Maharashtra, ~5.5% Karnataka),
  and amenity premium structures (15–25%) are modeled on real, publicly documented norms.
- Fraud labels are synthetic, built by injecting realistic fraud patterns (duplicate
  reposts, bait pricing, fake listings, scam-ring broker reuse, recycled descriptions)
  onto the synthetic listing base.
- **Construction monitoring uses procedurally generated stage imagery, not real drone or
  satellite photos.** Real satellite imagery (Sentinel-2, ~10m resolution) is too coarse
  to detect construction stage; production-grade monitoring needs sub-meter commercial
  imagery or drone footage, neither freely available in this build environment. Classical
  CV feature extraction stands in for a transfer-learned CNN (no GPU/deep-learning
  framework available here — production path noted below).
- The Lender API is a **simulated** response — no real lending decision engine is connected.

Treat this project as a demonstration of **modeling methodology and system architecture**,
not a real property valuation, fraud detection, or lending tool.

## Production path (what would change with real infrastructure/data access)

- Price/Fraud: replace synthetic base with licensed listing-aggregator data or scraped +
  legally-reviewed RERA/IGRS data feeds
- Construction Monitoring: swap classical CV features for a transfer-learned CNN
  (ResNet/EfficientNet) trained on real drone imagery; add temporal tracking across visits
- Lender API: real auth, real construction-verification data source (inspector reports or
  drone imagery pipeline), audit logging per RBI disbursement-linked lending norms
- Yield Recommender: replace simulated investor-interaction matrix with real user
  interaction/click data once the platform has actual users

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
python models/unified_graph.py

# launch the app
streamlit run app.py
```

## Project structure

```
prism/
├── app.py                          # Home dashboard
├── pages/
│   ├── 1_Buyer_App.py              # Consumer-facing property lookup
│   ├── 2_Investor_Dashboard.py     # Portfolio-style investment view
│   ├── 3_Lender_API.py             # Simulated disbursement-decision API
│   ├── 4_Price_Prediction.py       # Price model deep-dive
│   ├── 5_Rental_Yield.py           # Yield recommender deep-dive
│   ├── 6_Construction_Monitoring.py# Construction CV deep-dive
│   └── 7_Fraud_Detection.py        # Fraud model deep-dive
├── data/                           # Generators + generated CSVs
├── models/                         # Training scripts + saved .joblib models
├── utils/styling.py                # Shared visual identity
└── README.md
```
