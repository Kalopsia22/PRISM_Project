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
- **170 micro-markets** (9-10 per city), **7 property types** (Apartment, Villa, Independent House,
  Penthouse, Studio/1RK, Row House, Plot/Land)
- **8,500 properties**, **~13,000 listings** split across sale and rental markets, each with an
  associated transaction/entity record for AML analysis

## Modules

| Module | Approach | Key metric |
|---|---|---|
| Price Prediction | XGBoost regressor, pincode/micro-market/property-type features, SHAP explainability | MAPE 4.9%, R² 0.99 |
| Rent Prediction | Same feature schema, separate model trained on rentable inventory | MAPE 20.7%, R² 0.85 |
| Rental Yield & Investment Recommender | Risk-profile scoring + collaborative filtering (SVD), yield calibrated to Magicbricks Rental Index | 4 investor personas |
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

## Model evaluation: accuracy, precision, recall, F1, ROC-AUC

Every model-backed page shows these five classification metrics at the top, computed on a
held-out test set the model never trained on. Two of the four pages are genuine classifiers, so
the numbers mean exactly what they say; the other two are regression/ranking systems, where these
metrics don't natively apply — rather than skip them or fake a number, each uses a clearly
captioned, non-circular derived framing:

- **Fraud Detection** (genuine classifier): 97.1% accuracy, 87.9% precision, 92.4% recall, 90.1%
  F1, 0.977 ROC-AUC — fraud vs. clean listing classification.
- **AML Compliance** (genuine classifier): 95.3% accuracy, 87.7% precision, 83.4% recall, 85.5%
  F1, 0.924 ROC-AUC — flagged vs. clean transaction classification.
- **Price Prediction** (regression — no native classification target): framed as "does the model
  correctly call whether a listing is priced above or below the going rate for its market?", the
  same above/below-market test real AVMs are stress-tested on. For sale, this runs on the 13,609
  real scraped listings (never trained on) against each listing's own city's real median — and
  comes out only slightly better than chance (56.6% accuracy, 0.554 ROC-AUC), an honest result:
  the real data has no builder/amenity/age/metro fields, so a lot of the signal the model relies
  on simply isn't available for these rows. For rent, no real external data exists, so this runs
  on the model's own synthetic held-out test set instead (88.3% accuracy, 0.938 ROC-AUC) — clearly
  scoped as synthetic-only rather than implied to be real-world validation.
- **Rental Yield** (rule-based score + collaborative filtering — no native classification target):
  the risk-profile score itself is a deterministic weighted formula, so scoring it against its own
  inputs would be circular and meaningless. The one genuinely trained component — the SVD
  collaborative-filtering layer — is evaluated the standard recsys way: 20% of the simulated
  investor-locality interaction matrix is held out, hidden from the factorization, and the
  reconstruction is scored on whether it correctly classifies real vs. simulated interest for
  interactions it never saw (58.7% accuracy, 0.606 ROC-AUC — modest, reflecting how sparse a
  300-investor × 170-locality simulated matrix genuinely is).

The point of showing the weaker numbers alongside the strong ones is the same reason the fraud and
AML classifiers were deliberately recalibrated earlier in this project to avoid suspiciously
perfect scores: a model card that only ever reports good numbers isn't credible. See each page's
in-app caption for the exact framing, and `models/price_model.py` (`classification_metrics_from_
regression`) / `models/yield_recommender.py` (`evaluate_collaborative_filter`) for the code.

## Data methodology

All data in this project is synthetic, generated to be calibrated to real, publicly available
regulatory anchors rather than claimed as real transaction records — bulk registered-transaction
data is not available via a single public API in India (fragmented across state sub-registrar
and RERA portals), so prices are calibrated to realistic Ready Reckoner/circle-rate bands per
locality instead. RERA registration patterns, builder tiers, stamp duty rates by state, and
amenity premium structures are modeled on real, publicly documented norms. Listing-fraud labels
and AML-transaction labels are both synthetic, built by injecting realistic typologies onto the
synthetic base rather than sourced from real cases.

**Rental yield is calibrated directly against a live external source**: the
[Magicbricks Rental Index](https://www.magicbricks.com) (Jan–Mar 2026 report), which publishes
trailing rental yields for four cities — Chennai (4.87%), Kolkata (4.81%), Bengaluru (4.19%),
Hyderabad (4.06%). Those four (average locality price, reported yield%) pairs were fit with a
simple linear regression (`yield_pct = 6.35 - 0.00019 × circle_rate_per_sqft`) and applied at the
locality level using each locality's own circle rate — which reproduces the anchor cities' actual
yields within ~0.3-0.5pp and preserves the intra-city premium/affordable spread the anchors
themselves imply. The same report's qualitative city commentary (Bengaluru/Hyderabad running
"strong momentum", NCR/Mumbai past a demand surge into a softer, more volatile phase) sets a
per-city yield-volatility multiplier. This is scoped deliberately to yield only — rent growth and
sale-price appreciation are different metrics, and the appreciation-trend dataset (circle-rate-
driven, tier-based) is not conflated with the rent-growth figures the index reports.

**Sale prices are additionally grounded against a real scraped-listings dataset** (`data/
raw_real_estate_listings.csv`, ~14.5K rows across Bengaluru, Pune, Delhi NCR, Chennai, Kolkata,
Mumbai, Hyderabad). After cleaning (parsing price strings, filtering unparseable/outlier rows,
extracting BHK and property type — see `data/clean_real_estate_data.py`), 13,609 rows remain and
are used two ways:

1. **Locality recalibration** — 37 of PRISM's curated localities have ≥10 matching real listings
   by exact (city, locality) name. For those, the illustrative circle-rate band is replaced with
   one derived from the real data's own 25th-75th percentile price/sqft. Real listing prices are
   *final* asking prices, already including whatever amenity/builder/age premiums exist in that
   market, while PRISM's `circle_rate_per_sqft` represents a *base* rate the generator then
   multiplies by those same premium factors — so the real percentiles are deflated by ~1.337x
   (the generator's own empirically-measured average combined premium for apartments) before
   being used as the band, so the generator's markup reconstructs the real price level instead of
   compounding on top of it.
2. **Out-of-sample model validation** — the trained sale-price model (which has never seen this
   real data) is run against all 13,609 real listings and scored against their actual prices. The
   real data has no builder/amenity/age/metro-distance fields, so those are filled with neutral
   defaults matched to the synthetic training data's own feature averages; for localities with no
   name match, the circle-rate anchor falls back to that city's real-data median (also deflated by
   the same 1.337x factor) rather than PRISM's own curated locality average, which skews toward
   better-known, pricier neighborhoods. Results (predicted vs. actual, city-level breakdown) are
   shown on the Price Prediction page's "Real-World Validation" tab.

Treat this project as a demonstration of modeling methodology and system architecture, not a
real property valuation, fraud detection, AML screening, or investment tool.

## Running locally

```bash
pip install -r requirements.txt

# clean the real scraped-listings dataset first (feeds locality calibration below)
python data/clean_real_estate_data.py

# regenerate data (optional — CSVs are already included)
python data/generate_locality_data.py
python data/generate_listings_data.py
python data/generate_transactions_data.py

# retrain models (optional — .joblib files are already included; price_model.py
# also runs real-world validation as part of training, see console output)
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
├── data/                           # Generators + generated CSVs + real listings/calibration
├── models/                         # Training scripts + saved .joblib models (price_model.py
│                                    #   includes real-world validation)
├── utils/
│   ├── styling.py                  # Shared visual identity
│   ├── charts.py                   # Reusable chart helpers (gauge, radar, map, network graph)
│   └── geo.py                      # Real city coordinates for map visuals
└── README.md
```
