<div align="center">

# 🏠 PRISM
### Property Risk, Intelligence, Score & Monitoring

**A decision-support platform for Indian residential real estate — a unified property graph, four ML modules, and a single bureau-style PRISM Score, feeding two audience-specific surfaces.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-multipage-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-price%20%2F%20fraud-107C41)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Random%20Forest%20%2F%20SVD-F7931E?logo=scikitlearn&logoColor=white)

</div>

---

## 🖥 What you're looking at

```mermaid
flowchart LR
    subgraph INPUTS["📥 Inputs"]
        direction TB
        I1["Registry & RERA data"]
        I2["Listing photos/text"]
        I3["Circle-rate / infra data"]
        I4["Rental yield data"]
    end

    INPUTS --> GRAPH["🗺 Unified Property Graph<br/>geo-indexed by pincode / micro-market"]

    GRAPH --> M1["Price/Rent Predictor<br/>XGBoost + SHAP"]
    GRAPH --> M2["Fraud Detector<br/>XGBoost + TF-IDF"]
    GRAPH --> M3["Construction Monitor<br/>CV pipeline"]
    GRAPH --> M4["Yield Recommender<br/>Collaborative filtering"]

    M1 --> SCORE["⭐ Unified PRISM Score<br/>300–900"]
    M2 --> SCORE
    M3 --> SCORE
    M4 --> SCORE

    SCORE --> S1["🏡 Buyer App<br/>Buy/Rent"]
    SCORE --> S2["📊 Investor Dashboard<br/>Rental / Purchase"]

    style GRAPH fill:#0a1525,stroke:#00d4ff,color:#dce8f5
    style SCORE fill:#1a1206,stroke:#d4963a,color:#f0e4d0
    style S1 fill:#0a1525,stroke:#61dafb,color:#dce8f5
    style S2 fill:#0a1525,stroke:#61dafb,color:#dce8f5
```

---

## ⭐ Unified PRISM Score

The centerpiece: every property gets **one bureau-style score (300–900)**, combining four weighted components computed from the graph and the four ML modules feeding into it.

```mermaid
flowchart TB
    subgraph SCORE["Unified PRISM Score — 300 to 900"]
        direction LR
        C1["Price Fairness<br/>30%<br/>asking vs. model-fair value"]
        C2["Trust<br/>30%<br/>inverse fraud prob. + RERA/builder rep"]
        C3["Delivery Risk<br/>15%<br/>construction schedule adherence"]
        C4["Investment Value<br/>25%<br/>yield, appreciation, stability"]
    end

    style SCORE fill:#120c04,stroke:#d4963a,color:#f0e4d0
```

**Bands:** Excellent (750+) · Good (650–749) · Fair (550–649) · Needs Review (<550)

The score is deliberately calibrated to discriminate rather than cluster everyone at the top: fair value is a clean function of features with no seller markup baked in, while listings carry a realistic asking-price markup (roughly -3% to +12%) — so price fairness actually varies across listings instead of trivially matching by construction. Price Fairness is computed separately for the sale-side and rent-side listing when both exist.

---

## 📊 Feature Map

| Module | Approach | Key metric |
|---|---|---|
| 💰 **Price Prediction** | XGBoost regressor, pincode/micro-market/property-type features, SHAP explainability | MAPE 4.9%, R² 0.99 |
| 🏘 **Rent Prediction** | Same feature schema, separate model trained on rentable inventory | MAPE 20.7%, R² 0.85 |
| 📈 **Rental Yield & Investment Recommender** | Risk-profile scoring + collaborative filtering (SVD) | 4 investor personas |
| 🏗 **Construction Progress Monitoring** | Classical CV feature extraction + Random Forest classifier | 5-stage classification |
| 🚩 **Fraud Detection in Listings** | XGBoost classifier + TF-IDF/structural duplicate detection | AUC ~0.99, sale + rent listings |

> Rent prediction carries a meaningfully higher MAPE than sale price — this mirrors real rental markets, where landlord-level idiosyncrasy adds noise that locality/property features alone don't fully explain, unlike sale prices which track circle-rate anchors more tightly.

---

## 🗺 Coverage

- **18 cities** across Tier 1 (Mumbai, Bangalore, Delhi NCR, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad), Tier 2 (Jaipur, Lucknow, Chandigarh, Indore, Kochi, Surat), and Tier 3 (Bhubaneswar, Raipur, Ranchi, Dehradun)
- **68 micro-markets**, **7 property types** (Apartment, Villa, Independent House, Penthouse, Studio/1RK, Row House, Plot/Land)
- **5,440 properties**, **~8,400 listings** split across sale and rental markets

---

## 🖱 Surfaces

| Surface | Highlights | Stack |
|---|---|---|
| 🏡 **Buyer App** | Single-property lookup, Buy/Rent toggle, property-type filter, plain-language score breakdown for whichever mode is selected | Streamlit |
| 📊 **Investor Dashboard** | Rental Income view (yield/rent-trust weighted) + Purchase/Appreciation view (appreciation/price-fairness/delivery weighted) | Streamlit |
| 🔍 **Module deep-dives** | Sidebar pages 3–6 — each underlying model explorable on its own | Streamlit |

---

## 🚀 Quick Start

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

---

## 📁 Repository Structure

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

---

## 🏗 Architecture Notes

```mermaid
flowchart LR
    A["Raw inputs<br/>registry, RERA, listings, infra"] --> B["Unified Property Graph<br/>geo-indexed"]
    B --> C["4 ML modules<br/>trained independently"]
    C --> D["Unified PRISM Score<br/>weighted composite"]
    D --> E["Buyer App /<br/>Investor Dashboard"]

    style B fill:#0a1525,stroke:#00d4ff,color:#dce8f5
    style D fill:#120c04,stroke:#d4963a,color:#f0e4d0
```

- **Graph-first design** — a single geo-indexed property graph (pincode / micro-market) feeds all four models, so they share one consistent view of location, rather than each module querying inputs independently.
- **Independent module training** — price/rent, fraud, construction, and yield are separate models with different algorithms (XGBoost, Random Forest, SVD) rather than one monolithic model, so each can be retrained or swapped without touching the others.
- **Score as the integration layer** — the four modules never talk to each other directly; they only meet at the weighted Unified PRISM Score, which keeps the scoring logic auditable and each module's contribution explicit.
- **Scoped-down construction module** — classical CV feature extraction stands in for a transfer-learned CNN; the build environment had no GPU and ran out of disk space installing a deep-learning framework, so the module was intentionally scoped down rather than faked.

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology | Used for |
|---|---|---|
| App framework | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Multipage app — home, buyer, investor, 4 deep-dive pages |
| Language | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Data generation, model training, orchestration |
| Price/Fraud models | ![XGBoost](https://img.shields.io/badge/-XGBoost-107C41) | Price/rent regression, fraud classification |
| Explainability | ![SHAP](https://img.shields.io/badge/-SHAP-8A2BE2) | Feature attribution for price predictions |
| Text similarity | ![TF--IDF](https://img.shields.io/badge/-TF--IDF-4B8BBE) | Structural duplicate/fraud detection in listing text |
| Construction CV | ![scikit--learn](https://img.shields.io/badge/-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white) | 5-stage construction classification from procedural imagery |
| Yield recommender | ![SVD](https://img.shields.io/badge/-Collaborative%20Filtering%20(SVD)-2E86AB) | Investor-persona yield/investment matching |
| Data | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Property graph, feature tables, generated CSVs |
| Persistence | ![Joblib](https://img.shields.io/badge/-Joblib-orange) | Saved trained models (`.joblib`) |

</div>

---

## 🧭 Data Methodology & Known Limitations

- **All data is synthetic**, generated to be calibrated to real, publicly available regulatory anchors rather than claimed as real transaction records — bulk registered-transaction data isn't available via a single public API in India (it's fragmented across state sub-registrar and RERA portals), so prices are calibrated to realistic Ready Reckoner/circle-rate bands per locality instead.
- **Regulatory patterns are real** — RERA registration patterns, builder tiers, stamp duty rates by state, and amenity premium structures are modeled on real, publicly documented norms.
- **Fraud labels are synthetic** — built by injecting realistic fraud patterns onto the synthetic listing base, not sourced from real fraud cases.
- **Construction imagery is procedurally generated**, not real drone/satellite photos; classical CV feature extraction stands in for a transfer-learned CNN due to build-environment constraints (no GPU, insufficient disk space for a deep-learning framework).
- **This is a methodology demonstration** — treat PRISM as a demonstration of modeling methodology and system architecture, **not** a real property valuation, fraud detection, or investment tool.

<div align="center">

---

*Built with Streamlit · XGBoost · SHAP · scikit-learn · pandas*

</div>
