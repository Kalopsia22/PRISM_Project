"""
PRISM Data Layer — Transaction & Entity Dataset Generator
=============================================================
Generates one representative registered-transaction record per property,
drawn from a pool of synthetic buyer/seller entities, with injected
money-laundering typologies commonly flagged in Indian real estate:

  1. Undervaluation — declared/registered value materially below the
     circle-rate-implied fair value (the gap is the classic "on-money"
     cash component used to evade stamp duty and income tax)
  2. Rapid flip — property resold within a very short holding period at
     a large markup (layering: fast-cycling money through property)
  3. High cash component — cash portion of the consideration well above
     what Indian tax law practically allows for property (Income Tax Act
     Section 269SS/269ST restrict cash for property dealings)
  4. Structuring/smurfing — the same buyer entity conducts multiple
     purchases in a short rolling window, each kept individually small
  5. Shell-entity buyer — a newly incorporated company/LLP/trust with a
     generic, template-sounding name and no other visible footprint
  6. Circular transaction ring — a small cluster of entities repeatedly
     trading properties among themselves (classic layering ring; detected
     via graph analysis in models/aml_risk.py, not injected as a flat flag)

This mirrors the fraud-detection module's design (synthetic base,
regulation-grounded patterns) applied to transaction-level, not
listing-level, risk — the two are complementary: listing fraud asks "is
this ad real", transaction AML risk asks "is this money clean".
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(23)

ENTITY_TYPES = ["Individual", "Pvt Ltd Company", "LLP", "Partnership Firm", "Trust", "NRI Individual"]
ENTITY_TYPE_WEIGHTS = [0.62, 0.14, 0.08, 0.06, 0.04, 0.06]

INDIVIDUAL_FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Kavya",
                     "Arjun", "Divya", "Karan", "Neha", "Suresh", "Meera", "Ajay", "Pooja"]
INDIVIDUAL_LAST = ["Sharma", "Patel", "Reddy", "Iyer", "Singh", "Nair", "Gupta", "Menon",
                    "Kulkarni", "Rao", "Verma", "Pillai", "Joshi", "Desai"]

SHELL_PREFIXES = ["Sunrise", "Alpine", "Zenith", "Vertex", "Horizon", "Pinnacle", "Meridian",
                   "Cascade", "Novus", "Orbis", "Summit", "Crestline"]
SHELL_SUFFIXES = ["Trading", "Ventures", "Enterprises", "Holdings", "Consultancy", "Infra", "Realty"]
SHELL_ENTITY_SUFFIX = ["Pvt Ltd", "LLP"]


def build_entity_pool(n_individuals=5000, n_shell_ring=15, n_structuring_pool=25):
    """Most entities are ordinary individual buyers/sellers, each transacting
    only once or twice — a large pool keeps buyer-reuse rare by construction,
    so genuine structuring (the same handful of entities buying repeatedly)
    stands out instead of being buried in organic reuse. Two small dedicated
    pools simulate the injected patterns: a ring pool for circular trading,
    and a structuring pool for entities deliberately reused across several
    purchases."""
    entities = []
    entity_id = 1

    for _ in range(n_individuals):
        etype = RNG.choice(ENTITY_TYPES, p=ENTITY_TYPE_WEIGHTS)
        if etype in ("Individual", "NRI Individual"):
            name = f"{RNG.choice(INDIVIDUAL_FIRST)} {RNG.choice(INDIVIDUAL_LAST)}"
        else:
            name = f"{RNG.choice(SHELL_PREFIXES)} {RNG.choice(SHELL_SUFFIXES)} {RNG.choice(SHELL_ENTITY_SUFFIX)}"
        entities.append({
            "entity_id": entity_id, "entity_name": name, "entity_type": etype,
            "pan_hash": f"PAN{entity_id:06d}", "is_ring_member": 0, "is_structuring_member": 0, "ring_id": None,
            "incorporation_age_days": int(RNG.integers(30, 4000)) if etype != "Individual" else None,
        })
        entity_id += 1

    # ring pool: 3 rings of 5 entities each, deliberately shell-like
    n_rings = n_shell_ring // 5
    for ring in range(n_rings):
        for _ in range(5):
            name = f"{RNG.choice(SHELL_PREFIXES)} {RNG.choice(SHELL_SUFFIXES)} {RNG.choice(SHELL_ENTITY_SUFFIX)}"
            entities.append({
                "entity_id": entity_id, "entity_name": name, "entity_type": RNG.choice(["Pvt Ltd Company", "LLP"]),
                "pan_hash": f"PAN{entity_id:06d}", "is_ring_member": 1, "is_structuring_member": 0, "ring_id": ring,
                "incorporation_age_days": int(RNG.integers(15, 200)),
            })
            entity_id += 1

    # structuring pool: entities that will be deliberately reused across
    # several purchases to simulate smurfing
    for _ in range(n_structuring_pool):
        name = f"{RNG.choice(INDIVIDUAL_FIRST)} {RNG.choice(INDIVIDUAL_LAST)}"
        entities.append({
            "entity_id": entity_id, "entity_name": name, "entity_type": "Individual",
            "pan_hash": f"PAN{entity_id:06d}", "is_ring_member": 0, "is_structuring_member": 1, "ring_id": None,
            "incorporation_age_days": None,
        })
        entity_id += 1

    return pd.DataFrame(entities)


def build_transactions(properties: pd.DataFrame, entities: pd.DataFrame, flag_rate=0.13) -> pd.DataFrame:
    general_pool = entities[(entities["is_ring_member"] == 0) & (entities["is_structuring_member"] == 0)]
    ring_pool = entities[entities["is_ring_member"] == 1]
    structuring_pool = entities[entities["is_structuring_member"] == 1]

    n = len(properties)
    n_ring_txns = int(n * 0.04)  # ~4% of properties involved in ring trading
    ring_property_ids = set(RNG.choice(properties["property_id"].values, size=n_ring_txns, replace=False))

    rows = []
    txn_id = 700000

    for _, prop in properties.iterrows():
        txn_id += 1
        is_ring_txn = prop["property_id"] in ring_property_ids

        if is_ring_txn:
            ring_id = RNG.integers(0, ring_pool["ring_id"].nunique())
            ring_entities = ring_pool[ring_pool["ring_id"] == ring_id]
            buyer = ring_entities.sample(1, random_state=RNG.integers(0, 1_000_000)).iloc[0]
            seller = ring_entities[ring_entities["entity_id"] != buyer["entity_id"]].sample(
                1, random_state=RNG.integers(0, 1_000_000)).iloc[0]
        else:
            buyer = general_pool.sample(1, random_state=RNG.integers(0, 1_000_000)).iloc[0]
            seller = general_pool.sample(1, random_state=RNG.integers(0, 1_000_000)).iloc[0]

        fair_value = prop["total_price"]

        # baseline: clean transaction assumptions
        declared_value = fair_value * RNG.normal(1.0, 0.04)
        cash_pct = max(0, RNG.normal(3, 2))
        is_flip = RNG.random() < 0.08  # some organic quick resales exist even when clean
        holding_days = int(RNG.integers(15, 90)) if is_flip else int(RNG.integers(365, 4000))
        financing_type = RNG.choice(["Bank Loan", "NBFC Loan", "Self-funded", "Mixed"], p=[0.45, 0.15, 0.25, 0.15])

        aml_flag = 0
        pattern_types = []

        if is_ring_txn:
            aml_flag = 1
            pattern_types.append("circular_ring")
            declared_value = fair_value * RNG.uniform(0.55, 0.75)  # undervalued to mask true flow
            cash_pct = RNG.uniform(30, 55)
            holding_days = int(RNG.integers(20, 120))  # rapid cycling within the ring

        rows.append({
            "transaction_id": txn_id, "property_id": prop["property_id"],
            "buyer_entity_id": buyer["entity_id"], "buyer_name": buyer["entity_name"],
            "buyer_type": buyer["entity_type"], "buyer_pan_hash": buyer["pan_hash"],
            "buyer_incorporation_age_days": buyer["incorporation_age_days"],
            "seller_entity_id": seller["entity_id"], "seller_name": seller["entity_name"],
            "seller_pan_hash": seller["pan_hash"],
            "fair_value": round(fair_value, 0), "declared_value": round(declared_value, 0),
            "cash_component_pct": round(cash_pct, 1), "holding_period_days": holding_days,
            "financing_type": financing_type, "is_ring_txn": int(is_ring_txn),
            "aml_flag": aml_flag, "pattern_types": ",".join(pattern_types) if pattern_types else "none",
        })

    df = pd.DataFrame(rows)

    # additional injected patterns on top of the clean/ring base: undervaluation,
    # rapid flip + markup, high cash, structuring, shell buyer — sampled from the
    # remaining non-ring transactions
    remaining_idx = df[df["is_ring_txn"] == 0].index
    n_more_flags = int(len(df) * flag_rate) - df["aml_flag"].sum()
    n_more_flags = max(0, n_more_flags)
    flag_idx = RNG.choice(remaining_idx, size=min(n_more_flags, len(remaining_idx)), replace=False)
    chunks = np.array_split(flag_idx, 4)

    # 1. Undervaluation
    for idx in chunks[0]:
        df.loc[idx, "declared_value"] = round(df.loc[idx, "fair_value"] * RNG.uniform(0.55, 0.78), 0)
        df.loc[idx, "aml_flag"] = 1
        df.loc[idx, "pattern_types"] = "undervaluation"

    # 2. Rapid flip with markup
    for idx in chunks[1]:
        df.loc[idx, "holding_period_days"] = int(RNG.integers(15, 75))
        df.loc[idx, "declared_value"] = round(df.loc[idx, "fair_value"] * RNG.uniform(1.15, 1.45), 0)
        df.loc[idx, "aml_flag"] = 1
        df.loc[idx, "pattern_types"] = "rapid_flip"

    # 3. High cash component
    for idx in chunks[2]:
        df.loc[idx, "cash_component_pct"] = round(RNG.uniform(25, 60), 1)
        df.loc[idx, "aml_flag"] = 1
        df.loc[idx, "pattern_types"] = "high_cash"

    # 4. Shell-entity buyer (swap in a newly-incorporated shell-style entity)
    shell_candidates = entities[(entities["entity_type"].isin(["Pvt Ltd Company", "LLP"]))
                                  & (entities["incorporation_age_days"] < 250)
                                  & (entities["is_ring_member"] == 0)]
    for idx in chunks[3]:
        shell = shell_candidates.sample(1, random_state=RNG.integers(0, 1_000_000)).iloc[0]
        df.loc[idx, "buyer_entity_id"] = shell["entity_id"]
        df.loc[idx, "buyer_name"] = shell["entity_name"]
        df.loc[idx, "buyer_type"] = shell["entity_type"]
        df.loc[idx, "buyer_pan_hash"] = shell["pan_hash"]
        df.loc[idx, "buyer_incorporation_age_days"] = shell["incorporation_age_days"]
        df.loc[idx, "aml_flag"] = 1
        df.loc[idx, "pattern_types"] = "shell_entity_buyer"

    # 5. Structuring/smurfing — reassign small clusters of otherwise-unflagged
    # transactions to a dedicated pool of entities that each appear 4-6 times,
    # each purchase kept individually unremarkable (near-fair-value, low cash)
    remaining_after_4 = df[(df["aml_flag"] == 0) & (df["is_ring_txn"] == 0)].index
    n_structuring_entities = len(structuring_pool)
    txns_per_entity = 5
    n_structuring_txns = min(n_structuring_entities * txns_per_entity, len(remaining_after_4))
    structuring_idx = RNG.choice(remaining_after_4, size=n_structuring_txns, replace=False)
    idx_groups = np.array_split(structuring_idx, n_structuring_entities)

    for group, (_, entity) in zip(idx_groups, structuring_pool.iterrows()):
        for idx in group:
            df.loc[idx, "buyer_entity_id"] = entity["entity_id"]
            df.loc[idx, "buyer_name"] = entity["entity_name"]
            df.loc[idx, "buyer_type"] = entity["entity_type"]
            df.loc[idx, "buyer_pan_hash"] = entity["pan_hash"]
            df.loc[idx, "buyer_incorporation_age_days"] = entity["incorporation_age_days"]
            df.loc[idx, "aml_flag"] = 1
            df.loc[idx, "pattern_types"] = "structuring"

    df["buyer_txn_count"] = df["buyer_pan_hash"].map(df["buyer_pan_hash"].value_counts())

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["undervaluation_pct"] = (df["fair_value"] - df["declared_value"]) / df["fair_value"] * 100
    df["value_deviation_pct"] = (df["declared_value"] - df["fair_value"]) / df["fair_value"] * 100
    df["is_rapid_flip"] = (df["holding_period_days"] < 120).astype(int)
    df["is_shell_pattern"] = (
        df["buyer_type"].isin(["Pvt Ltd Company", "LLP"])
        & df["buyer_incorporation_age_days"].fillna(9999).lt(250)
    ).astype(int)
    return df


if __name__ == "__main__":
    props = pd.read_csv("/home/claude/prism/data/properties.csv")
    entities = build_entity_pool(n_individuals=5000, n_shell_ring=15, n_structuring_pool=25)
    txns = build_transactions(props, entities, flag_rate=0.13)
    txns = add_engineered_features(txns)

    entities.to_csv("/home/claude/prism/data/entities.csv", index=False)
    txns.to_csv("/home/claude/prism/data/transactions.csv", index=False)

    print(f"Generated {len(entities)} entities ({entities['is_ring_member'].sum()} ring members)")
    print(f"Generated {len(txns)} transactions")
    print(f"AML flag rate: {txns['aml_flag'].mean():.2%}")
    print(txns["pattern_types"].value_counts())
