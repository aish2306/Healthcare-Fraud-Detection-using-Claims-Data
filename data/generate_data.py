"""Generate synthetic stand-ins that match the REAL open-dataset schemas.

Run:  python -m data.generate_data        (writes CSV/JSON into data/synth/)

The data is engineered so that fraud labels correlate with detectable signals
(coding errors, charge outliers, excluded/risky providers). That is what lets
the full pipeline catch things on camera and exercises every detection stage

Swap in the real Kaggle/CMS/LEIE CSVs by pointing the loaders at them (see
loaders.py). Columns match data/schemas.py.
"""

import json
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "synth")
os.makedirs(OUT, exist_ok=True)

SPECIALTIES = ["Internal Medicine", "Cardiology", "Orthopedics", "Radiology", "Dermatology", "Podiatry"]
REGIONS = ["NE", "SE", "MW", "SW", "W", "NW"]
INSURANCE = ["Medicare", "Medicaid", "Private", "Self-Pay"]

# CPT codes with a "typical" charge per specialty-ish context, plus units norms.
CPT = {
    "99213": ("Office visit, low/moderate", 130),
    "99215": ("Office visit, high complexity", 270),
    "70553": ("MRI brain w/ & w/o contrast", 1400),
    "80053": ("Comprehensive metabolic panel", 50),
    "80048": ("Basic metabolic panel", 28),
    "93000": ("Electrocardiogram, complete", 45),
    "97110": ("Therapeutic exercise", 42),
    "27447": ("Total knee arthroplasty", 23000),
    "20610": ("Joint injection, major", 210),
    "11042": ("Debridement, skin/subcut", 180),
}
CPT_LIST = list(CPT.keys())
ICD10 = ["E11.9", "I10", "M54.5", "J06.9", "Z00.00", "R51.9", "J45.909", "S82.101A", "M17.11", "L03.115"]

# Diagnoses that legitimately support each procedure (aligned to policy coverage),
# so the necessity agent fires selectively rather than on every claim.
PLAUSIBLE_DX = {
    "99213": ["I10", "E11.9", "M54.5", "J06.9"], "99215": ["I10", "E11.9", "M54.5"],
    "70553": ["R51.9"],  # headache: covered only AFTER the 2025-07-01 policy revision (temporal showcase)
    "80053": ["E11.9"], "80048": ["E11.9"], "93000": ["I10"],
    "97110": ["M54.5", "S82.101A"], "27447": ["M17.11"],
    "20610": ["M17.11", "M54.5"], "11042": ["L03.115"],
}
# Non-covering diagnoses to inject a genuine medical-necessity failure
NONCOVER_DX = ["J06.9", "Z00.00", "I10"]

# NCCI-style bundled / mutually-exclusive pairs (column_2 should not be billed with column_1)
NCCI_PAIRS = [("80053", "80048"), ("99215", "99213")]


def luhn_npi(base9: str) -> str:
    """Compute a valid NPI (10-digit) check digit using the CMS 80840 prefix + Luhn."""
    s = "80840" + base9
    total, alt = 0, True  # double starting from rightmost of the 14-digit string
    for ch in reversed(s):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    check = (10 - (total % 10)) % 10
    return base9 + str(check)


def make_providers(n=60):
    rows = []
    for i in range(n):
        base9 = "".join(str(d) for d in RNG.integers(0, 10, size=9))
        npi = luhn_npi(base9)
        rows.append({
            "Provider_ID": f"P-{1000+i}",
            "NPI": npi,
            "Specialty": RNG.choice(SPECIALTIES),
            "Region": RNG.choice(REGIONS),
            "Facility_ID": f"F-{RNG.integers(1, 21):02d}",
            # latent fraud propensity: most providers clean, a handful risky
            "_bad": 1 if i < 6 else 0,
        })
    df = pd.DataFrame(rows)
    # mark 3 of the bad providers as OIG-excluded
    df["_excluded"] = 0
    bad_idx = df.index[df["_bad"] == 1].tolist()
    for j in bad_idx[:3]:
        df.loc[j, "_excluded"] = 1
    return df


def gen_claims(providers, n=1500):
    rows = []
    bad_npis = set(providers.loc[providers["_bad"] == 1, "NPI"])
    excl_npis = set(providers.loc[providers["_excluded"] == 1, "NPI"])
    # oversample bad providers a bit so we get enough positives
    weights = np.where(providers["_bad"].values == 1, 2.0, 1.0)
    weights = weights / weights.sum()

    for i in range(n):
        p = providers.iloc[RNG.choice(len(providers), p=weights)]
        is_bad = p["NPI"] in bad_npis
        on_leie = p["NPI"] in excl_npis
        dos = np.datetime64("2025-01-01") + np.timedelta64(int(RNG.integers(0, 364)), "D")

        # choose 1-2 procedure lines
        k = RNG.integers(1, 3)
        codes = list(RNG.choice(CPT_LIST, size=k, replace=False))
        units = [int(max(1, RNG.poisson(1.2))) for _ in codes]

        fraud_type = "No Fraud"
        coding_flag = 0

        # inject coding issues on a fraction (more often for bad providers)
        if RNG.random() < (0.18 if is_bad else 0.05):
            roll = RNG.random()
            if roll < 0.5:  # duplicate line
                codes.append(codes[0]); units.append(units[0]); coding_flag = 1; fraud_type = "Unbundling"
            else:  # NCCI unbundling pair
                c1, c2 = NCCI_PAIRS[RNG.integers(0, len(NCCI_PAIRS))]
                codes = [c1, c2]; units = [1, 1]; coding_flag = 1; fraud_type = "Unbundling"

        # charges: log-normal around the typical CPT charge; bad providers inflate
        submitted = 0.0
        for c, u in zip(codes, units):
            base = CPT[c][1]
            infl = RNG.lognormal(mean=0.0, sigma=0.25)
            if is_bad:
                infl *= RNG.uniform(1.3, 2.2)  # upcoding/overcharge signal
            submitted += base * u * infl
        submitted = round(submitted, 2)

        # phantom billing: bad provider, high units, no real encounter signal
        if is_bad and RNG.random() < 0.15:
            fraud_type = "Phantom Billing"
            units = [u + int(RNG.integers(3, 8)) for u in units]
            submitted = round(submitted * RNG.uniform(1.5, 2.5), 2)

        approved = round(submitted * RNG.uniform(0.55, 0.95), 2)

        # derive diagnoses that plausibly support the procedures (selective necessity)
        dx_pool = []
        for c in codes:
            dx_pool += PLAUSIBLE_DX.get(c, ["I10"])
        dx = list(dict.fromkeys(dx_pool))[:2] or ["I10"]
        necessity_inject = False
        # inject a genuine necessity failure sometimes (more often for bad providers)
        if RNG.random() < (0.20 if is_bad else 0.04):
            dx = [RNG.choice(NONCOVER_DX)]
            necessity_inject = True

        # latent fraud probability (logistic) from the planted signals
        charge_z = min(max((submitted - 300) / 500.0, 0.0), 3.0)  # capped so big-ticket codes don't force fraud
        logit = (-3.8 + 1.8 * is_bad + 1.4 * coding_flag + 0.3 * charge_z
                 + 1.3 * necessity_inject
                 + 3.0 * on_leie + (0.8 if fraud_type == "Phantom Billing" else 0.0))
        prob = 1 / (1 + np.exp(-logit))
        is_fraud = int(RNG.random() < prob)
        if is_fraud and necessity_inject and fraud_type == "No Fraud":
            fraud_type = "Medically Unnecessary"
        if is_fraud and fraud_type == "No Fraud":
            fraud_type = "Upcoding"
        if not is_fraud:
            fraud_type = "No Fraud"

        rows.append({
            "Claim_ID": f"CLM-{100000+i}",
            "Provider_ID": p["Provider_ID"],
            "NPI": p["NPI"],
            "Patient_ID": f"M-{RNG.integers(1, 4000)}",
            "Patient_Age": int(RNG.integers(1, 95)),
            "Patient_Gender": RNG.choice(["M", "F"]),
            "Insurance_Type": RNG.choice(INSURANCE, p=[0.45, 0.2, 0.3, 0.05]),
            "Date_Of_Service": np.datetime_as_string(dos, unit="D"),
            "ICD10_Codes": ";".join(dx),
            "CPT_Codes": ";".join(codes),
            "Units": ";".join(str(u) for u in units),
            "Submitted_Amount": submitted,
            "Approved_Amount": approved,
            "Is_Fraud": is_fraud,
            "Fraud_Type": fraud_type,
        })
    return pd.DataFrame(rows)


def gen_provider_baselines(providers, claims):
    """CMS Part B-style provider x HCPCS aggregates (peer baseline vault)."""
    recs = []
    exploded = []
    for _, c in claims.iterrows():
        for code, u in zip(c["CPT_Codes"].split(";"), c["Units"].split(";")):
            exploded.append({"NPI": c["NPI"], "HCPCS_Code": code,
                             "charge": c["Submitted_Amount"] / max(len(c["CPT_Codes"].split(";")), 1),
                             "units": int(u)})
    ex = pd.DataFrame(exploded).merge(providers[["NPI", "Provider_ID", "Specialty", "Region"]], on="NPI")
    grp = ex.groupby(["NPI", "Provider_ID", "Specialty", "Region", "HCPCS_Code"])
    agg = grp.agg(Avg_Submitted_Charge=("charge", "mean"),
                  Total_Services=("units", "sum")).reset_index()
    agg["Avg_Medicare_Payment"] = (agg["Avg_Submitted_Charge"] * 0.72).round(2)
    agg["Avg_Submitted_Charge"] = agg["Avg_Submitted_Charge"].round(2)
    return agg


def gen_leie(providers):
    recs = []
    for _, p in providers[providers["_excluded"] == 1].iterrows():
        recs.append({
            "NPI": p["NPI"],
            "Provider_Name": f"Provider {p['Provider_ID']}",
            "Exclusion_Type": RNG.choice(["1128(a)(1) Medicare fraud", "1128(a)(2) Patient abuse"]),
            "Exclusion_Date": "2024-06-01",
            "Reinstatement_Date": "",
        })
    return pd.DataFrame(recs)


def gen_policies():
    policies = [
        {"policy_id": "POL-IMG-001", "title": "Advanced brain imaging (MRI/CT)",
         "procedure_codes": ["70553"], "covered_dx_prefixes": ["G43", "I63", "S06", "C71", "G40"],
         "effective_date": "2024-01-01", "end_date": "2025-06-30",
         "text": "Advanced brain imaging is covered for migraine with complications, stroke, intracranial injury, neoplasm, or seizure. Uncomplicated headache (R51) requires failed conservative care. Routine exams (Z00) do not support necessity. Superseded 2025-07-01."},
        {"policy_id": "POL-IMG-002", "title": "Advanced brain imaging (MRI/CT) - revised",
         "procedure_codes": ["70553"], "covered_dx_prefixes": ["G43", "I63", "S06", "C71", "G40", "R51"],
         "effective_date": "2025-07-01", "end_date": "2026-12-31",
         "text": "Revised policy effective 2025-07-01: advanced brain imaging covered for the listed neurologic diagnoses and for persistent headache (R51) after documented conservative management. Routine exams (Z00) remain non-covered."},
        {"policy_id": "POL-EM-001", "title": "E/M level selection",
         "procedure_codes": ["99213", "99214", "99215"], "covered_dx_prefixes": ["*"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "E/M level must match documented complexity. A level-5 visit (99215) requires high-complexity decision-making; routine or preventive encounters (Z00) do not support level 5 and indicate upcoding."},
        {"policy_id": "POL-LAB-001", "title": "Metabolic panel testing",
         "procedure_codes": ["80053", "80048"], "covered_dx_prefixes": ["E11", "E10", "N18", "*"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Comprehensive (80053) includes all components of basic (80048); both should not be billed together. Covered for metabolic, diabetic, and renal monitoring."},
        {"policy_id": "POL-ORTH-001", "title": "Total knee arthroplasty",
         "procedure_codes": ["27447"], "covered_dx_prefixes": ["M17"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Total knee arthroplasty (27447) is covered for osteoarthritis of the knee (M17) after documented failed conservative therapy. High-dollar; subject to prepay review."},
        {"policy_id": "POL-INJ-001", "title": "Major joint injection",
         "procedure_codes": ["20610"], "covered_dx_prefixes": ["M17", "M54", "M25"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Major joint injection (20610) covered for joint osteoarthritis or effusion with documented indication."},
        {"policy_id": "POL-DERM-001", "title": "Skin debridement",
         "procedure_codes": ["11042"], "covered_dx_prefixes": ["L03", "L97", "I83"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Debridement (11042) covered for documented cellulitis, ulcer, or wound requiring removal of devitalized tissue."},
        {"policy_id": "POL-CARD-001", "title": "Electrocardiogram",
         "procedure_codes": ["93000"], "covered_dx_prefixes": ["I10", "I20", "I48", "R00"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "ECG (93000) covered for hypertension, arrhythmia, chest pain, or palpitations; repeated same-day ECGs require documentation."},
        {"policy_id": "POL-PT-001", "title": "Therapeutic exercise",
         "procedure_codes": ["97110"], "covered_dx_prefixes": ["M54", "M25", "S82"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Therapeutic exercise (97110) covered under an active plan of care for documented musculoskeletal impairment."},
        {"policy_id": "POL-RESP-001", "title": "Nebulizer / respiratory",
         "procedure_codes": ["94640"], "covered_dx_prefixes": ["J44", "J45", "J06"],
         "effective_date": "2024-01-01", "end_date": "2026-12-31",
         "text": "Respiratory treatment covered for asthma, COPD, or acute respiratory infection with documented obstruction."},
    ]
    return policies


def main():
    providers = make_providers()
    claims = gen_claims(providers)
    baselines = gen_provider_baselines(providers, claims)
    leie = gen_leie(providers)
    policies = gen_policies()

    # write public-facing files (drop the latent _bad/_excluded helper cols from claims)
    claims.to_csv(os.path.join(OUT, "claims.csv"), index=False)
    baselines.to_csv(os.path.join(OUT, "provider_baselines.csv"), index=False)
    leie.to_csv(os.path.join(OUT, "leie.csv"), index=False)
    providers.drop(columns=["_bad", "_excluded"]).to_csv(os.path.join(OUT, "providers.csv"), index=False)
    with open(os.path.join(OUT, "policies.json"), "w") as f:
        json.dump({"policies": policies}, f, indent=2)

    fr = claims["Is_Fraud"].mean()
    print(f"claims.csv            {len(claims)} rows, fraud rate {fr:.1%}")
    print(f"provider_baselines    {len(baselines)} rows")
    print(f"leie.csv              {len(leie)} excluded providers")
    print(f"policies.json         {len(policies)} policies")
    print(f"Fraud_Type counts:\n{claims['Fraud_Type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
