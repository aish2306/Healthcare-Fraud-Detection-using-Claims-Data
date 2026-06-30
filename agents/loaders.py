"""Loaders: read the synthetic stand-ins by default, or a real CSV if you point
an environment variable at it. Column aliases (data/schemas.py) are normalized so
real Kaggle/CMS/LEIE downloads map onto the canonical names the agents expect.

Env overrides:
    CLAIMS_CSV, BASELINES_CSV, LEIE_CSV, POLICIES_JSON
"""

import json
import os

import pandas as pd

from data.schemas import CLAIMS_ALIASES

HERE = os.path.dirname(os.path.dirname(__file__))
SYNTH = os.path.join(HERE, "data", "synth")


def _norm_columns(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    rename = {}
    for canonical, alts in aliases.items():
        for a in alts:
            if a in df.columns and a != canonical:
                rename[a] = canonical
                break
    return df.rename(columns=rename)


def load_claims() -> pd.DataFrame:
    path = os.environ.get("CLAIMS_CSV", os.path.join(SYNTH, "claims.csv"))
    df = pd.read_csv(path, dtype={"NPI": str})
    return _norm_columns(df, CLAIMS_ALIASES)


def load_baselines() -> pd.DataFrame:
    path = os.environ.get("BASELINES_CSV", os.path.join(SYNTH, "provider_baselines.csv"))
    return pd.read_csv(path, dtype={"NPI": str})


def load_leie() -> pd.DataFrame:
    path = os.environ.get("LEIE_CSV", os.path.join(SYNTH, "leie.csv"))
    try:
        return pd.read_csv(path, dtype={"NPI": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["NPI", "Provider_Name", "Exclusion_Type", "Exclusion_Date", "Reinstatement_Date"])


def load_policies() -> list:
    path = os.environ.get("POLICIES_JSON", os.path.join(SYNTH, "policies.json"))
    with open(path) as f:
        return json.load(f)["policies"]


def load_providers() -> pd.DataFrame:
    path = os.environ.get("PROVIDERS_CSV", os.path.join(SYNTH, "providers.csv"))
    return pd.read_csv(path, dtype={"NPI": str})
