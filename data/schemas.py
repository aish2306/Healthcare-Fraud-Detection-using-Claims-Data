"""Schema reference: how the synthetic stand-ins map onto the real open datasets.

The synthetic generator (generate_data.py) emits files whose columns match the
REAL datasets below, so you can drop a real download in place of the stand-in by
pointing the loader at it (see loaders.py, env vars). Nothing about the agents
changes — only the file path.

----------------------------------------------------------------------------
1. claims.csv  -- matches: Kaggle "Healthcare Fraud Detection" (Nudrat Abbas)
                  and NHIS "Healthcare Claims and Fraud" (Boniface Chosen)
   columns:
     Claim_ID, Provider_ID, NPI, Patient_ID, Patient_Age, Patient_Gender,
     Insurance_Type, Date_Of_Service, ICD10_Codes (semicolon-sep),
     CPT_Codes (semicolon-sep), Units (semicolon-sep), Submitted_Amount,
     Approved_Amount, Is_Fraud (0/1), Fraud_Type
   Fraud_Type in {No Fraud, Phantom Billing, Upcoding, Unbundling}  (NHIS-style typology)

2. provider_baselines.csv -- matches: CMS "Medicare Physician & Other
                             Practitioners" (Part B) provider-level aggregates
   columns:
     NPI, Provider_ID, Specialty, Region, HCPCS_Code,
     Avg_Submitted_Charge, Avg_Medicare_Payment, Total_Services

3. leie.csv -- matches: OIG "List of Excluded Individuals/Entities" (LEIE)
   columns:
     NPI, Provider_Name, Exclusion_Type, Exclusion_Date, Reinstatement_Date

4. policies.json -- matches: coverage policy corpus (CMS Medicare Coverage
                    Database / Synthea-derived guidelines) with effective windows
   each policy: policy_id, title, procedure_codes[], covered_dx_prefixes[],
                effective_date, end_date, text
----------------------------------------------------------------------------

Real-CSV column aliases the loader will accept (so real downloads map cleanly):
"""

# Accepted real-world column aliases -> canonical name used in code.
CLAIMS_ALIASES = {
    "Provider_ID": ["Provider_ID", "ProviderID", "Provider", "Rndrng_NPI"],
    "NPI": ["NPI", "Rndrng_NPI", "provider_npi"],
    "Date_Of_Service": ["Date_Of_Service", "DOS", "Service_Date", "ClaimStartDt"],
    "ICD10_Codes": ["ICD10_Codes", "Diagnosis", "ICD10", "DiagnosisCodes"],
    "CPT_Codes": ["CPT_Codes", "Procedure", "CPT", "HCPCS", "ProcedureCodes"],
    "Submitted_Amount": ["Submitted_Amount", "Claim_Amount", "Billed", "InscClaimAmtReimbursed"],
    "Approved_Amount": ["Approved_Amount", "Approved", "Paid_Amount"],
    "Is_Fraud": ["Is_Fraud", "PotentialFraud", "Fraud", "Label"],
}
