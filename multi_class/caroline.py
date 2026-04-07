# create_multiclass_xy_file.py

import gc
import os

import pandas as pd

# =========================================================
# Input files
# =========================================================
USE_SAMPLE = os.environ.get("USE_SAMPLE", "0") == "1"

if USE_SAMPLE:
    RAW_EHR_PATH = "gp_icd10_hesin_raw_data_with_ukb_sample.csv"
    OUTCOME_PATH = "fibrotic_gpX_hesinY_Y_visit_sample.csv"
else:
    RAW_EHR_PATH = "gp_icd10_hesin_raw_data_with_ukb.csv"
    OUTCOME_PATH = "fibrotic_gpX_hesinY_Y_visit.csv"

OUTPUT_PATH = "multiclass_outcome_file.csv"
CHUNK_SIZE = 50_000

RAW_EHR_PATH = os.environ.get("RAW_EHR_PATH", RAW_EHR_PATH)
OUTCOME_PATH = os.environ.get("OUTCOME_PATH", OUTCOME_PATH)
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", OUTPUT_PATH)

# =========================================================
# 7 disease columns
# =========================================================
DISEASE_COLS = [
    "CKD",
    "Cardiac_Fibrosis",
    "MASH",
    "Pulmonary_fibrosis",
    "SSc_Connective_Tissue",
    "Crohns_Disease",
    "Fibrosis_of_Skin",
]

CLASS_ID_MAP = {
    "CKD": 0,
    "Cardiac_Fibrosis": 1,
    "MASH": 2,
    "Pulmonary_fibrosis": 3,
    "SSc_Connective_Tissue": 4,
    "Crohns_Disease": 5,
    "Fibrosis_of_Skin": 6,
}

# Exact raw-column names after removing dots from fibrotic_codes.jpg ICD codes.
FIBROTIC_ICD_CODES = {
    "CKD": ["N18", "N181", "N182", "N183", "N184", "N185", "N189"],
    "Cardiac_Fibrosis": ["I40", "I409", "I420", "I423", "I424", "I425", "I514"],
    "MASH": ["K74", "K746", "K75", "K758", "K7581"],
    "Pulmonary_fibrosis": ["J841", "J8410", "J84112", "J8417", "J848", "J8489", "J849"],
    "SSc_Connective_Tissue": ["M34", "M340", "M341", "M342", "M348", "M349"],
    "Crohns_Disease": ["K50", "K501", "K508", "K509"],
    "Fibrosis_of_Skin": ["L905", "L91", "L910"],
}

ALL_FIBROTIC_CODES = set()
for codes in FIBROTIC_ICD_CODES.values():
    ALL_FIBROTIC_CODES.update(codes)


# =========================================================
# Helpers
# =========================================================
def parse_date_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def read_csv_flexible(path: str, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.read_csv(path, engine="python", **kwargs)


print(f"Using {'sample' if USE_SAMPLE else 'full'} data")
print(f"Raw EHR file: {RAW_EHR_PATH}")
print(f"Outcome file: {OUTCOME_PATH}")
print(f"Output file: {OUTPUT_PATH}")
print(f"Chunk size: {CHUNK_SIZE:,}")

# =========================================================
# Load outcome data and raw header
# =========================================================
outcome = read_csv_flexible(OUTCOME_PATH)
raw_header = read_csv_flexible(RAW_EHR_PATH, nrows=0)

outcome.columns = [c.strip() for c in outcome.columns]
raw_header.columns = [c.strip() for c in raw_header.columns]

required_raw = {"eid", "event_dt"}
required_outcome = {"eid", "event_dt"} | set(DISEASE_COLS)

missing_raw = required_raw - set(raw_header.columns)
missing_outcome = required_outcome - set(outcome.columns)

if missing_raw:
    raise ValueError(f"RAW file missing columns: {missing_raw}")
if missing_outcome:
    raise ValueError(f"OUTCOME file missing columns: {missing_outcome}")

outcome["eid"] = outcome["eid"].astype(str)
outcome["event_dt"] = parse_date_col(outcome["event_dt"])
outcome = outcome.dropna(subset=["eid", "event_dt"]).copy()

for c in DISEASE_COLS:
    outcome[c] = pd.to_numeric(outcome[c], errors="coerce").fillna(0).astype(int)

# =========================================================
# Step 1:
# In fibrotic_gpX_hesinY_Y_visit.csv, find each patient's earliest visit
# =========================================================
outcome = outcome.sort_values(["eid", "event_dt"]).copy()
earliest_idx = outcome.groupby("eid")["event_dt"].idxmin()
earliest_outcome = outcome.loc[earliest_idx].copy()

# =========================================================
# Step 2:
# Keep only rows where exactly ONE disease column == 1
# Exclude patients with multiple fibrotic diseases at earliest visit
# =========================================================
earliest_outcome["positive_count"] = earliest_outcome[DISEASE_COLS].sum(axis=1)
earliest_outcome = earliest_outcome.loc[earliest_outcome["positive_count"] == 1].copy()
earliest_outcome["disease"] = earliest_outcome[DISEASE_COLS].idxmax(axis=1)
earliest_outcome["class_id"] = earliest_outcome["disease"].map(CLASS_ID_MAP)

print(f"\nPatients after earliest-visit single-disease filter: {len(earliest_outcome):,}")
print(earliest_outcome["disease"].value_counts(dropna=False).sort_index())

# =========================================================
# Step 3:
# Check whether this earliest fibrotic visit is also the patient's first raw EHR visit
# If yes -> exclude
# We only keep patients whose fibrotic event is NOT their first raw visit
# =========================================================
first_raw_lookup = {}
total_rows_scanned = 0

for chunk in read_csv_flexible(
    RAW_EHR_PATH,
    usecols=["eid", "event_dt"],
    chunksize=CHUNK_SIZE,
):
    chunk.columns = [c.strip() for c in chunk.columns]
    total_rows_scanned += len(chunk)

    chunk["eid"] = chunk["eid"].astype(str)
    chunk["event_dt"] = parse_date_col(chunk["event_dt"])
    chunk = chunk.dropna(subset=["eid", "event_dt"])

    chunk_min = chunk.groupby("eid")["event_dt"].min()
    for eid, dt in chunk_min.items():
        if eid not in first_raw_lookup or dt < first_raw_lookup[eid]:
            first_raw_lookup[eid] = dt

    del chunk, chunk_min
    gc.collect()
    print(f"  scanned {total_rows_scanned:,} raw rows for first-visit lookup ...", end="\r")

first_raw_visit = pd.DataFrame(
    {
        "eid": list(first_raw_lookup.keys()),
        "first_raw_event_dt": list(first_raw_lookup.values()),
    }
)

multiclass = earliest_outcome.merge(first_raw_visit, on="eid", how="left")
multiclass = multiclass.loc[
    multiclass["event_dt"] > multiclass["first_raw_event_dt"]
].copy()
multiclass = multiclass.rename(columns={"event_dt": "diagnosis_event_dt"})

print(f"\nEligible multiclass patients with prior history: {len(multiclass):,}")

# =========================================================
# Step 4:
# Remove fibrotic ICD columns from raw EHR features
# =========================================================
raw_cols = raw_header.columns.tolist()
fibrotic_cols_found = ALL_FIBROTIC_CODES.intersection(set(raw_cols))
cols_to_keep = [c for c in raw_cols if c not in fibrotic_cols_found]
feature_cols = [c for c in cols_to_keep if c not in {"eid", "event_dt"}]

print(f"Fibrotic ICD columns found and removed: {len(fibrotic_cols_found)}")
print(f"Columns kept for X/Y rows: {len(cols_to_keep)}")

# =========================================================
# Step 5:
# Build one multiclass file containing both X rows and Y rows
# =========================================================
case_meta_cols = ["eid", "diagnosis_event_dt", "disease", "class_id", "first_raw_event_dt"] + DISEASE_COLS
case_meta = multiclass[case_meta_cols].copy()
eligible_eids = set(case_meta["eid"])

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

output_cols = (
    [
        "eid",
        "event_dt",
        "diagnosis_event_dt",
        "first_raw_event_dt",
        "record_type",
        "disease",
        "class_id",
    ]
    + DISEASE_COLS
    + feature_cols
)

written_header = False
x_row_count = 0
y_row_count = 0
x_patients = set()
y_patients = set()
total_rows_scanned = 0
total_rows_kept = 0

for chunk in read_csv_flexible(
    RAW_EHR_PATH,
    usecols=cols_to_keep,
    parse_dates=["event_dt"],
    chunksize=CHUNK_SIZE,
):
    chunk.columns = [c.strip() for c in chunk.columns]
    total_rows_scanned += len(chunk)

    chunk["eid"] = chunk["eid"].astype(str)
    chunk["event_dt"] = parse_date_col(chunk["event_dt"])
    chunk = chunk.dropna(subset=["eid", "event_dt"])

    filtered = chunk.loc[chunk["eid"].isin(eligible_eids)].copy()
    if filtered.empty:
        del chunk, filtered
        gc.collect()
        print(f"  processed {total_rows_scanned:,} raw rows, wrote {total_rows_kept:,} X/Y rows ...", end="\r")
        continue

    filtered = filtered.merge(case_meta, on="eid", how="left")

    x_rows = filtered.loc[filtered["event_dt"] < filtered["diagnosis_event_dt"]].copy()
    if not x_rows.empty:
        x_rows["record_type"] = "x_row"
        x_row_count += len(x_rows)
        x_patients.update(x_rows["eid"].unique().tolist())

    y_rows = filtered.loc[filtered["event_dt"] == filtered["diagnosis_event_dt"]].copy()
    if not y_rows.empty:
        y_rows["record_type"] = "y_row"
        y_row_count += len(y_rows)
        y_patients.update(y_rows["eid"].unique().tolist())

    frames_to_write = []
    if not x_rows.empty:
        frames_to_write.append(x_rows[output_cols])
    if not y_rows.empty:
        frames_to_write.append(y_rows[output_cols])

    if frames_to_write:
        batch_df = pd.concat(frames_to_write, ignore_index=True)
        total_rows_kept += len(batch_df)
        batch_df.to_csv(
            OUTPUT_PATH,
            index=False,
            mode="a",
            header=(not written_header),
        )
        written_header = True
        del batch_df

    del chunk, filtered, x_rows, y_rows
    gc.collect()
    print(f"  processed {total_rows_scanned:,} raw rows, wrote {total_rows_kept:,} X/Y rows ...", end="\r")

# =========================================================
# Summary
# =========================================================
if written_header:
    print(f"\nSaved multiclass X+Y file to: {OUTPUT_PATH}")
else:
    print(f"\nNo eligible X/Y rows were written to: {OUTPUT_PATH}")
print(f"Eligible patients kept: {len(multiclass):,}")
print(f"Patients with >=1 x_row written: {len(x_patients):,}")
print(f"Patients with >=1 y_row written: {len(y_patients):,}")
print(f"Total x_rows: {x_row_count:,}")
print(f"Total y_rows: {y_row_count:,}")
print(f"Total output rows: {x_row_count + y_row_count:,}")

print("\nCounts by disease:")
print(multiclass["disease"].value_counts(dropna=False).sort_index())

if written_header and os.path.exists(OUTPUT_PATH):
    preview = read_csv_flexible(OUTPUT_PATH, nrows=10)
    print("\nPreview:")
    print(preview)
