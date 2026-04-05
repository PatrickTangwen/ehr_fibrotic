# Development Document: 7 Disease-Specific Dataset Creation Pipeline

## 1. Project Overview

This pipeline builds 7 binary classification datasets (one per fibrotic disease) from UK Biobank EHR data. Each dataset contains **Case** patients (y=1, diagnosed with that disease) and **Control** patients (y=0, diagnosed with a different fibrotic disease). The goal is to predict whether a patient will develop a specific fibrotic disease based on their prior visit history, using models like HAN and GraphSAGE.

**Implementation**: `create_disease_datasets.ipynb`

---

## 2. Input Files

| File | Description | Columns | Size |
|---|---|---|---|
| `fibrotic_gpX_hesinY_Y_visit.csv` | Disease visit labels | 9 cols: `eid`, `event_dt`, 7 disease flags | Small |
| `gp_icd10_hesin_raw_data_with_ukb.csv` | Raw EHR data with UKB features | 2007 cols: `eid`, `event_dt`, ~1801 ICD codes, ~204 UKB static features | Very large (causes OOM if loaded at once) |
| `fibrotic_codes.jpg` | Reference table: 7 diseases and their ICD-10 codes | — | Image |
| Sample versions (`*_sample.csv`) | Small subsets for development/testing | Same schema | Small |

### Raw EHR Column Structure
- **Columns 1-2**: `eid` (patient ID), `event_dt` (visit date)
- **Columns 3-1803**: ICD-10 code indicator columns (binary 0/1), e.g., `A010`, `A045`, `N182`, `I420`
- **Columns 1804-2007**: UK Biobank static features (format: `6142.0.0`, `4653.0.0`, etc.), constant per patient across visits

---

## 3. 7 Target Diseases & ICD Code Mapping

ICD codes from `fibrotic_codes.jpg` are converted to column format by removing dots:

| Disease | Affected Organ | ICD Codes (column format) |
|---|---|---|
| CKD | Kidney | N018, N181, N182, N183, N184, N185, N189 |
| Cardiac_Fibrosis | Heart | I040, I409, I420, I423, I424, I425, I514 |
| MASH | Liver | K074, K746, K75, K758, K7581 |
| Pulmonary_fibrosis | Lung | J841, J8410, J84112, J8417, J848, J8489, J849 |
| SSc_Connective_Tissue | Multi-organ | M034, M340, M341, M342, M348, M349 |
| Crohns_Disease | Gut | K050, K501, K508, K509 |
| Fibrosis_of_Skin | Skin | L905, L091, L910 |

**Total: 39 codes to remove from features.** Only **exact string matches** are removed — e.g., `I421` (I42.1) is NOT a fibrotic code and must be kept.

---

## 4. Pipeline Steps

### Step 1: Assign Each Patient to ONE Disease

**Input**: `fibrotic_gpX_hesinY_Y_visit.csv`

**Logic**:
1. Sort by `eid` and `event_dt`
2. For each patient, take the **earliest** visit
3. At that visit, identify which disease column = 1 → assign that disease
4. **Edge case**: if multiple diseases = 1 at the earliest visit → **skip this patient entirely**

**Output**: `patient_assignment` dict: `{eid: (disease_name, diagnosis_date)}`

**Example**:
```
1000527, 2002-12-23, Pulmonary_fibrosis=1  →  assigned Pulmonary_fibrosis
1000971, 2007-08-18, CKD=1                 →  assigned CKD (later visit ignored)
1002442, 2010-07-25, Crohns_Disease=1       →  assigned Crohns_Disease
```

### Step 2: Filter Case Patients — Must Have Prior EHR History

**Input**: `gp_icd10_hesin_raw_data_with_ukb.csv` (only `eid` + `event_dt` columns, read in chunks)

**Logic**:
1. For each patient, find their **earliest-ever** visit date in the raw EHR
2. Compare with their diagnosis date from Step 1:
   - If `diagnosis_date <= earliest_raw_visit` → **EXCLUDE** (no prior history to use as X features)
   - If `diagnosis_date > earliest_raw_visit` → **KEEP** (has prior visits)

**Output**: `case_patients` dict (filtered subset of `patient_assignment`)

**Note**: Controls are **NOT** filtered — all assigned patients from other diseases are included regardless of prior history.

### Step 3: Identify Fibrotic ICD Columns to Remove

**Logic**:
1. Read only the header of the raw EHR file
2. Match all 39 fibrotic ICD codes against column names using **exact set membership**
3. Build `cols_to_keep` list (all columns minus matched fibrotic codes)

**Important**: Many fibrotic codes may not exist as columns in the data — these are simply skipped without error.

### Step 4: Build 7 Disease Datasets (Chunked)

For each disease D:

**Cases (y=1)**: Patients assigned to disease D who passed Step 2 filter
- **x_row**: all raw EHR visits with `event_dt < diagnosis_date` (prior history)
- **y_row**: the raw EHR visit with `event_dt == diagnosis_date` (diagnosis event)

**Controls (y=0)**: All patients assigned to ANY OTHER disease (from `patient_assignment`, no Step 2 filter)
- **control**: all their raw EHR visits

**Output**: `datasets/dataset_{disease_name}.csv`

### Step 5: Validation

Automated checks:
1. No fibrotic ICD columns leaked into output
2. No patient appears as both case and control in the same dataset
3. Every case patient has at least 1 x_row and y_row(s)
4. Summary statistics table printed

---

## 5. Memory Optimization (Chunked Processing)

The raw EHR file is too large to load into memory at once. The pipeline uses a **two-pass chunked approach**:

### Configurable Parameters (in Config cell)

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `100,000` | Rows per chunk when reading raw EHR |
| `CTRL_BATCH` | `500` | Number of control patients to accumulate before flushing to CSV |

### Pass 1: Step 2 — Earliest Date Computation (Chunked)

```python
for chunk in pd.read_csv(RAW_EHR_FILE, usecols=["eid", "event_dt"],
                          parse_dates=["event_dt"], chunksize=CHUNK_SIZE):
    # Only reads 2 columns per chunk
    # Incrementally updates earliest_raw dict
```

- Only loads `eid` + `event_dt` (2 columns) per chunk
- Updates a dict `{eid: earliest_date}` incrementally
- Memory footprint: ~one chunk of 2 columns + the dict

### Pass 2: Step 4 — Data Extraction (Chunked + Filtered)

```python
for chunk in pd.read_csv(RAW_EHR_FILE, usecols=cols_to_keep,
                          parse_dates=["event_dt"], chunksize=CHUNK_SIZE):
    mask = chunk["eid"].isin(all_relevant_eids)
    filtered = chunk[mask]  # Keep only assigned patients
    # ... accumulate filtered rows
    del chunk, mask, filtered
    gc.collect()
```

- Reads ~1992 columns per chunk (fibrotic codes already excluded via `usecols`)
- Filters each chunk to **only keep rows for assigned patients** (a small fraction of total)
- Explicitly `del` + `gc.collect()` after each chunk

### Pass 3: Step 4 — CSV Writing (Batched Append)

```python
# Case rows: write all at once (small)
case_df.to_csv(output_path, index=False, mode="w")

# Control rows: flush every CTRL_BATCH patients
batch_df.to_csv(output_path, index=False, mode="a", header=False)
del batch_df
gc.collect()
```

- Case patients are typically few → written in one batch
- Control patients are written in batches of `CTRL_BATCH` (default 500) via CSV append mode
- Each batch is freed from memory after writing

### Tuning Tips

- **Kernel still dies?** → Decrease `CHUNK_SIZE` to `20,000` or `10,000`
- **Running too slowly?** → Increase `CHUNK_SIZE` to `200,000` if memory allows
- **Control writing too slow?** → Increase `CTRL_BATCH` to `1000`

---

## 6. Output Format

For each disease, one CSV file in `datasets/`:

```
datasets/
├── dataset_CKD.csv
├── dataset_Cardiac_Fibrosis.csv
├── dataset_MASH.csv
├── dataset_Pulmonary_fibrosis.csv
├── dataset_SSc_Connective_Tissue.csv
├── dataset_Crohns_Disease.csv
└── dataset_Fibrosis_of_Skin.csv
```

### Column Layout

```
eid, event_dt, y_label, record_type, [non-fibrotic ICD codes...], [UKB static features...]
```

| Column | Description |
|---|---|
| `eid` | Patient ID |
| `event_dt` | Visit date |
| `y_label` | 1 = case (diagnosed with this disease), 0 = control |
| `record_type` | `x_row` (case prior visit), `y_row` (case diagnosis visit), `control` (control visit) |
| ICD codes | ~1762 non-fibrotic ICD indicator columns (binary 0/1) |
| UKB features | ~204 static features (constant per patient) |

### How Downstream Models Should Use This

1. **Group by `eid`**, sort by `event_dt` → get each patient's visit sequence
2. **y_label** → patient-level binary target for training
3. For case patients:
   - `x_row` records → input feature sequence (prior visit history)
   - `y_row` record → the diagnosis event (can be used or excluded depending on model design)
4. For control patients:
   - `control` records → all visits used as input feature sequence

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Multi-disease at earliest visit | Skip the patient entirely | Avoid ambiguity in label assignment |
| Control filtering | No prior-history filter for controls | Controls represent general disease population; filtering would reduce sample size unnecessarily |
| Y row in output | Included, marked as `record_type="y_row"` | Downstream models can decide whether to include or exclude |
| ICD code removal | Exact string match only | Prevent accidentally removing non-fibrotic codes that share a prefix (e.g., I421 ≠ I420) |
| Memory strategy | Chunked reading + batched CSV writes | Full dataset causes kernel OOM; filtering to relevant patients keeps accumulated data manageable |

---

## 8. Quick Start

1. Place input files in the project root:
   - `fibrotic_gpX_hesinY_Y_visit.csv`
   - `gp_icd10_hesin_raw_data_with_ukb.csv`

2. Open `create_disease_datasets.ipynb`

3. In the **Config cell**, set:
   ```python
   USE_SAMPLE = False   # True for testing with sample data
   CHUNK_SIZE = 100000  # Adjust based on available RAM
   ```

4. Run all cells sequentially

5. Output CSVs will be in `datasets/` folder
