# Plan: Create 7 Disease-Specific Datasets for Single-Label Prediction

## Context
We need to build 7 binary classification datasets (one per fibrotic disease) from EHR data. Each dataset has Case patients (y=1, diagnosed with that disease) and Control patients (y=0, diagnosed with a different fibrotic disease). The goal is to predict whether a patient will develop a specific fibrotic disease based on their prior visit history.

## Input Files
- `fibrotic_gpX_hesinY_Y_visit.csv` — disease visit labels (9 cols: eid, event_dt, 7 disease flags)
- `gp_icd10_hesin_raw_data_with_ukb.csv` — raw EHR data (2007 cols: eid, event_dt, ~1801 ICD codes, ~204 UKB static features)
- Sample versions: `*_sample.csv` for development/testing

## ICD Code Mapping (fibrotic_codes.jpg → column names)
Remove dots from ICD codes to match column format:

| Disease | ICD Codes (column format) |
|---|---|
| CKD | N18, N181, N182, N183, N184, N185, N189 |
| Cardiac_Fibrosis | I40, I409, I420, I423, I424, I425, I514 |
| MASH | K74, K746, K75, K758, K7581 |
| Pulmonary_fibrosis | J841, J8410, J84112, J8417, J848, J8489, J849 |
| SSc_Connective_Tissue | M34, M340, M341, M342, M348, M349 |
| Crohns_Disease | K50, K501, K508, K509 |
| Fibrosis_of_Skin | L905, L91, L910 |

**Total: 39 codes to remove.** Only EXACT matches — e.g., I421 (I42.1) is NOT fibrotic and must be kept.

Of these 39, only 15 exist in the sample raw data: I420, J841, J849, K501, K509, K746, K758, L905, L910, M349, N182, N183, N184, N185, N189. Non-existent columns are simply skipped.

## Pipeline Steps

### Step 1: Assign Each Patient to ONE Disease
- Load `fibrotic_gpX_hesinY_Y_visit.csv`
- For each eid, find the **earliest** visit (by event_dt)
- At that earliest visit, identify which disease column = 1 → that's the patient's assigned disease
- Result: `dict{eid: (disease_name, diagnosis_date)}`
- **Edge case**: if multiple diseases = 1 at the earliest visit → **skip this patient** (log a warning, exclude from all datasets)

**Example from sample data:**
- 1000527: only visit 2002-12-23, Pulmonary_fibrosis=1 → assigned Pulmonary_fibrosis
- 1000971: earliest visit 2007-08-18, CKD=1 → assigned CKD (ignore later 2010-10-19 visit)
- 1002442: earliest visit 2010-07-25, Crohns_Disease=1 → assigned Crohns_Disease (ignore later MASH visit)

### Step 2: Filter — Must Have Prior EHR History
- From raw EHR data, get each patient's **earliest** visit date (only need eid + event_dt columns, memory efficient)
- For each patient from Step 1: compare their diagnosis_date vs their earliest raw EHR visit date
- **EXCLUDE** if diagnosis_date == earliest_raw_visit (no prior history to use as features)
- **KEEP** if diagnosis_date > earliest_raw_visit (has prior visits for X)
- Result: filtered set of valid case patients

### Step 3: Build 7 Disease Datasets
For each disease D:

**Cases (y=1):** Patients assigned to disease D who passed Step 2 filter
- **X rows**: all raw EHR visits with event_dt < diagnosis_date
- **Y row**: the raw EHR visit with event_dt == diagnosis_date (used to mark y_label=1, but the features of Y row itself are the visit on that date)

**Controls (y=0):** Patients assigned to ANY OTHER disease (not D) — **no prior history filter needed**
- ALL their raw EHR visits are included as records

### Step 4: Feature Column Processing
For ALL rows (cases and controls):
- **Remove** all 39 fibrotic ICD code columns (exact match only)
- **Keep**: eid, event_dt, all non-fibrotic ICD codes (~1762 cols), all UKB static features (~204 cols)

### Output Format
For each disease, one CSV file: `dataset_{disease_name}.csv`

Columns: `eid, event_dt, y_label, record_type, [non-fibrotic ICD codes], [UKB features]`

Where:
- `y_label`: 1 for case patients, 0 for control patients (patient-level, same for all rows of that patient)
- `record_type`: "x_row" (case patient's pre-diagnosis visits), "y_row" (case patient's diagnosis visit), "control" (control patient's visits)

Downstream model training can:
- Group by eid, sort by event_dt to get visit sequences
- Use y_label for training target
- For case patients: use x_rows as input features, y_row as the prediction target event
- For control patients: use all rows as input features

## Implementation
- **Jupyter Notebook**: `create_disease_datasets.ipynb`
- Use pandas with chunked reading for memory efficiency on large files
- Parameterize with `USE_SAMPLE = True/False` flag to switch between sample and full data
- Print statistics at each step (patient counts, excluded counts, dataset sizes)
- Output: 7 CSV files in `datasets/` folder (e.g., `dataset_CKD.csv`, `dataset_Pulmonary_fibrosis.csv`, etc.)

### Notebook Sections
1. **Config & Imports** — file paths, USE_SAMPLE flag, fibrotic ICD code definitions
2. **Step 1: Patient-Disease Assignment** — load fibrotic visit file, assign each patient to one disease
3. **Step 2: Prior History Filter** — read eid+event_dt from raw EHR, filter case patients
4. **Step 3: Identify Fibrotic Columns** — match 39 fibrotic codes against raw EHR columns
5. **Step 4: Build Datasets** — for each disease, extract case X/Y rows and control rows, remove fibrotic columns, save CSV
6. **Step 5: Validation** — verify outputs, print summary statistics

## Verification
- Check no fibrotic ICD columns appear in output
- Check no patient appears as both case and control in same dataset
- Check every case patient has at least 1 x_row and exactly 1 y_row
- Print summary table: disease | n_cases | n_controls | n_total_rows

## Decisions (confirmed by user)
1. **Multi-disease at earliest visit** → Skip the patient entirely (exclude from all datasets)
2. **Control filtering** → Controls do NOT need Step 2 filter; include all controls regardless of prior history
3. **Y row** → Include in output, marked with record_type="y_row"
4. **Output format** → Jupyter Notebook, producing 7 CSV files (one per disease)
