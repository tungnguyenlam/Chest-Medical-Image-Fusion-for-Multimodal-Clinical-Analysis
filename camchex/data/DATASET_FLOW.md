# CaMCheX Dataset Flow

This diagram traces how the raw tables, text reports, labels, and local image files become the train, development, and test CSVs consumed by CaMCheX training.

## Dataset subsets (`--subset`)

`01_make_dataset.py` and `03_filter_existing_images.py` take `--subset {full, subset, kaggle}`. Default is **`subset`**.

| `--subset`     | Step 01 reads MIMIC metadata/reports from | Step 03 filters images against                       | Output CSVs                          | Where it's used                                    |
| -------------- | ----------------------------------------- | ---------------------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| `subset` | `data/subset/MIMIC-CXR*`                  | `data/subset/MIMIC-CXR-JPG/files`                    | `03_mimic_{train,development,test}`  | Cloud GPU (rented) and iteration-heavy machines    |
| `full`         | `data/MIMIC-CXR*`                         | `data/MIMIC-CXR-JPG/files`                           | `03_mimic_{train,development,test}`  | School server (holds the full credentialed copy)   |
| `kaggle`       | `data/MIMIC-CXR*` (same as `full`)        | `data/data-kaggle/official_data_iccv_final/files`    | `03_kaggle_{train,development,test}` | Machines with the kaggle re-host instead of MIMIC  |

`kaggle` is image-only: the kaggle re-host ships JPGs but no metadata or reports, so step 01 falls back to the full MIMIC roots (`data/MIMIC-CXR*`). Only step 03 differs. If a machine has only the kaggle images (and no MIMIC tree), step 01 with `--subset kaggle` will still fail at metadata read — step 01 fundamentally needs the MIMIC metadata and reports.

`03` filters against **one** source per invocation. The previous behavior (loop over both `mimic` and `kaggle` in a single run) is gone — run twice (`--subset full && --subset kaggle`) if both output CSV sets are needed.

Companion datasets (`data/CXR-LT/`, `data/MIMIC-IV-ED-2-2/`) are read from `data/` regardless of `--subset` — they are small and bundled whole.

### Building and distributing the subset

The subset is a deterministic patient-level 10% sample (seed 42, sampled by `subject_id` so a patient's full longitudinal record stays intact and there is no train/val leakage).

- `scripts/build_mimic_subset.py` — sample patients → copy matching JPGs and reports into `data/subset/` (mirrors the original `MIMIC-CXR-JPG/files/...` and `MIMIC-CXR/files/...` tree, plus the small CSVs at each dataset root) → write `manifest.json` (seed, fraction, counts, SHA-256 of the sampled `subject_id` list) → archive `subset/ CXR-LT/ MIMIC-IV-ED-2-2/` into password-protected split `.7z` volumes (`bundle-a3f9.7z.001`, `.002`, ... by default; AES-256, header encryption `-mhe=on`) → delete/recreate the private HuggingFace dataset repo `tungnguyenlam/tung-thesis` so prior history is not retained → upload all parts.
- `scripts/download_subset.py` — pulls either the split `.7z.*` volumes or a legacy single `.7z` from `tungnguyenlam/tung-thesis` and extracts straight into `data/` on the target machine.
- Secrets live in `.env` (see `.env.example`): `HF_TOKEN` (HuggingFace) and `DATA_PASSWORD` (7z). Both files are gitignored.
- Archive name is intentionally neutral (`bundle-a3f9.7z` by default) and the repo is **private** — public re-hosting of MIMIC would violate the PhysioNet DUA. Anyone with access must already be PhysioNet-credentialed.
- Bundle size is roughly 50–75 GB for the 10% slice, split into 10 GB parts by default. Sanity-check the actual total size before upload.

```mermaid
flowchart TD
    %% Raw source files
    subgraph raw["Raw inputs under data/"]
        metadata["MIMIC-CXR-JPG metadata<br/>data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv<br/>subject_id, study_id, dicom_id, ViewPosition, StudyDate, StudyTime"]
        reports["MIMIC-CXR text reports<br/>data/MIMIC-CXR/files/pXX/pSUBJECT/sSTUDY.txt"]
        section_parser["mimic-cxr/txt/section_parser.py<br/>section_text + custom_mimic_cxr_rules"]
        triage["MIMIC-IV-ED triage<br/>data/MIMIC-IV-ED-2-2/.../ed/triage.csv.gz"]
        edstays["MIMIC-IV-ED stays<br/>data/MIMIC-IV-ED-2-2/.../ed/edstays.csv.gz"]
        vitals["MIMIC-IV-ED vitals<br/>data/MIMIC-IV-ED-2-2/.../ed/vitalsign.csv.gz"]
        cxrlt_train["CXR-LT train.csv<br/>dicom_id + study_id + 26 labels"]
        cxrlt_dev["CXR-LT development.csv<br/>dicom_id + study_id + 26 labels"]
        cxrlt_test["CXR-LT test.csv<br/>dicom_id + study_id + 26 labels"]
    end

    %% Step 1
    subgraph step1["Step 1: python src/prepare/01_make_dataset.py"]
        study_base["Study-level CXR table<br/>drop duplicate study_id rows<br/>sort by subject_id, StudyDate<br/>compute PreviousStudy and StudyDateTime"]
        ed_join["ED context table<br/>triage merge edstays on subject_id, stay_id<br/>left-join to CXR studies by subject_id<br/>keep closest ED stay by time to StudyDateTime"]
        vital_join["Vitals fallback<br/>join vitalsign rows by subject_id, stay_id<br/>select nearest charttime row<br/>fill temperature, heartrate, resprate, o2sat, sbp, dbp, pain"]
        label_union["CXR-LT label union<br/>concat train + development + test<br/>attach split=train/development/test<br/>retain 26 disease labels"]
        report_parse["Parallel report parsing<br/>read report .txt by subject_id/study_id<br/>parse sections: findings, impression, last_paragraph, indication, history<br/>clean and lowercase text"]
        progress["data/data-camchex/01_progress.csv<br/>checkpoint after ED/vitals/report parsing"]
        report_fields["Final text fields<br/>report = findings, else impression, else last_paragraph<br/>clinical_indication = indication + history<br/>drop intermediate report sections"]
        dicom_expand["DICOM-level expansion<br/>merge metadata dicom_id + ViewPosition by study_id<br/>drop rows without report<br/>rename ClinicalIndication to clinical_indication"]
        labels_by_dicom["Attach CXR-LT labels by dicom_id<br/>merge full train/development/test label rows<br/>split column survives from CXR-LT"]
        merged01["data/data-camchex/01_merged.csv<br/>one row per image view<br/>metadata + ED fields + vitals + report + clinical_indication + ViewPosition + 26 labels + split"]
        path02["Add initial image path<br/>images/pXX/pSUBJECT/sSTUDY/DICOM.jpg"]
        train02_from_step1["data/data-camchex/02_train.csv"]
        dev02_from_step1["data/data-camchex/02_development.csv"]
        test02_from_step1["data/data-camchex/02_test.csv"]
    end

    metadata --> study_base
    triage --> ed_join
    edstays --> ed_join
    study_base --> ed_join
    vitals --> vital_join
    ed_join --> vital_join
    reports --> report_parse
    section_parser --> report_parse
    vital_join --> report_parse
    report_parse --> progress
    report_parse --> report_fields
    metadata --> dicom_expand
    report_fields --> dicom_expand
    cxrlt_train --> label_union
    cxrlt_dev --> label_union
    cxrlt_test --> label_union
    label_union --> labels_by_dicom
    dicom_expand --> labels_by_dicom
    labels_by_dicom --> merged01
    merged01 --> path02
    path02 -->|"split == train"| train02_from_step1
    path02 -->|"split == development"| dev02_from_step1
    path02 -->|"split == test"| test02_from_step1

    %% Optional Step 2
    subgraph step2["Optional Step 2: python src/prepare/02_split_dataset.py"]
        split_ids["Reload CXR-LT dicom_id split sets<br/>train.csv, development.csv, test.csv"]
        resplit["Filter 01_merged.csv by dicom_id membership<br/>recompute images/pXX/.../DICOM.jpg path"]
        train02["data/data-camchex/02_train.csv"]
        dev02["data/data-camchex/02_development.csv"]
        test02["data/data-camchex/02_test.csv"]
    end

    merged01 -. "rerun splitting without reparsing reports" .-> resplit
    cxrlt_train -.-> split_ids
    cxrlt_dev -.-> split_ids
    cxrlt_test -.-> split_ids
    split_ids --> resplit
    resplit --> train02
    resplit --> dev02
    resplit --> test02
    train02_from_step1 -->|"same target files if Step 2 is skipped"| train02
    dev02_from_step1 -->|"same target files if Step 2 is skipped"| dev02
    test02_from_step1 -->|"same target files if Step 2 is skipped"| test02

    %% Step 3
    subgraph images["Image source directories checked by Step 3"]
        mimic_images["MIMIC image source<br/>data/MIMIC-CXR-JPG/files<br/>usually symlinked to full JPG storage"]
        kaggle_images["Kaggle image source<br/>data/data-kaggle/official_data_iccv_final/files<br/>alternate subset if present"]
    end

    subgraph step3["Step 3: python src/prepare/03_filter_existing_images.py"]
        strip_prefix["For each 02 split<br/>strip leading images/ from path"]
        filter_mimic["Check file exists under MIMIC source<br/>rewrite path for camchex cwd:<br/>../data/MIMIC-CXR-JPG/files/pXX/.../DICOM.jpg"]
        filter_kaggle["Check file exists under Kaggle source<br/>rewrite path for camchex cwd:<br/>../data/data-kaggle/official_data_iccv_final/files/pXX/.../DICOM.jpg"]
        train03_mimic["data/data-camchex/03_mimic_train.csv"]
        dev03_mimic["data/data-camchex/03_mimic_development.csv"]
        test03_mimic["data/data-camchex/03_mimic_test.csv"]
        train03_kaggle["data/data-camchex/03_kaggle_train.csv"]
        dev03_kaggle["data/data-camchex/03_kaggle_development.csv"]
        test03_kaggle["data/data-camchex/03_kaggle_test.csv"]
    end

    train02 --> strip_prefix
    dev02 --> strip_prefix
    test02 --> strip_prefix
    mimic_images --> filter_mimic
    kaggle_images --> filter_kaggle
    strip_prefix --> filter_mimic
    strip_prefix --> filter_kaggle
    filter_mimic --> train03_mimic
    filter_mimic --> dev03_mimic
    filter_mimic --> test03_mimic
    filter_kaggle --> train03_kaggle
    filter_kaggle --> dev03_kaggle
    filter_kaggle --> test03_kaggle

    %% Training consumption
    subgraph training["Training and evaluation consumption from camchex/ cwd"]
        config["camchex/config.yaml<br/>data.datamodule_cfg paths<br/>default points to ../data/data-camchex/03_mimic_*.csv"]
        dm_init["CaMCheXDataModule.__init__<br/>read train_df_path, devel_df_path, pred_df_path<br/>load BioBERT tokenizer"]
        fit_setup["setup('fit' or 'validate')<br/>train_df -> CaMCheXDataset with train transforms<br/>devel_df -> CaMCheXDataset with val transforms"]
        pred_setup["setup('predict')<br/>pred_df_path -> CaMCheXDataset with val transforms"]
        study_group["CaMCheXDataset<br/>group CSV rows by study_id<br/>sample up to 4 views per study<br/>load images from rewritten path<br/>encode ViewPosition ids"]
        text_tokens["Clinical text inputs<br/>clinical_indication tokenized to max_length 384<br/>vitals + gender text tokenized to max_length 128"]
        labels["Targets<br/>26 CXR-LT multi-label columns<br/>one label vector per study"]
        train_loader["train_dataloader<br/>shuffled training batches"]
        val_loader["val_dataloader<br/>development validation batches"]
        pred_loader["predict_dataloader<br/>test prediction batches"]
    end

    train03_mimic --> config
    dev03_mimic --> config
    test03_mimic --> config
    train03_kaggle -. "alternate if config paths are changed" .-> config
    dev03_kaggle -. "alternate if config paths are changed" .-> config
    test03_kaggle -. "alternate if config paths are changed" .-> config
    config --> dm_init
    dm_init --> fit_setup
    dm_init --> pred_setup
    fit_setup --> study_group
    pred_setup --> study_group
    study_group --> text_tokens
    study_group --> labels
    text_tokens --> train_loader
    labels --> train_loader
    text_tokens --> val_loader
    labels --> val_loader
    text_tokens --> pred_loader
    labels --> pred_loader
```

## Read This Diagram

- The pipeline scripts moved out of `camchex/data/` and now live at `src/prepare/0{1,2,3,4}_*.py`. They are shared by the legacy `camchex/` and the refactored `training/` paths; the diagram labels above use the current paths.
- `01_make_dataset.py` is the expensive step because it parses every report. Its `01_progress.csv` checkpoint is saved after report parsing so a late crash does not force the slow part to be re-derived manually.
- `02_split_dataset.py` is optional when step 1 finishes normally because step 1 already writes `02_train.csv`, `02_development.csv`, and `02_test.csv`. Run step 2 when you want to rebuild splits from `01_merged.csv` without reparsing reports.
- `03_filter_existing_images.py` is what makes the CSVs machine-aware. It drops rows whose image file is absent on the selected image source and rewrites `path` values so training can open them from the `camchex/` working directory.
- `04_build_prior_aware_dataset.py` is optional, used only by the prior-aware model. It collapses the 03 CSV to one row per `study_id`, joins each study with its `PreviousStudy`, pre-tokenizes BioBERT inputs for both, and writes `prior_aware_{train,development,test}.parquet`. The runtime `PriorAwareDataset` only has to decode JPEGs.
  - **Label space is selectable.** It defaults to the CXR-LT 2023 26-label set. To build a CXR-LT 2024 prior-aware dataset, first relabel/re-split the prepared CSVs with `scripts/relabel_prepared_cxrlt.py` (which writes `cxrlt2024_task1/{train,val,test}.csv`), then run stage 4 against those with the matching label set:
    ```bash
    python src/prepare/04_build_prior_aware_dataset.py \
      --in-dir data/data-camchex/cxrlt2024_task1 --in-prefix "" --splits train val test \
      --out-dir data/data-camchex --out-prefix prior_aware_cxrlt2024_task1_ \
      --cxr-lt-version cxr-lt-2024 --cxr-lt-label-set task1
    ```
    The 2024 release is the *same images* as 2023 — only the labels and the train/val/test assignment differ — so the expensive image-channel cache (`../cache/channels`) and frozen text-embedding cache (`../cache/text_embeddings`) are keyed by image path / text content and are **reused verbatim**; only this lightweight parquet is rebuilt. Prior linkage is re-resolved here because it is resolved *within a split*, and 2024 reassigns splits. A sidecar `prior_aware_cxrlt2024_task1_label_metadata.json` records the resolved class order and per-split positive counts (drop the train `class_instance_nums`/`total_instance_num` straight into a model's `loss_init_args`).
- The active default training files are `03_mimic_train.csv`, `03_mimic_development.csv`, and `03_mimic_test.csv`, because those are the paths in `camchex/config.yaml`. The Kaggle `03_kaggle_*` files are alternate outputs if that source directory exists and the config is changed to point at them.
- `CaMCheXDataset` groups the final CSV rows by `study_id`, so the dataloader returns one study sample with up to four image views, view-position IDs, clinical indication tokens, vitals/gender tokens, and the 26-label target vector.
- The merged CSVs retain a cleaned `report` column, but the current `CaMCheXDataset` returns `clinical_indication` and vitals/gender text, not the full `report` text.
