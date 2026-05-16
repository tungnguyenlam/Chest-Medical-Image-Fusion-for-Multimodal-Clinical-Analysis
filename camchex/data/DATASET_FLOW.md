# CaMCheX Dataset Flow

This diagram traces how the raw tables, text reports, labels, and local image files become the train, development, and test CSVs consumed by CaMCheX training.

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
    subgraph step1["Step 1: python camchex/data/01_make_dataset.py"]
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
    subgraph step2["Optional Step 2: python camchex/data/02_split_dataset.py"]
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

    subgraph step3["Step 3: python camchex/data/03_filter_existing_images.py"]
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

- `01_make_dataset.py` is the expensive step because it parses every report. Its `01_progress.csv` checkpoint is saved after report parsing so a late crash does not force the slow part to be re-derived manually.
- `02_split_dataset.py` is optional when step 1 finishes normally because step 1 already writes `02_train.csv`, `02_development.csv`, and `02_test.csv`. Run step 2 when you want to rebuild splits from `01_merged.csv` without reparsing reports.
- `03_filter_existing_images.py` is what makes the CSVs machine-aware. It drops rows whose image file is absent on the selected image source and rewrites `path` values so training can open them from the `camchex/` working directory.
- The active default training files are `03_mimic_train.csv`, `03_mimic_development.csv`, and `03_mimic_test.csv`, because those are the paths in `camchex/config.yaml`. The Kaggle `03_kaggle_*` files are alternate outputs if that source directory exists and the config is changed to point at them.
- `CaMCheXDataset` groups the final CSV rows by `study_id`, so the dataloader returns one study sample with up to four image views, view-position IDs, clinical indication tokens, vitals/gender tokens, and the 26-label target vector.
- The merged CSVs retain a cleaned `report` column, but the current `CaMCheXDataset` returns `clinical_indication` and vitals/gender text, not the full `report` text.
