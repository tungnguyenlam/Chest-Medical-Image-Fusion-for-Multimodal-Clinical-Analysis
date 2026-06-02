from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display


FRONTAL_VIEWS = {"AP", "PA"}
LATERAL_VIEWS = {"LATERAL", "LL", "LPO", "RPO"}
MAJOR_VIEWS = ["AP", "PA", "LATERAL"]

BASE_ID_COLUMNS = [
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
]
NON_LABEL_COLUMNS = set(BASE_ID_COLUMNS + ["path", "fpath", "split", "source_split", "task"])


def get_notebook_paths(expected_dir_name: str = "00-examine-data") -> tuple[Path, Path]:
    cwd = Path.cwd()
    if cwd.name != expected_dir_name:
        raise RuntimeError(f"Please run this notebook from the {expected_dir_name} directory")
    root_dir = cwd.parents[1]
    return root_dir, root_dir / "data"


def load_first_existing(paths: list[Path]) -> tuple[pd.DataFrame | None, Path | None]:
    for candidate_path in paths:
        if candidate_path.exists():
            return pd.read_csv(candidate_path), candidate_path
    return None, None


def load_csv_map(base_dir: Path, csv_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(base_dir / filename) for name, filename in csv_files.items()}


def detect_path_column(df: pd.DataFrame) -> str:
    for column in ("path", "fpath"):
        if column in df.columns:
            return column
    raise KeyError("Expected an image path column named 'path' or 'fpath'")


def label_columns(df: pd.DataFrame, extra_exclude: set[str] | None = None) -> list[str]:
    excluded = set(NON_LABEL_COLUMNS)
    if extra_exclude:
        excluded.update(extra_exclude)
    return [column for column in df.columns if column not in excluded]


def no_finding_column(label_columns_: list[str]) -> str | None:
    for candidate in ("No Finding", "Normal"):
        if candidate in label_columns_:
            return candidate
    return None


def normalize_study_id_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text[1:] if text.startswith("s") and text[1:].isdigit() else text


def add_normalized_study_id(df: pd.DataFrame, column: str = "study_id") -> pd.DataFrame:
    return df.assign(study_id_norm=df[column].map(normalize_study_id_value))


def lookup_previous_study_date(prev_id, date_lookup: dict) -> pd.Timestamp:
    if pd.isna(prev_id):
        return pd.NaT
    try:
        return date_lookup.get(int(prev_id), pd.NaT)
    except (ValueError, TypeError):
        return pd.NaT


def format_count_pct(count: int | float, pct: int | float) -> str:
    if pd.isna(count) or pd.isna(pct):
        return ""
    return f"{int(round(count)):,}\n({pct:.1f}%)"


def annotate_bar_containers(ax, labels_by_container: list[list[str]], orientation: str) -> None:
    for container, labels in zip(ax.containers, labels_by_container):
        for bar, label in zip(container.patches, labels):
            if bar is None or not label:
                continue
            if orientation == "horizontal":
                label = label.replace("\n", " ")
                x = bar.get_x() + bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                ax.annotate(label, (x, y), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
            else:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + bar.get_height()
                ax.annotate(label, (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    if orientation == "horizontal":
        ax.margins(x=0.18)
    else:
        ax.margins(y=0.18)


def labels_by_hue(
    plot_df: pd.DataFrame,
    category_col: str,
    hue_col: str,
    count_col: str,
    pct_col: str,
    category_order,
    hue_order,
) -> list[list[str]]:
    keyed = plot_df.set_index([hue_col, category_col])
    labels = []
    for hue_value in hue_order:
        hue_labels = []
        for category_value in category_order:
            key = (hue_value, category_value)
            if key in keyed.index:
                row = keyed.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                hue_labels.append(format_count_pct(row[count_col], row[pct_col]))
            else:
                hue_labels.append("")
        labels.append(hue_labels)
    return labels


def labels_for_categories(
    plot_df: pd.DataFrame,
    category_col: str,
    count_col: str,
    pct_col: str,
    category_order,
) -> list[str]:
    keyed = plot_df.set_index(category_col)
    labels = []
    for category_value in category_order:
        if category_value in keyed.index:
            row = keyed.loc[category_value]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            labels.append(format_count_pct(row[count_col], row[pct_col]))
        else:
            labels.append("")
    return labels


def normalize_view_position(series: pd.Series) -> pd.Series:
    return series.fillna("Missing").astype(str).str.strip().str.upper().replace({"": "Missing"})


def format_view_combo(views: tuple[str, ...]) -> str:
    return " + ".join(views) if views else "Missing"


def has_any_view(views: tuple[str, ...], candidates: set[str]) -> bool:
    return bool(set(views) & candidates)


def parse_mimic_study_datetime(df: pd.DataFrame) -> pd.Series:
    if "StudyDate" not in df.columns or "StudyTime" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    date_text = df["StudyDate"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    time_text = (
        df["StudyTime"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.split(".")
        .str[0]
        .str.zfill(6)
        .str[:6]
    )
    return pd.to_datetime(date_text + time_text, format="%Y%m%d%H%M%S", errors="coerce")


def preview_dataset(name: str, df: pd.DataFrame, head_rows: int = 3) -> None:
    print(f"\n{name}")
    print(f"shape: {df.shape}")
    print(f"duplicate dicom_id rows: {df['dicom_id'].duplicated().sum()}")
    display(df.head(head_rows))

    missing = df.isna().sum().rename("missing_count").loc[lambda series: series > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values detected.")
    else:
        missing_df = missing.to_frame().assign(missing_pct=lambda frame: frame["missing_count"] / len(df) * 100)
        display(missing_df)


def build_split_overview(split_frames: dict[str, pd.DataFrame], label_columns_: list[str]) -> pd.DataFrame:
    rows = []
    no_finding = no_finding_column(label_columns_)
    for split_name, df in split_frames.items():
        positive_labels_per_image = df[label_columns_].sum(axis=1)
        row = {
            "split": split_name,
            "rows": len(df),
            "unique_subjects": df["subject_id"].nunique(),
            "unique_studies": df["study_id"].nunique(),
            "unique_dicoms": df["dicom_id"].nunique(),
            "missing_ViewPosition": df["ViewPosition"].isna().sum(),
            "missing_ViewCodeMeaning": df["ViewCodeSequence_CodeMeaning"].isna().sum(),
            "avg_labels_per_image": positive_labels_per_image.mean(),
            "median_labels_per_image": positive_labels_per_image.median(),
        }
        if no_finding:
            row[f"{no_finding}_count"] = int(df[no_finding].sum())
            row[f"{no_finding}_rate_pct"] = df[no_finding].mean() * 100
        rows.append(row)

    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def summarize_labels(df: pd.DataFrame, split_name: str, label_columns_: list[str]) -> pd.DataFrame:
    summary = (
        df[label_columns_]
        .mean()
        .mul(100)
        .rename("positive_rate_pct")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "label"})
    )
    summary.insert(0, "split", split_name)
    summary["positive_count"] = summary["label"].map(df[label_columns_].sum().to_dict()).astype(int)
    return summary[["split", "label", "positive_count", "positive_rate_pct"]]


def plot_label_prevalence(
    split_frames: dict[str, pd.DataFrame],
    label_columns_: list[str],
    reference_df: pd.DataFrame,
    top_n: int = 15,
) -> None:
    label_summary = pd.concat(
        [summarize_labels(df, split_name, label_columns_) for split_name, df in split_frames.items()],
        ignore_index=True,
    )
    reference_summary = summarize_labels(reference_df, "global", label_columns_)
    label_order = reference_summary.head(top_n)["label"].tolist()
    hue_order = list(split_frames)
    plot_df = label_summary[label_summary["label"].isin(label_order)].copy()

    plt.figure(figsize=(12, max(7, top_n * 0.95)))
    ax = sns.barplot(data=plot_df, x="positive_rate_pct", y="label", hue="split", order=label_order, hue_order=hue_order)
    annotate_bar_containers(
        ax,
        labels_by_hue(plot_df, "label", "split", "positive_count", "positive_rate_pct", label_order, hue_order),
        orientation="horizontal",
    )
    plt.title(f"Top {top_n} labels by global prevalence")
    plt.xlabel("Positive rate (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_global_label_prevalence(df: pd.DataFrame, label_columns_: list[str], top_n: int = 26) -> None:
    plot_df = summarize_labels(df, "global", label_columns_).head(top_n)
    label_order = plot_df["label"].tolist()

    plt.figure(figsize=(10, max(7, top_n * 0.32)))
    ax = sns.barplot(data=plot_df, x="positive_rate_pct", y="label", order=label_order, color="C0")
    annotate_bar_containers(
        ax,
        [labels_for_categories(plot_df, "label", "positive_count", "positive_rate_pct", label_order)],
        orientation="horizontal",
    )
    plt.title(f"Global top {top_n} labels")
    plt.xlabel("Positive rate (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_labels_per_image(split_frames: dict[str, pd.DataFrame], label_columns_: list[str], include_global: bool = True) -> None:
    frames = dict(split_frames)
    if include_global:
        frames["global"] = pd.concat(split_frames.values(), ignore_index=True)

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (split_name, df) in zip(axes, frames.items()):
        label_counts = df[label_columns_].sum(axis=1).astype(int)
        counts = label_counts.value_counts().sort_index()
        pct = counts / len(df) * 100
        bars = ax.bar(counts.index.astype(str), counts.values)
        ax.bar_label(
            bars,
            labels=[format_count_pct(count, pct_value) for count, pct_value in zip(counts.values, pct.values)],
            padding=3,
            fontsize=8,
        )
        ax.margins(y=0.18)
        ax.set_title(split_name)
        ax.set_xlabel("Positive labels per image")
        ax.set_ylabel("Image count")

    plt.tight_layout()
    plt.show()


def summarize_view_positions(df: pd.DataFrame, split_name: str, top_n: int = 8) -> pd.DataFrame:
    view_counts = (
        df["ViewPosition"]
        .fillna("Missing")
        .value_counts()
        .head(top_n)
        .rename_axis("ViewPosition")
        .reset_index(name="count")
    )
    view_counts.insert(0, "split", split_name)
    view_counts["rate_pct"] = view_counts["count"] / len(df) * 100
    return view_counts


def plot_view_positions(split_frames: dict[str, pd.DataFrame], reference_df: pd.DataFrame, top_n: int = 6) -> None:
    view_order = summarize_view_positions(reference_df, "global", top_n=top_n)["ViewPosition"].tolist()
    plot_df = pd.concat(
        [summarize_view_positions(df, split_name, top_n=999) for split_name, df in split_frames.items()],
        ignore_index=True,
    )
    plot_df = plot_df[plot_df["ViewPosition"].isin(view_order)].copy()
    hue_order = list(split_frames)

    plt.figure(figsize=(12, 5.5))
    ax = sns.barplot(
        data=plot_df,
        x="ViewPosition",
        y="rate_pct",
        hue="split",
        order=view_order,
        hue_order=hue_order,
    )
    annotate_bar_containers(
        ax,
        labels_by_hue(plot_df, "ViewPosition", "split", "count", "rate_pct", view_order, hue_order),
        orientation="vertical",
    )
    plt.title(f"Top {top_n} view positions by global frequency")
    plt.xlabel("")
    plt.ylabel("Rate (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_global_view_positions(df: pd.DataFrame, top_n: int = 8) -> None:
    plot_df = summarize_view_positions(df, "global", top_n=top_n)
    view_order = plot_df["ViewPosition"].tolist()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=plot_df, x="ViewPosition", y="rate_pct", order=view_order, color="C0")
    annotate_bar_containers(
        ax,
        [labels_for_categories(plot_df, "ViewPosition", "count", "rate_pct", view_order)],
        orientation="vertical",
    )
    plt.title(f"Global top {top_n} view positions")
    plt.xlabel("")
    plt.ylabel("Rate (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def compute_overlap(split_frames: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    split_names = list(split_frames)
    key_sets = {split_name: set(df[key]) for split_name, df in split_frames.items()}

    overlap_matrix = pd.DataFrame(index=split_names, columns=split_names, dtype=int)
    for left_name in split_names:
        for right_name in split_names:
            overlap_matrix.loc[left_name, right_name] = len(key_sets[left_name] & key_sets[right_name])

    return overlap_matrix


def summarize_no_finding_conflicts(split_frames: dict[str, pd.DataFrame], label_columns_: list[str]) -> pd.DataFrame:
    no_finding = no_finding_column(label_columns_)
    if no_finding is None:
        return pd.DataFrame()
    other_labels = [label for label in label_columns_ if label != no_finding]
    rows = []

    for split_name, df in split_frames.items():
        no_finding_count = int(df[no_finding].sum())
        conflicts = int(((df[no_finding] == 1) & (df[other_labels].sum(axis=1) > 0)).sum())
        rows.append(
            {
                "split": split_name,
                f"rows_with_{no_finding}": no_finding_count,
                f"rows_with_{no_finding}_and_other_label": conflicts,
                f"conflict_rate_within_{no_finding}_pct": conflicts / max(no_finding_count, 1) * 100,
            }
        )

    return pd.DataFrame(rows)


def summarize_submission_file(df: pd.DataFrame, split_name: str, label_columns_: list[str]) -> pd.DataFrame:
    submission_label_columns = [column for column in df.columns if column != "dicom_id"]
    rows = [
        {
            "split": split_name,
            "rows": len(df),
            "label_columns_match_train": submission_label_columns == label_columns_,
            "min_score": df[submission_label_columns].min().min(),
            "mean_score": df[submission_label_columns].mean().mean(),
            "max_score": df[submission_label_columns].max().max(),
        }
    ]
    return pd.DataFrame(rows)


def build_study_view_table(df: pd.DataFrame, label_columns_: list[str]) -> pd.DataFrame:
    temp = df.assign(_view=normalize_view_position(df["ViewPosition"]))
    group_keys = ["subject_id", "study_id"]

    counts = temp.groupby(group_keys, as_index=False).agg(image_count=("dicom_id", "size"), unique_dicoms=("dicom_id", "nunique"))
    views = temp.groupby(group_keys)["_view"].agg(lambda values: tuple(sorted(set(values)))).rename("views").reset_index()
    study_labels = temp.groupby(group_keys)[label_columns_].max().sum(axis=1).rename("positive_labels_per_study").reset_index()

    study_table = counts.merge(views, on=group_keys).merge(study_labels, on=group_keys)
    study_table["view_combo"] = study_table["views"].map(format_view_combo)
    study_table["has_frontal"] = study_table["views"].map(lambda views: has_any_view(views, FRONTAL_VIEWS))
    study_table["has_lateral"] = study_table["views"].map(lambda views: has_any_view(views, LATERAL_VIEWS))
    study_table["has_pa"] = study_table["views"].map(lambda views: "PA" in views)
    study_table["has_ap"] = study_table["views"].map(lambda views: "AP" in views)
    study_table["has_frontal_and_lateral"] = study_table["has_frontal"] & study_table["has_lateral"]
    return study_table


def summarize_study_views(split_frames: dict[str, pd.DataFrame], label_columns_: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    summary_rows = []

    for split_name, df in split_frames.items():
        study_table = build_study_view_table(df, label_columns_).assign(split=split_name)
        tables.append(study_table)
        summary_rows.append(
            {
                "split": split_name,
                "studies": len(study_table),
                "single_image_studies_count": int((study_table["image_count"] == 1).sum()),
                "single_image_studies_pct": (study_table["image_count"] == 1).mean() * 100,
                "median_images_per_study": study_table["image_count"].median(),
                "frontal_and_lateral_studies_count": int(study_table["has_frontal_and_lateral"].sum()),
                "frontal_and_lateral_studies_pct": study_table["has_frontal_and_lateral"].mean() * 100,
                "pa_and_lateral_studies_pct": (study_table["has_pa"] & study_table["has_lateral"]).mean() * 100,
                "ap_and_lateral_studies_pct": (study_table["has_ap"] & study_table["has_lateral"]).mean() * 100,
                "median_positive_labels_per_study": study_table["positive_labels_per_study"].median(),
            }
        )

    all_studies = pd.concat(tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return all_studies, summary


def summarize_view_combos(study_table: pd.DataFrame, scope_col: str = "split", top_order=None) -> pd.DataFrame:
    combo_counts = study_table.groupby([scope_col, "view_combo"]).size().rename("study_count").reset_index()
    totals = study_table.groupby(scope_col).size().rename("total_studies")
    combo_counts = combo_counts.merge(totals, on=scope_col)
    combo_counts["study_rate_pct"] = combo_counts["study_count"] / combo_counts["total_studies"] * 100
    if top_order is not None:
        combo_counts = combo_counts[combo_counts["view_combo"].isin(top_order)].copy()
    return combo_counts


def plot_top_view_combos(study_table: pd.DataFrame, top_n: int = 12) -> None:
    combo_order = study_table.groupby("view_combo").size().sort_values(ascending=False).head(top_n).index.tolist()
    plot_df = summarize_view_combos(study_table, top_order=combo_order)
    hue_order = [split_name for split_name in study_table["split"].dropna().unique() if split_name in plot_df["split"].unique()]

    plt.figure(figsize=(13, max(5, top_n * 0.35)))
    ax = sns.barplot(data=plot_df, x="study_count", y="view_combo", hue="split", order=combo_order, hue_order=hue_order)
    annotate_bar_containers(
        ax,
        labels_by_hue(plot_df, "view_combo", "split", "study_count", "study_rate_pct", combo_order, hue_order),
        orientation="horizontal",
    )
    plt.title(f"Top {top_n} study view combinations")
    plt.xlabel("Study count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_global_top_view_combos(study_table: pd.DataFrame, top_n: int = 12) -> None:
    global_table = study_table.drop(columns=["split"], errors="ignore").assign(split="global")
    combo_order = global_table["view_combo"].value_counts().head(top_n).index.tolist()
    plot_df = summarize_view_combos(global_table, top_order=combo_order)

    plt.figure(figsize=(11, max(5, top_n * 0.35)))
    ax = sns.barplot(data=plot_df, x="study_count", y="view_combo", order=combo_order, color="C0")
    annotate_bar_containers(
        ax,
        [labels_for_categories(plot_df, "view_combo", "study_count", "study_rate_pct", combo_order)],
        orientation="horizontal",
    )
    plt.title(f"Global top {top_n} study view combinations")
    plt.xlabel("Study count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def build_study_timeline(images_df: pd.DataFrame, label_columns_: list[str], scope_columns: list[str] | None = None) -> pd.DataFrame:
    scope_columns = scope_columns or []
    temp = images_df.assign(_view=normalize_view_position(images_df["ViewPosition"]))
    group_keys = scope_columns + ["subject_id", "study_id"]

    base = temp.groupby(group_keys, as_index=False).agg(
        image_count=("dicom_id", "size"),
        study_datetime=("study_datetime", "min"),
        first_dicom_id=("dicom_id", "first"),
    )
    views = temp.groupby(group_keys)["_view"].agg(lambda values: tuple(sorted(set(values)))).rename("views").reset_index()
    labels = temp.groupby(group_keys)[label_columns_].max().sum(axis=1).rename("positive_labels_per_study").reset_index()

    timeline = base.merge(views, on=group_keys).merge(labels, on=group_keys)
    timeline["view_combo"] = timeline["views"].map(format_view_combo)
    sort_columns = scope_columns + ["subject_id", "study_datetime", "study_id"]
    timeline = timeline.sort_values(sort_columns, na_position="last")
    subject_group = scope_columns + ["subject_id"]
    timeline["days_since_previous_study"] = timeline.groupby(subject_group)["study_datetime"].diff().dt.total_seconds().div(86400)
    return timeline


def summarize_patient_timelines(
    study_timeline: pd.DataFrame,
    scope_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_columns = scope_columns or []
    patient_group = scope_columns + ["subject_id"]
    patient_summary = study_timeline.groupby(patient_group, as_index=False).agg(
        study_count=("study_id", "nunique"),
        first_study_datetime=("study_datetime", "min"),
        last_study_datetime=("study_datetime", "max"),
        total_images=("image_count", "sum"),
    )
    patient_summary["span_days"] = (
        patient_summary["last_study_datetime"] - patient_summary["first_study_datetime"]
    ).dt.total_seconds().div(86400)

    if scope_columns:
        timeline_summary = (
            patient_summary.groupby(scope_columns)
            .agg(
                patients=("subject_id", "nunique"),
                patients_with_multiple_studies=("study_count", lambda values: int((values > 1).sum())),
                median_studies_per_patient=("study_count", "median"),
                max_studies_per_patient=("study_count", "max"),
                median_span_days=("span_days", "median"),
                max_span_days=("span_days", "max"),
            )
            .reset_index()
        )
    else:
        timeline_summary = pd.DataFrame(
            [
                {
                    "scope": "global",
                    "patients": patient_summary["subject_id"].nunique(),
                    "patients_with_multiple_studies": int((patient_summary["study_count"] > 1).sum()),
                    "median_studies_per_patient": patient_summary["study_count"].median(),
                    "max_studies_per_patient": patient_summary["study_count"].max(),
                    "median_span_days": patient_summary["span_days"].median(),
                    "max_span_days": patient_summary["span_days"].max(),
                }
            ]
        )

    timeline_summary["multiple_study_patient_pct"] = (
        timeline_summary["patients_with_multiple_studies"] / timeline_summary["patients"] * 100
    )
    return patient_summary, timeline_summary


def label_pair_correlation_table(df: pd.DataFrame, label_columns_: list[str]) -> pd.DataFrame:
    corr = df[label_columns_].corr()
    pairs = []
    for left_idx, left_label in enumerate(label_columns_):
        for right_label in label_columns_[left_idx + 1 :]:
            pairs.append(
                {
                    "left_label": left_label,
                    "right_label": right_label,
                    "correlation": corr.loc[left_label, right_label],
                    "cooccurrence_count": int(((df[left_label] == 1) & (df[right_label] == 1)).sum()),
                    "left_positive_count": int(df[left_label].sum()),
                    "right_positive_count": int(df[right_label].sum()),
                }
            )
    return pd.DataFrame(pairs).dropna(subset=["correlation"])


def view_conditioned_prevalence(
    split_frames: dict[str, pd.DataFrame],
    label_columns_: list[str],
    major_views: list[str] = MAJOR_VIEWS,
) -> pd.DataFrame:
    rows = []
    for split_name, df in split_frames.items():
        temp = df.assign(_view=normalize_view_position(df["ViewPosition"]))
        for view_name, view_df in temp[temp["_view"].isin(major_views)].groupby("_view"):
            for label in label_columns_:
                positives = int(view_df[label].sum())
                rows.append(
                    {
                        "split": split_name,
                        "ViewPosition": view_name,
                        "label": label,
                        "row_count": len(view_df),
                        "positive_count": positives,
                        "positive_rate_pct": positives / len(view_df) * 100 if len(view_df) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_view_conditioned_prevalence(prevalence_df: pd.DataFrame, label_order: list[str], title: str) -> None:
    plot_df = prevalence_df[prevalence_df["label"].isin(label_order)].copy()
    hue_order = [view for view in MAJOR_VIEWS if view in plot_df["ViewPosition"].unique()]

    plt.figure(figsize=(12, max(6, len(label_order) * 0.45)))
    ax = sns.barplot(
        data=plot_df,
        x="positive_rate_pct",
        y="label",
        hue="ViewPosition",
        order=label_order,
        hue_order=hue_order,
    )
    annotate_bar_containers(
        ax,
        labels_by_hue(plot_df, "label", "ViewPosition", "positive_count", "positive_rate_pct", label_order, hue_order),
        orientation="horizontal",
    )
    plt.title(title)
    plt.xlabel("Positive rate (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def resolve_relative_path(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def summarize_image_paths(split_frames: dict[str, pd.DataFrame], image_root: Path) -> pd.DataFrame:
    rows = []
    for split_name, df in split_frames.items():
        path_column = detect_path_column(df)
        paths = df[path_column].dropna().astype(str)
        exists_flags = paths.map(lambda value: resolve_relative_path(image_root, value).is_file())
        rows.append(
            {
                "split": split_name,
                "rows": len(df),
                "path_column": path_column,
                "path_values": len(paths),
                "unique_paths": paths.nunique(),
                "duplicate_path_rows": int(paths.duplicated().sum()),
                "jpg_extension_rows": int(paths.str.lower().str.endswith(".jpg").sum()),
                "existing_files": int(exists_flags.sum()),
                "missing_files": int((~exists_flags).sum()),
                "existing_file_pct": exists_flags.mean() * 100 if len(exists_flags) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_report_linkage(
    split_frames: dict[str, pd.DataFrame],
    global_df: pd.DataFrame,
    study_list_df: pd.DataFrame,
    mimic_cxr_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    study_list_for_join = study_list_df.rename(columns={"path": "report_path"}).copy()
    study_list_for_join["study_id_norm"] = study_list_for_join["study_id"].map(normalize_study_id_value)

    unique_studies_df = pd.concat(
        [
            add_normalized_study_id(df[["subject_id", "study_id"]].drop_duplicates()).assign(split=split_name)
            for split_name, df in split_frames.items()
        ],
        ignore_index=True,
    )
    global_unique_studies_df = add_normalized_study_id(global_df[["subject_id", "study_id"]].drop_duplicates()).assign(scope="global")

    join_cols = ["subject_id", "study_id_norm"]
    linked_reports_df = unique_studies_df.merge(
        study_list_for_join[["subject_id", "study_id_norm", "report_path"]],
        on=join_cols,
        how="left",
    )
    global_linked_reports_df = global_unique_studies_df.merge(
        study_list_for_join[["subject_id", "study_id_norm", "report_path"]],
        on=join_cols,
        how="left",
    )

    for linked_df in [linked_reports_df, global_linked_reports_df]:
        linked_df["has_report_index"] = linked_df["report_path"].notna()
        linked_df["report_file_exists"] = linked_df["report_path"].fillna("").map(
            lambda value: (mimic_cxr_dir / value).is_file() if value else False
        )

    report_link_summary_df = (
        linked_reports_df.groupby("split")
        .agg(
            studies=("study_id", "nunique"),
            studies_with_report_index=("has_report_index", "sum"),
            studies_with_report_file=("report_file_exists", "sum"),
        )
        .reset_index()
    )
    report_link_summary_df["report_index_pct"] = (
        report_link_summary_df["studies_with_report_index"] / report_link_summary_df["studies"] * 100
    )
    report_link_summary_df["report_file_pct"] = (
        report_link_summary_df["studies_with_report_file"] / report_link_summary_df["studies"] * 100
    )

    global_report_link_summary_df = (
        global_linked_reports_df.groupby("scope")
        .agg(
            studies=("study_id", "nunique"),
            studies_with_report_index=("has_report_index", "sum"),
            studies_with_report_file=("report_file_exists", "sum"),
        )
        .reset_index()
    )
    global_report_link_summary_df["report_index_pct"] = (
        global_report_link_summary_df["studies_with_report_index"] / global_report_link_summary_df["studies"] * 100
    )
    global_report_link_summary_df["report_file_pct"] = (
        global_report_link_summary_df["studies_with_report_file"] / global_report_link_summary_df["studies"] * 100
    )
    return report_link_summary_df, global_report_link_summary_df, linked_reports_df, global_linked_reports_df


def read_report_text(mimic_cxr_dir: Path, report_path: str, max_chars: int = 3000) -> str | None:
    if not report_path:
        return None
    full_path = mimic_cxr_dir / report_path
    if not full_path.is_file():
        return None
    return full_path.read_text(errors="replace")[:max_chars]


def add_bar_counts(ax, *, min_count=1):
    for container in ax.containers:
        labels = [f"{int(v)}" if v >= min_count else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=2, fontsize=8, rotation=90)


def plot_dis_study_per_subject(meta):
    studies_per_subject = meta.groupby("subject_id")["study_id"].nunique()
    print(max(studies_per_subject))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.histplot(studies_per_subject, bins=30, kde=False, ax=axes[0])
    add_bar_counts(axes[0])
    axes[0].set_title("Distribution of Number of Studies per Subject (log y)")
    axes[0].set_xlim(0, 160)
    axes[0].set_yscale("log")

    sns.histplot(studies_per_subject, bins=30, kde=False, ax=axes[1])
    add_bar_counts(axes[1])
    axes[1].set_title("Distribution of Number of Studies per Subject (linear y)")
    axes[1].set_xlim(0, 160)

    plt.tight_layout()
    plt.show()


def plot_dis_image_per_subject(meta):
    images_per_subject = meta.groupby("subject_id")["dicom_id"].nunique()
    print(max(images_per_subject))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.histplot(images_per_subject, bins=30, kde=False, ax=axes[0])
    add_bar_counts(axes[0])
    axes[0].set_title("Distribution of Number of Images per Subject (log y)")
    axes[0].set_xlim(0, 180)
    axes[0].set_yscale("log")
    sns.histplot(images_per_subject, bins=30, kde=False, ax=axes[1])
    add_bar_counts(axes[1])
    axes[1].set_title("Distribution of Number of Images per Subject (linear y)")
    axes[1].set_xlim(0, 180)
    plt.tight_layout()
    plt.show()


def plot_dis_image_per_study(meta):
    images_per_study = meta.groupby("study_id")["dicom_id"].nunique()
    print(max(images_per_study))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.histplot(images_per_study, bins=30, kde=False, ax=axes[0])
    add_bar_counts(axes[0])
    axes[0].set_title("Distribution of Number of Images per Study (log y)")
    axes[0].set_xlim(0, 12)
    axes[0].set_yscale("log")
    sns.histplot(images_per_study, bins=30, kde=False, ax=axes[1])
    add_bar_counts(axes[1])
    axes[1].set_title("Distribution of Number of Images per Study (linear y)")
    axes[1].set_xlim(0, 12)
    plt.tight_layout()
    plt.show()


def plot_sample_images_procedure(meta, mimic_cxr_jpg_dir: Path | str):
    procedure_col = "PerformedProcedureStepDescription"
    jpg_root = Path(mimic_cxr_jpg_dir) / "files"
    procedures = meta[procedure_col].dropna().unique()
    ncols = 3
    nrows = math.ceil(len(procedures) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, procedure in zip(axes, procedures):
        row = meta.loc[meta[procedure_col] == procedure].iloc[0]
        subject_id = str(row["subject_id"])
        study_id = normalize_study_id_value(row["study_id"])
        dicom_id = str(row["dicom_id"])
        view = row.get("ViewPosition", "unknown view")
        img_path = jpg_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"

        if not img_path.exists():
            ax.set_title(f"{procedure}\nmissing image")
            ax.axis("off")
            continue

        img = plt.imread(img_path)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{procedure}\n View: {view}")
        ax.axis("off")

    for ax in axes[len(procedures) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_sample_images_view_position(meta, mimic_cxr_jpg_dir: Path | str):
    view_col = "ViewPosition"
    jpg_root = Path(mimic_cxr_jpg_dir) / "files"
    views = meta[view_col].dropna().unique()
    ncols = 3
    nrows = math.ceil(len(views) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, view in zip(axes, views):
        row = meta.loc[meta[view_col] == view].iloc[0]
        subject_id = str(row["subject_id"])
        study_id = normalize_study_id_value(row["study_id"])
        dicom_id = str(row["dicom_id"])
        img_path = jpg_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"

        if not img_path.exists():
            ax.set_title(f"{view}\nmissing image")
            ax.axis("off")
            continue
        img = plt.imread(img_path)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"View: {view}")
        ax.axis("off")

    for ax in axes[len(views) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
