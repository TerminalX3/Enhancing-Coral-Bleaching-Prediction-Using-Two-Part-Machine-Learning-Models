from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

SOURCE_URL = "https://datadocs.bco-dmo.org/dataset/773466/file/B11vA82u7y2Owp/global_bleaching_environmental.csv"
SOURCE_SHA256 = "78cc014b1e7887f26f4c44a40d4d9213212f3c6217541c6a1b69e723b61bfacb"
CLEAN_SHA256 = "2fb9f55f6f6fb222f581b9c65a0d7acd3b51857e91430c58ea30c36a1847bfc3"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_source(source_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    if sha256(source_bytes) != SOURCE_SHA256:
        raise ValueError("Source differs from the frozen study version; audit it before use.")
    raw = pd.read_csv(io.BytesIO(source_bytes), low_memory=False)
    frame = raw.replace(r"(?i)^\s*(?:nd|na|n/a|null)\s*$", np.nan, regex=True)
    target = pd.to_numeric(frame["Percent_Bleaching"], errors="coerce")
    missing = target.isna()
    out_of_range = target.notna() & ~target.between(0, 100)
    frame["Percent_Bleaching"] = target
    frame = frame.loc[target.between(0, 100)].copy()
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Survey_Year"] = dates.dt.year
    day = dates.dt.dayofyear
    frame["DOY_Sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["DOY_Cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["Bleaching_Class"] = pd.cut(
        frame["Percent_Bleaching"],
        bins=[-0.001, 0, 10, 50, 100],
        labels=["None", "Low", "Medium", "High"],
        include_lowest=True,
    )
    report = {
        "source_url": SOURCE_URL,
        "dataset_doi": "10.26008/1912/bco-dmo.773466.2",
        "dataset_version": "2 (2022-10-14)",
        "license": "CC-BY-4.0",
        "source_sha256": sha256(source_bytes),
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "missing_target_rows_removed": int(missing.sum()),
        "out_of_range_target_rows_removed": int(out_of_range.sum()),
        "exact_duplicate_rows_removed": duplicates,
        "clean_rows": int(len(frame)),
        "clean_columns": int(len(frame.columns)),
        "unparsed_retained_dates": int(dates.isna().sum()),
        "negative_SSTA_retained": int((pd.to_numeric(frame["SSTA"], errors="coerce") < 0).sum()),
        "negative_TSA_retained": int((pd.to_numeric(frame["TSA"], errors="coerce") < 0).sum()),
        "expected_clean_sha256": CLEAN_SHA256,
    }
    return frame, report

MISSING = {"", "nan", "none", "null", "nd", "na", "n/a", "<na>"}


def _id(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if value.lower() in MISSING else value


def build_groups(frame: pd.DataFrame, link_coordinates: bool = True) -> np.ndarray:


    n = len(frame)
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    keys = [[] for _ in range(n)]
    for column in ("Site_ID", "Reef_ID", "Sample_ID"):
        if column in frame:
            for i, value in enumerate(frame[column]):
                value = _id(value)
                if value is not None:
                    keys[i].append(f"{column}:{value}")
    if link_coordinates and {"Latitude_Degrees", "Longitude_Degrees"} <= set(frame):
        lat = pd.to_numeric(frame["Latitude_Degrees"], errors="coerce").to_numpy()
        lon = pd.to_numeric(frame["Longitude_Degrees"], errors="coerce").to_numpy()
        for i, (a, b) in enumerate(zip(lat, lon)):
            if np.isfinite(a) and np.isfinite(b) and abs(a) <= 90 and abs(b) <= 180:
                keys[i].append(f"coordinate:{float(a):.17g},{float(b):.17g}")
    seen = {}
    for i, row_keys in enumerate(keys):
        for key in row_keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    roots = np.array([find(i) for i in range(n)])
    canonical = {}

    def rank(key):
        return (0 if key.startswith("Site_ID:") else 1, key)
    for i, root in enumerate(roots):
        key = min(keys[i], key=rank) if keys[i] else f"unidentified_row:{i}"
        if root not in canonical or rank(key) < rank(canonical[root]):
            canonical[root] = key
    return np.asarray(["component:" + canonical[r] for r in roots], dtype=object)


def prepare_samples(frame: pd.DataFrame, features: list[str],
                    target: str = "Percent_Bleaching"):


    if frame["Sample_ID"].isna().any():
        raise ValueError("Missing Sample_ID requires a separately documented policy")
    counts = frame.groupby("Sample_ID", sort=True)[features + [target]].nunique(dropna=False)
    conflicts = counts.index[counts.gt(1).any(axis=1)].tolist()
    eligible = frame.loc[~frame["Sample_ID"].isin(conflicts)]
    prepared = eligible.drop_duplicates("Sample_ID", keep="first").copy()

    prepared = prepared.sort_values("Sample_ID", kind="stable").reset_index(drop=True)
    report = {
        "input_rows": int(len(frame)),
        "unique_sample_ids": int(frame["Sample_ID"].nunique()),
        "ambiguous_sample_ids": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in conflicts],
        "ambiguous_source_rows": int(frame["Sample_ID"].isin(conflicts).sum()),
        "replicate_rows_collapsed": int(len(eligible) - len(prepared)),
        "analysis_samples": int(len(prepared)),
        "conflict_columns": {c: int(counts[c].gt(1).sum()) for c in counts if counts[c].gt(1).any()},
    }
    return prepared, report


def prepare_input(source, output, download):
    if source.exists():
        source_bytes = source.read_bytes()
    elif download:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            source_bytes = response.read()
        if sha256(source_bytes) != SOURCE_SHA256:
            raise ValueError('Downloaded source differs from the frozen study version.')
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(source_bytes)
    else:
        raise FileNotFoundError(f'{source} does not exist. Supply --source, --data, or --download.')
    frame, report = prepare_source(source_bytes)
    output_bytes = frame.to_csv(index=False).encode('utf-8')
    report['reconstructed_clean_sha256'] = sha256(output_bytes)
    if sha256(output_bytes) != CLEAN_SHA256:
        raise ValueError('Reconstruction differs from the frozen clean file. Use the dependency versions in --help.')
    destination = output / 'coral_bleaching_clean.csv'
    if destination.exists() and destination.read_bytes() != output_bytes:
        raise FileExistsError('Refusing to overwrite a different existing cleaned CSV.')
    if not destination.exists():
        destination.write_bytes(output_bytes)
    write_json(output / 'source_preparation.json', report)
    return destination

SEED = 20261020

TARGET = "Percent_Bleaching"

THERMAL = [
    "ClimSST", "Temperature_Kelvin", "Temperature_Mean", "Temperature_Minimum",
    "Temperature_Maximum", "Temperature_Kelvin_Standard_Deviation", "SSTA",
    "SSTA_Standard_Deviation", "SSTA_Mean", "SSTA_Minimum", "SSTA_Maximum",
    "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation", "SSTA_FrequencyMax",
    "SSTA_FrequencyMean", "SSTA_DHW", "SSTA_DHW_Standard_Deviation",
    "SSTA_DHWMax", "SSTA_DHWMean", "TSA", "TSA_Standard_Deviation",
    "TSA_Minimum", "TSA_Maximum", "TSA_Mean", "TSA_Frequency",
    "TSA_Frequency_Standard_Deviation", "TSA_FrequencyMax", "TSA_FrequencyMean",
    "TSA_DHW", "TSA_DHW_Standard_Deviation", "TSA_DHWMax", "TSA_DHWMean",
]

TEMPORAL = ["Survey_Year", "DOY_Sin", "DOY_Cos"]

ENVIRONMENTAL = ["Distance_to_Shore", "Turbidity", "Cyclone_Frequency", "Depth_m", "Windspeed"]

GEOGRAPHIC = ["Latitude_Degrees", "Longitude_Degrees"]

FEATURES = [
    "Latitude_Degrees", "Longitude_Degrees", "Distance_to_Shore", "Turbidity",
    "Cyclone_Frequency", "Depth_m", "ClimSST", "Temperature_Kelvin",
    "Temperature_Mean", "Temperature_Minimum", "Temperature_Maximum",
    "Temperature_Kelvin_Standard_Deviation", "Windspeed", "SSTA",
    "SSTA_Standard_Deviation", "SSTA_Mean", "SSTA_Minimum", "SSTA_Maximum",
    "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation", "SSTA_FrequencyMax",
    "SSTA_FrequencyMean", "SSTA_DHW", "SSTA_DHW_Standard_Deviation", "SSTA_DHWMax",
    "SSTA_DHWMean", "TSA", "TSA_Standard_Deviation", "TSA_Minimum", "TSA_Maximum",
    "TSA_Mean", "TSA_Frequency", "TSA_Frequency_Standard_Deviation", "TSA_FrequencyMax",
    "TSA_FrequencyMean", "TSA_DHW", "TSA_DHW_Standard_Deviation", "TSA_DHWMax",
    "TSA_DHWMean", "Survey_Year", "DOY_Sin", "DOY_Cos",
]

CURRENT_THERMAL = ["Temperature_Kelvin", "SSTA", "SSTA_Frequency", "SSTA_DHW", "TSA", "TSA_Frequency", "TSA_DHW"]

FEATURE_SETS = {
    "full42": FEATURES,
    "thermal_only": [x for x in FEATURES if x in THERMAL],
    "no_geography": [x for x in FEATURES if x not in GEOGRAPHIC],
    "no_year": [x for x in FEATURES if x != "Survey_Year"],
    "restricted_window": [x for x in FEATURES if x not in THERMAL or x in CURRENT_THERMAL],
}

STRUCTURES = [
    {"max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 10.0},
    {"max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
]

GATES = np.round(np.arange(0.2, 0.801, 0.05), 2).tolist()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_regressor(loss, structure, seed):
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=200, learning_rate=0.07, early_stopping=False,
        random_state=seed, **STRUCTURES[structure],
    )


def make_classifier(structure, seed):
    return HistGradientBoostingClassifier(
        loss="log_loss", max_iter=200, learning_rate=0.07,
        early_stopping=False, random_state=seed, **STRUCTURES[structure],
    )


def clip(prediction):
    return np.clip(np.asarray(prediction), 0.0, 100.0)


def prepare(data_path):
    raw = pd.read_csv(data_path, low_memory=False)
    raw["source_row"] = np.arange(len(raw))
    if not raw[TARGET].between(0, 100).all():
        raise ValueError("Target must be finite and within [0,100].")
    frame, report = prepare_samples(raw, FEATURES, TARGET)
    frame["component"] = build_groups(frame, link_coordinates=True)
    frame["analysis_row"] = np.arange(len(frame))
    report.update({
        "components": int(frame.component.nunique()),
        "largest_component_samples": int(frame.groupby("component").size().max()),
        "zero_samples": int(frame[TARGET].eq(0).sum()),
        "positive_samples": int(frame[TARGET].gt(0).sum()),
        "high_severity_samples": int(frame[TARGET].gt(50).sum()),
        "target_mean": float(frame[TARGET].mean()),
        "target_median": float(frame[TARGET].median()),
        "year_min": int(frame.Survey_Year.min()),
        "year_max": int(frame.Survey_Year.max()),
        "input_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    })
    return frame, report


def build_folds(frame, output, seed):
    folds = np.full(len(frame), -1, dtype=int)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y = frame[TARGET].to_numpy()
    groups = frame.component.to_numpy()
    inner_rows = []
    inner_map = {}
    for outer, (train, test) in enumerate(splitter.split(frame, y, groups)):
        assert not set(groups[train]) & set(groups[test])
        folds[test] = outer
        inner = GroupKFold(n_splits=3, shuffle=True, random_state=seed + 100 + outer)
        splits = list(inner.split(frame.iloc[train], y[train], groups[train]))
        inner_map[outer] = splits
        for inner_fold, (itrain, ivalid) in enumerate(splits):
            assert not set(groups[train[itrain]]) & set(groups[train[ivalid]])
            inner_rows.extend({"outer_fold": outer, "inner_fold": inner_fold,
                               "analysis_row": int(i), "Sample_ID": int(frame.Sample_ID.iloc[i]),
                               "component": str(groups[i])} for i in train[ivalid])
    assert (folds >= 0).all()
    frame["outer_fold"] = folds
    keep = ["analysis_row", "source_row", "Sample_ID", "Site_ID", "Reef_ID", "component", "outer_fold",
            "Latitude_Degrees", "Longitude_Degrees", "Country_Name", "Ecoregion_Name", "Survey_Year", TARGET]
    frame[keep].to_csv(output / "outer_fold_assignments.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_fold_assignments.csv", index=False)
    return inner_map


def choose(rows):

    return min(rows, key=lambda r: r["inner_mae"])


def run_fold(frame, outer, splits, feature_set, output_dir, seed, threads, force=False):
    start = time.monotonic()
    output = Path(output_dir) / "folds" / f"fold_{outer}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "complete.json"
    if checkpoint.exists() and not force:
        print(f"fold {outer}: cached complete", flush=True)
        return outer
    features = FEATURE_SETS[feature_set]
    train_idx = np.flatnonzero(frame.outer_fold.to_numpy() != outer)
    test_idx = np.flatnonzero(frame.outer_fold.to_numpy() == outer)
    X = frame[features].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    xt, yt = X[train_idx], y[train_idx]
    xv, yv = X[test_idx], y[test_idx]
    fit_seed = seed + 1000 + outer
    direct = {(loss, s): np.full(len(train_idx), np.nan)
              for loss in ["absolute_error", "squared_error"] for s in range(2)}
    severity = {key: np.full(len(train_idx), np.nan) for key in direct}
    occurrence = {s: np.full(len(train_idx), np.nan) for s in range(2)}
    print(f"fold {outer}: start, train={len(yt)}, test={len(yv)}, features={len(features)}", flush=True)
    with threadpool_limits(limits=threads):
        for inner_fold, (itrain, ivalid) in enumerate(splits):
            positive = itrain[yt[itrain] > 0]
            for (loss, s), pred in direct.items():
                model = make_regressor(loss, s, fit_seed)
                model.fit(xt[itrain], yt[itrain])
                pred[ivalid] = clip(model.predict(xt[ivalid]))
                model = make_regressor(loss, s, fit_seed)
                model.fit(xt[positive], yt[positive])
                severity[(loss, s)][ivalid] = clip(model.predict(xt[ivalid]))
            for s, pred in occurrence.items():
                model = make_classifier(s, fit_seed)
                model.fit(xt[itrain], yt[itrain] > 0)
                pred[ivalid] = model.predict_proba(xt[ivalid])[:, 1]
            print(f"fold {outer}: inner {inner_fold + 1}/3 finished ({time.monotonic() - start:.0f}s)", flush=True)

        candidates = []
        selected = {}
        for name, loss in [("direct_hgb_mae", "absolute_error"), ("direct_hgb_mse", "squared_error")]:
            rows = [{"model": name, "loss": loss, "structure": s,
                     "inner_mae": float(mean_absolute_error(yt, direct[(loss, s)]))} for s in range(2)]
            candidates.extend(rows)
            selected[name] = choose(rows)
        hard_rows = []
        soft_rows = []
        for cs in range(2):
            for rs in range(2):
                for threshold in GATES:
                    pred = np.where(occurrence[cs] >= threshold, severity[("absolute_error", rs)], 0.0)
                    hard_rows.append({"model": "hard_hurdle", "classifier_structure": cs,
                                      "regressor_structure": rs, "loss": "absolute_error",
                                      "threshold": threshold, "inner_mae": float(mean_absolute_error(yt, pred))})
                pred = occurrence[cs] * severity[("squared_error", rs)]
                soft_rows.append({"model": "soft_hurdle", "classifier_structure": cs,
                                  "regressor_structure": rs, "loss": "squared_error",
                                  "inner_mae": float(mean_absolute_error(yt, pred))})
        candidates.extend(hard_rows + soft_rows)
        selected["hard_hurdle"] = choose(hard_rows)
        selected["soft_hurdle"] = choose(soft_rows)
        pd.DataFrame(candidates).to_csv(output / "inner_tuning.csv", index=False)
        write_json(output / "selected_settings.json", selected)

        predictions = frame.iloc[test_idx][["analysis_row", "Sample_ID", "component", "outer_fold", TARGET]].copy()
        predictions["global_median"] = np.median(yt)
        predictions["global_mean"] = np.mean(yt)
        for name in ["direct_hgb_mae", "direct_hgb_mse"]:
            setting = selected[name]
            model = make_regressor(setting["loss"], setting["structure"], fit_seed)
            model.fit(xt, yt)
            predictions[name] = clip(model.predict(xv))
        fitted_classifiers = {}
        for name in ["hard_hurdle", "soft_hurdle"]:
            setting = selected[name]
            cs = setting["classifier_structure"]
            if cs not in fitted_classifiers:
                classifier = make_classifier(cs, fit_seed)
                classifier.fit(xt, yt > 0)
                fitted_classifiers[cs] = classifier
            probability = fitted_classifiers[cs].predict_proba(xv)[:, 1]
            regressor = make_regressor(setting["loss"], setting["regressor_structure"], fit_seed)
            regressor.fit(xt[yt > 0], yt[yt > 0])
            positive_prediction = clip(regressor.predict(xv))
            predictions[f"{name}_probability"] = probability
            predictions[f"{name}_positive_severity"] = positive_prediction
            predictions[name] = (np.where(probability >= setting["threshold"], positive_prediction, 0.0)
                                 if name == "hard_hurdle" else probability * positive_prediction)
        for name, cls in [("random_forest", RandomForestRegressor), ("extra_trees", ExtraTreesRegressor)]:
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  cls(n_estimators=200, min_samples_leaf=2, max_features=0.8,
                                      random_state=fit_seed, n_jobs=1))
            model.fit(xt, yt)
            predictions[name] = clip(model.predict(xv))
        predictions.to_csv(output / "predictions.csv", index=False)
    duration = time.monotonic() - start
    write_json(checkpoint, {"outer_fold": outer, "seconds": duration,
                            "train_samples": len(yt), "test_samples": len(yv),
                            "feature_set": feature_set, "features": features})
    summary = ", ".join(f"{m}={mean_absolute_error(yv, predictions[m]):.3f}"
                        for m in ["direct_hgb_mae", "hard_hurdle", "soft_hurdle"])
    print(f"fold {outer}: complete ({duration:.0f}s); MAE {summary}", flush=True)
    return outer

MODEL_NAMES = ["global_median", "global_mean", "direct_hgb_mae", "direct_hgb_mse",
               "random_forest", "extra_trees", "hard_hurdle", "soft_hurdle"]


def regression_metrics(y, prediction, groups):
    y, prediction = np.asarray(y), np.asarray(prediction)
    errors = np.abs(y - prediction)
    group_mae = pd.Series(errors).groupby(np.asarray(groups)).mean().mean()
    return {
        "samples": int(len(y)), "components": int(pd.Series(groups).nunique()),
        "mae_pp": float(errors.mean()), "rmse_pp": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)), "group_macro_mae_pp": float(group_mae),
        "positive_mae_pp": float(errors[y > 0].mean()) if (y > 0).any() else None,
        "high_severity_mae_pp": float(errors[y > 50].mean()) if (y > 50).any() else None,
        "zero_mae_pp": float(errors[y == 0].mean()) if (y == 0).any() else None,
        "bias_pp": float((prediction - y).mean()),
    }


def summarize(output, bootstrap_reps, seed):
    prediction = pd.concat([pd.read_csv(output / "folds" / f"fold_{fold}" / "predictions.csv")
                            for fold in range(5)], ignore_index=True).sort_values("analysis_row")
    assert len(prediction) == prediction.analysis_row.nunique()
    assert prediction[MODEL_NAMES].notna().all().all()
    prediction.to_csv(output / "oof_predictions.csv", index=False)
    y, groups = prediction[TARGET].to_numpy(), prediction.component.to_numpy()
    rows = []
    for fold in ["pooled", 0, 1, 2, 3, 4]:
        subset = prediction if fold == "pooled" else prediction.loc[prediction.outer_fold == fold]
        for model in MODEL_NAMES:
            rows.append({"fold": fold, "model": model, **regression_metrics(subset[TARGET], subset[model], subset.component)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    write_json(output / "metrics.json", rows)
    occurrence_rows = []
    for fold in ["pooled", 0, 1, 2, 3, 4]:
        subset = prediction if fold == "pooled" else prediction.loc[prediction.outer_fold == fold]
        binary = subset[TARGET].gt(0).astype(int)
        for model in ["hard_hurdle", "soft_hurdle"]:
            p = subset[f"{model}_probability"].to_numpy()
            occurrence_rows.append({"fold": fold, "model": model, "roc_auc": float(roc_auc_score(binary, p)),
                                    "average_precision": float(average_precision_score(binary, p)),
                                    "brier_score": float(brier_score_loss(binary, p)),
                                    "positive_prevalence": float(binary.mean())})
    pd.DataFrame(occurrence_rows).to_csv(output / "occurrence_metrics.csv", index=False)


    _, group_index = np.unique(groups, return_inverse=True)
    n_groups = group_index.max() + 1
    counts = np.bincount(group_index)
    loss_sums = np.column_stack([np.bincount(group_index, weights=np.abs(y - prediction[m]))
                                for m in MODEL_NAMES])
    generator = np.random.default_rng(seed + 50000)
    draws = np.empty((bootstrap_reps, len(MODEL_NAMES)))
    for begin in range(0, bootstrap_reps, 100):
        end = min(begin + 100, bootstrap_reps)
        selected = generator.integers(0, n_groups, size=(end - begin, n_groups))
        draws[begin:end] = loss_sums[selected].sum(axis=1) / counts[selected].sum(axis=1, keepdims=True)
    ci_rows = []
    for i, model in enumerate(MODEL_NAMES):
        lo, hi = np.quantile(draws[:, i], [.025, .975])
        ci_rows.append({"model": model, "mae_pp": float(np.abs(y - prediction[model]).mean()),
                        "ci_low": float(lo), "ci_high": float(hi), "bootstrap_replicates": bootstrap_reps})
    pd.DataFrame(ci_rows).to_csv(output / "mae_cluster_bootstrap.csv", index=False)
    comparisons = [(a, b) for a in ["hard_hurdle", "soft_hurdle", "direct_hgb_mae", "direct_hgb_mse", "random_forest", "extra_trees"]
                   for b in ["global_median", "direct_hgb_mae", "direct_hgb_mse", "random_forest", "extra_trees"] if a != b]
    difference_rows = []
    for a, b in comparisons:
        difference = draws[:, MODEL_NAMES.index(a)] - draws[:, MODEL_NAMES.index(b)]
        lo, hi = np.quantile(difference, [.025, .975])
        difference_rows.append({"model_a": a, "model_b": b,
                                "mae_difference_a_minus_b_pp": float(np.abs(y - prediction[a]).mean() - np.abs(y - prediction[b]).mean()),
                                "ci_low": float(lo), "ci_high": float(hi),
                                "bootstrap_replicates": bootstrap_reps})
    pd.DataFrame(difference_rows).to_csv(output / "paired_mae_cluster_bootstrap.csv", index=False)
    print(metrics.loc[metrics.fold == "pooled", ["model", "mae_pp", "rmse_pp", "r2", "group_macro_mae_pp"]].to_string(index=False), flush=True)


def run_ablation_fold(frame, outer, output_dir, seed, threads, ablations, force=False):
    output = Path(output_dir)
    checkpoint = output / "ablations" / f"fold_{outer}_predictions.csv"
    if checkpoint.exists() and not force:
        print(f"ablation fold {outer}: cached", flush=True)
        return outer
    settings = json.loads((output / "folds" / f"fold_{outer}" / "selected_settings.json").read_text())
    train = frame.outer_fold.to_numpy() != outer
    y, yv = frame.loc[train, TARGET].to_numpy(), frame.loc[~train, TARGET].to_numpy()
    predictions = frame.loc[~train, ["analysis_row", "Sample_ID", "component", "outer_fold", TARGET]].copy()
    fit_seed = seed + 1000 + outer
    with threadpool_limits(limits=threads):
        for feature_set in ablations:
            features = FEATURE_SETS[feature_set]
            xt, xv = frame.loc[train, features].to_numpy(float), frame.loc[~train, features].to_numpy(float)
            setting = settings["direct_hgb_mae"]
            model = make_regressor(setting["loss"], setting["structure"], fit_seed)
            model.fit(xt, y)
            predictions[f"direct_hgb_mae__{feature_set}"] = clip(model.predict(xv))
            setting = settings["hard_hurdle"]
            classifier = make_classifier(setting["classifier_structure"], fit_seed)
            classifier.fit(xt, y > 0)
            probability = classifier.predict_proba(xv)[:, 1]
            model = make_regressor(setting["loss"], setting["regressor_structure"], fit_seed)
            model.fit(xt[y > 0], y[y > 0])
            positive = clip(model.predict(xv))
            predictions[f"hard_hurdle__{feature_set}"] = np.where(probability >= setting["threshold"], positive, 0.0)
            print(f"ablation fold {outer}: {feature_set} complete", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(checkpoint, index=False)
    return outer


def summarize_ablations(output):
    prediction = pd.concat([pd.read_csv(output / "ablations" / f"fold_{fold}_predictions.csv") for fold in range(5)])
    prediction = prediction.sort_values("analysis_row")
    prediction.to_csv(output / "ablation_oof_predictions.csv", index=False)
    model_columns = [c for c in prediction if "__" in c]
    rows = []
    for fold in ["pooled", 0, 1, 2, 3, 4]:
        subset = prediction if fold == "pooled" else prediction.loc[prediction.outer_fold == fold]
        for column in model_columns:
            model, features = column.split("__")
            rows.append({"fold": fold, "model": model, "feature_set": features, "n_features": len(FEATURE_SETS[features]),
                         **regression_metrics(subset[TARGET], subset[column], subset.component)})
    pd.DataFrame(rows).to_csv(output / "ablation_metrics.csv", index=False)
    print(pd.DataFrame(rows).query("fold == 'pooled'")[["model", "feature_set", "mae_pp"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the grouped coral bleaching analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Python 3.12\n"
            "pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 "
            "joblib==1.5.3 threadpoolctl==3.6.0\n"
            "python coral_bleaching.py --download\n"
            "python coral_bleaching.py --source global_bleaching_environmental.csv\n"
            "python coral_bleaching.py --data coral_bleaching_clean.csv"
        ),
    )
    parser.add_argument("--data", type=Path, help="Optional cleaned CSV instead of the raw source.")
    parser.add_argument("--source", type=Path, default=Path("global_bleaching_environmental.csv"))
    parser.add_argument("--download", action="store_true", help="Download the frozen public CSV if --source is absent.")
    parser.add_argument("--output", type=Path, default=Path("coral_bleaching_results"))
    parser.add_argument("--features", choices=FEATURE_SETS, default="full42")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--stage", choices=["prepare", "main", "ablations", "all", "summarize"], default="all")
    parser.add_argument("--ablations", nargs="+", default=["thermal_only", "no_geography", "no_year", "restricted_window"], choices=list(FEATURE_SETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = args.data if args.data is not None else prepare_input(args.source, args.output, args.download)
    frame, report = prepare(data_path)
    config = {
        "seed": args.seed, "feature_set": args.features, "features": FEATURE_SETS[args.features],
        "feature_sets": FEATURE_SETS, "outer_folds": 5, "inner_folds": 3,
        "splitter": "GroupKFold(shuffle=True), connected components; seed and seed+100+outer",
        "hgb_structures": STRUCTURES, "hgb_max_iter": 200, "hgb_learning_rate": .07,
        "hgb_early_stopping": False, "gate_candidates": GATES, "selection_metric": "inner OOF observation MAE",
        "forest_parameters": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": .8},
        "bootstrap_replicates": args.bootstrap_reps, "bootstrap_unit": "connected location component",
        "bootstrap_interpretation": "percentile fixed-fit CI, conditional on predictions; no model refitting",
        "workers": args.workers, "threads_per_worker": args.threads,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "input_sha256": report["input_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    old_config_path = args.output / "run_configuration.json"
    if old_config_path.exists() and not args.force:
        old = json.loads(old_config_path.read_text())
        for key in ["seed", "feature_set", "features", "input_sha256", "hgb_structures", "hgb_max_iter", "hgb_learning_rate", "hgb_early_stopping", "gate_candidates", "forest_parameters", "outer_folds", "inner_folds"]:
            if old.get(key) != config[key]:
                raise ValueError(f"Existing run has different {key}; choose new output directory or --force.")
    write_json(old_config_path, config)
    inner_map = build_folds(frame, args.output, args.seed)
    write_json(args.output / "data_preparation.json", report)
    print(f"Prepared {len(frame)} unique samples / {frame.component.nunique()} location components", flush=True)
    if args.stage in ["main", "all"]:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_fold, frame, outer, inner_map[outer], args.features,
                                   str(args.output), args.seed, args.threads, args.force) for outer in range(5)]
            for future in as_completed(futures):
                future.result()
    if args.stage in ["main", "all", "summarize"]:
        summarize(args.output, args.bootstrap_reps, args.seed)
    if args.stage in ["ablations", "all"]:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_ablation_fold, frame, outer, str(args.output), args.seed,
                                   args.threads, args.ablations, args.force) for outer in range(5)]
            for future in as_completed(futures):
                future.result()
        summarize_ablations(args.output)

if __name__ == "__main__":
    main()
