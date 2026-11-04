"""Shared helpers for the Spaceship Titanic project (AI6102).

Every notebook starts with:
    from ss_utils import *
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY = ["RoomService", "Spa", "VRDeck"]
BASIC = ["FoodCourt", "ShoppingMall"]
CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]  # string categoricals
LOG_PATH = "results/experiment_log.csv"


# ----------------------------------------------------------------------------
# 1. Loading and parsing
# ----------------------------------------------------------------------------
def load_data(data_dir="."):
    """Read train/test and stack them into one frame with a 'split' column."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train["split"], test["split"] = "train", "test"
    df = pd.concat([train, test], ignore_index=True)
    return parse_basic(df)


def _to_float_bool(s):
    """'True'/'False'/True/False/NaN -> 1.0/0.0/NaN."""
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})


def parse_basic(df):
    """Split composite columns. These parsed columns are needed for imputation
    whichever feature groups are used later."""
    df = df.copy()
    df["CryoSleep"] = _to_float_bool(df["CryoSleep"])
    df["VIP"] = _to_float_bool(df["VIP"])
    if "Transported" in df:
        df["Transported"] = _to_float_bool(df["Transported"])
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"], df["CabinNum"], df["Side"] = cabin[0], cabin[1].astype(float), cabin[2]
    df["Surname"] = df["Name"].str.split().str[-1]
    return df


# ----------------------------------------------------------------------------
# 2. Imputation strategies (used in experiment E2)
# ----------------------------------------------------------------------------
def impute_native(df):
    """Leave NaNs for models that handle them (LightGBM, XGBoost, CatBoost)."""
    return df.copy(), {}


def impute_simple(df):
    """Median for numbers, mode for categories. The generic baseline."""
    df = df.copy()
    for c in ["Age", "CabinNum"] + SPEND:
        df[c] = df[c].fillna(df[c].median())
    for c in ["CryoSleep", "VIP"] + CAT_COLS:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df, {}


def _fill_from_key(df, col, key):
    """Fill NaNs in col with the most common value among rows sharing key."""
    known = df.dropna(subset=[col, key])
    if known.empty:
        return 0
    mode = known.groupby(key)[col].agg(lambda s: s.mode().iloc[0])
    mask = df[col].isna() & df[key].notna()
    fill = df.loc[mask, key].map(mode)
    df.loc[mask, col] = fill
    return int(fill.notna().sum())


def impute_logic(df, cryo_from_zero_spend=True):
    """Domain-rule imputation. Returns (df, counts) where counts records how
    many values each rule filled (report this in E2)."""
    df = df.copy()
    n = {}
    nan_before = int(df[SPEND + ["CryoSleep", "HomePlanet", "Deck", "Side"]].isna().sum().sum())

    # Rule 1: asleep passengers spend nothing
    m = df["CryoSleep"] == 1
    n["R1 cryo -> spend 0"] = int(df.loc[m, SPEND].isna().sum().sum())
    df.loc[m, SPEND] = df.loc[m, SPEND].fillna(0)

    # Rule 2: any spending means awake; all-zero spending suggests asleep
    spent = (df[SPEND].fillna(0) > 0).any(axis=1)
    m = df["CryoSleep"].isna() & spent
    n["R2a spend>0 -> awake"] = int(m.sum())
    df.loc[m, "CryoSleep"] = 0.0
    if cryo_from_zero_spend:
        all_zero = df[SPEND].notna().all(axis=1) & (df[SPEND].sum(axis=1) == 0)
        m = df["CryoSleep"].isna() & all_zero
        n["R2b zero spend -> asleep"] = int(m.sum())
        df.loc[m, "CryoSleep"] = 1.0

    # Rule 3: children (<=12) spend nothing
    m = df["Age"] <= 12
    n["R3 child -> spend 0"] = int(df.loc[m, SPEND].isna().sum().sum())
    df.loc[m, SPEND] = df.loc[m, SPEND].fillna(0)

    # Rule 4: home planet from group, then surname, then deck
    n["R4a planet <- group"] = _fill_from_key(df, "HomePlanet", "GroupId")
    n["R4b planet <- surname"] = _fill_from_key(df, "HomePlanet", "Surname")
    n["R4c planet <- deck"] = _fill_from_key(df, "HomePlanet", "Deck")

    # Rule 5: cabin parts from group mates
    n["R5 deck <- group"] = _fill_from_key(df, "Deck", "GroupId")
    n["R5 side <- group"] = _fill_from_key(df, "Side", "GroupId")
    n["R5 surname <- group"] = _fill_from_key(df, "Surname", "GroupId")

    # Rule 6: whatever is left -> generic fill
    left = int(df[SPEND + ["CryoSleep", "HomePlanet", "Deck", "Side"]].isna().sum().sum())
    n["R6 generic fill (rest)"] = left
    df, _ = impute_simple(df)
    n["(missing before, these columns)"] = nan_before
    return df, n


IMPUTERS = {"native": impute_native, "simple": impute_simple, "logic": impute_logic}


# ----------------------------------------------------------------------------
# 3. Feature groups (used in experiment E1)
# ----------------------------------------------------------------------------
BASE_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"] + SPEND
GROUP_FEATURES = {
    "G1": ["Deck", "Side", "CabinNum", "CabinRegion"],
    "G2": ["GroupSize", "IsSolo", "FamilySize"],
    "G3": [f"log_{c}" for c in SPEND] + ["log_TotalSpend", "NoSpend", "log_Luxury", "log_Basic"],
    "G4": ["AgeGroup", "IsChild"],
}


def add_features(df, region_width=300):
    """Compute every engineered column. Which ones a model sees is decided
    later by feature_list(), so the ablation only changes the column list."""
    df = df.copy()
    # G1 cabin
    df["CabinRegion"] = (df["CabinNum"] // region_width)
    # G2 group / family (uses train+test together: no target involved)
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("size")
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
    df["FamilySize"] = df.groupby("Surname")["Surname"].transform("size")
    df["FamilySize"] = df["FamilySize"].fillna(df["GroupSize"])  # unknown surname
    # G3 spending
    df["TotalSpend"] = df[SPEND].sum(axis=1, min_count=1)
    df["Luxury"] = df[LUXURY].sum(axis=1, min_count=1)
    df["Basic"] = df[BASIC].sum(axis=1, min_count=1)
    for c in SPEND + ["TotalSpend", "Luxury", "Basic"]:
        df[f"log_{c}"] = np.log1p(df[c])
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    # G4 age
    df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 17, 25, 40, 200], labels=False)
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    return df


def feature_list(groups=("G1", "G2", "G3", "G4")):
    cols = list(BASE_FEATURES)
    for g in groups:
        cols += GROUP_FEATURES[g]
    return cols


def build(df_raw, impute="logic", groups=("G1", "G2", "G3", "G4")):
    """One call from parsed data to model inputs.
    Returns X_train, y, groups (GroupId), X_test, test_ids, impute_counts."""
    df, counts = IMPUTERS[impute](df_raw)
    df = add_features(df)
    cols = feature_list(groups)
    X = df[cols].copy()
    for c in cols:
        if c in CAT_COLS:
            X[c] = X[c].astype("category")
    tr = (df["split"] == "train").to_numpy()
    X_tr, X_te = X[tr].reset_index(drop=True), X[~tr].reset_index(drop=True)
    y = df.loc[tr, "Transported"].astype(int).to_numpy()
    grp = df.loc[tr, "GroupId"].to_numpy()
    ids = df.loc[~tr, "PassengerId"].to_numpy()
    return X_tr, y, grp, X_te, ids, counts


# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
def prepare(X_tr, X_te=None, model="lgbm"):
    """Convert features into the form each library expects."""
    cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    if model in ("logreg", "rf", "mlp"):
        allX = pd.concat([X_tr, X_te]) if X_te is not None else X_tr
        allX = pd.get_dummies(allX, columns=cats, dtype=float)
        allX = allX.fillna(allX.median())
        a, b = allX.iloc[: len(X_tr)], allX.iloc[len(X_tr):]
        return a.reset_index(drop=True), (b.reset_index(drop=True) if X_te is not None else None)
    if model == "cat":
        def to_str(X):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return X
        return to_str(X_tr), (to_str(X_te) if X_te is not None else None)
    return X_tr, X_te  # lgbm, xgb, hgb use pandas categories directly


def get_model(model="lgbm", seed=0, params=None, cat_cols=None):
    p = dict(params or {})
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=p.get("C", 1.0)))
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 500),
                                      min_samples_leaf=p.get("min_samples_leaf", 2),
                                      n_jobs=-1, random_state=seed)
    if model == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                             early_stopping=True, max_iter=500, random_state=seed))
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed, **p)
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, verbose=-1, **p)
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, enable_categorical=True, tree_method="hist",
                             eval_metric="logloss", **p)
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, cat_features=cat_cols, **p)
    raise ValueError(model)


# ----------------------------------------------------------------------------
# 5. Cross-validation
# ----------------------------------------------------------------------------
def get_folds(y, groups, scheme="group", n_splits=5, seed=0):
    if scheme == "group":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def run_cv(model, X, y, groups, params=None, seeds=(0, 1, 2), scheme="group", n_splits=5, verbose=True):
    """Repeated k-fold CV. Returns dict with mean/std accuracy over all folds,
    out-of-fold probabilities (averaged over seeds) and run time."""
    Xp, _ = prepare(X, None, model)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"] if model == "cat" else None
    t0 = time.time()
    fold_scores, oof = [], np.zeros(len(y))
    for s in seeds:
        oof_s = np.zeros(len(y))
        for tr, va in get_folds(y, groups, scheme, n_splits, seed=s):
            m = get_model(model, seed=s, params=params, cat_cols=cat_cols)
            m.fit(Xp.iloc[tr], y[tr])
            oof_s[va] = m.predict_proba(Xp.iloc[va])[:, 1]
            fold_scores.append(accuracy_score(y[va], oof_s[va] > 0.5))
        oof += oof_s / len(seeds)
    res = {"mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)),
           "folds": np.array(fold_scores),
           "oof": oof, "oof_acc": float(accuracy_score(y, oof > 0.5)),
           "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"{model:7s} CV acc = {res['mean']:.4f} ± {res['std']:.4f}  "
              f"(oof {res['oof_acc']:.4f}, {res['seconds']}s)")
    return res


def paired_test(res_a, res_b, name_a="A", name_b="B"):
    """Compare two run_cv results made with the SAME seeds and scheme
    (so fold k is the same split in both). Paired t-test over folds."""
    from scipy.stats import ttest_rel
    d = res_b["folds"] - res_a["folds"]
    t, p = ttest_rel(res_b["folds"], res_a["folds"])
    print(f"{name_b} - {name_a}: {d.mean():+.4f} accuracy (better on {(d > 0).sum()}/{len(d)} folds), p = {p:.3f}")
    return d.mean(), p


def fit_predict_test(model, X_tr, y, X_te, params=None, seeds=(0, 1, 2, 3, 4)):
    """Train on all training rows with several seeds; return mean test probability."""
    a, b = prepare(X_tr, X_te, model)
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"] if model == "cat" else None
    preds = []
    for s in seeds:
        m = get_model(model, seed=s, params=params, cat_cols=cat_cols)
        m.fit(a, y)
        preds.append(m.predict_proba(b)[:, 1])
    return np.mean(preds, axis=0)


# ----------------------------------------------------------------------------
# 6. Logging and submission
# ----------------------------------------------------------------------------
def log_result(run_id, experiment, description, model, res, lb_score=None, path=LOG_PATH):
    """Append one row to the experiment log (the source of every report table)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = pd.DataFrame([{
        "run_id": run_id, "date": time.strftime("%Y-%m-%d %H:%M"), "experiment": experiment,
        "description": description, "model": model,
        "cv_mean": round(res["mean"], 5), "cv_std": round(res["std"], 5),
        "oof_acc": round(res["oof_acc"], 5), "seconds": res["seconds"], "lb_score": lb_score,
    }])
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def make_submission(ids, proba, name, threshold=0.5, out_dir="submissions"):
    os.makedirs(out_dir, exist_ok=True)
    sub = pd.DataFrame({"PassengerId": ids, "Transported": proba > threshold})
    path = os.path.join(out_dir, f"{name}.csv")
    sub.to_csv(path, index=False)
    print(f"Saved {path}: {len(sub)} rows, {sub['Transported'].mean():.3f} predicted True")
    return path
