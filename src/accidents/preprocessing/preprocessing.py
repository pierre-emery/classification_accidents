"""
Ce module applique les transformations suivantes :
  1. Imputation des NaN numeriques (VITESSE_AUTOR, HEURE) par tirage aleatoire
     dans un pool de valeurs observees uniquement sur l'ensemble train.
  2. Retrait des colonnes qui fuitent la cible (NB_MORTS, NB_BLESSES_*, etc.)
  3. Encodage des variables categorielles :
     - target encoding pour REG_ADM et MRC (beaucoup de modalites)
     - one-hot pour les autres categorielles a faible cardinalite
  4. Standardisation (z-score) des variables numeriques
"""

from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Colonnes qui encodent directement la cible GRAVITE_3, Les garder donne un score artificiellement parfait.
LEAKAGE_COLS = [
    "NB_MORTS",
    "NB_BLESSES_GRAVES",
    "NB_BLESSES_LEGERS",
    "NB_VICTIMES_TOTAL",
    "NB_DECES_PIETON",
    "NB_BLESSES_PIETON",
    "NB_VICTIMES_PIETON",
    "NB_DECES_MOTO",
    "NB_BLESSES_MOTO",
    "NB_VICTIMES_MOTO",
    "NB_DECES_VELO",
    "NB_BLESSES_VELO",
    "NB_VICTIMES_VELO",
]

# target encoding
HIGH_CARD_CAT_COLS = ["REG_ADM", "MRC"]

# Variables a cycle naturel : encodage sin/cos
# (valeur max = periode du cycle)
CYCLIC_COLS = {
    "HEURE": 24,   # 0..23
    "MOIS": 12,    # 1..12
}
 
# JR_SEMN_ACCDN encode en strings (LU, MA, ...) il faut d'abord le mapper en entiers avant l'encodage cyclique
JOUR_SEMAINE_MAP = {
    "LU": 0, "MA": 1, "ME": 2, "JE": 3,
    "VE": 4, "SA": 5, "DI": 6,
}
 
# one-hot
LOW_CARD_CAT_COLS = [
    "SAISON",            # 4 valeurs (on aurait pu faire comme avec les autres cycles mais on pense que on-hot suffit)
    "CD_GENRE_ACCDN",
    "CD_ETAT_SURFC",
    "CD_ECLRM",
    "CD_ENVRN_ACCDN",
    "CD_CATEG_ROUTE",
    "CD_ASPCT_ROUTE",
    "CD_LOCLN_ACCDN",
    "CD_CONFG_ROUTE",
    "CD_COND_METEO",
    "LOC_COTE_QD",       # A / B
    "LOC_DETACHEE",      # O / N
    "LOC_IMPRECISION",   # O / N
]
 

def drop_leakage(X: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in LEAKAGE_COLS if c in X.columns]
    return X.drop(columns=cols_present)


def fit_random_imputation(
    X_train: pd.DataFrame,
    seed: int = 3795,
) -> dict:
    """La fonction sert à imputer VITESSE_AUTOR et HEURE.
 
    L'idée est que pour chaque colonne a imputer, on memorise les valeurs
    observees dans le training set (eventuellement stratifiees par une autre colonne).
    Cet "apprentissage" ne touche jamais val/test, donc pas de leakage.
 
    Pour VITESSE_AUTOR : on stratifie par CD_CATEG_ROUTE pour preserver la
    distribution conditionnelle (les différentes routes ont différentes vitesses).
 
    Pour HEURE : on fait un tirage global (pas de stratification).
    """
    imputation_state = {"seed": seed, "pools": {}}
 
    # VITESSE_AUTOR : pool par CD_CATEG_ROUTE
    if "VITESSE_AUTOR" in X_train.columns:
        pools_vitesse = {}
        if "CD_CATEG_ROUTE" in X_train.columns:
            for cat, group in X_train.groupby("CD_CATEG_ROUTE"):
                known = group["VITESSE_AUTOR"].dropna().values
                if len(known) > 0:
                    pools_vitesse[cat] = known
        # pool global pour fallback (si une categorie de route n'existe qu'en val/test)
        pools_vitesse["_global"] = X_train["VITESSE_AUTOR"].dropna().values
        imputation_state["pools"]["VITESSE_AUTOR"] = pools_vitesse
 
    # HEURE : pool global
    if "HEURE" in X_train.columns:
        imputation_state["pools"]["HEURE"] = {
            "_global": X_train["HEURE"].dropna().values,
        }
 
    return imputation_state
 
 
def apply_random_imputation(
    X: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    """Applique l'imputation par tirage aleatoire sur X, en utilisant les
    pools appris sur le train."""
    X = X.copy()
    rng = np.random.default_rng(seed=state["seed"])
 
    # VITESSE_AUTOR : tirage stratifie par CD_CATEG_ROUTE
    if "VITESSE_AUTOR" in state["pools"]:
        pools = state["pools"]["VITESSE_AUTOR"]
        global_pool = pools["_global"]
        n_nan_before = X["VITESSE_AUTOR"].isna().sum()
 
        if "CD_CATEG_ROUTE" in X.columns:
            for cat, group in X.groupby("CD_CATEG_ROUTE"):
                mask_nan = group["VITESSE_AUTOR"].isna()
                n_to_fill = mask_nan.sum()
                if n_to_fill == 0:
                    continue
                # utiliser le pool de la categorie si dispo, sinon le pool global
                pool = pools.get(cat, global_pool)
                sampled = rng.choice(pool, size=n_to_fill)
                X.loc[mask_nan.index[mask_nan], "VITESSE_AUTOR"] = sampled
 
        # fallback pour les NaN restants (CD_CATEG_ROUTE manquant)
        n_restants = X["VITESSE_AUTOR"].isna().sum()
        if n_restants > 0:
            X.loc[X["VITESSE_AUTOR"].isna(), "VITESSE_AUTOR"] = rng.choice(
                global_pool, size=n_restants)
 
        if n_nan_before > 0:
            print(f"VITESSE_AUTOR : {n_nan_before} NaN imputés")
 
    # HEURE : tirage global
    if "HEURE" in state["pools"]:
        n_nan = X["HEURE"].isna().sum()
        if n_nan > 0:
            pool = state["pools"]["HEURE"]["_global"]
            X.loc[X["HEURE"].isna(), "HEURE"] = rng.choice(pool, size=n_nan)
            print(f"HEURE : {n_nan} NaN imputés")
 
    return X

def apply_cyclic_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """Encodage cyclique sin/cos pour les variables a cycle naturel.
 
    Une variable cyclique comme HEURE (0..23) a le probleme suivant :
    23h et 0h sont "proches" dans le temps mais tres eloignes numeriquement.
    L'encodage cyclique resout ca en projetant la valeur sur un cercle :
 
        sin(2*pi*val / periode)
        cos(2*pi*val / periode)
 
    Resultat : 23h et 0h deviennent deux points proches dans l'espace 2D
    (sin, cos), comme ils devraient l'etre semantiquement.
 
    Aucune statistique apprise ici, pas de fit/apply separes.
    """
    X = X.copy()
 
    # HEURE et MOIS : valeurs deja numeriques
    for col, period in CYCLIC_COLS.items():
        if col not in X.columns:
            continue
        values = X[col].astype(float)
        X[col + "_sin"] = np.sin(2 * np.pi * values / period)
        X[col + "_cos"] = np.cos(2 * np.pi * values / period)
        X = X.drop(columns=[col])
        print(f"Encodage cyclique applique a {col} (periode {period})")
 
    # JR_SEMN_ACCDN : strings (LU, MA, ...) -> mapper d'abord en entiers
    if "JR_SEMN_ACCDN" in X.columns:
        # certaines lignes peuvent avoir "Inconnu" apres le cleaning
        jours_num = X["JR_SEMN_ACCDN"].map(JOUR_SEMAINE_MAP)
        # Inconnu -> NaN -> on met sin=0, cos=0 (point au centre du cercle, neutre)
        X["JR_SEMN_sin"] = np.where(
            jours_num.notna(),
            np.sin(2 * np.pi * jours_num.fillna(0) / 7),
            0.0,
        )
        X["JR_SEMN_cos"] = np.where(
            jours_num.notna(),
            np.cos(2 * np.pi * jours_num.fillna(0) / 7),
            0.0,
        )
        X = X.drop(columns=["JR_SEMN_ACCDN"])
        print("Encodage cyclique applique a JR_SEMN_ACCDN (periode 7)")
 
    return X 
 
def fit_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cols: list[str],
    smoothing: float = 10.0,
) -> dict[str, dict]:
    """Calcule un encodage cible (target encoding) sur X_train.
 
    Pour chaque categorie d'une colonne, on remplace la valeur par la
    proportion d'une classe de reference (ici "Grave") parmi les lignes
    de cette categorie. Un lissage bayesien est applique pour eviter les
    estimations instables sur les categories rares :
 
        encoded = (n_cat * mean_cat + smoothing * mean_global)
                  / (n_cat + smoothing)
 
    On encode la probabilite P(GRAVITE_3 == "Grave") car c'est la classe
    la plus informative et rare (les variations entre quartiers/regions
    devraient surtout se voir sur les cas graves).
    """
    y_binary = (y_train == "Grave").astype(int)
    global_mean = y_binary.mean()
 
    encodings = {"_global_mean": global_mean}
    for col in cols:
        if col not in X_train.columns:
            continue
        df_tmp = pd.DataFrame({col: X_train[col].values, "y": y_binary.values})
        agg = df_tmp.groupby(col)["y"].agg(["mean", "count"])
        smoothed = (
            (agg["count"] * agg["mean"] + smoothing * global_mean)
            / (agg["count"] + smoothing)
        )
        encodings[col] = smoothed.to_dict()
    return encodings
 
 
def apply_target_encoding(
    X: pd.DataFrame,
    encodings: dict,
    cols: list[str],
) -> pd.DataFrame:

    X = X.copy()
    global_mean = encodings["_global_mean"]
    for col in cols:
        if col not in X.columns:
            continue
        mapping = encodings[col]
        # categories inconnues (presentes en val/test mais pas en train)
        # -> moyenne globale comme valeur neutre
        X[col + "_te"] = X[col].map(mapping).fillna(global_mean)
        X = X.drop(columns=[col])
    return X
 
 
def fit_one_hot(X_train: pd.DataFrame, cols: list[str]) -> dict[str, list]:

    categories = {}
    for col in cols:
        if col in X_train.columns:
            categories[col] = sorted(X_train[col].astype(str).unique().tolist())
    return categories
 
 
def apply_one_hot(
    X: pd.DataFrame,
    categories: dict[str, list],
) -> pd.DataFrame:

    X = X.copy()
    for col, cats in categories.items():
        if col not in X.columns:
            continue
        col_str = X[col].astype(str)
        # modalites inconnues -> "_other" pour rester coherent entre train/val/test
        col_str = col_str.where(col_str.isin(cats), other="_other")
        cats_with_other = cats + (["_other"] if "_other" not in cats else [])
        dummies = pd.get_dummies(
            pd.Categorical(col_str, categories=cats_with_other),
            prefix=col,
            dtype=float,
        )
        dummies.index = X.index
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    return X
 
 
def preprocess_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    # 1. imputation des NaN numeriques (sans leakage)
    imputation_state = fit_random_imputation(X_train)
    print("--- Imputation train ---")
    X_train = apply_random_imputation(X_train, imputation_state)
    print("--- Imputation val ---")
    X_val = apply_random_imputation(X_val, imputation_state)
    print("--- Imputation test ---")
    X_test = apply_random_imputation(X_test, imputation_state)

    # 2. retrait des colonnes de fuite
    X_train = drop_leakage(X_train)
    X_val = drop_leakage(X_val)
    X_test = drop_leakage(X_test)
    print(f"Colonnes de fuite retirees. Shape apres : {X_train.shape}")
 
    # 3. encodage
    # encodage cyclique (HEURE, MOIS, JR_SEMN_ACCDN)
    X_train = apply_cyclic_encoding(X_train)
    X_val = apply_cyclic_encoding(X_val)
    X_test = apply_cyclic_encoding(X_test)

    # target encoding 
    target_cols_present = [c for c in HIGH_CARD_CAT_COLS if c in X_train.columns]
    if target_cols_present:
        encodings = fit_target_encoding(X_train, y_train, target_cols_present)
        X_train = apply_target_encoding(X_train, encodings, target_cols_present)
        X_val = apply_target_encoding(X_val, encodings, target_cols_present)
        X_test = apply_target_encoding(X_test, encodings, target_cols_present)
        print(f"Target encoding applique a : {target_cols_present}")
 
    # one-hot encoding
    oh_cols_present = [c for c in LOW_CARD_CAT_COLS if c in X_train.columns]
    if oh_cols_present:
        categories = fit_one_hot(X_train, oh_cols_present)
        X_train = apply_one_hot(X_train, categories)
        X_val = apply_one_hot(X_val, categories)
        X_test = apply_one_hot(X_test, categories)
        print(f"One-hot applique a : {oh_cols_present}")
        print(f"Nombre de colonnes apres one-hot : {X_train.shape[1]}")
 
    # 4. alignement final des colonnes (securite) puis scaling
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
 
    non_num = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise ValueError(
            f"Colonnes non-numeriques restantes apres encodage : {non_num}. "
            "Ajoute-les a LOW_CARD_CAT_COLS ou HIGH_CARD_CAT_COLS."
        )
 
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
 
    cols = X_train.columns
    X_train_proc = pd.DataFrame(X_train_arr, columns=cols, index=X_train.index)
    X_val_proc = pd.DataFrame(X_val_arr, columns=cols, index=X_val.index)
    X_test_proc = pd.DataFrame(X_test_arr, columns=cols, index=X_test.index)
 
    print(f"Standardisation appliquee. Shape finale : {X_train_proc.shape}")
    return X_train_proc, X_val_proc, X_test_proc