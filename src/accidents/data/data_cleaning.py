from __future__ import annotations
 
from pathlib import Path
import pandas as pd
import numpy as np
 

# Colonnes avec >60 % de valeurs manquantes du graphique de data_exploration
COLS_TROP_DE_NAN = [
    "BORNE_KM_ACCDN",      
    "CD_PNT_CDRNL_ROUTE",  
    "SFX_NO_CIVIQ_ACCDN",  
    "NO_ROUTE",             
    "CD_ZON_TRAVX_ROUTR",  
    "CD_SIT_PRTCE_ACCDN",  
    "CD_POSI_ACCDN",         
    "CD_ETAT_CHASS",        
    "NB_METRE_DIST_ACCD",  
    "CD_PNT_CDRNL_REPRR",  
]
 
# Colonnes d'identification ou textuelles qu'on trouve juste inutile
COLS_IDENTIFICATION = [
    "NO_SEQ_COLL",           # identifiant unique, pas de valeur analytique
    "CD_MUNCP",              # code municipal, redondant avec REG_ADM/MRC
    "NO_CIVIQ_ACCDN",        # numéro civique, trop granulaire, 57% NaN
    "RUE_ACCDN",             # nom de rue, trop de valeurs uniques
    "ACCDN_PRES_DE",         # intersection proche, texte libre, 33% NaN
    "TP_REPRR_ACCDN",        # type de repère, 38% NaN, peu informatif
]
 
# Colonnes redondantes
COLS_REDONDANTES = [
    "LOC_X",                 # redondant avec LOC_LONG
    "LOC_Y",                 # redondant avec LOC_LAT
]
 
 
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes inutilisables (trop de NaN),
    les identifiants textuels et les colonnes redondantes.
    """
    cols_to_drop = COLS_TROP_DE_NAN + COLS_IDENTIFICATION + COLS_REDONDANTES
    cols_present = [c for c in cols_to_drop if c in df.columns]
    cols_absent = [c for c in cols_to_drop if c not in df.columns]
 
    if cols_absent:
        print(f"Colonnes déjà absentes (ignorées) : {cols_absent}")
 
    df = df.drop(columns=cols_present)
    print(f"{len(cols_present)} colonnes supprimées, {df.shape[1]} colonnes restantes")
    return df
 
 
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme DT_ACCDN et HEURE_ACCDN en features exploitables.
    - DT_ACCDN ("2012/02/01") devient mois, saison
    - HEURE_ACCDN ("02:00:00-02:59:00" ou "Non précisé") devient heure entière (0-23)
    - JR_SEMN_ACCDN est déjà le jour de la semaine (LU, MA, ...), on le garde
    - AN existe déjà comme colonne entière, on le garde
    """
    dt = pd.to_datetime(df["DT_ACCDN"], format="%Y/%m/%d", errors="coerce")
    df["MOIS"] = dt.dt.month
 
    saison_map = {
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Ete", 7: "Ete", 8: "Ete",
        9: "Automne", 10: "Automne", 11: "Automne",
    }
    df["SAISON"] = df["MOIS"].map(saison_map)
 
    def extract_hour(val):
        if pd.isna(val) or "Non" in str(val):
            return np.nan
        try:
            return int(str(val).split(":")[0])
        except (ValueError, IndexError):
            return np.nan
 
    df["HEURE"] = df["HEURE_ACCDN"].apply(extract_hour)
 
    df = df.drop(columns=["DT_ACCDN", "HEURE_ACCDN"])
 
    n_heure_nan = df["HEURE"].isna().sum()
    n_mois_nan = df["MOIS"].isna().sum()
    print(f"Colonnes créées : MOIS, SAISON, HEURE")
    print(f"HEURE : {n_heure_nan} NaN ({n_heure_nan/len(df)*100:.2f}%)")
    print(f"MOIS  : {n_mois_nan} NaN ({n_mois_nan/len(df)*100:.2f}%)")
    print(f"Colonnes supprimées : DT_ACCDN, HEURE_ACCDN")
    return df
 
 
# Ces colonnes sont des codes numériques mais représentent des catégories
COLS_CODES_CATEGORIELS = [
    "CD_GENRE_ACCDN",       # genre d'accident
    "CD_ETAT_SURFC",        # état de la surface
    "CD_ECLRM",             # éclairage
    "CD_ENVRN_ACCDN",       # environnement
    "CD_CATEG_ROUTE",       # catégorie de route
    "CD_ASPCT_ROUTE",       # aspect de la route
    "CD_LOCLN_ACCDN",       # localisation longitudinale
    "CD_CONFG_ROUTE",       # configuration de la route
    "CD_COND_METEO",        # conditions météo
]
 
 
def retype_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes CD_* de float64 vers string (catégoriel).
    Les NaN restent NaN, les valeurs numériques deviennent des strings
    comme "11", "21", etc. pour éviter que le ML les traite comme continus.
    """
    for col in COLS_CODES_CATEGORIELS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda x: str(int(x)) if pd.notna(x) else np.nan
        )
    print(f"{len(COLS_CODES_CATEGORIELS)} colonnes CD_* converties en catégoriel (string)")
    return df
 
 
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratégie d'imputation :
    - Colonnes catégorielles (string/object): NaN devient catégorie "Inconnu"
    - VITESSE_AUTOR (37% NaN) : tirage aléatoire selon la distribution par CD_CATEG_ROUTE
      pour préserver la distribution réelle sans créer de pic artificiel
    - HEURE (si NaN, "Non précisé") : tirage aléatoire selon la distribution réelle
    - LOC_LONG / LOC_LAT (0.01% NaN) ici on supprime les lignes
    Commentaire : Enfait on voulait utiliser pour VITESSE_AUTOR et HEURE simplement la valeur médiane
    et remplacer tout les NaN par cette valeur mais ça créeait un immense pic artificiel on a donc préféré le tirage aléatoire
    """
    n_avant = len(df)
 
    df = df.dropna(subset=["LOC_LONG", "LOC_LAT"])
    n_apres_coord = len(df)
    print(f"{n_avant - n_apres_coord} lignes supprimées (coordonnées manquantes)")
 
    # VITESSE_AUTOR : tirage aléatoire selon la distribution par catégorie de route
    if "VITESSE_AUTOR" in df.columns:
        n_vit_nan = df["VITESSE_AUTOR"].isna().sum()
        rng = np.random.default_rng(seed=3795)
 
        if "CD_CATEG_ROUTE" in df.columns:
            for cat, group in df.groupby("CD_CATEG_ROUTE"):
                mask_nan = group["VITESSE_AUTOR"].isna()
                n_to_fill = mask_nan.sum()
                if n_to_fill == 0:
                    continue
                known = group["VITESSE_AUTOR"].dropna().values
                if len(known) > 0:
                    sampled = rng.choice(known, size=n_to_fill)
                else:
                    # Fallback : distribution globale
                    sampled = rng.choice(df["VITESSE_AUTOR"].dropna().values, size=n_to_fill)
                df.loc[mask_nan.index[mask_nan], "VITESSE_AUTOR"] = sampled
 
        # Fallback final pour les NaN restants (ex: CD_CATEG_ROUTE aussi manquant)
        n_restants = df["VITESSE_AUTOR"].isna().sum()
        if n_restants > 0:
            known_global = df["VITESSE_AUTOR"].dropna().values
            df.loc[df["VITESSE_AUTOR"].isna(), "VITESSE_AUTOR"] = rng.choice(
                known_global, size=n_restants
            )
            print(f"VITESSE_AUTOR : {n_restants} NaN restants remplis par tirage sur la distribution globale")
 
        print(f"VITESSE_AUTOR : {n_vit_nan} NaN imputés (tirage aléatoire par catégorie de route)")
 
    # HEURE : tirage aléatoire selon la distribution réelle des heures connues
    if "HEURE" in df.columns:
        n_h_nan = df["HEURE"].isna().sum()
        if n_h_nan > 0:
            rng_h = np.random.default_rng(seed=3795)
            known_hours = df["HEURE"].dropna().values
            df.loc[df["HEURE"].isna(), "HEURE"] = rng_h.choice(known_hours, size=n_h_nan)
        print(f"HEURE : {n_h_nan} NaN remplacés par tirage aléatoire selon la distribution réelle")
 
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "GRAVITE"]
    for col in cat_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna("Inconnu")
            print(f"{col} : {n_nan} NaN changé en 'Inconnu'")
 
    remaining_nan = df.isna().sum()
    remaining_nan = remaining_nan[remaining_nan > 0]
    if len(remaining_nan) > 0:
        print(f"NaN restants :")
        for col, n in remaining_nan.items():
            print(f"    {col}: {n}")
    else:
        print(f"Plus aucun NaN dans le DataFrame")
 
    return df
 
 

 
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes dupliquées exactes."""
    n_avant = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup = n_avant - len(df)
    print(f"{n_dup} doublons supprimés, {len(df)} lignes restantes")
    return df
 
 
# on regroupe mortel et blessures graves comme Grave et tous les dommages matériels uniquement comme Materiel.
# Note : le jeu de données contient des variantes d'écriture ("Léger", "Grave") qu'il faut aussi mapper.
GRAVITE_MAP = {
    "Mortel": "Grave",
    "Blessures graves": "Grave",
    "Grave": "Grave",
    "Blessures légères": "Leger",
    "Léger": "Leger",
    "Dommages matériels seulement": "Materiel",
    "Dommages matériels inférieurs au seuil de rapportage": "Materiel",
}
 
 
def regroup_gravite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les 5 niveaux de gravité en 3 classes :
    - Grave   : Mortel + Blessures graves
    - Leger   : Blessures légères
    - Materiel : DMS + DMISR
    """
    df["GRAVITE_3"] = df["GRAVITE"].map(GRAVITE_MAP)
 
    unmapped = df["GRAVITE_3"].isna().sum()
    if unmapped > 0:
        print(f"{unmapped} valeurs de GRAVITE non reconnues !")
        print(f"Valeurs inconnues : {df.loc[df['GRAVITE_3'].isna(), 'GRAVITE'].unique()}")
 
    print(f"GRAVITE regroupée en 3 classes (GRAVITE_3) :")
    print(df["GRAVITE_3"].value_counts().to_string(header=False))
    return df
 
 
def run_full_pipeline(
    csv_path: str | Path,
    save_dir: str | Path = "data/clean",
    filename: str = "collisions_clean.csv",
) -> pd.DataFrame:
    """
    Exécute toutes les étapes de nettoyage et sauvegarde le résultat.
    """
    print("On commence le nettoyage")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nDonnées chargées : {df.shape[0]:,} lignes × {df.shape[1]} colonnes\n")
 
    df = drop_columns(df)
    df = parse_datetime(df)
    df = retype_codes(df)
    df = impute_missing(df)
    df = drop_duplicates(df)
    df = regroup_gravite(df)
 
    print(f"Netoyage terminé")
    print(f"Shape finale : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"NaN restants : {df.isna().sum().sum()}")
    print(f"{'=' * 60}")
 
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    df.to_csv(out_path, index=False)
    print(f"\nSauvegardé : {out_path}  ({out_path.stat().st_size / 1e6:.1f} Mo)")
 
    return df