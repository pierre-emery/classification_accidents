from __future__ import annotations
 
from pathlib import Path
import requests
import pandas as pd
 
 
URL_CSV = (
    "https://donnees.montreal.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b"
    "/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/collisions_routieres.csv"
)
 
 
def repo_root(start: Path | None = None) -> Path:
    """
    Remonte jusqu'a la racine du depot git.
    Permet de toujours ecrire dans data/ peu importe d'ou le script est lance.
    """
    p = (start or Path(__file__).resolve()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()
 
 
def fetch_collisions(
    raw_dir: str | Path = "data/raw",
    filename: str = "collisions_routieres.csv",
    force: bool = False,
) -> Path:
    """
    Telecharge le CSV des collisions routieres de Montreal (depuis 2012).
 
    Parameters
    ----------
    raw_dir  : dossier de destination (relatif a la racine du depot)
    filename : nom du fichier local
    force    : re-telecharger meme si le fichier existe deja
 
    Returns
    -------
    Path vers le fichier telecharge
    """
    root = repo_root()
    raw_dir = Path(raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
 
    out_path = raw_dir / filename
 
    if out_path.exists() and not force:
        print(f"Fichier deja present : {out_path}")
        return out_path
 
    print(f"Telechargement depuis :\n   {URL_CSV}")
    tmp = out_path.with_suffix(".csv.part")
 
    with requests.get(URL_CSV, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r   {pct:.1f}%  ({downloaded/1e6:.1f} Mo)", end="", flush=True)
 
    tmp.replace(out_path)
    print(f"\nSauvegarde : {out_path}  ({out_path.stat().st_size / 1e6:.1f} Mo)")
    return out_path
 
 
def load_collisions(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Charge le CSV brut en DataFrame.
    Passe tous les kwargs a pd.read_csv (ex. nrows=50_000 pour tester).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Lancez d'abord fetch_collisions() pour le telecharger."
        )
    df = pd.read_csv(path, low_memory=False, **kwargs)
    print(f"Charge : {df.shape[0]:,} lignes x {df.shape[1]} colonnes  ({path.name})")
    return df
 