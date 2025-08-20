
import zipfile
from pathlib import Path
import pandas as pd
import logging
from charset_normalizer import from_path
import re
import matplotlib.pyplot as plt
from typing import Sequence
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("./histo_tirages_from_fdj/raw")
SILVER_DIR = Path("./histo_tirages_from_fdj/silver")
GOLD_DIR = Path("./histo_tirages_from_fdj/gold")
OUTPUT_FILE = GOLD_DIR / "full_histo.csv"

REQUIRED_COLS = [
    "date_de_tirage",
    "boule_1",
    "boule_2",
    "boule_3",
    "boule_4",
    "boule_5",
    "etoile_1",
    "etoile_2",
]

DATE_PATTERNS = [
    (re.compile(r"^\d{8}$"), "%Y%m%d"),                 # AAAAMMDD
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),   # JJ/MM/AAAA
]

# utiliser le 04 février 2020 pour vérifier la cohérence des données avec la FDJ
DATE_MIN = pd.Timestamp("2004-02-13")
#DATE_MIN = pd.Timestamp("2020-02-04")

# Dézippe chaque archive .zip de raw vers silver
def unzip_all_from_raw_to_silver():
    written_csvs = []
    for z in sorted(RAW_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                member = zf.namelist()[0]  # on prend directement le premier fichier
                target_path = SILVER_DIR / Path(member).name
                with zf.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
                written_csvs.append(target_path)
                logger.info(f"Traitement de l'archive {z.name} -> extraction du fichier {target_path.name}")
        except Exception as e:
            logger.error(f"Erreur avec {z.name}: {e}")
    return written_csvs

# Lecture d'un fichier csv avec au préalable détection de l'encodage et en sortie nettoyage des noms de colonne
def read_csv(path: Path) -> pd.DataFrame:  
    result = from_path(str(path)).best()
    encoding = result.encoding if result else "utf-8"
    logger.info(f"Lecture de {path.name} avec encodage détecté : {encoding}")
    df = pd.read_csv(path, sep=";", encoding=encoding, dtype=str, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    return df

# Devine le format de date en se basant sur un échantillon des 20 premières lignes
def guess_date_format(series: pd.Series) -> str | None:
    sample = series.dropna().astype(str).head(20)
    for pat, format in DATE_PATTERNS:
        if sample.apply(lambda x: bool(pat.match(x))).all():
            return format
    return None  # inconnu → fallback


# Vérifie et nettoie un DataFrame :
#1 -> toutes les colonnes requises doivent être présentes
#2 -> `date_de_tirage` doit être une date
#3 -> les autres colonnes doivent être des entiers
def clean_csv(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    df_clean = df.copy()

    # Vérification colonnes
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: colonnes manquantes: {missing}")

    # Hack pour traiter le cas où les dates sont au format dd/mm/aa et non dd/mm/aaaa
    df_clean["date_de_tirage"] = df_clean["date_de_tirage"].str.replace(
        r"^(\d{2}/\d{2}/)(\d{2})$",
        lambda m: f"{m.group(1)}20{m.group(2)}",
        regex=True
    )

    # Vérification et uniformisation des dates
    _format = guess_date_format(df_clean["date_de_tirage"])
    if _format:
        df_clean["date_de_tirage"] = pd.to_datetime(df_clean["date_de_tirage"], format=_format, errors="coerce")
        logger.info(f"{file_path.name}: parsing de date avec format détecté {_format}")
    else:
        df_clean["date_de_tirage"] = pd.NaT
        df_clean["date_de_tirage"] = pd.to_datetime(df_clean["date_de_tirage"])
        logger.warning(f"{file_path.name}: format de date non reconnu")

    # Vérification des doublons pour s'assurer de ne pas avoir deux fois le même tirage dans les données
    non_null = df_clean["date_de_tirage"].notna()
    dup_mask = non_null & df_clean["date_de_tirage"].duplicated(keep=False)
    if dup_mask.any():
        dups = (
            df_clean.loc[dup_mask, "date_de_tirage"]
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
        )
        logger.error(f"{file_path.name}: doublons détectés sur date_de_tirage:\n{dups.to_string()}")
        raise ValueError(f"{file_path.name}: doublons de dates détectés ({dups.sum()} lignes concernées)")

    # Conversion des numéros en entiers
    for col in REQUIRED_COLS[1:]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce", downcast="integer")
        invalid_numbers = df_clean[col].isna().sum()
        if invalid_numbers > 0:
            logger.warning(f"{file_path.name}: {invalid_numbers} valeurs invalides dans {col}")

    # Remontées de logs
    bad_mask = df_clean["date_de_tirage"].isna()
    bad_dates = bad_mask.sum()
    if bad_dates:
        invalid_values = df.loc[bad_mask, "date_de_tirage"].unique().tolist()
        logger.warning(
            f"{file_path.name}: {bad_dates} dates invalides remplacées par NaT. "
            f"Valeurs concernées: {invalid_values}"
        )

    for col in REQUIRED_COLS[1:]:
        bad_nums = df_clean[col].isna().sum()
        if bad_nums:
            logger.warning(f"{file_path.name}: {bad_nums} valeurs invalides dans {col}")

    return df_clean

# Concatène les colonnes requises de tous les CSV de ./silver/ vers ./gold/full_histo.csv
def concatenate_silver_to_gold():
    csvs = sorted(SILVER_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {SILVER_DIR}.")
    frames = []
    for p in csvs:
        df = read_csv(p)              # lecture
        df = clean_csv(df, p)         # nettoyage + validation

        frames.append(df[REQUIRED_COLS])
        logger.info(f"Colonnes prises de {p.name}: {REQUIRED_COLS}")

    if not frames:
        raise RuntimeError("Aucun DataFrame valide à concaténer.")

    full = pd.concat(frames, ignore_index=True).drop_duplicates()

    try:
        full = full.sort_values("date_de_tirage")
    except Exception:
        pass

    full.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.info(f"Fichier généré: {OUTPUT_FILE}")

# Eclaircit une couleur RGB (0..1) vers le blanc selon `factor` (0=inchangé, 1=blanc)
def _lighten(color, factor):
    r, g, b = color
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

# Colormap dégradé clair → couleur de base (même teinte)
def _gradient_cmap(base_color):
    light = _lighten(base_color, 0.75)
    return LinearSegmentedColormap.from_list("grad", [light, base_color], N=256)

# Analyse des données
# Par "slot" :  on considère les 5 "emplacements" indépendamment (boule_1, boule_2, etc.) => nombre total d’apparitions
# Par "tirage" : on considère chaque tirage dans son ensemble => nombre de tirages contenant le numéro au moins une fois
# Au final les résultats sont identiques à un facteur 5 près
def data_analysis(csv_path: str | Path,
                  boule_cols: Sequence[str] = ("boule_1","boule_2","boule_3","boule_4","boule_5")) -> tuple[pd.DataFrame, pd.DataFrame]:

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str, encoding="utf-8")

    # Colonnes boules en numérique
    for c in boule_cols:
        if c not in df.columns:
            raise KeyError(f"Colonne manquante: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filtrage sur DATE_MIN (pour rappel, utiliser le 04/02/2020 pour comparaison avec la FDJ)
    if "date_de_tirage" in df.columns:
        df["date_de_tirage"] = pd.to_datetime(df["date_de_tirage"], errors="coerce", format="%Y-%m-%d")
        df = df[df["date_de_tirage"] >= DATE_MIN]

    # Méthode par "slot"
    stacked = pd.concat([df[c] for c in boule_cols], axis=0).dropna().astype(int)
    stacked = stacked[(stacked >= 1) & (stacked <= 50)] #garde fou mais les données FDJ semblent clean

    total_slots = len(stacked)
    counts_slot = stacked.value_counts().reindex(range(1, 51), fill_value=0).sort_index()
    percentages_slot = counts_slot / total_slots * 100.0

    res_slot = pd.DataFrame({
        "numero": np.arange(1, 51),
        "occurrences": counts_slot.values,
        "pourcentage": percentages_slot.values
    })

    # Méthode par "tirage"
    n_tirages = len(df)

    presence_par_tirage = {}

    for n in range(1, 51):
        is_present = (df[list(boule_cols)] == n).any(axis=1)
        count = is_present.sum()
        presence_par_tirage[n] = count

    counts_tirage = pd.Series(presence_par_tirage).sort_index()
    percentages_tirage = counts_tirage / n_tirages * 100.0

    res_tirage = pd.DataFrame({
        "numero": np.arange(1, 51),
        "occurrences": counts_tirage.values,
        "pourcentage": percentages_tirage.values
    })

    # Fonction de tracé
    def plot_frequencies(res: pd.DataFrame, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(10, 10))  # carré pour LinkedIn
        order_desc = res.sort_values("pourcentage", ascending=False)
        top_set = set(order_desc.iloc[:5]["numero"])
        bot_set = set(order_desc.iloc[-5:]["numero"])

        TOP_COLOR = (0.00, 0.55, 0.20) # vert
        MID_COLOR = (0.20, 0.45, 0.85) # bleu
        BOT_COLOR = (0.80, 0.25, 0.25) # rouge
        cmap_top = _gradient_cmap(TOP_COLOR)
        cmap_mid = _gradient_cmap(MID_COLOR)
        cmap_bot = _gradient_cmap(BOT_COLOR)

        res_sorted = res.sort_values("numero", ascending=True).reset_index(drop=True)
        y = np.arange(1, 51)
        bars = ax.barh(y, res_sorted["pourcentage"].values, color=(0,0,0,0), edgecolor="none")

        xmax = float(res_sorted["pourcentage"].max())
        for rect, (num, pct) in zip(bars, zip(res_sorted["numero"], res_sorted["pourcentage"])):
            if num in top_set:
                cmap = cmap_top
            elif num in bot_set:
                cmap = cmap_bot
            else:
                cmap = cmap_mid

            grad = np.linspace(0, 1, 256).reshape(1, -1)
            x0, x1 = 0, pct
            y0 = rect.get_y()
            y1 = y0 + rect.get_height()

            im = ax.imshow(grad, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", cmap=cmap)
            im.set_clip_path(rect)
            rect.set_edgecolor((0,0,0,0.15))
            rect.set_linewidth(0.8)

            ax.text(rect.get_width() + xmax * 0.01, rect.get_y() + rect.get_height()/2, f"{pct:.2f}%", va="center", ha="left", fontsize=8)

        ax.set_yticks(y, [str(i) for i in y])
        ax.set_yticklabels([str(i) for i in y], fontsize=8)
        ax.set_ylim(0.5, 50.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(5.0, xmax * 1.15))
        ax.set_title(title, fontsize=14, weight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")  # PNG carré (format Linkedin)
        plt.show()
        plt.close(fig)

    # Dataviz
    date_min_str = DATE_MIN.strftime("%d/%m/%Y")

    plot_frequencies(
        res_slot,
        f"Fréquence d'apparition des numéros par slot à l'EuroMillions depuis le {date_min_str}",
        "frequence_par_slot.png"
    )

    plot_frequencies(
        res_tirage,
        f"Fréquence d'apparition des numéros par tirage de l'EuroMillions depuis le {date_min_str}",
        "frequence_par_tirage.png"
    )

    return res_slot, res_tirage

def main():
    unzip_all_from_raw_to_silver()
    concatenate_silver_to_gold()
    res = data_analysis(OUTPUT_FILE)
    #print(res)

if __name__ == "__main__":
    main()
