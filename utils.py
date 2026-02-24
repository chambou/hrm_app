import json
import numpy as np
import pandas as pd
import re
from scipy.interpolate import griddata
from astropy.io import fits

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

def candels_catalog_tansform(filename):
    """
    Clean and transform a Candels catalog for photometric analysis.

    Parameters
    ----------
    filename : path to file

    Returns
    -------
    pandas.DataFrame
        Cleaned and transformed catalog.
    """

    # Open FITS
    hdul = fits.open(filename)
    data = hdul[1].data
    # Convert type 
    data = np.array(data, dtype=data.dtype.newbyteorder('='))
    # Convert to panda dataframe
    df = pd.DataFrame(data)
    df.rename(columns={'RA': 'ra'}, inplace=True)
    df.rename(columns={'Dec': 'dec'}, inplace=True)
    df.rename(columns={'magR': 'SDSSrMag'}, inplace=True)
    df.rename(columns={'magI': 'SDSSiMag'}, inplace=True)
    df.rename(columns={'magH': 'hSyntMag'}, inplace=True)
    df['jSyntMag'] = df['hSyntMag']

    return df

def load_best_sr_from_bestjson(json_path, sci_index=0):
    """
 Load a JSON file produced by summarize(include_all=False)
    (see json_summary.py)
    that contains only the best asterism per field.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    sci_index : int
        Index of the science source to use (0 if only one).

    Returns
    -------
    best_sr_per_field : np.ndarray
        1D array of SR values (one per field).
    nfields_declared : int
        Number of fields declared in the JSON ('nfields' key if present).
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    nfields_declared = int(data.get("nfields", 0))

    if "best_by_field" in data:
        best_list = data["best_by_field"]
    else:
        raise KeyError("Invalid JSON structure: missing 'best_by_field'.")

    sr_list = []

    for entry in best_list:
        metrics = entry["metrics"]
        strehl = metrics["strehl"]  
        if strehl is None or len(strehl) == 0:
            continue
        sr = strehl[sci_index]
        if sr is None:
            continue
        sr_list.append(float(sr))

    best_sr_per_field = np.array(sr_list, dtype=float)
    return best_sr_per_field, nfields_declared

def load_best_fwhm_from_bestjson(json_path, sci_index=0):
    """
 Load a JSON file produced by summarize(include_all=False)
    (see json_summary.py)
    that contains only the best asterism per field.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    sci_index : int
        Index of the science source to use (0 if only one).

    Returns
    -------
    best_sr_per_field : np.ndarray
        1D array of SR values (one per field).
    nfields_declared : int
        Number of fields declared in the JSON ('nfields' key if present).
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    nfields_declared = int(data.get("nfields", 0))

    if "best_by_field" in data:
        best_list = data["best_by_field"]
    else:
        raise KeyError("Invalid JSON structure: missing 'best_by_field'.")

    fw_list = []

    for entry in best_list:
        metrics = entry["metrics"]
        fwhm = metrics["fwhm"][0]
        if fwhm is None or len(fwhm) == 0:
            continue
        fw = fwhm[sci_index]
        if fw is None:
            continue
        fw_list.append(float(fw))

    best_fwhm_per_field = np.array(fw_list, dtype=float)
    return best_fwhm_per_field, nfields_declared

def empty_fields(P):
    mask = np.zeros(len(P))
    for k in range(0,len(P)):
        if type(P[k]) == np.int16:
            mask[k] = 1
    return mask.astype(bool)

def generate_SR_map(N_ra, N_dec, field_name, seeing_conditions):
    P = np.load(
        DATA_DIR / f"asterism_data_{field_name}_catalog_outer120arcsecs_inner20arcsecs_noPCAM.npy",
        allow_pickle=True
    )
    RA = np.load(DATA_DIR / f"asterism_data_{field_name}_catalog_outer120arcsecs_inner20arcsecs_noPCAM_RA_positions.npy")
    DEC = np.load(DATA_DIR / f"asterism_data_{field_name}_catalog_outer120arcsecs_inner20arcsecs_noPCAM_DEC_positions.npy")

    SR = load_best_sr_from_bestjson(
        DATA_DIR / f"HRM_{field_name}_120_20_{seeing_conditions}_blur_best_asterisms_updated.json"
    )[0]
    FWHM = load_best_fwhm_from_bestjson(
        DATA_DIR / f"HRM_{field_name}_120_20_{seeing_conditions}_blur_best_asterisms_updated.json"
    )[0]

    mask = empty_fields(P)
    RA_all = np.concatenate((RA[~mask], RA[mask]))
    DEC_all = np.concatenate((DEC[~mask], DEC[mask]))
    SR_all = np.concatenate((SR, np.zeros(np.sum(mask))))
    FWHM_all = np.concatenate((FWHM, 500 * np.ones(np.sum(mask))))

    points = np.column_stack((RA_all, DEC_all))
    values_SR = SR_all
    values_FWHM = FWHM_all

    RA_i = np.linspace(RA_all.min(), RA_all.max(), N_ra)
    DEC_i = np.linspace(DEC_all.min(), DEC_all.max(), N_dec)
    RA_grid, DEC_grid = np.meshgrid(RA_i, DEC_i)

    Z_sr = np.clip(griddata(points, values_SR, (RA_grid, DEC_grid), method="cubic"), 0, 1)
    Z_sr = np.ma.masked_invalid(Z_sr)

    Z_fwhm = griddata(points, values_FWHM, (RA_grid, DEC_grid), method="linear")
    Z_fwhm = np.ma.masked_invalid(Z_fwhm)

    return (
        RA_i.astype(np.float32),
        DEC_i.astype(np.float32),
        RA_grid.astype(np.float32),
        DEC_grid.astype(np.float32),
        Z_sr.astype(np.float32),
        Z_fwhm.astype(np.float32),
    )

def generate_SR_map_from_json(N_ra, N_dec, field_name, seeing_conditions, min_ngs=1):
    """
    min_ngs:
      - 3 => only asterisms with exactly 3 NGS? (we interpret as at least 3, i.e. ==3 because max is 3 here)
      - 2 => allow 2 or 3
      - 1 => allow 1,2,3 (current behavior)
    """
    # 1) full positions (fixed grid indexing)
    RA_full = np.load(DATA_DIR / f"asterism_data_{field_name}_catalog_outer120arcsecs_inner20arcsecs_noPCAM_RA_positions.npy")
    DEC_full = np.load(DATA_DIR / f"asterism_data_{field_name}_catalog_outer120arcsecs_inner20arcsecs_noPCAM_DEC_positions.npy")

    # 2) JSON best
    json_path = DATA_DIR / f"HRM_{field_name}_120_20_{seeing_conditions}_blur_best_asterisms_updated.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    nfields_total = int(data.get("nfields_total", len(RA_full)))
    SR_full   = np.full(nfields_total, np.nan, dtype=float)
    FWHM_full = np.full(nfields_total, np.nan, dtype=float)

    # 3) fill via rec_field_idx, BUT filter on n_stars
    for entry in data["best_by_field"]:
        n_stars = int(entry.get("n_stars", 0))
        if n_stars < int(min_ngs):
            continue  # rejected by NGS filter

        i = int(entry["rec_field_idx"])
        strehl = entry["metrics"].get("strehl", [])
        fwhm   = entry["metrics"].get("fwhm", [])

        # sci_index=0
        SR_full[i] = float(strehl[0]) if strehl else np.nan
        # fwhm is [[...]]
        FWHM_full[i] = float(fwhm[0][0]) if (fwhm and fwhm[0]) else np.nan

    # 4) mask = fields without value (either missing or filtered out)
    mask = ~np.isfinite(SR_full)

    # 5) default values for interpolation
    SR_all   = SR_full.copy()
    FWHM_all = FWHM_full.copy()
    SR_all[mask] = 0.0
    FWHM_all[mask] = 500.0

    points = np.column_stack([RA_full[:nfields_total], DEC_full[:nfields_total]])

    RA_i = np.linspace(np.nanmin(RA_full), np.nanmax(RA_full), N_ra)
    DEC_i = np.linspace(np.nanmin(DEC_full), np.nanmax(DEC_full), N_dec)
    RA_grid, DEC_grid = np.meshgrid(RA_i, DEC_i)

    Z_sr = np.clip(griddata(points, SR_all, (RA_grid, DEC_grid), method="cubic"), 0, 1)
    Z_sr = np.ma.masked_invalid(Z_sr)

    Z_fwhm = griddata(points, FWHM_all, (RA_grid, DEC_grid), method="linear")
    Z_fwhm = np.ma.masked_invalid(Z_fwhm)

    return RA_i, DEC_i, RA_grid, DEC_grid, Z_sr, Z_fwhm


def load_stars(field_name):
    data = candels_catalog_tansform(DATA_DIR / f"{field_name}_stars.fits")
    return data["ra"].to_numpy(), data["dec"].to_numpy(), data["hSyntMag"].to_numpy()


def load_galaxies(field_name):
    redshift_path = DATA_DIR / f"{field_name}_redshift.txt"

    col_names = []
    with open(redshift_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                m = re.match(r"#\s*\d+\s+(\w+)", line)
                if m:
                    col_names.append(m.group(1))
            else:
                break

    df = pd.read_csv(
        redshift_path,
        comment="#",
        sep=r"\s+",
        engine="python",
        header=None
    )
    df.columns = col_names
    df = df[["RA", "DEC", "z_best"]]
    return df["RA"].values, df["DEC"].values, df["z_best"].values


    # Step 3: Assign column names
    df.columns = col_names

    # Step 4: Keep only RA, DEC, z_best
    df = df[["RA", "DEC", "z_best"]]

    return df["RA"].values, df["DEC"].values, df["z_best"].values