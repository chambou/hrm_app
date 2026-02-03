import json
import numpy as np
import pandas as pd
import re
from scipy.interpolate import griddata
from astropy.io import fits

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

    # Collect all other info
    P = np.load('./data/asterism_data_'+field_name+'_catalog_outer120arcsecs_inner20arcsecs_noPCAM.npy', allow_pickle = True)
    RA = np.load('./data//asterism_data_'+field_name+'_catalog_outer120arcsecs_inner20arcsecs_noPCAM_RA_positions.npy')
    DEC = np.load('./data/asterism_data_'+field_name+'_catalog_outer120arcsecs_inner20arcsecs_noPCAM_DEC_positions.npy')
    SR = load_best_sr_from_bestjson('./data/HRM_'+field_name+'_120_20_'+seeing_conditions+'_blur_best_asterisms_updated.json')[0]
    FWHM = load_best_fwhm_from_bestjson('./data//HRM_'+field_name+'_120_20_'+seeing_conditions+'_blur_best_asterisms_updated.json')[0]

    # Handle fileds with no stars
    mask = empty_fields(P)
    RA_all = np.concatenate((RA[~mask],RA[mask]))
    DEC_all = np.concatenate((DEC[~mask],DEC[mask]))
    SR_all = np.concatenate((SR,np.zeros(np.sum(mask))))
    FWHM_all = np.concatenate((FWHM,500*np.ones(np.sum(mask))))

    # Interpolation on Known points
    points = np.column_stack((RA_all , DEC_all))
    values_SR = SR_all
    values_FWHM = FWHM_all

    RA_i = np.linspace(RA_all.min(), RA_all.max(), N_ra)
    DEC_i = np.linspace(DEC_all.min(), DEC_all.max(), N_dec)
    RA, DEC = np.meshgrid(RA_i, DEC_i)

    Z_sr = np.clip(griddata(points, values_SR, (RA, DEC), method="cubic"),0,1)
    Z_sr = np.ma.masked_invalid(Z_sr)

    Z_fwhm = griddata(points, values_FWHM, (RA, DEC), method="linear")
    Z_fwhm = np.ma.masked_invalid(Z_fwhm)

    return RA_i.astype(np.float32),DEC_i.astype(np.float32), RA.astype(np.float32), DEC.astype(np.float32), Z_sr.astype(np.float32), Z_fwhm.astype(np.float32)


def load_stars(field_name):
    # Collect catalog to plot stars
    data = candels_catalog_tansform('./data/'+field_name+'_stars.fits')
    return data['ra'].to_numpy(), data['dec'].to_numpy(), data['hSyntMag'].to_numpy()


def load_galaxies(field_name):
    # Step 1: Extract column names from the header
    col_names = []
    with open('./data/'+field_name+'_redshift.txt', "r") as f:
        for line in f:
            if line.startswith("#"):
                # Match lines like "# 1 file" or "# 3 RA (something)"
                m = re.match(r"#\s*\d+\s+(\w+)", line)
                if m:
                    col_names.append(m.group(1))
            else:
                break  # stop at first non-comment line

    # Step 2: Read the data
    df = pd.read_csv(
        './data/'+field_name+'_redshift.txt',
        comment="#",
        sep="\s+",
        engine="python",
        header=None
    )

    # Step 3: Assign column names
    df.columns = col_names

    # Step 4: Keep only RA, DEC, z_best
    df = df[["RA", "DEC", "z_best"]]

    return df["RA"].values, df["DEC"].values, df["z_best"].values