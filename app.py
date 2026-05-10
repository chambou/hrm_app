import numpy as np
import os
import plotly.graph_objects as go
import copy
from dash import Dash, dcc, html, Input, Output, State
from utils import (
    generate_SR_map_from_json,
    load_galaxies,
    load_stars,
)

# ============================================================
# CONFIG
# ============================================================
PORT = int(os.environ.get("PORT", 8050))
HOST = os.environ.get("HOST", "127.0.0.2")

DEFAULT_BAND = "K"

min_FWHM = 10
max_FWHM = 20
nbin_SR = 40
nbin_FWHM = 40
N_ra, N_dec = 300, 300
bckg_color = "#212121"#"rgba(20, 20, 20, 1)"
color_features = "#065464"#"rgba(23, 240, 186, 0.6)"
bckg_color_control = "#065464"#"rgba(23, 240, 186, 0.6)"
DEFAULT_FIELD = "COSMOS"
DEFAULT_SEEING = "median"

FIELD_CACHE = {}
# ============================================================
# UTILS
# ============================================================
def get_field_catalogs(field_name, ra_grid_local, dec_grid_local):
    cache_key = field_name

    if cache_key not in FIELD_CACHE:
        gal_ra, gal_dec, gal_z = load_galaxies(field_name)
        star_ra, star_dec, star_mag = load_stars(field_name)

        FIELD_CACHE[cache_key] = {
            "gal_ra": gal_ra,
            "gal_dec": gal_dec,
            "gal_z": gal_z,
            "star_ra": star_ra,
            "star_dec": star_dec,
            "star_mag": star_mag,
        }

    data = FIELD_CACHE[cache_key]

    ra_min, ra_max = ra_grid_local.min(), ra_grid_local.max()
    dec_min, dec_max = dec_grid_local.min(), dec_grid_local.max()

    gal_mask_bounds = (
        (data["gal_ra"] >= ra_min) & (data["gal_ra"] <= ra_max) &
        (data["gal_dec"] >= dec_min) & (data["gal_dec"] <= dec_max)
    )

    star_mask_bounds = (
        (data["star_ra"] >= ra_min) & (data["star_ra"] <= ra_max) &
        (data["star_dec"] >= dec_min) & (data["star_dec"] <= dec_max)
    )

    return (
        data["gal_ra"][gal_mask_bounds],
        data["gal_dec"][gal_mask_bounds],
        data["gal_z"][gal_mask_bounds],
        data["star_ra"][star_mask_bounds],
        data["star_dec"][star_mask_bounds],
        data["star_mag"][star_mask_bounds],
    )

def z_at_galaxies(mask, Z, ra_grid_local, dec_grid_local, gal_ra_local, gal_dec_local):
    ix = np.abs(ra_grid_local[:, None] - gal_ra_local[mask]).argmin(axis=0)
    iy = np.abs(dec_grid_local[:, None] - gal_dec_local[mask]).argmin(axis=0)
    return Z[iy, ix]

def make_base_heatmap(plot_type, Z_sr, Z_fwhm, min_SR, max_SR, ra_grid_local, dec_grid_local):
    fig = go.Figure()

    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
        label_cursor = "SR"
        label_plot = "Strehl Ratio"
        min_Z = min_SR
        max_Z = max_SR
    else:
        Z = Z_fwhm
        label_Z = "FWHM (mas)"
        label_cursor = "FWHM"
        label_plot = "FWHM (mas)"
        min_Z = min_FWHM
        max_Z = max_FWHM

    fig.add_trace(
        go.Heatmap(
            x=ra_grid_local,
            y=dec_grid_local,
            z=Z,
            zmin=min_Z,        # set minimum of color scale
            zmax=max_Z,        # set maximum of color scale
            colorscale="Viridis",
            colorbar=dict(title=label_Z),
            hovertemplate=(
                "RA: %{x:.5f}°<br>"
                "Dec: %{y:.5f}°<br>"
                + label_cursor + ": %{z:.6f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis=dict(title="RA (deg)", autorange="reversed", gridcolor=color_features),
        yaxis=dict(title="Dec (deg)", scaleanchor="x", gridcolor=color_features),
        template="plotly_dark",
        paper_bgcolor=bckg_color,
        plot_bgcolor=bckg_color,
        margin=dict(l=70, r=40, t=100, b=60),  # increase top margin
    )

    return fig


def add_overlays(fig, gal_ra, gal_dec, star_ra, star_dec, star_mag, gal_mask=None, show_gal=False, show_stars=True):

    if show_gal and gal_mask is not None and np.any(gal_mask):
        # Halo
        fig.add_trace(
            go.Scattergl(
                x=gal_ra[gal_mask],
                y=gal_dec[gal_mask],
                mode="markers",
                marker=dict(
                    size=6,
                    color="white",
                    opacity=0.15,
                    symbol="circle",
                    line=dict(width=0)
                ),
                showlegend=False
            )
        )

        # Core
        fig.add_trace(
            go.Scattergl(
                x=gal_ra[gal_mask],
                y=gal_dec[gal_mask],
                mode="markers",
                marker=dict(
                    size=2,
                    color="white",
                    opacity=0.9,
                    symbol="circle",
                    line=dict(width=0)
                ),
                name="Selected galaxies",
                showlegend=False 
            )
        )


    if show_stars:
        fig.add_trace(
            go.Scattergl(
                x=star_ra,
                y=star_dec,
                mode="markers",
                marker=dict(
                    size=6,
                    color=star_mag,
                    colorscale="Plasma",
                    reversescale=True,
                    showscale=True,
                    colorbar=dict(
                        title="NGS - H-Magnitude",
                        orientation="h",   #
                        x=0.5,              # center
                        y=-0.08,            # below plot
                        xanchor="center",
                        yanchor="top",
                        len=0.6             # width of colorbar
                    ),
                    line=dict(width=0)
                ),
                name="Natural guide stars",
                showlegend=False 
            )
        )
    return fig

# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__)
app.title = "HARMONI"
server = app.server

app.layout = html.Div(
    className="app-container",
    style={
        "width": "99vw",
        "height": "99vh",
        "backgroundColor": bckg_color,
        "color": "#E0E0E0",
        "padding": "12px",
        "boxSizing": "border-box",
        "fontFamily": "Arial",
        "display": "flex",
        "gap": "0px",
    },
    children=[
        # Stores: keep current maps + base figs
        dcc.Store(id="map-store"),

        # ---------------- Left column: Title + Sky map
        html.Div(
            className="left-panel",
            style={
                "flex": "1",
                "height": "100%",
                "paddingRight": "6px",
                "marginTop": "5px",
                "display": "flex",
                "flexDirection": "column",
            },
            children=[
                # Title
                html.Div(
                    style={"padding": "8px 12px"},
                    children=[
                        html.Div(
                            "HARMONI",
                            className="title-main",
                            style={"fontSize": 28, "fontWeight": "800"},
                        ),
                        html.Div(
                            "Predicted performance of the Multi-Conjugate Adaptive Optics - Single OB",
                            className="title-sub",
                            style={"fontSize": 14, "color": "#BBBBBB"},
                        ),
                    ],
                ),

                # Sky map
                dcc.Graph(
                    id="sky-map",
                    figure=go.Figure(),  # filled by callback
                    style={"flex": "1"},
                    config={"scrollZoom": True},
                ),
            ],
        ),

        # ---------------- Vertical divider
        html.Div(
            style={
                "width": "2px",
                "backgroundColor": color_features,
                "height": "100%",
                "minHeight": "100%",
                "marginRight": "10px",
            }
        ),

        # ---------------- Right column: Controls + bottom plot
        html.Div(
            className="right-panel",
            style={
                "flex": "1",
                "height": "100%",
                "display": "flex",
                "flexDirection": "column",
                "gap": "0px",
            },
            children=[
                # Top right: Controls
                html.Div(
                    className="controls-panel",
                    style={
                        "flex": "0 0 auto",
                        "padding": "20px",
                        "backgroundColor": bckg_color_control,
                        "borderRadius": "8px",
                    },
                    children=[
                        # ---- RA/Dec row
                        html.Div(
                            className="coord-row",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "12px",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Label("RA (deg):"),
                                dcc.Input(
                                    id="input-ra",
                                    type="number",
                                    step=0.001,
                                    placeholder="RA",
                                    style={
                                        "width": "110px",
                                        "height": "34px",
                                        "padding": "4px 8px",
                                        "borderRadius": "6px",
                                        "border": "1px solid #555",
                                        "backgroundColor": "#2b2b2b",
                                        "color": "white",
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Label("Dec (deg):"),
                                dcc.Input(
                                        id="input-dec",
                                        type="number",
                                        step=0.001,
                                        placeholder="Dec",
                                        style={
                                            "width": "110px",
                                            "height": "34px",
                                            "padding": "4px 8px",
                                            "borderRadius": "6px",
                                            "border": "1px solid #555",
                                            "backgroundColor": "#2b2b2b",
                                            "color": "white",
                                            "fontSize": "13px",
                                        },
                                    ),
                                html.Button(
                                    "Evaluate",
                                    id="eval-button",
                                    style={
                                        "height": "34px",
                                        "padding": "0 14px",
                                        "backgroundColor": "#0b7285",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "fontSize": "13px",
                                        "fontWeight": "500",
                                        "cursor": "pointer",
                                        "transition": "0.2s",
                                        "boxShadow": "0 0 0 rgba(0,0,0,0)",
                                    },
                                ),
                                html.Div(
                                    id="eval-output",
                                    style={
                                        "marginLeft": "10px",
                                        "fontSize": "13px",
                                        "color": "#e9ecef",
                                        "minWidth": "120px",
                                    },
                                ),
                            ],
                        ),

                        # ---- Options row 
                        html.Div(
                            className="options-row",
                            style={
                                "marginTop": "16px",
                                "display": "grid",
                                "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                                "gap": "10px 16px",
                                "alignItems": "start",
                                "fontSize": "13px",
                            },
                            children=[
                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Field:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="field-name",
                                                    options=[
                                                        {"label": "COSMOS", "value": "COSMOS"},
                                                        {"label": "UDS", "value": "UDS"},
                                                        {"label": "GOODSS", "value": "GOODSS"},
                                                    ],
                                                    value=DEFAULT_FIELD,
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Band:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="band",
                                                    options=[
                                                        {"label": "K", "value": "K"},
                                                        {"label": "H", "value": "H"},
                                                        {"label": "J", "value": "J"},
                                                    ],
                                                    value=DEFAULT_BAND,
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Seeing:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="seeing-conditions",
                                                    options=[
                                                        {"label": "Median", "value": "median"},
                                                        {"label": "Q1", "value": "Q1"},
                                                    ],
                                                    value=DEFAULT_SEEING,
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("NGS:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="ngs-mode",
                                                    options=[
                                                        {"label": "3", "value": 3},
                                                        {"label": "2–3", "value": 2},
                                                        {"label": "all", "value": 1},
                                                    ],
                                                    value=1,
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Display:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.Checklist(
                                                    id="display-options",
                                                    options=[
                                                        {"label": "Galaxies", "value": "gal"},
                                                        {"label": "Stars", "value": "stars"},
                                                    ],
                                                    value=["stars"],
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                    inputStyle={"marginRight": "2px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Plot:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="plot-type",
                                                    options=[
                                                        {"label": "SR", "value": "strehl"},
                                                        {"label": "FWHM", "value": "fwhm"},
                                                    ],
                                                    value="strehl",
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Histogram:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "6px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="hist-mode",
                                                    options=[
                                                        {"label": "Cumu.", "value": "cumu"},
                                                        {"label": "Diff.", "value": "diff"},
                                                    ],
                                                    value="cumu",
                                                    inline=True,
                                                    labelStyle={"marginRight": "1px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ---- Slider
                        html.Div(
                            style={"marginTop": "14px"},
                            children=[
                                html.Label(
                                    "Galaxy redshift",
                                    style={
                                        "fontWeight": "bold",
                                        "fontSize": "13px",
                                        "marginBottom": "6px",
                                        "display": "block",
                                    },
                                ),
                                html.Div(
                                    style={"marginTop": "12px"},
                                    children=[
                                        dcc.RangeSlider(
                                            id="z-slider",
                                            min=0.0,
                                            max=10.0,
                                            step=1,
                                            value=[0, 10.0],
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                                "template": "{value}",
                                            },
                                            marks={
                                                0: {"label": "0", "style": {"fontSize": "11px"}},
                                                2: {"label": "2", "style": {"fontSize": "11px"}},
                                                4: {"label": "4", "style": {"fontSize": "11px"}},
                                                6: {"label": "6", "style": {"fontSize": "11px"}},
                                                8: {"label": "8", "style": {"fontSize": "11px"}},
                                                10: {"label": "10", "style": {"fontSize": "11px"}},
                                            },
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # ---------------- Horizontal divider
                html.Div(
                    style={
                        "height": "2px",
                        "backgroundColor": color_features,
                        "margin": "6px 0",
                    }
                ),

                # ---------------- Bottom right: Galaxy plot
                html.Div(
                    style={
                        "flex": "1",
                        "height": "100%",
                        "padding": "20px 40px 40px 40px",
                        "marginTop": "-10px",
                    },
                    children=[
                        dcc.Graph(
                            id="galaxy-z-plot",
                            style={"height": "100%"},
                        )
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================
# CALLBACK 1: recompute maps when seeing and ngs mode change (ONLY Z arrays + figs)
# ============================================================
@app.callback(
    Output("map-store", "data"),
    Input("field-name", "value"),
    Input("band", "value"),
    Input("seeing-conditions", "value"),
    Input("ngs-mode", "value"),
)
def recompute_maps(field_name, band, seeing_conditions_value, min_ngs):
    ra_grid_new, dec_grid_new, _, _, Z_sr, Z_fwhm = generate_SR_map_from_json(
        N_ra, N_dec, field_name, seeing_conditions_value,
        min_ngs=int(min_ngs),
        band=band,
    )

    Z_sr = np.array(Z_sr, dtype=float)
    Z_fwhm = np.array(Z_fwhm, dtype=float)

    min_SR = float(np.nanmin(Z_sr))
    max_SR = float(np.nanmax(Z_sr))

    map_data = {
        "field_name": field_name,
        "ra_grid": ra_grid_new.tolist(),
        "dec_grid": dec_grid_new.tolist(),
        "Z_sr": Z_sr.tolist(),
        "Z_fwhm": Z_fwhm.tolist(),
        "min_SR": min_SR,
        "max_SR": max_SR,
        "min_ngs": int(min_ngs),
        "band": band,
        "seeing": seeing_conditions_value,
    }

    return map_data

# ============================================================
# CALLBACK 2: evaluate metric at RA/Dec
# ============================================================
@app.callback(
    Output("eval-output", "children"),
    Input("eval-button", "n_clicks"),
    State("input-ra", "value"),
    State("input-dec", "value"),
    State("map-store", "data"),
    Input("plot-type", "value"),
)
def evaluate_position(_, ra_val, dec_val, map_data, plot_type):
    if ra_val is None or dec_val is None:
        return "Enter RA & Dec"
    if map_data is None:
        return "Map not ready"
    
    ra_grid_local = np.array(map_data["ra_grid"])
    dec_grid_local = np.array(map_data["dec_grid"])

    if not (ra_grid_local.min() <= ra_val <= ra_grid_local.max() and dec_grid_local.min() <= dec_val <= dec_grid_local.max()):
        return "Outside map"

    Z_sr = np.array(map_data["Z_sr"], dtype=float)
    Z_fwhm = np.array(map_data["Z_fwhm"], dtype=float)

    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
    else:
        Z = Z_fwhm
        label_Z = "FWHM (mas)"

    ix = np.abs(ra_grid_local - ra_val).argmin()
    iy = np.abs(dec_grid_local - dec_val).argmin()
    z_val = Z[iy, ix]
    return f"{label_Z} = {z_val:.6f}"

# ============================================================
# CALLBACK 3: update sky map overlays + histogram
# ============================================================
@app.callback(
    Output("sky-map", "figure"),
    Output("galaxy-z-plot", "figure"),
    Input("z-slider", "value"),
    Input("display-options", "value"),
    Input("plot-type", "value"),
    Input("hist-mode", "value"),
    Input("map-store", "data"),
)
def update_galaxies(z_range, display_options, plot_type, hist_mode, map_data):
    if map_data is None :
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", paper_bgcolor=bckg_color, plot_bgcolor=bckg_color)
        return empty, empty

    Z_sr = np.array(map_data["Z_sr"], dtype=float)
    Z_fwhm = np.array(map_data["Z_fwhm"], dtype=float)
    Z_sr = np.array(Z_sr, dtype=float)
    Z_fwhm = np.array(Z_fwhm, dtype=float)
    min_SR = float(np.nanmin(Z_sr))
    max_SR = float(np.nanmax(Z_sr))
    
    band = map_data.get("band", DEFAULT_BAND)

    ra_grid_local = np.array(map_data["ra_grid"])
    dec_grid_local = np.array(map_data["dec_grid"])

    field_name = map_data["field_name"]

    gal_ra_local, gal_dec_local, gal_z_local, star_ra_local, star_dec_local, star_mag_local = get_field_catalogs(
        field_name,
        ra_grid_local,
        dec_grid_local,
    )

    if plot_type == "strehl":
        Z = Z_sr
        label_plot = "Strehl Ratio"
        min_Z = min_SR
        max_Z = max_SR
        nbin_Z = nbin_SR
    else:
        Z = Z_fwhm
        label_plot = "FWHM (mas)"
        min_Z = min_FWHM
        max_Z = max_FWHM
        nbin_Z = nbin_FWHM

    zmin, zmax = z_range
    mask = (gal_z_local >= zmin) & (gal_z_local <= zmax)

    show_gal = "gal" in (display_options or [])
    show_stars = "stars" in (display_options or [])

    fig = make_base_heatmap(
        plot_type,
        Z_sr,
        Z_fwhm,
        min_SR,
        max_SR,
        ra_grid_local,
        dec_grid_local,
    )
    fig = add_overlays(
        fig,
        gal_ra_local, gal_dec_local,
        star_ra_local, star_dec_local, star_mag_local,
        gal_mask=mask,
        show_gal=show_gal,
        show_stars=show_stars,
    )

    if np.any(mask):
        zg = z_at_galaxies(mask, Z, ra_grid_local, dec_grid_local, gal_ra_local, gal_dec_local)

        if plot_type == "fwhm":
            zg_clipped = np.where(zg < min_Z, min_Z - 1, zg)
            zg_clipped = np.where(zg_clipped > max_Z, max_Z + 1, zg_clipped)
        else:
            zg_clipped = zg

        is_cumu = (hist_mode == "cumu")
        cumu_direction = "decreasing" if (is_cumu and plot_type == "strehl") else "increasing"

        gal_fig = go.Figure(
            go.Histogram(
                x=zg_clipped,
                nbinsx=nbin_Z,
                histnorm="percent",
                cumulative=dict(enabled=is_cumu, direction=cumu_direction),
                marker=dict(color="#831ec7", opacity=0.85),
            )
        )

        if not is_cumu:
            y_title = "Galaxies (%) per bin"
        else:
            y_title = "Galaxies with SR ≥ x (%)" if plot_type == "strehl" else "Galaxies with FWHM ≤ x (%)"

        gal_fig.update_layout(
            title=f"{label_plot} value for selected galaxies in band {band} - Single OB",
            xaxis=dict(title=label_plot, gridcolor=color_features, range=[min_Z, max_Z]),
            yaxis=dict(title=y_title, gridcolor=color_features, range=[0, 100] if is_cumu else None),
            template="plotly_dark",
            paper_bgcolor=bckg_color,
            plot_bgcolor=bckg_color,
            margin=dict(l=60, r=30, t=35, b=50)
        )
    else:
        gal_fig = go.Figure()
        gal_fig.update_layout(
            title="No galaxies selected",
            template="plotly_dark",
            paper_bgcolor=bckg_color,
            plot_bgcolor=bckg_color
        )

    return fig, gal_fig

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT, use_reloader=False)