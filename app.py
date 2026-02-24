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

min_FWHM = 10
max_FWHM = 20
nbin_SR = 40
nbin_FWHM = 40
N_ra, N_dec = 300, 300
bckg_color = "#212121"#"rgba(20, 20, 20, 1)"
color_features = "#065464"#"rgba(23, 240, 186, 0.6)"
bckg_color_control = "#065464"#"rgba(23, 240, 186, 0.6)"
field_name  ="COSMOS"
DEFAULT_SEEING = "median"

# ============================================================
# INITIAL MAP (to define fixed RA/Dec grids & bounds)
# ============================================================
ra_grid, dec_grid, RA, DEC, Z_sr0, Z_fwhm0 = generate_SR_map_from_json(
    N_ra, N_dec, field_name, DEFAULT_SEEING)

# SR range depends on map values -> will be recomputed when seeing changes
min_SR0 = float(np.min(Z_sr0))
max_SR0 = float(np.max(Z_sr0))

# ============================================================
# LOAD CATALOGS ONCE + FIXED MASK ONCE (bounds do not change)
# ============================================================
gal_ra, gal_dec, gal_z = load_galaxies(field_name)
star_ra, star_dec, star_mag = load_stars(field_name)

ra_min, ra_max = ra_grid.min(), ra_grid.max()
dec_min, dec_max = dec_grid.min(), dec_grid.max()

gal_mask_bounds = (
    (gal_ra >= ra_min) & (gal_ra <= ra_max) &
    (gal_dec >= dec_min) & (gal_dec <= dec_max)
)
gal_ra = gal_ra[gal_mask_bounds]
gal_dec = gal_dec[gal_mask_bounds]
gal_z = gal_z[gal_mask_bounds]

star_mask_bounds = (
    (star_ra >= ra_min) & (star_ra <= ra_max) &
    (star_dec >= dec_min) & (star_dec <= dec_max)
)
star_ra = star_ra[star_mask_bounds]
star_dec = star_dec[star_mask_bounds]
star_mag = star_mag[star_mask_bounds]

# ============================================================
# UTILS
# ============================================================
def nearest_z(ra_val, dec_val, Z):
    ix = np.abs(ra_grid - ra_val).argmin()
    iy = np.abs(dec_grid - dec_val).argmin()
    return Z[iy, ix]

def z_at_galaxies(mask,Z):
    ix = np.abs(ra_grid[:, None] - gal_ra[mask]).argmin(axis=0)
    iy = np.abs(dec_grid[:, None] - gal_dec[mask]).argmin(axis=0)
    return Z[iy, ix]

def make_base_heatmap(plot_type, Z_sr, Z_fwhm, min_SR, max_SR):
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
            x=ra_grid,
            y=dec_grid,
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


def add_overlays(fig, gal_mask=None, show_gal=False, show_stars=True):

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
app.title = f"HARMONI – {field_name}"
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
        dcc.Store(id="basefig-store"),

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
                            f"HARMONI – {field_name} field",
                            className="title-main",
                            style={"fontSize": 28, "fontWeight": "800"},
                        ),
                        html.Div(
                            "Predicted performance of the Multi-Conjugate Adaptive Optics at 2.2 microns",
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
                        "paddingLeft": "40px",
                        "padding": "40px",
                        "backgroundColor": bckg_color_control,
                        "borderRadius": "8px",
                    },
                    children=[
                        # ---- RA/Dec row
                        html.Div(
                            className="coord-row",
                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Label("RA (deg):"),
                                dcc.Input(
                                    id="input-ra",
                                    type="number",
                                    step=0.001,
                                    style={"backgroundColor": "white", "color": "black"},
                                ),
                                html.Label("Dec (deg):"),
                                dcc.Input(
                                    id="input-dec",
                                    type="number",
                                    step=0.001,
                                    style={"backgroundColor": "white", "color": "black"},
                                ),
                                html.Button(
                                    "Evaluate metric",
                                    id="eval-button",
                                    style={
                                        "backgroundColor": "#333",
                                        "color": "white",
                                        "border": "1px solid #555",
                                        "padding": "6px 12px",
                                    },
                                ),
                                html.Div(id="eval-output", style={"marginLeft": "12px"}),
                            ],
                        ),

                        # ---- Options row (clean)
                        html.Div(
                            className="options-row",
                            style={"marginTop": "20px"},
                            children=[
                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Seeing conditions:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="seeing-conditions",
                                                    options=[
                                                        {"label": "Median", "value": "median"},
                                                        {"label": "Q1", "value": "Q1"},
                                                    ],
                                                    value=DEFAULT_SEEING,
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("NGS asterisms:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="ngs-mode",
                                                    options=[
                                                        {"label": "Only 3 NGS", "value": 3},
                                                        {"label": "2–3 NGS", "value": 2},
                                                        {"label": "1–3 NGS", "value": 1},
                                                    ],
                                                    value=1,
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Display options:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.Checklist(
                                                    id="display-options",
                                                    options=[
                                                        {"label": "Show Galaxies", "value": "gal"},
                                                        {"label": "Show Stars", "value": "stars"},
                                                    ],
                                                    value=["stars"],
                                                    inline=True,
                                                    inputStyle={"marginRight": "6px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Plot type:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="plot-type",
                                                    options=[
                                                        {"label": "Strehl Ratio", "value": "strehl"},
                                                        {"label": "FWHM", "value": "fwhm"},
                                                    ],
                                                    value="strehl",
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="opt-block",
                                    children=[
                                        html.Label("Histogram mode:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="hist-mode",
                                                    options=[
                                                        {"label": "Differential", "value": "diff"},
                                                        {"label": "Cumulative", "value": "cumu"},
                                                    ],
                                                    value="diff",
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ---- Slider
                        html.Div(
                            style={"marginTop": "20px"},
                            children=[
                                html.Label("Galaxy redshift range", style={"fontWeight": "bold"}),
                                html.Div(
                                    style={"marginTop": "12px"},
                                    children=[
                                        dcc.RangeSlider(
                                            id="z-slider",
                                            min=0.0,
                                            max=10.0,
                                            step=1,
                                            value=[0, 10.0],
                                            tooltip={"placement": "bottom", "always_visible": True},
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
                    style={"flex": "1", "height": "100%", "padding": "40px"},
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
    Output("basefig-store", "data"),
    Input("seeing-conditions", "value"),
    Input("ngs-mode", "value"),
)
def recompute_maps(seeing_conditions_value, min_ngs):
    _, _, _, _, Z_sr, Z_fwhm = generate_SR_map_from_json(
        N_ra, N_dec, field_name, seeing_conditions_value, min_ngs=int(min_ngs)
    )

    min_SR = float(np.min(Z_sr))
    max_SR = float(np.max(Z_sr))

    base_strehl = make_base_heatmap("strehl", Z_sr, Z_fwhm, min_SR, max_SR)
    base_fwhm   = make_base_heatmap("fwhm",   Z_sr, Z_fwhm, min_SR, max_SR)

    map_data = {
        "Z_sr": Z_sr.tolist(),
        "Z_fwhm": Z_fwhm.tolist(),
        "min_SR": min_SR,
        "max_SR": max_SR,
        "min_ngs": int(min_ngs),
        "seeing": seeing_conditions_value,
    }
    base_figs = {
        "strehl": base_strehl.to_dict(),
        "fwhm": base_fwhm.to_dict(),
    }
    return map_data, base_figs

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

    if not (ra_grid.min() <= ra_val <= ra_grid.max() and dec_grid.min() <= dec_val <= dec_grid.max()):
        return "Outside map"

    Z_sr = np.array(map_data["Z_sr"])
    Z_fwhm = np.array(map_data["Z_fwhm"])

    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
    else:
        Z = Z_fwhm
        label_Z = "FWHM (mas)"

    z_val = nearest_z(ra_val, dec_val, Z)
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
    Input("basefig-store", "data"),
)
def update_galaxies(z_range, display_options, plot_type, hist_mode, map_data, base_figs):
    if map_data is None or base_figs is None:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", paper_bgcolor=bckg_color, plot_bgcolor=bckg_color)
        return empty, empty

    Z_sr = np.array(map_data["Z_sr"])
    Z_fwhm = np.array(map_data["Z_fwhm"])
    min_SR = float(map_data["min_SR"])
    max_SR = float(map_data["max_SR"])

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
    mask = (gal_z >= zmin) & (gal_z <= zmax)

    show_gal = "gal" in (display_options or [])
    show_stars = "stars" in (display_options or [])

    fig = go.Figure(base_figs[plot_type])
    fig = add_overlays(fig, gal_mask=mask, show_gal=show_gal, show_stars=show_stars)

    if np.any(mask):
        zg = z_at_galaxies(mask, Z)

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
                marker=dict(color="orange", opacity=0.9),
            )
        )

        if not is_cumu:
            y_title = "Galaxies (%) per bin"
        else:
            y_title = "Galaxies with SR ≥ x (%)" if plot_type == "strehl" else "Galaxies with FWHM ≤ x (%)"

        gal_fig.update_layout(
            title=label_plot + " value for selected galaxies at 2.2 microns",
            xaxis=dict(title=label_plot, gridcolor=color_features, range=[min_Z, max_Z]),
            yaxis=dict(title=y_title, gridcolor=color_features, range=[0, 100] if is_cumu else None),
            template="plotly_dark",
            paper_bgcolor=bckg_color,
            plot_bgcolor=bckg_color,
            margin=dict(l=60, r=30, t=50, b=50)
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