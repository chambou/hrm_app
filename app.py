import numpy as np
import plotly.graph_objects as go
import webbrowser
from utils import *
from dash import Dash, dcc, html, Input, Output, State

# ============================================================
# CONFIG
# ============================================================
HOST = "127.0.0.1"
PORT = 8050
min_FWHM = 10
max_FWHM = 20
nbin_SR = 40
nbin_FWHM = 40
N_ra, N_dec = 600, 600
bckg_color = "#212121"#"rgba(20, 20, 20, 1)"
color_features = "#065464"#"rgba(23, 240, 186, 0.6)"
bckg_color_control = "#065464"#"rgba(23, 240, 186, 0.6)"
field_name  ="COSMOS"
seeing_conditions = 'median'

# ============================================================
# MAP DATA
# ============================================================

ra_grid, dec_grid, RA, DEC, Z_sr, Z_fwhm = generate_SR_map(N_ra,N_dec,field_name,seeing_conditions)
Z = Z_sr
min_SR = Z_sr.min()
max_SR = Z_sr.max()

# ============================================================
# GALAXY CATALOG
# ============================================================

gal_ra, gal_dec, gal_z = load_galaxies(field_name)
# Mask
ra_min, ra_max = ra_grid.min(), ra_grid.max()
dec_min, dec_max = dec_grid.min(), dec_grid.max()
mask = (
    (gal_ra >= ra_min) & (gal_ra <= ra_max) &
    (gal_dec >= dec_min) & (gal_dec <= dec_max)
)

gal_ra  = gal_ra[mask]
gal_dec = gal_dec[mask]
gal_z   = gal_z[mask]

# ============================================================
# STAR CATALOG
# ============================================================

star_ra, star_dec, star_mag = load_stars(field_name)

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

def make_sky_figure(gal_mask=None, show_gal=False, show_stars=True, plot_type="strehl"):
    fig = go.Figure()

    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
        label_cursor = "SR"
        label_plot = "Strehl Ratio"
        min_Z = min_SR
        max_Z = max_SR
    elif plot_type == "fwhm":
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


    fig.update_layout(
        xaxis=dict(title="RA (deg)", autorange="reversed", gridcolor=color_features),
        yaxis=dict(title="Dec (deg)", scaleanchor="x", gridcolor=color_features),
        template="plotly_dark",
        paper_bgcolor=bckg_color,
        plot_bgcolor=bckg_color,
        margin=dict(l=70, r=40, t=100, b=60),  # increase top margin
    )

    # Add a "title box" annotation
    fig.add_annotation(
        x=-0.1, 
        y=1.2,                     # slightly above plot
        xref="paper",
        yref="paper",
        text=(
            "<b>HARMONI – "+field_name+" field</b><br>"
            "<span style='font-size:14px; color:#BBBBBB'>"
            "Predicted performance of the Multi-Conjugate Adaptive Optics at 2.2 microns </span>"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font=dict(size=28, family="Arial Black, Arial, sans-serif", color="#FFFFFF"),
        bgcolor=bckg_color,           # dark box behind title
        bordercolor=bckg_color,
        borderwidth=1,
        borderpad=6,
    )
    return fig

# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__)

app.layout = html.Div(
    style={
        "width": "99vw",
        "height": "99vh",
        "backgroundColor": bckg_color,
        "color": "#E0E0E0",
        "padding": "12px",
        "boxSizing": "border-box",
        "fontFamily": "Arial",
        "display": "flex",  # horizontal split
        "gap": "0px"
    },
    children=[

        # ---------------- Left column: Sky map
        html.Div(
            style={
                "flex": "1",
                "height": "100%",
                "paddingRight": "6px",
                "marginTop": "5px"
            },
            children=[
                dcc.Graph(
                    id="sky-map",
                    figure=make_sky_figure(),
                    style={"height": "100%"},
                    config={"scrollZoom": True}
                )
            ]
        ),

        # ---------------- Vertical divider
        html.Div(
            style={
                "width": "2px",
                "backgroundColor": color_features,
                "margin": "0 6px"
            }
        ),

        # ---------------- Right column: Controls + bottom plot
        html.Div(
            style={
                "flex": "1",
                "height": "100%",
                "display": "flex",
                "flexDirection": "column",
                "gap": "0px"
            },
            children=[

                # Top right: Controls
                html.Div(
                    style={
                        "flex": "0 0 auto",
                        "padding": "30px",
                        "backgroundColor": bckg_color_control,
                        "borderRadius": "8px"
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Label("RA (deg):"),
                                dcc.Input(
                                    id="input-ra",
                                    type="number",
                                    step=0.001,
                                    style={"backgroundColor": "white", "color": "black"}
                                ),
                                html.Label("Dec (deg):"),
                                dcc.Input(
                                    id="input-dec",
                                    type="number",
                                    step=0.001,
                                    style={"backgroundColor": "white", "color": "black"}
                                ),
                                html.Button(
                                    "Evaluate metric",
                                    id="eval-button",
                                    style={
                                        "backgroundColor": "#333",
                                        "color": "white",
                                        "border": "1px solid #555",
                                        "padding": "6px 12px"
                                    }
                                ),
                                html.Div(id="eval-output", style={"marginLeft": "12px"})
                            ]
                        ),



                        # ---------------- Middle part: two columns of checkboxes
                        html.Div(
                            style={
                                "display": "flex",
                                "marginTop": "20px",
                                "gap": "60px"  # horizontal space between left/right columns
                            },
                            children=[

                                # Middle-left: Galaxies / Stars
                                html.Div(
                                    children=[
                                        html.Label("Display options:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.Checklist(
                                                    id="display-options",
                                                    options=[
                                                        {"label": "Show Galaxies", "value": "gal"},
                                                        {"label": "Show Stars", "value": "stars"}
                                                    ],
                                                    value=["stars"],  # default: stars visible
                                                    inline=True,
                                                    inputStyle={"marginRight": "6px"}
                                                )
                                            ]
                                        )
                                    ]
                                ),

                                # Middle-right: Strehl / FWHM
                                html.Div(
                                    children=[
                                        html.Label("Plot type:", style={"fontWeight": "bold"}),
                                        html.Div(
                                            style={"marginTop": "12px"},
                                            children=[
                                                dcc.RadioItems(
                                                    id="plot-type",
                                                    options=[
                                                        {"label": "Strehl Ratio", "value": "strehl"},
                                                        {"label": "FWHM", "value": "fwhm"}
                                                    ],
                                                    value="strehl",  # default selection
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"}  # space between options
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),


                        # ---------------- Bottom part -----------------

                        html.Div(style={"marginTop": "20px"}, children=[
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
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ]
                            )
                        ])
                    ]
                ),

                # ---------------- Horizontal divider
                html.Div(
                    style={
                        "height": "2px",
                        "backgroundColor": color_features,
                        "margin": "6px 0"
                    }
                ),

                # Bottom right: Galaxy plot
                html.Div(
                    style={"flex": "1", "height": "100%","padding": "40px"},
                    children=[
                        dcc.Graph(
                            id="galaxy-z-plot",
                            style={"height": "100%"}
                        )
                    ]
                )

            ]
        )

    ]
)


# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("eval-output", "children"),
    Input("eval-button", "n_clicks"),
    State("input-ra", "value"),
    State("input-dec", "value"),
    Input("plot-type", "value")
)
def evaluate_position(_, ra_val, dec_val,plot_type):
    if ra_val is None or dec_val is None:
        return "Enter RA & Dec"

    if not (ra_grid.min() <= ra_val <= ra_grid.max() and
            dec_grid.min() <= dec_val <= dec_grid.max()):
        return "Outside map"


    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
        label_cursor = "SR"
        label_plot = "Strehl Ratio"
        min_Z = min_SR
        max_Z = max_SR
        nbin_Z = nbin_SR
    elif plot_type == "fwhm":
        Z = Z_fwhm
        label_Z = "FWHM (mas)"
        label_cursor = "FWHM"
        label_plot = "FWHM (mas)"
        min_Z = min_FWHM
        max_Z = max_FWHM
        nbin_Z = nbin_FWHM

    z_val = nearest_z(ra_val, dec_val,Z)
    return f"{label_Z} = {z_val:.6f}"

@app.callback(
    Output("sky-map", "figure"),
    Output("galaxy-z-plot", "figure"),
    Input("z-slider", "value"),
    Input("display-options", "value"),
    Input("plot-type", "value")
)
def update_galaxies(z_range, display_options,plot_type):

    if plot_type == "strehl":
        Z = Z_sr
        label_Z = "SR"
        label_cursor = "SR"
        label_plot = "Strehl Ratio"
        min_Z = min_SR
        max_Z = max_SR
        nbin_Z = nbin_SR
    elif plot_type == "fwhm":
        Z = Z_fwhm
        label_Z = "FWHM (mas)"
        label_cursor = "FWHM"
        label_plot = "FWHM (mas)"
        min_Z = min_FWHM
        max_Z = max_FWHM
        nbin_Z = nbin_FWHM
        
    zmin, zmax = z_range
    mask = (gal_z >= zmin) & (gal_z <= zmax)

    show_gal = "gal" in display_options
    show_stars = "stars" in display_options

    sky_fig = make_sky_figure(mask, show_gal=show_gal, show_stars=show_stars, plot_type=plot_type)

    if np.any(mask):
        zg = z_at_galaxies(mask,Z)

        if plot_type == "fwhm":
            # Transform zg: clip outside range to min_Z-1 / max_Z+1
            zg_clipped = np.where(zg < min_Z, min_Z - 1, zg)
            zg_clipped = np.where(zg_clipped > max_Z, max_Z + 1, zg_clipped)
        else:
            zg_clipped = zg

        gal_fig = go.Figure(
            go.Histogram(
                x=zg_clipped,
                nbinsx=nbin_Z,                  # number of bins
                histnorm='percent',         # normalize to percentage
                marker=dict(
                    color="orange", opacity=0.9
                ),
            )
        )

        gal_fig.update_layout(
            title= label_plot + " value for selected galaxies at 2.2 microns",
            xaxis=dict(title=label_plot,gridcolor=color_features,range=[min_Z, max_Z]),
            yaxis=dict(title="Sky Coverage (in % per bins)",gridcolor=color_features),
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

    return sky_fig, gal_fig

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Open a public tunnel to your Dash app
    #public_url = ngrok.connect(8050)
    #print(f"Public URL: {public_url}")
    #app.run(port=8050)
    url = f"http://{HOST}:{PORT}/"
    print(f"Opening {url}")
    webbrowser.open(url)
    app.run(debug=True, host=HOST, port=PORT)
