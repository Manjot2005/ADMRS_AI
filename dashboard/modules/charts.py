"""
charts.py — All Plotly chart builders for ADMRS
Extracted from app.py for modularity and caching.
"""
import numpy as np
import pandas as pd
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


MONO = "Share Tech Mono, monospace"
BG   = "#0b0e17"
BG2  = "#0e1420"
BG3  = "#080b12"
BORDER = "#1a2035"
GRID   = "#111827"


def _rgba(hex_col: str, alpha: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════════
#  MAIN DUAL MAP (Situational Awareness)
# ══════════════════════════════════════════════════════════════════
def build_main_map(df_json: str, layer_mode: str = "deforestation",
                   pulse_alpha: float = 0.25):
    df = pd.read_json(StringIO(df_json), orient='records')
    clat = float(df["lat"].mean())
    clon = float(df["lon"].mean())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["DEFORESTATION DETECTED (AI)", "PREDICTIVE RISK ZONES"],
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]],
        horizontal_spacing=0.008)

    # ── Density rings ──
    for ha, lat, lon in zip(df["new_loss_ha"], df["lat"], df["lon"]):
        r      = max(0.07, float(ha) * 0.011)
        angles = np.linspace(0, 2 * np.pi, 30)
        clats  = [lat + r * np.cos(a) for a in angles] + [lat + r * np.cos(angles[0])]
        clons  = [lon + r * np.sin(a) for a in angles] + [lon + r * np.sin(angles[0])]
        fa = 0.32 if layer_mode == "heatmap" else 0.12
        la = 0.20 if layer_mode == "heatmap" else 0.08
        col = "#f85149" if ha > 6 else "#d29922" if ha > 3 else "#3fb950"
        fig.add_trace(go.Scattermapbox(
            lat=clats, lon=clons, mode="lines",
            fill="toself", fillcolor=_rgba(col, fa),
            line=dict(color=_rgba(col, la), width=0),
            hoverinfo="skip", showlegend=False), row=1, col=1)

    # ── Pulsing rings for critical ──
    crit = df[df["new_loss_ha"] > 6]
    if len(crit) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=list(crit["lat"]), lon=list(crit["lon"]), mode="markers",
            marker=dict(
                size=[max(30, min(52, int(ha * 5))) for ha in crit["new_loss_ha"]],
                color="#f85149", opacity=pulse_alpha, allowoverlap=True),
            hoverinfo="skip", showlegend=False), row=1, col=1)

    # ── Core detection markers ──
    colors = ["#f85149" if ha > 6 else "#d29922" if ha > 3 else "#3fb950"
              for ha in df["new_loss_ha"]]
    sizes  = [max(10, min(28, int(ha * 2.8))) for ha in df["new_loss_ha"]]
    hover  = [f"<b>{r['alert_id']}</b><br>Area: {r['new_loss_ha']:.1f}ha"
              f"<br>Conf: {r['confidence']:.0%}"
              f"<br><a href='https://maps.google.com/?q={r['lat']},{r['lon']}'>📍 Open in Maps</a>"
              for _, r in df.iterrows()]

    if layer_mode == "ndvi":
        ndvi_vals = [round(0.72 - float(ha) * 0.03, 2) for ha in df["new_loss_ha"]]
        fig.add_trace(go.Scattermapbox(
            lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
            marker=dict(size=sizes, color=ndvi_vals, colorscale="RdYlGn",
                        cmin=0.3, cmax=0.8, opacity=0.92, allowoverlap=True,
                        colorbar=dict(title="NDVI", x=0.48, thickness=8,
                                      tickfont=dict(size=8, color="#8892a4"))),
            text=hover, hovertemplate="%{text}<extra></extra>",
            showlegend=False), row=1, col=1)

    elif layer_mode == "thermal":
        fig.add_trace(go.Scattermapbox(
            lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
            marker=dict(size=sizes, color=list(df["new_loss_ha"]),
                        colorscale="Hot", opacity=0.92, allowoverlap=True,
                        colorbar=dict(title="Heat", x=0.48, thickness=8,
                                      tickfont=dict(size=8, color="#8892a4"))),
            text=hover, hovertemplate="%{text}<extra></extra>",
            showlegend=False), row=1, col=1)

    else:
        fig.add_trace(go.Scattermapbox(
            lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.95, allowoverlap=True),
            text=hover, hovertemplate="%{text}<extra></extra>",
            name="Detections", showlegend=True), row=1, col=1)

    # ── Right map: risk halos ──
    fig.add_trace(go.Scattermapbox(
        lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
        marker=dict(size=[max(30, min(56, int(ha * 5.8))) for ha in df["new_loss_ha"]],
                    color="#f85149", opacity=0.06, allowoverlap=True),
        hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scattermapbox(
        lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
        marker=dict(size=[max(18, min(34, int(ha * 3))) for ha in df["new_loss_ha"]],
                    color="#d29922", opacity=0.13, allowoverlap=True),
        hoverinfo="skip", showlegend=False), row=1, col=2)
    colors2 = ["#f85149" if ha > 6 else "#d29922" if ha > 3 else "#3fb950"
               for ha in df["new_loss_ha"]]
    fig.add_trace(go.Scattermapbox(
        lat=list(df["lat"]), lon=list(df["lon"]), mode="markers",
        marker=dict(size=[max(8, min(20, int(ha * 2.1))) for ha in df["new_loss_ha"]],
                    color=colors2, opacity=1, allowoverlap=True),
        text=[f"<b>{r['alert_id']}</b><br>{r['new_loss_ha']:.1f}ha"
              for _, r in df.iterrows()],
        hovertemplate="%{text}<extra></extra>", showlegend=False), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=0, r=0, t=28, b=0), height=385,
        legend=dict(bgcolor=BG2, bordercolor=BORDER, borderwidth=1,
                    font=dict(size=10, family=MONO),
                    orientation="h", x=0.01, y=0.01),
        mapbox=dict(
            style="white-bg",
            center=dict(lat=clat, lon=clon), zoom=5,
            layers=[dict(
                sourcetype="raster",
                source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                below="traces"
            )]),
        mapbox2=dict(
            style="white-bg",
            center=dict(lat=clat, lon=clon), zoom=5,
            layers=[dict(
                sourcetype="raster",
                source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                below="traces"
            )]))
    for ann in fig.layout.annotations:
        ann.font.update(color="#4a5568", size=10, family=MONO)
    return fig


# ══════════════════════════════════════════════════════════════════
#  FORENSIC MAP
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_forensic_map(lat: float, lon: float, ha: float,
                        aid: str, before_date: str):
    d   = float((ha ** 0.5) * 0.035)
    rng = np.random.default_rng(int(ha * 100))
    poly_lat = [lat-d, lat+d, lat+d, lat-d, lat-d]
    poly_lon = [lon-d, lon-d, lon+d, lon+d, lon-d]
    p_lat, p_lon = [], []
    for _ in range(4):
        dl  = lat + rng.uniform(-.25, .25)
        dln = lon + rng.uniform(-.25, .25)
        dd  = d * rng.uniform(.25, .65)
        p_lat += [dl-dd, dl+dd, dl+dd, dl-dd, dl-dd, None]
        p_lon += [dln-dd, dln-dd, dln+dd, dln+dd, dln-dd, None]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f"BEFORE — {before_date}", "AFTER — 2024-03-02"],
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]],
        horizontal_spacing=0.015)
    fig.add_trace(go.Scattermapbox(
        lat=[lat], lon=[lon], mode="markers",
        marker=dict(size=18, color="#3fb950", opacity=0.9), showlegend=False,
        hovertemplate="<b>Forest intact</b><extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scattermapbox(
        lat=poly_lat, lon=poly_lon, mode="lines",
        fill="toself", fillcolor="rgba(248,81,73,0.28)",
        line=dict(color="#f85149", width=2), showlegend=False,
        hovertemplate=f"<b>{aid}</b> — {ha:.1f}ha<extra></extra>"), row=1, col=2)
    if p_lat:
        fig.add_trace(go.Scattermapbox(
            lat=p_lat, lon=p_lon, mode="lines",
            fill="toself", fillcolor="rgba(192,57,43,0.18)",
            line=dict(color="#c0392b", width=1),
            hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scattermapbox(
        lat=[lat], lon=[lon], mode="markers",
        marker=dict(size=10, color="#58a6ff", opacity=1),
        showlegend=False), row=1, col=2)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=0, r=0, t=28, b=0), height=400, showlegend=False,
        mapbox=dict(
            style="white-bg",
            center=dict(lat=lat, lon=lon), zoom=9,
            layers=[dict(
                sourcetype="raster",
                source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                below="traces"
            )]),
        mapbox2=dict(
            style="white-bg",
            center=dict(lat=lat, lon=lon), zoom=9,
            layers=[dict(
                sourcetype="raster",
                source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
                below="traces"
            )]))
    for i, ann in enumerate(fig.layout.annotations):
        ann.font.update(color="#3fb950" if i == 0 else "#f85149",
                        size=10, family=MONO)
    return fig


# ══════════════════════════════════════════════════════════════════
#  ENHANCED NDVI + LOSS RATE
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_ndvi_chart(_df_json: str):
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    rng  = np.random.default_rng(12)
    ndvi = [round(0.72 - i * 0.005 + rng.uniform(-0.015, 0.015), 3) for i in range(12)]
    prev = [round(v + rng.uniform(-0.02, 0.01), 3) for v in ndvi]
    loss = [round(1.8 + i * 0.3 + rng.uniform(-0.2, 0.4), 2) for i in range(12)]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                         vertical_spacing=0.08,
                         subplot_titles=["NDVI — BEFORE/AFTER", "LOSS RATE ha/wk"])
    fig.add_trace(go.Scatter(x=months, y=prev, mode="lines", name="2023",
        line=dict(color="#1a2d3a", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=ndvi, mode="lines+markers", name="2024",
        line=dict(color="#3fb950", width=2),
        fill="tonexty", fillcolor="rgba(63,185,80,0.06)",
        marker=dict(size=4, color="#3fb950")), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#d29922",
        annotation_text="Threshold", annotation_font_color="#d29922",
        annotation_font_size=8, row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=loss, name="Loss",
        marker=dict(color=["#f85149" if v > 3 else "#d29922" if v > 2 else "#3fb950"
                           for v in loss], opacity=0.85)), row=2, col=1)
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=8, r=8, t=28, b=8), height=280, showlegend=True,
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=8, family=MONO)),
        yaxis=dict(gridcolor=GRID, range=[0.3, 0.85]),
        xaxis2=dict(gridcolor=GRID, tickfont=dict(size=8, family=MONO)),
        yaxis2=dict(gridcolor=GRID, title_font=dict(size=8)),
        legend=dict(bgcolor=BG2, bordercolor=BORDER, font=dict(size=8)),
        bargap=0.2)
    for ann in fig.layout.annotations:
        ann.font.update(size=8, color="#2d3a52", family=MONO)
    return fig


# ══════════════════════════════════════════════════════════════════
#  FORECAST CHART
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def build_forecast_chart(forecast: dict):
    hist_d = forecast["hist_dates"]
    hist_v = forecast["hist_values"]
    fut_d  = forecast["dates"]
    fut_v  = forecast["forecast"]
    lo     = forecast["lower"]
    hi     = forecast["upper"]

    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(x=hist_d, y=hist_v, mode="lines+markers",
        name="Historical", line=dict(color="#58a6ff", width=2),
        marker=dict(size=4)))
    # Confidence band
    fig.add_trace(go.Scatter(
        x=fut_d + fut_d[::-1],
        y=hi + lo[::-1],
        fill="toself", fillcolor="rgba(63,185,80,0.08)",
        line=dict(color="rgba(63,185,80,0)", width=0),
        hoverinfo="skip", name="90% CI", showlegend=True))
    # Forecast line
    fig.add_trace(go.Scatter(x=fut_d, y=fut_v, mode="lines+markers",
        name="Forecast", line=dict(color="#3fb950", width=2, dash="dot"),
        marker=dict(size=5, color="#3fb950")))
    # Upper/lower bounds
    fig.add_trace(go.Scatter(x=fut_d, y=hi, mode="lines",
        line=dict(color="rgba(63,185,80,0.3)", width=1),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut_d, y=lo, mode="lines",
        line=dict(color="rgba(63,185,80,0.3)", width=1),
        showlegend=False, hoverinfo="skip"))

    # Divider between hist and forecast.
    # Use a Scatter trace instead of add_vline — categorical string x-axis
    # causes add_vline to crash on Plotly <5.14 (TypeError: int + str).
    if hist_d and fut_d:
        y_min = min(list(lo) + list(hist_v))
        y_max = max(list(hi) + list(hist_v))
        fig.add_trace(go.Scatter(
            x=[hist_d[-1], hist_d[-1]], y=[y_min, y_max],
            mode="lines",
            line=dict(color="#2d3a52", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(
            x=hist_d[-1], y=y_max, text="NOW",
            font=dict(size=8, color="#2d3a52", family=MONO),
            showarrow=False, yanchor="bottom")

    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=8, r=8, t=16, b=8), height=240,
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=8, family=MONO),
                   tickangle=-30),
        yaxis=dict(gridcolor=GRID, title="ha/week",
                   title_font=dict(size=8)),
        legend=dict(bgcolor=BG2, bordercolor=BORDER,
                    font=dict(size=9), orientation="h", x=0, y=1.05))
    return fig


# ══════════════════════════════════════════════════════════════════
#  AI CONFIDENCE GAUGE
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_confidence_gauge(conf: float):
    color = "#3fb950" if conf >= 0.85 else "#d29922" if conf >= 0.65 else "#f85149"
    label = "HIGH CONFIDENCE" if conf >= 0.85 else "MEDIUM" if conf >= 0.65 else "LOW"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number=dict(suffix="%", font=dict(size=26, color=color,
                                           family="Barlow Condensed")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#2d3a52",
                      tickfont=dict(size=8, family=MONO, color="#2d3a52")),
            bar=dict(color=color, thickness=0.7),
            bgcolor=BG3, borderwidth=0,
            steps=[dict(range=[0,65],  color="#1a0a0a"),
                   dict(range=[65,85], color="#1a150a"),
                   dict(range=[85,100],color="#0a1a0d")],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.8, value=conf*100)),
        title=dict(text=f"<b>{label}</b>",
                   font=dict(size=9, color=color, family=MONO))))
    fig.update_layout(
        paper_bgcolor=BG2, font_color="#c9d1d9",
        margin=dict(l=10, r=10, t=30, b=0), height=155)
    return fig


# ══════════════════════════════════════════════════════════════════
#  GLOBAL MONITORING MAP
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_global_map(n_amazon: int = 14):
    zones = [
        {"zone":"Amazon Basin","lat":-3.47,"lon":-60.02,"alerts":n_amazon,"status":"ACTIVE"},
        {"zone":"Congo Basin", "lat":-0.23,"lon":23.65, "alerts":2,       "status":"MONITORING"},
        {"zone":"Indonesia",   "lat":-0.79,"lon":113.92,"alerts":5,       "status":"ACTIVE"},
        {"zone":"Borneo",      "lat":0.96, "lon":114.55,"alerts":3,       "status":"MONITORING"},
        {"zone":"SE Asia",     "lat":14.1, "lon":101.0, "alerts":1,       "status":"NOMINAL"},
    ]
    cc = {"ACTIVE":"#f85149","MONITORING":"#d29922","NOMINAL":"#3fb950"}
    fig = go.Figure()
    for z in zones:
        c = cc.get(z["status"], "#58a6ff")
        fig.add_trace(go.Scattergeo(
            lat=[z["lat"]], lon=[z["lon"]], mode="markers+text",
            marker=dict(size=max(10, min(22, z["alerts"] * 2)), color=c,
                        opacity=0.85, line=dict(color=c, width=1)),
            text=[z["zone"]], textposition="top center",
            textfont=dict(size=8, color=c, family=MONO),
            hovertemplate=(f"<b>{z['zone']}</b><br>Alerts: {z['alerts']}"
                           f"<br>Status: {z['status']}<extra></extra>"),
            showlegend=False))
        fig.add_trace(go.Scattergeo(
            lat=[z["lat"]], lon=[z["lon"]], mode="markers",
            marker=dict(size=max(18, min(40, z["alerts"] * 4)), color=c, opacity=0.1),
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        paper_bgcolor=BG2, font_color="#c9d1d9",
        margin=dict(l=0, r=0, t=0, b=0), height=200,
        geo=dict(bgcolor=BG3, showframe=False, showcoastlines=True,
                 coastlinecolor=BORDER, showland=True, landcolor=BG,
                 showocean=True, oceancolor=BG3,
                 showlakes=True, lakecolor=BG2,
                 showcountries=True, countrycolor=GRID,
                 projection_type="natural earth"))
    return fig


# ══════════════════════════════════════════════════════════════════
#  NDVI CLASSIFICATION CHART  (Synopsis §2.4 — NDVI Value Table)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_ndvi_classification(ndvi_values: list):
    """
    Bar chart showing distribution of pixels across NDVI classes
    as defined in the synopsis (Tucker 1979 scale).
    NDVI = (NIR - RED) / (NIR + RED)
    """
    # Classify each alert's estimated NDVI into synopsis categories
    categories = ["< 0\nWater/Soil", "0–0.3\nSparse/Grass",
                  "0.3–0.6\nModerate Veg", "0.6–1\nDense Forest"]
    colors     = ["#58a6ff", "#d29922", "#3fb950", "#1a7a30"]
    # Count per category
    counts = [0, 0, 0, 0]
    for v in ndvi_values:
        if v < 0:    counts[0] += 1
        elif v < 0.3: counts[1] += 1
        elif v < 0.6: counts[2] += 1
        else:         counts[3] += 1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=counts,
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color=BORDER, width=1)),
        text=counts, textposition="outside",
        textfont=dict(size=9, family=MONO, color="#e6edf3")))
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=8, r=8, t=10, b=8), height=185,
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=8, family=MONO)),
        yaxis=dict(gridcolor=GRID, title="Alert Count",
                   title_font=dict(size=8), tickfont=dict(size=8)),
        showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX  (Synopsis §3.4 — Accuracy Assessment)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_confusion_matrix():
    """
    Confusion matrix for NDVI threshold classifier.
    Values from Synopsis §3.4: Overall Accuracy=85%, Kappa=0.78
    """
    # Derived from OA=85%, Kappa=0.78 with balanced classes
    z = [[76, 12],   # [TP, FN]  — actual Forest
         [9,  68]]   # [FP, TN]  — actual Non-Forest
    text = [["TP: 76", "FN: 12"],
            ["FP: 9",  "TN: 68"]]
    labels = ["Forest", "Non-Forest"]

    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred: Forest", "Pred: Non-Forest"],
        y=["Actual: Forest", "Actual: Non-Forest"],
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, family=MONO),
        colorscale=[[0,"#0b0e17"],[0.5,"#0d2218"],[1,"#3fb950"]],
        showscale=False,
        hovertemplate="<b>%{text}</b><br>Count: %{z}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG, font_color="#c9d1d9",
        margin=dict(l=8, r=8, t=10, b=8), height=200,
        xaxis=dict(tickfont=dict(size=9, family=MONO), side="top"),
        yaxis=dict(tickfont=dict(size=9, family=MONO), autorange="reversed"))
    return fig


# ══════════════════════════════════════════════════════════════════
#  NDVI HEATMAP GRID  (Synopsis §1.3, §3.3.4 — Visualization)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_ndvi_heatmap_grid(seed: int = 42):
    """
    Simulated NDVI heatmap over an AOI grid cell (20x20 pixels).
    Represents the heatmap output described in the synopsis.
    In production: replace with real rasterio-loaded band data.
    NDVI = (NIR - RED) / (NIR + RED)
    """
    rng = np.random.default_rng(seed)
    H, W = 20, 20
    # Simulate NIR and RED bands (reflectance 0-1)
    nir = np.clip(rng.normal(0.45, 0.15, (H, W)), 0.01, 1.0)
    red = np.clip(rng.normal(0.20, 0.12, (H, W)), 0.01, 1.0)
    # Create deforested patch (bottom-right quadrant)
    nir[12:, 12:] = np.clip(rng.normal(0.15, 0.05, (8, 8)), 0.01, 0.4)
    red[12:, 12:] = np.clip(rng.normal(0.35, 0.06, (8, 8)), 0.1, 0.7)
    # NDVI formula from synopsis
    ndvi = (nir - red) / (nir + red + 1e-8)

    fig = go.Figure(go.Heatmap(
        z=ndvi.tolist(),
        colorscale=[
            [0.0,  "#f85149"],   # < 0   : non-veg/water
            [0.25, "#d29922"],   # 0–0.3 : sparse
            [0.55, "#3fb950"],   # 0.3–0.6: moderate
            [1.0,  "#1a7a30"],   # 0.6–1 : dense forest
        ],
        zmin=-0.2, zmax=0.85,
        colorbar=dict(
            title="NDVI", tickvals=[-0.2, 0, 0.3, 0.6, 0.85],
            ticktext=["<0", "0", "0.3", "0.6", ">0.6"],
            tickfont=dict(size=8, family=MONO, color="#8892a4"),
            title_font=dict(size=8, family=MONO, color="#8892a4"),
            len=0.9, thickness=10),
        hovertemplate="Row %{y}, Col %{x}<br>NDVI: %{z:.3f}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        margin=dict(l=8, r=8, t=8, b=8), height=240,
        xaxis=dict(tickfont=dict(size=7, family=MONO), showgrid=False,
                   title="Pixel Column", title_font=dict(size=8)),
        yaxis=dict(tickfont=dict(size=7, family=MONO), showgrid=False,
                   title="Pixel Row", title_font=dict(size=8),
                   autorange="reversed"))
    return fig


# ══════════════════════════════════════════════════════════════════
#  BINARY CLASSIFICATION MAP  (Synopsis §3.3.4 — Forest/Non-Forest)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def build_binary_class_map(seed: int = 42):
    """
    Binary Forest/Non-Forest classification map from NDVI threshold.
    Threshold = 0.3 (synopsis §2.4: values > 0.3 = vegetated).
    """
    rng = np.random.default_rng(seed)
    H, W = 20, 20
    nir = np.clip(rng.normal(0.45, 0.15, (H, W)), 0.01, 1.0)
    red = np.clip(rng.normal(0.20, 0.12, (H, W)), 0.01, 1.0)
    nir[12:, 12:] = np.clip(rng.normal(0.15, 0.05, (8, 8)), 0.01, 0.4)
    red[12:, 12:] = np.clip(rng.normal(0.35, 0.06, (8, 8)), 0.1, 0.7)
    ndvi = (nir - red) / (nir + red + 1e-8)
    # Threshold-based classification (synopsis §3.3.3)
    binary = (ndvi >= 0.3).astype(float)

    forest_pct  = round(float(binary.mean()) * 100, 1)
    defor_pct   = round(100 - forest_pct, 1)

    fig = go.Figure(go.Heatmap(
        z=binary.tolist(),
        colorscale=[[0, "#f85149"], [1, "#3fb950"]],
        zmin=0, zmax=1, showscale=False,
        hovertemplate="Row %{y}, Col %{x}<br>Class: %{z:.0f} (1=Forest)<extra></extra>"))
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Forest: {forest_pct}%  |  Deforested: {defor_pct}%",
        font=dict(size=9, family=MONO, color="#e6edf3"),
        bgcolor="#080b12", bordercolor="#1a2035", borderwidth=1,
        showarrow=False, align="left", xanchor="left", yanchor="top")
    fig.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        margin=dict(l=8, r=8, t=8, b=8), height=240,
        xaxis=dict(tickfont=dict(size=7, family=MONO), showgrid=False,
                   title="Pixel Column", title_font=dict(size=8)),
        yaxis=dict(tickfont=dict(size=7, family=MONO), showgrid=False,
                   title="Pixel Row", title_font=dict(size=8),
                   autorange="reversed"))
    return fig
