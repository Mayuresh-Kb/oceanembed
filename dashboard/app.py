"""Research data portal for OceanEmbed saved reconstruction products."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data_access import RUNS, load_collocations, load_metrics, load_prediction_cube, load_surface_fields
from dashboard.research_portal import RESEARCH_VARIABLES, research_csv, research_layer


st.set_page_config(page_title="OceanEmbed Research Portal", page_icon="🌊", layout="wide")
st.markdown("""<style>
.stApp{background:#071827;color:#e9f4ff;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.block-container{padding-top:1.25rem;max-width:1500px}
.main-brand{display:flex;align-items:center;gap:13px;margin:4px 0 22px}.main-brand svg{width:43px;height:43px;stroke:#19d2d4;stroke-width:2.4;fill:none;stroke-linecap:round}.main-brand-title{font-size:2rem;font-weight:720;line-height:1;color:#edf8ff}.main-brand-subtitle{font-size:.88rem;color:#aac2d3;margin-top:5px}.page-title{font-size:1.62rem;font-weight:680;color:#edf8ff;margin:0 0 4px}.page-subtitle{font-size:.92rem;color:#aac2d3;margin:0 0 20px}
.card{background:#0d2538;border:1px solid #1e425b;border-radius:12px;padding:14px;min-height:105px;overflow-wrap:anywhere}.card-title{font-size:.76rem;color:#aac2d3;font-weight:600;letter-spacing:.07em}.card-value{font-size:1.25rem;font-weight:680;margin:8px 0;color:#edf8ff}.card-detail{font-size:.86rem;color:#75d8e8;line-height:1.35}.insight-card{min-height:184px;box-sizing:border-box;background:linear-gradient(145deg,#10314a,#0b2236);border:1px solid #22516d;border-radius:14px;padding:18px;margin-bottom:16px;overflow:hidden}.insight-icon{height:28px}.insight-icon svg{width:27px;height:27px;stroke:#22d3d5;stroke-width:1.8;fill:none;stroke-linecap:round;stroke-linejoin:round}.insight-title{font-size:1.05rem;font-weight:680;color:#85ebf2;margin:9px 0 7px}.insight-body{font-size:.9rem;line-height:1.45;color:#d8e9f5}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061e34 0%,#071827 100%);border-right:1px solid #16425c}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .brand{padding:20px 8px 12px}.wave{font-size:2.2rem;line-height:1;color:#19d2d4;letter-spacing:-.35em}.brand-title{font-weight:720;color:#edf8ff;font-size:1.32rem;line-height:1.12;margin-top:12px}.brand-subtitle{color:#80a9c4;font-size:.78rem;margin-top:6px}[data-testid="stSidebar"] [role="radiogroup"]{gap:.35rem}[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent;border-radius:7px;padding:8px 10px;color:#bdd2e2}[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#0a476a;color:#dffaff;border-left:3px solid #19d2d4}[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading saved research products…")
def artifacts(run_name: str):
    return load_prediction_cube(run_name), load_metrics(run_name), load_collocations(run_name)


@st.cache_data(show_spinner="Loading surface fields…")
def surface_artifacts(run_name: str):
    return load_surface_fields(run_name)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def compact_card(title: str, value: str, detail: str) -> None:
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div><div class='card-value'>{value}</div><div class='card-detail'>{detail}</div></div>", unsafe_allow_html=True)


def insight_icon(concept: str) -> str:
    paths = {
        "SST": "<path d='M12 3v10.5a4 4 0 1 1-3 0V3a3 3 0 0 1 3-3z'/><path d='M9 9h3'/>",
        "SSS": "<path d='M12 2s6 6.6 6 11a6 6 0 1 1-12 0c0-4.4 6-11 6-11z'/><path d='M9 15c.8 1 2 1.5 3.4 1.2'/>",
        "SSH / SLA": "<path d='M2 8c3-3 5 3 8 0s5 3 8 0 3 0 4 0'/><path d='M2 15c3-3 5 3 8 0s5 3 8 0 3 0 4 0'/>",
        "Currents": "<path d='M4 8c3-5 12-5 15 0 3 5-6 8-10 4-3-4 3-7 7-4'/><path d='M17 5l2 3-3 1'/>",
        "Surface winds": "<path d='M3 8h12c4 0 4-5 0-5-2 0-3 1-3 2'/><path d='M3 13h17c3 0 3 4 0 4-2 0-3-1-3-2'/><path d='M3 18h9'/>",
        "Thermocline": "<path d='M4 4h16M4 10h16M4 16h16'/><path d='M12 4v12'/><path d='M9 13l3 3 3-3'/>",
        "Stratification": "<path d='M4 5h16M4 10h16M4 15h16M4 20h16'/><circle cx='8' cy='7.5' r='1'/><circle cx='16' cy='17.5' r='1'/>",
        "Upwelling": "<path d='M4 18c3-3 5 3 8 0s5 3 8 0'/><path d='M12 17V5'/><path d='M8 9l4-4 4 4'/>",
        "Downwelling": "<path d='M4 18c3-3 5 3 8 0s5 3 8 0'/><path d='M12 5v12'/><path d='M8 13l4 4 4-4'/>",
        "Ocean eddies": "<path d='M19 9a7 7 0 1 0 1 6'/><path d='M16 5l3 4-5 1'/><path d='M14 11a3 3 0 1 1-1 4'/>",
        "Subsurface temperature": "<path d='M3 7c3-3 5 3 8 0s5 3 10 0'/><path d='M12 10v8'/><path d='M9 15l3 3 3-3'/><path d='M5 20h14'/>",
        "Marine heatwaves": "<path d='M12 2c3 4 6 6 6 11a6 6 0 1 1-12 0c0-3 2-5 4-7 0 3 2 4 2 4 1-2 0-5 0-8z'/>",
    }
    return f"<svg viewBox='0 0 24 24' aria-hidden='true'>{paths.get(concept, paths['Subsurface temperature'])}</svg>"


def research_figure(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, *, label: str, units: str, cmap: str, symmetric: bool, vectors: tuple[np.ndarray, np.ndarray] | None) -> go.Figure:
    finite = field[np.isfinite(field)]
    low = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
    high = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if symmetric:
        edge = max(abs(low), abs(high), 0.1); low, high = -edge, edge
    if high <= low: high = low + 1.0
    scale = {"temperature": "Turbo", "salinity": "Cividis", "height": "RdBu", "speed": "Viridis"}[cmap]
    figure = go.Figure(go.Heatmap(z=field, x=longitude, y=latitude, colorscale=scale, zmin=low, zmax=high, zmid=0.0 if symmetric else None, colorbar={"title": units, "thickness": 16}, hovertemplate="%{y:.2f}°N, %{x:.2f}°E<br>" + label + ": %{z:.3f} " + units + "<extra></extra>"))
    if vectors is not None:
        u, v = vectors; step = max(1, int(max(len(latitude), len(longitude)) / 12)); arrow_length = 0.10 * min(float(np.ptp(latitude)), float(np.ptp(longitude))); xs, ys = [], []
        for row in range(0, len(latitude), step):
            for column in range(0, len(longitude), step):
                east, north = float(u[row, column]), float(v[row, column]); speed = float(np.hypot(east, north))
                if np.isfinite(speed) and speed > 0:
                    xs.extend([longitude[column], longitude[column] + arrow_length * east / speed, None]); ys.extend([latitude[row], latitude[row] + arrow_length * north / speed, None])
        figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#e5fbff", "width": 1}, name="Direction", hoverinfo="skip"))
    figure.update_layout(template="plotly_dark", paper_bgcolor="#071827", plot_bgcolor="#071827", height=500, margin={"l": 55, "r": 25, "t": 36, "b": 45}, title={"text": label + " — hover to inspect", "font": {"size": 16}}, xaxis={"title": "Longitude (°E)", "showgrid": False}, yaxis={"title": "Latitude (°N)", "showgrid": False, "scaleanchor": "x", "scaleratio": 1})
    return figure


def profile_figure(depths: list[float], values: np.ndarray) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5, 4)); figure.patch.set_facecolor("#0d2538"); axis.set_facecolor("#0d2538")
    axis.plot(values, depths, "o-", color="#75d8e8"); axis.invert_yaxis(); axis.grid(alpha=.2, color="white")
    axis.set(xlabel="Temperature (°C)", ylabel="Depth (m)", title="Reconstructed vertical profile")
    axis.tick_params(colors="#dcecf5"); axis.xaxis.label.set_color("#dcecf5"); axis.yaxis.label.set_color("#dcecf5"); axis.title.set_color("#dcecf5")
    return figure


st.markdown("""<div class='main-brand'><svg viewBox='0 0 48 48' aria-label='OceanEmbed wave logo'><path d='M5 17c7-8 13 7 20 0s13 8 20 0'/><path d='M5 25c7-8 13 7 20 0s13 8 20 0'/><path d='M5 33c7-8 13 7 20 0s13 8 20 0'/></svg><div><div class='main-brand-title'>OCEANEMBED</div><div class='main-brand-subtitle'>Research data portal for subsurface ocean temperature</div></div></div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div class='brand'><div class='wave'>≋≋</div><div class='brand-title'>OCEANEMBED</div><div class='brand-subtitle'>Research data portal</div></div>""", unsafe_allow_html=True)
navigation = st.sidebar.radio("Navigation", ["Dataset Explorer", "Validation", "About"], label_visibility="collapsed")
st.sidebar.divider(); st.sidebar.markdown("#### Active research dataset")
selected_run = "Bay of Bengal: 150-day experiment"
active_run = RUNS[selected_run]; lat_min, lat_max, lon_min, lon_max = active_run["region_bounds"]
st.sidebar.markdown(f"**{active_run['region_title']} dataset**"); st.sidebar.caption(f"Coverage: {lat_min:g}–{lat_max:g}°N · {lon_min:g}–{lon_max:g}°E"); st.sidebar.caption("CMEMS GLORYS12V1 reanalysis with ERA5 wind inputs.")
cube, metrics, collocations = artifacts(selected_run); bundle, surface = surface_artifacts(selected_run)

if navigation == "Dataset Explorer":
    page_header("Dataset Explorer", "Explore and download reconstructed ocean fields across depth and time.")
    times = [str(value)[:10] for value in cube.time.values]
    latitude, longitude = bundle.latitude, bundle.longitude
    date_col, variable_col = st.columns([1.1, 3.2])
    selected_time = date_col.selectbox("Analysis date", times, index=len(times) - 1)
    variable = variable_col.radio("Map field", RESEARCH_VARIABLES, horizontal=True)
    depths = [float(value) for value in cube.depth.values]
    if "reconstruction_depth" not in st.session_state or st.session_state["reconstruction_depth"] not in depths:
        st.session_state["reconstruction_depth"] = 10.0 if 10.0 in depths else depths[0]
    selected_depth = float(st.session_state["reconstruction_depth"])
    source_date_index = int(np.where(bundle.times.astype("datetime64[D]") == np.datetime64(selected_time))[0][0])
    prediction = cube["oceanembed_temperature"].sel(time=np.datetime64(selected_time), depth=selected_depth).values
    field, label, units, cmap, symmetric, vectors, surface_depth, source = research_layer(variable, prediction, surface, source_date_index)
    export_depth = selected_depth if surface_depth is None else surface_depth
    valid_cells, total_cells = int(np.isfinite(field).sum()), int(field.size)

    map_column, metadata_column = st.columns([2.15, 1], gap="large")
    with map_column:
        st.markdown(f"#### Interactive subsurface-temperature map — {active_run['region_title']}")
        depth_note = f" at {export_depth:g} m" if export_depth is not None else " at the surface"
        st.caption(f"{label}{depth_note} on {selected_time}. Hover over the map for the value at each grid cell; use the legend at right to interpret colours.")
        st.plotly_chart(research_figure(field, latitude, longitude, label=label, units=units, cmap=cmap, symmetric=symmetric, vectors=vectors), width="stretch", config={"scrollZoom": True, "displaylogo": False})
        if variable == "Reconstructed subsurface temperature":
            st.select_slider("Reconstruction depth — discrete model output levels", options=depths, key="reconstruction_depth", format_func=lambda value: f"{value:g} m", help="OceanEmbed predicts these defined levels; it is not continuous depth output.")
        else:
            st.caption("This is a surface field; its depth is fixed at 0 m.")

    with metadata_column:
        st.markdown("#### Dataset details")
        compact_card(
            "DATASET",
            active_run["region_title"],
            str(active_run.get("dataset_name", "CMEMS GLORYS12V1 Global Ocean Physics Reanalysis")),
        )
        compact_card("FIELD", label, f"Source: {source} · Unit: {units}")
        compact_card("SELECTED AREA", f"{latitude.min():.2f}–{latitude.max():.2f}°N", f"{longitude.min():.2f}–{longitude.max():.2f}°E · {len(latitude)} × {len(longitude)} cells")
        compact_card("DATA COVERAGE", f"{valid_cells:,} valid cells", f"{100 * valid_cells / max(total_cells, 1):.1f}% of selected grid cells contain a value")

    with st.expander("Inspect a location and vertical temperature profile"):
        input_a, input_b = st.columns(2)
        inspect_lat = input_a.number_input("Latitude (°N)", min_value=float(latitude.min()), max_value=float(latitude.max()), value=float(np.mean(latitude)), step=float(np.median(np.diff(latitude))))
        inspect_lon = input_b.number_input("Longitude (°E)", min_value=float(longitude.min()), max_value=float(longitude.max()), value=float(np.mean(longitude)), step=float(np.median(np.diff(longitude))))
        row, column = int(np.argmin(np.abs(latitude - inspect_lat))), int(np.argmin(np.abs(longitude - inspect_lon)))
        first, second = st.columns(2)
        with first:
            st.caption(f"Nearest grid cell: {latitude[row]:.2f}°N, {longitude[column]:.2f}°E · {selected_time}")
            st.write(f"Selected field value: **{field[row, column]:.3f} {units}**")
        with second:
            profile = cube["oceanembed_temperature"].sel(time=np.datetime64(selected_time)).isel(latitude=row, longitude=column).values
            figure = profile_figure(depths, profile); st.pyplot(figure, width="stretch"); plt.close(figure)

    st.markdown("#### Generate research dataset")
    st.caption("Download the field currently shown on the map as a research-ready CSV. Every row includes coordinates, units, data source, validity state, and experiment label.")
    filename_depth = "surface" if export_depth is None else f"{export_depth:g}m"
    st.download_button("Download displayed research dataset (.csv)", research_csv(field=field, latitude=latitude, longitude=longitude, variable=label, units=units, source=source, date=selected_time, depth_metres=export_depth, run_label=str(metrics["label"])), file_name=f"oceanembed_{label.lower().replace(' ', '_')}_{selected_time}_{filename_depth}.csv", mime="text/csv")
    st.info("Research-use boundary: these reconstructed temperature products may support studies of ocean variability and correlations with other phenomena. They do not establish causal links to disasters and do not predict disasters.")

if navigation == "Validation":
    page_header("Independent Validation", "Compare reconstructed temperature with quality-controlled ARGO float observations.")
    st.caption("ARGO is a global network of autonomous ocean floats. It was not used for training, normalization, or model selection in this prototype.")
    a, b, c, d = st.columns(4); a.metric("Root Mean Squared Error", f"{float(metrics['rmse_c']):.2f} °C"); b.metric("Mean Absolute Error", f"{float(metrics['mae_c']):.2f} °C"); c.metric("Average bias", f"{float(metrics['bias_c']):+.2f} °C"); d.metric("Matched observations", metrics["valid_temperature_depth_pairs"])
    profiles = sorted({(item["relative_path"], item["profile_index"]) for item in collocations})
    if profiles:
        st.markdown("#### Compare one observed ARGO profile")
        chosen = st.selectbox("Choose an ARGO float profile", profiles, format_func=lambda item: f"Observed profile {item[1]} — source file {Path(item[0]).name}")
        values = sorted([item for item in collocations if (item["relative_path"], item["profile_index"]) == chosen], key=lambda item: float(item["depth_metres_prototype"]))
        figure, axis = plt.subplots(figsize=(6, 5)); axis.plot([float(item["argo_temperature_c"]) for item in values], [float(item["depth_metres_prototype"]) for item in values], "o-", label="ARGO float observation"); axis.plot([float(item["oceanembed_temperature_c"]) for item in values], [float(item["depth_metres_prototype"]) for item in values], "o-", label="OceanEmbed reconstruction"); axis.invert_yaxis(); axis.grid(alpha=.3); axis.legend(); axis.set(xlabel="Temperature (°C)", ylabel="Pressure used as approximate depth (m; prototype)"); st.pyplot(figure); plt.close(figure)
        st.markdown("#### Download validation data")
        stream = StringIO(); writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        download_one, download_two = st.columns(2); download_one.download_button("Download displayed ARGO comparison (.csv)", stream.getvalue().encode("utf-8"), file_name="argo_profile_comparison.csv", mime="text/csv"); download_two.download_button("Download validation metrics (.json)", json.dumps(metrics, indent=2).encode("utf-8"), file_name="argo_validation_metrics.json", mime="application/json")

if navigation == "About":
    page_header("About the research portal", "A lower-cost, large-area complement to sparse in-situ subsurface measurements.")
    cards = [("Research purpose", "Subsurface temperature", "OceanEmbed provides gridded reconstructed subsurface-temperature products that researchers can inspect and export for oceanographic analysis."), ("Why this matters", "Currents", "In-situ instruments provide valuable direct profiles but have limited spatial and temporal coverage. A gridded reconstruction can support broader-scale research alongside observations."), ("Data basis", "SST", "The current proof of concept uses GLORYS and ERA5 reanalysis inputs. GLORYS remains a training reference, not perfect ground truth."), ("Responsible interpretation", "Thermocline", "Researchers may investigate correlations with marine hazards or other events, but OceanEmbed does not infer causation or issue disaster forecasts or warnings.")]
    for start in range(0, len(cards), 2):
        columns = st.columns(2, gap="large")
        for column, (title, icon_concept, body) in zip(columns, cards[start:start + 2]):
            with column: st.markdown(f"<div class='insight-card'><div class='insight-icon'>{insight_icon(icon_concept)}</div><div class='insight-title'>{title}</div><div class='insight-body'>{body}</div></div>", unsafe_allow_html=True)
