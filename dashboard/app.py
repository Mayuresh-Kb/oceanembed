"""Compact multi-page OceanEmbed reconstruction dashboard."""
from __future__ import annotations

from datetime import date
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
from dashboard.decision_support import build_decision_summary, layer_csv, nearest_cell, reference_mean, regions_csv
from dashboard.insights import OCEAN_INSIGHTS
from preprocessing.config import load_config

st.set_page_config(page_title="OceanEmbed", page_icon="🌊", layout="wide")
st.markdown("""<style>
.stApp{background:#071827;color:#e9f4ff;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.block-container{padding-top:1.25rem;max-width:1500px}
.eyebrow{color:#75d8e8;font-size:.78rem;font-weight:650;letter-spacing:.1em}.badge{background:#123951;border:1px solid #1e425b;border-radius:15px;padding:4px 10px;color:#9de8f1;font-size:.76rem;font-weight:500}.muted{color:#aac2d3}
.main-brand{display:flex;align-items:center;gap:13px;margin:4px 0 22px}.main-brand svg{width:43px;height:43px;stroke:#19d2d4;stroke-width:2.4;fill:none;stroke-linecap:round}.main-brand-title{font-size:2rem;font-weight:720;line-height:1;color:#edf8ff}.main-brand-subtitle{font-size:.88rem;color:#aac2d3;margin-top:5px;font-weight:400}.page-title{font-size:1.62rem;font-weight:680;color:#edf8ff;margin:0 0 4px}.page-subtitle{font-size:.92rem;color:#aac2d3;margin:0 0 20px}
.context{background:#0d2538;border:1px solid #1e425b;border-radius:12px;padding:12px}.card{background:#0d2538;border:1px solid #1e425b;border-radius:12px;padding:14px;min-height:112px;overflow-wrap:anywhere}.card-title{font-size:.76rem;color:#aac2d3;font-weight:600;letter-spacing:.07em}.card-value{font-size:1.4rem;font-weight:680;margin:8px 0;color:#edf8ff}.card-detail{font-size:.86rem;color:#75d8e8;line-height:1.35}.priority{background:#0d2538;border-left:4px solid #f0aa3c;border-radius:8px;padding:10px;margin-bottom:8px}.high{border-left-color:#ed5d62}.insight-card{min-height:184px;box-sizing:border-box;background:linear-gradient(145deg,#10314a,#0b2236);border:1px solid #22516d;border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 6px 18px rgba(0,0,0,.12);overflow:hidden}.insight-icon{height:28px}.insight-icon svg{width:27px;height:27px;stroke:#22d3d5;stroke-width:1.8;fill:none;stroke-linecap:round;stroke-linejoin:round}.insight-title{font-size:1.05rem;font-weight:680;color:#85ebf2;margin:9px 0 7px}.insight-body{font-size:.9rem;line-height:1.45;color:#d8e9f5}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061e34 0%,#071827 100%);border-right:1px solid #16425c}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .brand{padding:20px 8px 12px}.wave{font-size:2.2rem;line-height:1;color:#19d2d4;letter-spacing:-.35em}.brand-title{font-weight:720;color:#edf8ff;font-size:1.32rem;line-height:1.12;margin-top:12px}.brand-subtitle{color:#80a9c4;font-size:.78rem;margin-top:6px}[data-testid="stSidebar"] [role="radiogroup"]{gap:.35rem}[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent;border-radius:7px;padding:8px 10px;color:#bdd2e2}[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#0a476a;color:#dffaff;border-left:3px solid #19d2d4}[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading analysis artifacts…")
def artifacts(run_name: str):
    return load_prediction_cube(run_name), load_metrics(run_name), load_collocations(run_name)


@st.cache_data(show_spinner="Loading source fields…")
def surface_artifacts(run_name: str):
    return load_surface_fields(run_name)


def compact_card(title: str, value: str, detail: str) -> None:
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div><div class='card-value'>{value}</div><div class='card-detail'>{detail}</div></div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def insight_icon(concept: str) -> str:
    """Return a small inline SVG icon; no external icon asset is required."""
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


def context_figure(run: dict) -> go.Figure:
    """Self-contained named ocean context map using exact configured bounds."""
    lat_min, lat_max, lon_min, lon_max = run["region_bounds"]
    view_lat_min, view_lat_max, view_lon_min, view_lon_max = run["context_bounds"]
    title = str(run["region_title"])
    figure = go.Figure()
    figure.add_shape(type="rect", x0=lon_min, x1=lon_max, y0=lat_min, y1=lat_max, line={"color":"#75d8e8","width":3}, fillcolor="rgba(117,216,232,.14)")
    figure.add_annotation(x=(lon_min+lon_max)/2, y=(lat_min+lat_max)/2, text=f"<b>{title}</b><br>{lat_min:g}–{lat_max:g}°N<br>{lon_min:g}–{lon_max:g}°E", showarrow=False, font={"color":"#e9f4ff","size":13})
    figure.add_annotation(x=view_lon_min+2, y=view_lat_min+2, text="Coordinate context map", showarrow=False, font={"color":"#aac2d3","size":10}, xanchor="left", yanchor="bottom")
    figure.update_layout(template="plotly_dark", paper_bgcolor="#0d2538", plot_bgcolor="#0d2538", margin={"l":35,"r":15,"t":28,"b":30}, height=225, title={"text":f"Study area: {title}","font":{"size":15}}, xaxis={"title":"°E","range":[view_lon_min,view_lon_max],"dtick":10,"showgrid":True,"gridcolor":"#17384d"}, yaxis={"title":"°N","range":[view_lat_min,view_lat_max],"dtick":5,"showgrid":True,"gridcolor":"#17384d","scaleanchor":"x","scaleratio":1})
    return figure


def analysis_figure(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, *, label: str, units: str, cmap: str, symmetric: bool, regions: list[dict], vectors: tuple[np.ndarray,np.ndarray] | None) -> go.Figure:
    finite = field[np.isfinite(field)]
    if symmetric:
        edge = max(float(np.nanpercentile(np.abs(finite),98)),.1) if finite.size else 1.; zmin,zmax,zmid=-edge,edge,0.
    else:
        zmin=float(np.nanpercentile(finite,2)) if finite.size else 0.; zmax=float(np.nanpercentile(finite,98)) if finite.size else 1.; zmax=zmax if zmax>zmin else zmin+1.; zmid=None
    scale={"temperature":"Turbo","anomaly":[[0,"#2166ac"],[.5,"#f7f7f7"],[1,"#b2182b"]],"speed":"Viridis","ssh":"RdBu"}[cmap]
    figure=go.Figure(go.Heatmap(z=field,x=longitude,y=latitude,colorscale=scale,zmin=zmin,zmax=zmax,zmid=zmid,colorbar={"title":units,"thickness":16},hovertemplate="%{y:.2f}°N, %{x:.2f}°E<br>"+label+": %{z:.2f} "+units+"<extra></extra>"))
    for region in regions:
        colour="#ed5d62" if region["severity"]=="HIGH ANOMALY" else "#f0aa3c"
        figure.add_shape(type="rect",x0=region["lon_min"],x1=region["lon_max"],y0=region["lat_min"],y1=region["lat_max"],line={"color":colour,"width":2})
    if vectors is not None:
        u,v=vectors; step=max(1,int(max(len(latitude),len(longitude))/12)); length=.10*min(float(np.ptp(latitude)),float(np.ptp(longitude))); xs=[];ys=[]
        for r in range(0,len(latitude),step):
            for c in range(0,len(longitude),step):
                east,north=float(u[r,c]),float(v[r,c]); speed=float(np.hypot(east,north))
                if np.isfinite(speed) and speed>0: xs.extend([longitude[c],longitude[c]+length*east/speed,None]);ys.extend([latitude[r],latitude[r]+length*north/speed,None])
        figure.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line={"color":"#e5fbff","width":1},name="Direction",hoverinfo="skip"))
    figure.update_layout(template="plotly_dark",paper_bgcolor="#071827",plot_bgcolor="#071827",height=440,margin={"l":55,"r":25,"t":36,"b":45},title={"text":label+" — zoom and hover to inspect","font":{"size":16}},xaxis={"title":"Longitude (°E)","showgrid":False},yaxis={"title":"Latitude (°N)","showgrid":False,"scaleanchor":"x","scaleratio":1})
    return figure


def layer_data(name: str, prediction: np.ndarray, summary: dict, surface: dict[str,np.ndarray], date_index: int):
    if name=="Temperature": return prediction,"Subsurface temperature","°C","temperature",False,None
    if name=="Anomaly": return summary["anomaly"],"Temperature anomaly","°C from short reference","anomaly",True,None
    if name=="Δ Previous": return summary["change"] if summary["change"] is not None else np.full_like(prediction,np.nan),"Change from previous analysis","°C","anomaly",True,None
    if name=="SST": return surface["sst"][date_index],"Surface temperature input","°C","temperature",False,None
    if name=="SSH": return surface["ssh"][date_index],"Sea surface height","m","ssh",True,None
    u,v=("current_u","current_v") if name=="Currents" else ("wind_u","wind_v")
    return np.hypot(surface[u][date_index],surface[v][date_index]),f"{name} speed","m/s","speed",False,(surface[u][date_index],surface[v][date_index])


def profile_figure(depths: list[float], values: np.ndarray) -> plt.Figure:
    figure,axis=plt.subplots(figsize=(5,4));figure.patch.set_facecolor("#0d2538");axis.set_facecolor("#0d2538");axis.plot(values,depths,"o-",color="#75d8e8");axis.invert_yaxis();axis.grid(alpha=.2,color="white");axis.set(xlabel="Temperature (°C)",ylabel="Depth (m)",title="Reconstructed temperature profile");axis.tick_params(colors="#dcecf5");axis.xaxis.label.set_color("#dcecf5");axis.yaxis.label.set_color("#dcecf5");axis.title.set_color("#dcecf5");return figure


st.markdown("""<div class='main-brand'><svg viewBox='0 0 48 48' aria-label='OceanEmbed wave logo'><path d='M5 17c7-8 13 7 20 0s13 8 20 0'/><path d='M5 25c7-8 13 7 20 0s13 8 20 0'/><path d='M5 33c7-8 13 7 20 0s13 8 20 0'/></svg><div><div class='main-brand-title'>OCEANEMBED</div><div class='main-brand-subtitle'>Subsurface Ocean Intelligence</div></div></div>""", unsafe_allow_html=True)
st.sidebar.markdown("""<div class='brand'><div class='wave'>≋≋</div><div class='brand-title'>OCEANEMBED</div><div class='brand-subtitle'>Subsurface Ocean Intelligence</div></div>""", unsafe_allow_html=True)
navigation = st.sidebar.radio("Navigation", ["Ocean Analysis", "Validation", "Ocean Insights", "About"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("#### Active analysis dataset")
selected_run = st.sidebar.selectbox("Choose experiment run", ["Bay of Bengal: 150-day experiment", "Full NIO: 30-day demo"], help="This choice controls the maps, dates, validation evidence, and downloadable results shown throughout the dashboard.")
active_run = RUNS[selected_run]
lat_min, lat_max, lon_min, lon_max = active_run["region_bounds"]
st.sidebar.markdown(f"**{active_run['region_title']}**  ")
st.sidebar.caption(f"{lat_min:g}–{lat_max:g}°N · {lon_min:g}–{lon_max:g}°E")
if "150-day" in selected_run:
    st.sidebar.caption("150-day experiment: Bay of Bengal regional patch.")
else:
    st.sidebar.caption("30-day full North Indian Ocean demonstration.")
cube, metrics, collocations = artifacts(selected_run)
bundle, surface = surface_artifacts(selected_run)
config = load_config(RUNS[selected_run]["config"])

if navigation == "Ocean Analysis":
    page_header("Ocean Analysis", "Explore reconstructed subsurface temperature across depth and time.")
    date_col, layer_col = st.columns([1.1, 3.2])
    times = [str(value)[:10] for value in cube.time.values]
    available_dates = {date.fromisoformat(value) for value in times}
    selected_time = date_col.selectbox("Saved analysis date", times, index=len(times) - 1)
    layer_name = layer_col.radio("Map layer", ["Temperature", "Anomaly", "Δ Previous", "SST", "SSH", "Currents", "Winds"], horizontal=True)
    if selected_time in times:
        if "reconstruction_depth" not in st.session_state or st.session_state["reconstruction_depth"] not in list(cube.depth.values.astype(float)):
            st.session_state["reconstruction_depth"] = 10.0 if 10.0 in list(cube.depth.values.astype(float)) else float(cube.depth.values[0])
        depths = [float(value) for value in cube.depth.values]
        selected_depth = float(st.session_state["reconstruction_depth"])
        time_index = times.index(selected_time)
        depth_index = depths.index(selected_depth)
        date_index = int(np.where(bundle.times.astype("datetime64[D]") == np.datetime64(selected_time))[0][0])
        latitude, longitude = bundle.latitude, bundle.longitude
        prediction = cube["oceanembed_temperature"].sel(time=np.datetime64(selected_time), depth=selected_depth).values
        previous = None if time_index == 0 else cube["oceanembed_temperature"].isel(time=time_index - 1).sel(depth=selected_depth).values
        reference = reference_mean(bundle.temperature, depth_index, list(config["alerts"]["reference_day_indices"]))
        summary = build_decision_summary(prediction, reference, previous, latitude, longitude, config["alerts"])
        field, label, units, cmap, symmetric, vectors = layer_data(layer_name, prediction, summary, surface, date_index)
        alert = summary["alert"]

        map_column, summary_column = st.columns([2.15, 1], gap="large")
        with map_column:
            st.markdown(f"#### Interactive subsurface-temperature map — {active_run['region_title']}")
            st.caption(f"{label} at {selected_depth:g} m on {selected_time}. Hover over the map for the value at each grid cell; use the legend at right to interpret colours.")
            st.plotly_chart(analysis_figure(field, latitude, longitude, label=label, units=units, cmap=cmap, symmetric=symmetric, regions=summary["regions"], vectors=vectors), width="stretch", config={"scrollZoom": True, "displaylogo": False})
            st.select_slider("Reconstruction depth — discrete model output levels", options=depths, key="reconstruction_depth", format_func=lambda value: f"{value:g} m", help="OceanEmbed predicts these defined levels; it is not continuous depth output.")

        with summary_column:
            st.markdown("#### Situation summary")
            compact_card(
                "WHAT IS HAPPENING?",
                str(alert["status"]).replace(" ANOMALY", " anomaly").title(),
                f"The regional average is {float(alert.get('mean_anomaly_c', 0)):+.2f} °C compared with the configured short GLORYS reference mean.",
            )
            largest = summary["regions"][0] if summary["regions"] else None
            compact_card(
                "WHERE SHOULD I LOOK?",
                f"{len(summary['regions'])} area(s) for review",
                f"The largest connected area is about {float(largest.get('area_km2_approx', 0)):,.0f} km²." if largest else "No connected ocean area crosses the configured anomaly threshold.",
            )
            change = "No earlier saved date" if summary["change_mean_c"] is None else f"{float(summary['change_mean_c']):+.2f} °C since previous date"
            detail = "This is the first saved analysis date." if summary["change_mean_c"] is None else f"{100 * float(summary['changed_fraction']):.1f}% of valid ocean cells changed by at least {float(summary['change_threshold_c']):g} °C."
            compact_card("WHAT CHANGED?", change, detail)
            st.caption("These are model-generated temperature-anomaly review cues, not official hazard warnings.")

        st.markdown("#### Review priorities")
        if summary["regions"]:
            priority_columns = st.columns(min(3, len(summary["regions"])))
            for number, region in enumerate(summary["regions"][:3], start=1):
                style = "high" if region["severity"] == "HIGH ANOMALY" else ""
                with priority_columns[number - 1]:
                    st.markdown(f"<div class='priority {style}'><b>{region['severity']}</b><br>Region {number:02d}<br><span class='muted'>{region['lat_min']:.2f}–{region['lat_max']:.2f}°N · {region['lon_min']:.2f}–{region['lon_max']:.2f}°E<br>Mean anomaly {region['mean_anomaly_c']:+.2f} °C</span></div>", unsafe_allow_html=True)
        else:
            st.success("No connected review region crosses the configured threshold.")
        st.caption("Model-generated temperature-anomaly review cues only; not official hazard events.")

        with st.expander("Inspect a location and vertical temperature profile"):
            left, right = st.columns(2)
            inspect_lat = left.number_input("Latitude (°N)", min_value=float(latitude.min()), max_value=float(latitude.max()), value=float(np.mean(latitude)), step=float(np.median(np.diff(latitude))))
            inspect_lon = right.number_input("Longitude (°E)", min_value=float(longitude.min()), max_value=float(longitude.max()), value=float(np.mean(longitude)), step=float(np.median(np.diff(longitude))))
            row, column = nearest_cell(latitude, longitude, inspect_lat, inspect_lon)
            first, second = st.columns(2)
            with first:
                st.caption(f"{latitude[row]:.2f}°N, {longitude[column]:.2f}°E · {selected_time}")
                st.write(f"Temperature: **{prediction[row, column]:.2f} °C**")
                st.write(f"Anomaly: **{summary['anomaly'][row, column]:+.2f} °C**")
            with second:
                figure = profile_figure(depths, cube["oceanembed_temperature"].sel(time=np.datetime64(selected_time)).isel(latitude=row, longitude=column).values)
                st.pyplot(figure, width="stretch")
                plt.close(figure)

        st.markdown("#### Download analysis data")
        st.caption("Download the currently displayed map values or the detected temperature-anomaly review regions as CSV files.")
        export_one, export_two = st.columns(2)
        export_one.download_button("Download displayed map layer (.csv)", layer_csv(layer=field, latitude=latitude, longitude=longitude, layer_name=label, units=units, date=selected_time, depth=selected_depth, run_label=str(metrics["label"])), file_name=f"oceanembed_{selected_time}_{selected_depth:g}m_layer.csv", mime="text/csv")
        export_two.download_button("Download review regions (.csv)", regions_csv(summary["regions"], date=selected_time, depth=selected_depth, run_label=str(metrics["label"])), file_name=f"oceanembed_{selected_time}_{selected_depth:g}m_regions.csv", mime="text/csv")

if navigation == "Validation":
    page_header("Validation", "Compare OceanEmbed reconstructions with independent ARGO float observations.")
    st.caption("ARGO is a global network of autonomous ocean floats that measure temperature below the surface. These observations were kept separate from training.")
    a, b, c, d = st.columns(4)
    a.metric("Root Mean Squared Error (RMSE)", f"{float(metrics['rmse_c']):.2f} °C", help="Typical size of prediction errors; lower is better.")
    b.metric("Mean Absolute Error (MAE)", f"{float(metrics['mae_c']):.2f} °C", help="Average absolute difference between OceanEmbed and ARGO temperatures; lower is better.")
    c.metric("Average bias", f"{float(metrics['bias_c']):+.2f} °C", help="Positive means the reconstruction is warmer on average; negative means cooler.")
    d.metric("Matched observations", metrics["valid_temperature_depth_pairs"], help="Quality-controlled ARGO temperature-depth measurements matched to model output.")
    profiles = sorted({(item["relative_path"], item["profile_index"]) for item in collocations})
    if profiles:
        st.markdown("##### Compare one observed ARGO profile with OceanEmbed")
        st.caption("Choose one quality-controlled float profile. The chart compares its observed temperatures at several depths with OceanEmbed’s reconstruction at the same nearby location and date.")
        chosen = st.selectbox("Choose an ARGO float profile", profiles, format_func=lambda item: f"Observed profile {item[1]} — source file {Path(item[0]).name}")
        values = sorted([item for item in collocations if (item["relative_path"], item["profile_index"]) == chosen], key=lambda item: float(item["depth_metres_prototype"]))
        figure, axis = plt.subplots(figsize=(6,5));axis.plot([float(item["argo_temperature_c"]) for item in values],[float(item["depth_metres_prototype"]) for item in values],"o-",label="ARGO float observation");axis.plot([float(item["oceanembed_temperature_c"]) for item in values],[float(item["depth_metres_prototype"]) for item in values],"o-",label="OceanEmbed reconstruction");axis.invert_yaxis();axis.grid(alpha=.3);axis.legend();axis.set(xlabel="Temperature (°C)",ylabel="Pressure used as approximate depth (m; prototype)");st.pyplot(figure);plt.close(figure)
        st.markdown("##### Download validation data")
        st.caption("Download the quality-controlled matched observations used in the currently displayed comparison, or the complete validation-metric summary for this run.")
        stream = StringIO();writer = csv.DictWriter(stream, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        download_one, download_two = st.columns(2)
        download_one.download_button("Download displayed ARGO comparison (.csv)", stream.getvalue().encode("utf-8"), file_name="argo_profile_comparison.csv", mime="text/csv")
        download_two.download_button("Download validation metrics (.json)", json.dumps(metrics, indent=2).encode("utf-8"), file_name="argo_validation_metrics.json", mime="application/json")

if navigation == "Ocean Insights":
    page_header("Ocean Insights", "A quick visual guide to the ocean ideas used by OceanEmbed.")
    insight_items = list(OCEAN_INSIGHTS.items())
    for start in range(0, len(insight_items), 3):
        columns = st.columns(3, gap="large")
        for column, (concept, explanation) in zip(columns, insight_items[start:start + 3]):
            with column:
                st.markdown(f"<div class='insight-card'><div class='insight-icon'>{insight_icon(concept)}</div><div class='insight-title'>{concept}</div><div class='insight-body'>{explanation}</div></div>", unsafe_allow_html=True)

if navigation == "About":
    page_header("About OceanEmbed", "A student-built proof of concept for exploring subsurface ocean conditions.")
    about_cards = [
        ("Subsurface temperature", "Subsurface temperature", "OceanEmbed reconstructs the temperature below the sea surface at defined depths, from 0 m to 1,000 m."),
        ("What informs it", "Currents", "The proof of concept combines sea-surface temperature, salinity, height, currents, and winds from GLORYS and ERA5 reanalysis fields."),
        ("Scientific evidence", "Thermocline", "GLORYS is the training reference reanalysis. Quality-controlled ARGO float profiles are used separately as independent validation evidence."),
        ("Responsible use", "Marine heatwaves", "This system provides reconstruction and temperature-anomaly review cues only. It is not a cyclone, tsunami, flood, or official disaster-warning system."),
    ]
    for start in range(0, len(about_cards), 2):
        columns = st.columns(2, gap="large")
        for column, (title, icon_concept, body) in zip(columns, about_cards[start:start + 2]):
            with column:
                st.markdown(f"<div class='insight-card'><div class='insight-icon'>{insight_icon(icon_concept)}</div><div class='insight-title'>{title}</div><div class='insight-body'>{body}</div></div>", unsafe_allow_html=True)
