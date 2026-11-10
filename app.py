# Serology results dashboard — filters update the charts immediately (no refresh button)
import os

import pandas as pd
import streamlit as st

import analysis as an


st.set_page_config(
    page_title="Serology results dashboard",
    page_icon="🧫",
    layout="wide",
)


@st.cache_data(show_spinner="Reading and preparing data...")
def load_from_path(path):
    raw, n_drop = an.read_raw_from_path(path)
    result = an.prepare_dataset(raw, n_drop=n_drop)
    result["source"] = f"Local file: {path}"
    return result


@st.cache_data(show_spinner="Reading the uploaded file...")
def load_from_upload(filename, data):
    raw, n_drop = an.read_raw_from_bytes(filename, data)
    result = an.prepare_dataset(raw, n_drop=n_drop)
    result["source"] = f"Uploaded file: {filename}"
    return result


@st.cache_data(show_spinner="Reading the Google Sheet via a public link...")
def load_from_google():
    raw, n_drop, url = an.read_raw_from_google()
    result = an.prepare_dataset(raw, n_drop=n_drop)
    result["source"] = "Google Sheet (public CSV export)"
    result["source_url"] = url
    return result


def _or_all(selected):
    """Empty selection means all values (pass None to the filter)."""
    return list(selected) if selected else None


@st.cache_data(show_spinner="Updating charts and tables...")
def compute_views(
    prepared,
    trips,
    projects,
    sample_types,
    provinces,
    result_cols,
    project_field,
    hide_untested,
    group_counts_by,
    map_col,
    include_titer_neg,
):
    trips = _or_all(trips)
    projects = _or_all(projects)
    sample_types = _or_all(sample_types)
    provinces = _or_all(provinces)
    if result_cols:
        lookup = dict(an.RESULT_COLUMNS)
        results = [(c, lookup[c]) for c in result_cols if c in lookup]
    else:
        results = list(an.RESULT_COLUMNS)
    filtered = an.filter_frame(
        prepared,
        trips=trips,
        projects=projects,
        sample_types=sample_types,
        provinces=provinces,
        project_field=project_field,
    )
    title = "main projects" if project_field == "main_project" else "projects"
    caption = an.filter_caption(trips, projects, sample_types, provinces, project_title=title)
    extra = [c for c in [
        "id", "trip", "project", "main_project", "sample_type",
        "collection_date", "province", "district", "tumbon", "lat", "lon",
    ] if c in filtered.columns]
    views = an.build_dashboard_views(
        filtered, results, an.RESULT_LOOKUP, an.SEROLOGY_COLUMNS, caption, len(prepared),
        hide_untested=hide_untested,
        group_counts_by=group_counts_by,
        map_col=map_col,
        map_label=an.RESULT_LOOKUP.get(map_col, map_col),
        include_titer_neg=include_titer_neg,
        preview_rows=an.TABLE_PREVIEW_ROWS,
        extra_cols=extra,
    )
    views["n_filtered"] = int(len(filtered))
    views["caption"] = caption
    views["n_results"] = len(results)
    return views


def _display_table(table, columns):
    if table is None or table.empty:
        return table
    keep = [c for c in columns if c in table.columns]
    return table[keep]


st.title("Serology results dashboard")
st.caption(
    "Browse infection results by sample type and filter by trip, project, and province "
    "from the left sidebar. Percent positive uses the number tested as the denominator "
    "and is shown as xx% (pos/tested), with a Wilson 95% confidence interval."
)

# ----- Data source -----
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload an Excel or CSV file (use this if you do not want to make the sheet public)",
    type=["xlsx", "xls", "csv"],
    help="The Data for analysis tab is read automatically from Excel files.",
)

local_file = (os.environ.get("LOCAL_FILE") or "").strip()
data = None
load_error = None

if uploaded is not None:
    data = load_from_upload(uploaded.name, uploaded.getvalue())
elif local_file:
    data = load_from_path(local_file)
else:
    try:
        data = load_from_google()
    except Exception as exc:
        load_error = exc

if data is None:
    st.warning(
        "The Google Sheet could not be read. It may not be shared as "
        "Anyone with the link can view. Please upload an .xlsx or .csv file in the sidebar."
    )
    if load_error is not None:
        st.caption(f"Details: {load_error}")
    st.info(
        "If you have already shared the sheet, check that the permission is "
        "Anyone with the link → Viewer, then reload this page."
    )
    st.stop()

prepared = data["prepared"]
st.sidebar.caption(data["source"])
if data.get("n_drop"):
    st.sidebar.caption(f"Dropped {data['n_drop']:,} rows whose key columns were empty")

# ----- Filters -----
st.sidebar.header("Filters")
st.sidebar.caption("Leave a box empty to include all values for that filter.")

project_mode = st.sidebar.radio(
    "Project filter",
    options=["project", "main_project"],
    format_func=lambda v: "Project" if v == "project" else "Main project",
    index=0,
)
project_field = project_mode
project_label = "Main project" if project_field == "main_project" else "Project"

trip_sel = st.sidebar.multiselect("Trip", options=an.trip_options(prepared))
proj_sel = st.sidebar.multiselect(project_label, options=an.project_options(prepared, project_field))
type_sel = st.sidebar.multiselect("Type of samples", options=an.sample_type_options(prepared))
prov_sel = st.sidebar.multiselect("Province", options=an.province_options(prepared))
path_sel = st.sidebar.multiselect(
    "Tests",
    options=[col for col, _label in an.RESULT_COLUMNS],
    format_func=lambda c: an.RESULT_LOOKUP.get(c, c),
)

map_options = path_sel if path_sel else [col for col, _label in an.RESULT_COLUMNS]
default_map = an.SEROLOGY_COLUMNS[0] if an.SEROLOGY_COLUMNS[0] in map_options else map_options[0]
map_col = st.sidebar.selectbox(
    "Pathogen for the map",
    options=map_options,
    index=map_options.index(default_map) if default_map in map_options else 0,
    format_func=lambda c: an.RESULT_LOOKUP.get(c, c),
)
group_by = st.sidebar.radio(
    "Stack counts by",
    options=["sample_type", "trip"],
    format_func=lambda v: "Sample type" if v == "sample_type" else "Trip",
    index=0,
)
hide_untested = st.sidebar.checkbox("Hide untested sample types in tables and prevalence", value=True)
include_titer_neg = st.sidebar.checkbox("Include negative (N) results in the titer chart", value=True)

views = compute_views(
    prepared,
    tuple(trip_sel),
    tuple(proj_sel),
    tuple(type_sel),
    tuple(prov_sel),
    tuple(path_sel),
    project_field,
    hide_untested,
    group_by,
    map_col,
    include_titer_neg,
)

kpis = views["kpis"]
st.info(
    f"**{views['n_filtered']:,} rows match the current filters**, "
    f"out of {data['n_rows']:,} rows in total.  \n"
    f"{views['caption']}"
)

# ----- Overview -----
st.header("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{kpis['n_rows']:,}")
c2.metric("Trips", f"{kpis['n_trips']:,}")
c3.metric("Projects", f"{kpis['n_projects']:,}")
c4.metric("Provinces", f"{kpis['n_provinces']:,}")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Tests performed", f"{kpis['n_tested']:,}")
c6.metric("Positive results", f"{kpis['n_pos']:,}")
pct_txt = kpis.get("pct_label") or "—"
ci_txt = "" if kpis["ci_low"] is None else f"{kpis['ci_low']:.1f}–{kpis['ci_high']:.1f}"
c7.metric("Overall percent positive", pct_txt, delta=f"95% CI {ci_txt}" if ci_txt else None, delta_color="off")
c8.metric("Points with usable coordinates", f"{kpis['n_coords']:,}")
st.caption(
    "Overall percent positive pools every selected test, so it is an overview figure, "
    "not the prevalence of a single pathogen. See the table below for prevalence by test. "
    "Confidence intervals are Wilson 95% intervals. Percentages are shown as xx% (pos/tested)."
)

st.subheader("Prevalence summary")
summary = views["summary"]
summary_view = _display_table(summary, an.SUMMARY_DISPLAY_COLUMNS)
st.dataframe(summary_view, width='stretch', hide_index=True)
if summary is not None and not summary.empty:
    st.download_button(
        "Download summary table as CSV",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="serology_summary.csv",
        mime="text/csv",
    )

# ----- Charts -----
st.header("Prevalence")
st.plotly_chart(views["figs"]["prevalence"], width='stretch')
st.caption(
    "Bars show percent positive as xx% (pos/tested). Whiskers are Wilson 95% confidence intervals."
)

st.header("Positive / negative / not tested counts")
st.plotly_chart(views["figs"]["counts"], width='stretch')
if views["n_results"] > 1:
    st.caption(
        "When more than one test is selected, counts in this chart are pooled across those tests, "
        "so one sample may be counted more than once. Hover a Positive bar to see xx% (pos/tested)."
    )

st.header("Percent positive by trip")
st.plotly_chart(views["figs"]["trip"], width='stretch')
st.caption(
    "Trips are ordered by the first collection date in each trip. "
    "Trips without a date are listed at the end. Hover a point to see xx% (pos/tested)."
)
trip_table = views.get("trip_table")
trip_view = _display_table(trip_table, an.TRIP_DISPLAY_COLUMNS)
if trip_view is not None:
    st.subheader("Prevalence by trip")
    st.dataframe(trip_view, width='stretch', hide_index=True)
    if trip_table is not None and not trip_table.empty:
        st.download_button(
            "Download trip table as CSV",
            data=trip_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="serology_by_trip.csv",
            mime="text/csv",
            key="download_trip",
        )

st.header("Sample collection map")
st.plotly_chart(views["figs"]["map"], width='stretch')
st.caption(
    f"Red = positive, green = negative, for «{an.RESULT_LOOKUP.get(map_col, map_col)}». "
    "Serology marker size follows the titer. Invalid or missing coordinates are omitted. "
    "Hover a point to see trip, project, sample type, province, district, and subdistrict."
)

st.header("Titer distribution")
st.plotly_chart(views["figs"]["titer"], width='stretch')
st.caption("This chart uses only the selected serology tests (STG / MTG / SFG).")

# ----- Data table -----
st.header("Filtered data")
preview = views["preview"]
full = views["full"]
n_show = 0 if preview is None else len(preview)
n_full = 0 if full is None else len(full)
st.caption(f"Showing the first {n_show:,} of {n_full:,} filtered rows")
if preview is not None:
    st.dataframe(preview, width='stretch', hide_index=True)
if full is not None and not full.empty:
    st.download_button(
        "Download filtered data as CSV",
        data=full.to_csv(index=False).encode("utf-8-sig"),
        file_name="serology_filtered_data.csv",
        mime="text/csv",
        key="download_filtered",
    )

st.divider()
st.caption(
    "Positive definition: P/Positive or a titer of 1:50 or higher is positive; "
    "N/Negative is negative; blank / Waiting / no sample is not tested; "
    "Suspected is excluded from the percent-positive denominator."
)
