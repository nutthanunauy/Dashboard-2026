import math
import re
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


BLANK_LABEL = "Unspecified"

SAMPLE_TYPE_ORDER = [
    "Rodent", "Dog", "Cat", "Chigger", "Tick",
    "Flea", "Lice", "Other mite", "Soil", "Water",
]

TITER_PATTERN = re.compile(r"^\s*1\s*:\s*([\d,]+)\s*$")

COLOR_POS = "#C44E52"
COLOR_NEG = "#4C9A68"
COLOR_NT = "#B0B0B0"
COLOR_STATUS = {"Positive": COLOR_POS, "Negative": COLOR_NEG, "Not tested / excluded": COLOR_NT}

CHART_FONT = "Sarabun, Noto Sans Thai, Tahoma, DejaVu Sans"


def is_blank(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def normalize_text(value):
    if is_blank(value):
        return ""
    return str(value).strip()


def normalize_label(value):
    if is_blank(value):
        return BLANK_LABEL
    return str(value).strip()


def normalize_trip(value):
    """แปลงเลขทริปให้อยู่ในรูปข้อความเดียวกัน เช่น 1.0 เป็น '1'"""
    if is_blank(value):
        return BLANK_LABEL
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return BLANK_LABEL
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    try:
        number = float(text.replace(",", ""))
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return text


def to_number(value):
    if is_blank(value):
        return math.nan
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return math.nan


def is_valid_lonlat(lat, lon):
    if lat is None or lon is None:
        return False
    if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in (lat, lon)):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def unique_headers(headers):
    """ถ้าชื่อคอลัมน์ซ้ำ ให้เติม .1 .2 แบบที่ pandas ทำ"""
    seen = {}
    out = []
    for name in headers:
        raw = "Unnamed" if name is None else str(name).strip()
        if raw == "":
            raw = "Unnamed"
        n = seen.get(raw, 0)
        out.append(raw if n == 0 else f"{raw}.{n}")
        seen[raw] = n + 1
    return out



def is_blank_row(series):
    return all(is_blank(v) for v in series)


def drop_blank_rows(df, key_cols):
    """ตัดแถวที่คอลัมน์หลักว่างทั้งหมด (รวมช่องที่เป็นข้อความว่างจาก Google Sheet)

    dropna(how="all") อย่างเดียวไม่พอ เพราะชีตมักให้ค่าเป็นสตริงว่าง ไม่ใช่ค่าว่างของ pandas
    ถ้ามีค่าค้างในคอลัมน์ที่ไม่ใช้ แถวนั้นก็ยังควรถูกตัดเมื่อคอลัมน์หลักว่างหมด
    """
    if df is None or df.empty:
        return df, 0
    present = [c for c in key_cols if c in df.columns]
    if not present:
        mask = df.apply(is_blank_row, axis=1)
    else:
        mask = df[present].apply(is_blank_row, axis=1)
    n_drop = int(mask.sum())
    return df.loc[~mask].copy(), n_drop



def parse_titer(value):
    """อ่านค่า titer เป็นตัวเลขส่วนหลัง เช่น 1:400 ได้ 400 ถ้าไม่ใช่ titer คืน None"""
    text = normalize_text(value)
    if not text:
        return None
    match = TITER_PATTERN.match(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_date(value):
    """แปลงวันที่จากชีตให้เป็น Timestamp ถ้าแปลงไม่ได้จะได้ค่าว่าง"""
    if is_blank(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, date):
        return pd.Timestamp(value)
    return pd.to_datetime(value, errors="coerce")


def classify_result(value, titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as):
    """แยกผลเป็น positive / negative / not_tested

    ค่า titer ที่มากกว่าหรือเท่ากับ titer_cutoff ถือว่าบวก
    ค่าที่ยังไม่ชัวร์ เช่น Suspected ใช้ตาม setting treat_suspected_as
    """
    if is_blank(value):
        return "not_tested"

    text = normalize_text(value)
    lowered = text.lower()
    compact = lowered.replace(" ", "")

    not_tested_set = {normalize_text(v).lower() for v in not_tested_text if normalize_text(v)}
    not_tested_set |= {v.replace(" ", "") for v in not_tested_set}
    if lowered in not_tested_set or compact in not_tested_set:
        return "not_tested"

    if lowered in {"suspected", "suspect"}:
        return treat_suspected_as

    pos_set = {normalize_text(v).lower() for v in positive_text}
    neg_set = {normalize_text(v).lower() for v in negative_text}
    if lowered in pos_set:
        return "positive"
    if lowered in neg_set:
        return "negative"

    titer = parse_titer(text)
    if titer is not None:
        return "positive" if titer >= float(titer_cutoff) else "negative"

    return "not_tested"


def add_result_classes(df, result_columns, titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as):
    """สร้างคอลัมน์สถานะและค่า titer ของแต่ละผลตรวจ"""
    warnings = []
    out = df.copy()
    known_extra = {"suspected", "suspect"}
    for col, _label in result_columns:
        if col not in out.columns:
            raise ValueError(f"Result column not found: {col}")
        classes = out[col].map(
            lambda v, col=col: classify_result(
                v, titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as
            )
        )
        odd = []
        pos_set = {normalize_text(v).lower() for v in list(positive_text) + list(negative_text) + list(not_tested_text)}
        for raw in out[col]:
            if is_blank(raw):
                continue
            text = normalize_text(raw)
            lowered = text.lower()
            known = lowered in pos_set or lowered in known_extra or parse_titer(text) is not None
            cls = classify_result(raw, titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as)
            if cls == "not_tested" and not known:
                odd.append(text)
        if odd:
            vc = pd.Series(odd).value_counts()
            warnings.append((col, vc.head(10).to_dict(), int(vc.sum())))
        out[f"__class__{col}"] = classes
        out[f"__titer__{col}"] = out[col].map(parse_titer)
    return out, warnings


def prepare_frame(df, col_sample, col_trip, col_project, result_columns,
                  titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as,
                  col_main_project=None, col_province=None, col_district=None, col_tumbon=None,
                  col_lat=None, col_lon=None, col_date=None, col_id=None):
    """เตรียมตารางหลัก: ชื่อที่ทำให้เป็นรูปแบบเดียวกัน, ผลบวก/ลบ, วันที่ และพิกัด"""
    optional = {
        "main_project": col_main_project,
        "province": col_province,
        "district": col_district,
        "tumbon": col_tumbon,
        "id": col_id,
    }
    needed = [col_sample, col_trip, col_project] + [c for c, _ in result_columns]
    for col in optional.values():
        if col:
            needed.append(col)
    if col_lat:
        needed.append(col_lat)
    if col_lon:
        needed.append(col_lon)
    if col_date:
        needed.append(col_date)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("These columns were not found in the sheet: " + ", ".join(missing))

    # เก็บคอลัมน์ที่ต้องใช้ และสำเนาคอลัมน์ผลดิบไว้ด้วย
    out = df.loc[:, [c for i, c in enumerate(needed) if c not in needed[:i]]].copy()
    out["sample_type"] = out[col_sample].map(normalize_label)
    out["trip"] = out[col_trip].map(normalize_trip)
    out["project"] = out[col_project].map(normalize_label)
    for dest, col in optional.items():
        if col:
            out[dest] = out[col].map(normalize_label)
        else:
            out[dest] = BLANK_LABEL
    out["lat"] = out[col_lat].map(to_number) if col_lat else math.nan
    out["lon"] = out[col_lon].map(to_number) if col_lon else math.nan
    out["coord_ok"] = [is_valid_lonlat(a, b) for a, b in zip(out["lat"], out["lon"])]
    out["collection_date"] = out[col_date].map(parse_date) if col_date else pd.NaT

    # วันที่แทนของแต่ละทริป = วันเก็บตัวอย่างวันแรกในทริปนั้น (ใช้เรียงกราฟตามทริป)
    trip_dates = (
        out.loc[out["collection_date"].notna(), ["trip", "collection_date"]]
        .groupby("trip")["collection_date"].min()
    )
    out["trip_date"] = out["trip"].map(trip_dates)

    out, warnings = add_result_classes(
        out, result_columns, titer_cutoff, positive_text, negative_text, not_tested_text, treat_suspected_as
    )
    return out, warnings


def filter_frame(df, trips=None, projects=None, sample_types=None, provinces=None, project_field="project"):
    """กรองแถวตามตัวเลือก ถ้าตัวใดเป็น None หรือว่าง ถือว่าไม่จำกัดค่านั้น"""
    out = df
    if trips:
        out = out[out["trip"].isin(list(trips))]
    if projects:
        if project_field not in out.columns:
            raise ValueError(f"Project filter column not found: {project_field}")
        out = out[out[project_field].isin(list(projects))]
    if sample_types:
        out = out[out["sample_type"].isin(list(sample_types))]
    if provinces:
        out = out[out["province"].isin(list(provinces))]
    return out


def sample_type_sort_key(name):
    if name in SAMPLE_TYPE_ORDER:
        return (0, SAMPLE_TYPE_ORDER.index(name))
    if name == BLANK_LABEL:
        return (2, 0)
    return (1, str(name).lower())


def trip_sort_key(trip, trip_date=None):
    """เรียงทริปตามวันเก็บตัวอย่างวันแรก ถ้าไม่มีวันที่ให้ไปท้ายแล้วเรียงตามเลขทริป"""
    try:
        num = int(trip)
    except (TypeError, ValueError):
        num = 10 ** 9
    if trip_date is not None and not pd.isna(trip_date):
        return (0, pd.Timestamp(trip_date), num, str(trip))
    return (1, pd.Timestamp.max, num, str(trip))


def trip_options(df):
    dates = df.groupby("trip")["trip_date"].min() if "trip_date" in df.columns else {}
    vals = list(df["trip"].unique())

    def key(v):
        return trip_sort_key(v, dates.get(v) if hasattr(dates, "get") else dates[v] if v in dates else None)

    return sorted(vals, key=key)


def project_options(df, field="project"):
    vals = [v for v in df[field].unique() if v != BLANK_LABEL]
    extra = [BLANK_LABEL] if BLANK_LABEL in set(df[field]) else []
    return extra + sorted(vals, key=lambda s: str(s).lower())


def province_options(df):
    vals = [v for v in df["province"].unique() if v != BLANK_LABEL]
    extra = [BLANK_LABEL] if BLANK_LABEL in set(df["province"]) else []
    return extra + sorted(vals, key=lambda s: str(s).lower())


def sample_type_options(df):
    return sorted(df["sample_type"].unique(), key=sample_type_sort_key)


def wilson_ci(n_pos, n, z=1.959963984540054):
    """ช่วงเชื่อมั่น 95% แบบ Wilson สำหรับสัดส่วน คืนค่าเป็นร้อยละ (ต่ำ, สูง)"""
    n = int(n) if n is not None else 0
    if n <= 0:
        return (None, None)
    n_pos = max(0, min(int(n_pos), n))
    p = n_pos / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (round(lo * 100.0, 1), round(hi * 100.0, 1))


def format_pct_pos_tested(pct, n_pos, n_tested):
    """Format as '13.8% (778/5,620)'. Returns '—' when there are no tests."""
    if pct is None or n_tested is None or int(n_tested) == 0:
        return "—"
    return f"{float(pct):.1f}% ({int(n_pos):,}/{int(n_tested):,})"


SUMMARY_DISPLAY_COLUMNS = [
    "Sample type", "Test", "Rows", "Tested", "Positive", "Negative",
    "Not tested / excluded", "% positive (pos/tested)", "95% CI",
]

TRIP_DISPLAY_COLUMNS = [
    "Trip", "First collection date", "Sample type", "Test",
    "Rows", "Tested", "Positive", "Negative",
    "% positive (pos/tested)", "95% CI",
]


def _summary_row(sample_type, test_label, n_all, n_pos, n_neg, n_nt, extra=None):
    n_tested = n_pos + n_neg
    pct = (n_pos / n_tested * 100.0) if n_tested else None
    if pct is not None:
        pct = round(pct, 1)
    lo, hi = wilson_ci(n_pos, n_tested) if n_tested else (None, None)
    row = {
        "Sample type": sample_type,
        "Test": test_label,
        "Rows": n_all,
        "Tested": n_tested,
        "Positive": n_pos,
        "Negative": n_neg,
        "Not tested / excluded": n_nt,
        "% positive": pct,
        "CI low": lo,
        "CI high": hi,
        "95% CI": "—" if lo is None else f"{lo:.1f}–{hi:.1f}",
        "% positive (pos/tested)": format_pct_pos_tested(pct, n_pos, n_tested),
    }
    if extra:
        row.update(extra)
    return row


def summarize_one(df, result_col, result_label, hide_untested_types=True):
    """Summary table by sample type for one test, including Wilson 95% CI."""
    class_col = f"__class__{result_col}"
    rows = []
    types = sorted(df["sample_type"].unique(), key=sample_type_sort_key)
    for stype in types:
        sub = df[df["sample_type"] == stype]
        n_all = int(len(sub))
        n_pos = int((sub[class_col] == "positive").sum())
        n_neg = int((sub[class_col] == "negative").sum())
        n_nt = int((sub[class_col] == "not_tested").sum())
        n_tested = n_pos + n_neg
        if hide_untested_types and n_tested == 0:
            continue
        rows.append(_summary_row(stype, result_label, n_all, n_pos, n_neg, n_nt))
    return pd.DataFrame(rows)


def summarize_many(df, result_columns, hide_untested_types=True):
    parts = [
        summarize_one(df, col, label, hide_untested_types=False)
        for col, label in result_columns
    ]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if hide_untested_types and not table.empty:
        table = table[table["Tested"] > 0].reset_index(drop=True)
    return table


def summarize_by_trip(df, result_columns, hide_untested_types=True):
    """Prevalence table by trip, sample type, and test."""
    if df is None or df.empty or not result_columns:
        return pd.DataFrame(columns=TRIP_DISPLAY_COLUMNS)
    trips = _trip_order(df)
    dates = df.groupby("trip")["trip_date"].min() if "trip_date" in df.columns else {}
    types = sorted(df["sample_type"].unique(), key=sample_type_sort_key)
    rows = []
    for col, label in result_columns:
        class_col = f"__class__{col}"
        for trip in trips:
            trip_date = dates[trip] if trip in dates else None
            if trip_date is not None and not pd.isna(trip_date):
                date_label = pd.Timestamp(trip_date).strftime("%Y-%m-%d")
            else:
                date_label = ""
            for stype in types:
                part = df[(df["trip"] == trip) & (df["sample_type"] == stype)]
                if part.empty:
                    continue
                n_pos = int((part[class_col] == "positive").sum())
                n_neg = int((part[class_col] == "negative").sum())
                n_nt = int((part[class_col] == "not_tested").sum())
                n_tested = n_pos + n_neg
                if hide_untested_types and n_tested == 0:
                    continue
                rows.append(_summary_row(
                    stype, label, int(len(part)), n_pos, n_neg, n_nt,
                    extra={"Trip": trip, "First collection date": date_label},
                ))
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=TRIP_DISPLAY_COLUMNS)
    ordered = [c for c in TRIP_DISPLAY_COLUMNS if c in table.columns]
    rest = [c for c in table.columns if c not in ordered]
    return table[ordered + rest]


def compute_kpis(df, result_columns):
    """Overview figures for the filtered data."""
    n_rows = int(len(df))
    n_trips = int(df["trip"].nunique()) if n_rows else 0
    n_projects = int(df["project"].nunique()) if n_rows else 0
    n_main = int(df["main_project"].nunique()) if n_rows and "main_project" in df.columns else 0
    n_prov = int(df["province"].nunique()) if n_rows and "province" in df.columns else 0
    n_coords = int(df["coord_ok"].sum()) if n_rows and "coord_ok" in df.columns else 0
    n_tested = 0
    n_pos = 0
    for col, _label in result_columns:
        cls = df[f"__class__{col}"]
        n_tested += int(((cls == "positive") | (cls == "negative")).sum())
        n_pos += int((cls == "positive").sum())
    pct = (n_pos / n_tested * 100.0) if n_tested else None
    if pct is not None:
        pct = round(pct, 1)
    lo, hi = wilson_ci(n_pos, n_tested) if n_tested else (None, None)
    return {
        "n_rows": n_rows,
        "n_trips": n_trips,
        "n_projects": n_projects,
        "n_main_projects": n_main,
        "n_provinces": n_prov,
        "n_coords": n_coords,
        "n_tested": n_tested,
        "n_pos": n_pos,
        "pct": pct,
        "ci_low": lo,
        "ci_high": hi,
        "pct_label": format_pct_pos_tested(pct, n_pos, n_tested),
    }


def filter_caption(trips, projects, sample_types, provinces, project_title="projects"):
    def chunk(label_all, values, limit=8):
        if not values:
            return label_all
        shown = ", ".join(str(v) for v in values[:limit])
        if len(values) > limit:
            shown += "…"
        return shown

    parts = [
        chunk("All trips", trips),
        chunk(f"All {project_title}", projects),
        chunk("All sample types", sample_types),
        chunk("All provinces", provinces),
    ]
    return " · ".join(parts)


def apply_chart_style(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=CHART_FONT, size=13),
        height=height,
        margin=dict(l=50, r=30, t=70, b=80),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def fig_prevalence(table, title):
    """Bar chart of percent positive with 95% CI and 'xx% (pos/tested)' labels."""
    fig = go.Figure()
    if table is None or table.empty:
        fig.update_layout(title="No data for the current filters")
        return apply_chart_style(fig)
    used = table[table["Tested"] > 0].copy()
    if used.empty:
        fig.update_layout(title="No tested sample types for the current filters")
        return apply_chart_style(fig)

    types = sorted(used["Sample type"].unique(), key=sample_type_sort_key)
    pathogens = list(dict.fromkeys(used["Test"].tolist()))
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Dark2 + px.colors.qualitative.Set1

    def bar_text(row):
        return format_pct_pos_tested(row["% positive"], row["Positive"], row["Tested"])

    def bar_hover(row):
        return f"{bar_text(row)}<br>95% CI: {row['95% CI']}"

    if len(pathogens) == 1:
        sub = used.set_index("Sample type").reindex(types).dropna(subset=["Tested"])
        err_plus = (sub["CI high"] - sub["% positive"]).clip(lower=0)
        err_minus = (sub["% positive"] - sub["CI low"]).clip(lower=0)
        labels = [bar_text(row) for _, row in sub.iterrows()]
        hovers = [bar_hover(row) for _, row in sub.iterrows()]
        fig.add_bar(
            x=list(sub.index),
            y=sub["% positive"],
            marker_color="#3D6CB9",
            error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color="#333333", thickness=1.2, width=6),
            text=labels,
            textposition="outside",
            customdata=hovers,
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
            name=pathogens[0],
        )
        ymax = max(float(sub["CI high"].max()), float(sub["% positive"].max()), 5)
        fig.update_yaxes(range=[0, min(100, ymax * 1.35)])
        fig.update_traces(cliponaxis=False)
        fig.update_layout(title=title, showlegend=False)
    else:
        n = len(pathogens)
        cols = 3 if n > 2 else n
        rows = int(math.ceil(n / cols))
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=pathogens,
            shared_yaxes=False, vertical_spacing=0.16, horizontal_spacing=0.06,
        )
        global_max = 5
        for i, patho in enumerate(pathogens):
            r, c = divmod(i, cols)
            sub = used[used["Test"] == patho]
            xs, ys, ep, em, labels, hovers = [], [], [], [], [], []
            for stype in types:
                hit = sub[sub["Sample type"] == stype]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                xs.append(stype)
                ys.append(float(row["% positive"]))
                ep.append(max(0.0, float(row["CI high"]) - float(row["% positive"])))
                em.append(max(0.0, float(row["% positive"]) - float(row["CI low"])))
                labels.append(bar_text(row))
                hovers.append(bar_hover(row))
                global_max = max(global_max, float(row["CI high"]), float(row["% positive"]))
            fig.add_trace(
                go.Bar(
                    x=xs, y=ys, marker_color=palette[i % len(palette)],
                    error_y=dict(type="data", symmetric=False, array=ep, arrayminus=em, color="#444444", thickness=1, width=5),
                    text=labels, textposition="outside",
                    customdata=hovers,
                    name=patho, showlegend=False,
                    hovertemplate="%{x}<br>%{fullData.name}<br>%{customdata}<extra></extra>",
                    cliponaxis=False,
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_yaxes(title_text="Percent positive", range=[0, min(100, global_max * 1.55)])
        fig.update_xaxes(tickangle=-25)
        fig.update_layout(title=title)
        return apply_chart_style(fig, height=max(480, 340 * rows))

    fig.update_layout(
        title=title,
        xaxis_title="Sample type",
        yaxis_title="Percent positive (Wilson 95% CI)",
    )
    return apply_chart_style(fig)


def _trip_order(df):
    dates = df.groupby("trip")["trip_date"].min() if "trip_date" in df.columns else {}
    trips = list(df["trip"].unique())
    return sorted(trips, key=lambda t: trip_sort_key(t, dates[t] if t in dates else None))


def trip_axis_label(trip, trip_date):
    if trip_date is not None and not pd.isna(trip_date):
        return f"{trip} ({pd.Timestamp(trip_date).strftime('%Y-%m-%d')})"
    return str(trip)


def fig_stacked_counts(df, result_columns, title, group_by="sample_type", hide_untested_types=False):
    """Stacked counts of positive / negative / not tested, by sample type or trip."""
    if df is None or df.empty or not result_columns:
        fig = go.Figure()
        fig.update_layout(title="No data for the current filters")
        return apply_chart_style(fig)

    if group_by == "trip":
        cats = _trip_order(df)
        dates = df.groupby("trip")["trip_date"].min()
        labels = [trip_axis_label(t, dates[t] if t in dates else None) for t in cats]
        key = "trip"
    else:
        cats = sorted(df["sample_type"].unique(), key=sample_type_sort_key)
        labels = cats
        key = "sample_type"

    pos, neg, nt, keep_idx, pct_labels = [], [], [], [], []
    for i, cat in enumerate(cats):
        sub = df[df[key] == cat]
        n_pos = n_neg = n_nt = 0
        for col, _label in result_columns:
            cls = sub[f"__class__{col}"]
            n_pos += int((cls == "positive").sum())
            n_neg += int((cls == "negative").sum())
            n_nt += int((cls == "not_tested").sum())
        if hide_untested_types and (n_pos + n_neg) == 0:
            continue
        keep_idx.append(i)
        pos.append(n_pos)
        neg.append(n_neg)
        nt.append(n_nt)
        pct = (n_pos / (n_pos + n_neg) * 100.0) if (n_pos + n_neg) else None
        if pct is not None:
            pct = round(pct, 1)
        pct_labels.append(format_pct_pos_tested(pct, n_pos, n_pos + n_neg))
    labels = [labels[i] for i in keep_idx]

    fig = go.Figure()
    if not labels:
        fig.update_layout(title="No tested groups for the current filters")
        return apply_chart_style(fig)
    fig.add_bar(
        name="Positive", x=labels, y=pos, marker_color=COLOR_POS,
        text=pos, textposition="inside",
        customdata=pct_labels,
        hovertemplate="Positive: %{y:,}<br>%{customdata}<extra></extra>",
    )
    fig.add_bar(
        name="Negative", x=labels, y=neg, marker_color=COLOR_NEG,
        text=neg, textposition="inside",
        customdata=pct_labels,
        hovertemplate="Negative: %{y:,}<br>Percent positive: %{customdata}<extra></extra>",
    )
    fig.add_bar(
        name="Not tested / excluded", x=labels, y=nt, marker_color=COLOR_NT,
        text=nt, textposition="inside",
        hovertemplate="Not tested / excluded: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Trip (ordered by first collection date)" if group_by == "trip" else "Sample type",
        yaxis_title="Number of test results" if len(result_columns) > 1 else "Number of samples",
        bargap=0.2,
    )
    if group_by == "trip" and len(labels) > 25:
        fig.update_xaxes(tickangle=-60)
        fig.update_layout(height=560, margin=dict(b=140))
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_traces(cliponaxis=False)
    height = 520 if group_by != "trip" or len(labels) <= 25 else 600
    return apply_chart_style(fig, height=height)


def fig_by_trip(df, result_columns, title, hide_untested_types=True):
    """Percent positive by trip, ordered by first collection date, colored by sample type."""
    fig = go.Figure()
    if df is None or df.empty or not result_columns:
        fig.update_layout(title="No data for the current filters")
        return apply_chart_style(fig)

    trips = _trip_order(df)
    dates = df.groupby("trip")["trip_date"].min()
    xlabels = [trip_axis_label(t, dates[t] if t in dates else None) for t in trips]
    types = sorted(df["sample_type"].unique(), key=sample_type_sort_key)
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set2

    def trip_hover(trip, stype, pct, n_pos, n_tested, lo, hi, test_label=None):
        parts = [f"Trip {trip}"]
        if test_label:
            parts.append(test_label)
        parts.append(f"Sample type: {stype}")
        parts.append(format_pct_pos_tested(round(pct, 1), n_pos, n_tested))
        parts.append(f"95% CI: {lo:.1f}–{hi:.1f}")
        return "<br>".join(parts)

    pathogens = result_columns
    if len(pathogens) == 1:
        col, label = pathogens[0]
        class_col = f"__class__{col}"
        ymax = 5
        traces = 0
        for i, stype in enumerate(types):
            xs, ys, el, eh, hovers = [], [], [], [], []
            sub = df[df["sample_type"] == stype]
            for trip, xlab in zip(trips, xlabels):
                part = sub[sub["trip"] == trip]
                n_pos = int((part[class_col] == "positive").sum())
                n_neg = int((part[class_col] == "negative").sum())
                n_tested = n_pos + n_neg
                if n_tested == 0:
                    continue
                pct = n_pos / n_tested * 100.0
                lo, hi = wilson_ci(n_pos, n_tested)
                xs.append(xlab)
                ys.append(round(pct, 1))
                el.append(max(0.0, round(pct, 1) - lo))
                eh.append(max(0.0, hi - round(pct, 1)))
                hovers.append(trip_hover(trip, stype, pct, n_pos, n_tested, lo, hi))
                ymax = max(ymax, hi, pct)
            if hide_untested_types and not xs:
                continue
            traces += 1
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=stype, marker=dict(size=8, color=palette[i % len(palette)]),
                line=dict(color=palette[i % len(palette)], width=2),
                error_y=dict(type="data", symmetric=False, array=eh, arrayminus=el, color=palette[i % len(palette)], thickness=1, width=4),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hovers,
            ))
        if traces == 0:
            fig.update_layout(title="No trips with test results for the current filters")
            return apply_chart_style(fig)
        fig.update_layout(
            title=title,
            xaxis_title="Trip (ordered by first collection date)",
            yaxis_title="Percent positive (Wilson 95% CI)",
            yaxis=dict(range=[0, min(100, ymax * 1.25)]),
        )
        if len(trips) > 20:
            fig.update_xaxes(tickangle=-60)
            fig.update_layout(margin=dict(b=140), xaxis=dict(rangeslider=dict(visible=True)))
        return apply_chart_style(fig, height=560)

    cols = 1 if len(pathogens) == 1 else (2 if len(pathogens) <= 6 else 2)
    rows = int(math.ceil(len(pathogens) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[lab for _c, lab in pathogens],
                        shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.06)
    for pi, (col, label) in enumerate(pathogens):
        r, c = divmod(pi, cols)
        class_col = f"__class__{col}"
        for i, stype in enumerate(types):
            xs, ys, hovers = [], [], []
            sub = df[df["sample_type"] == stype]
            for trip, xlab in zip(trips, xlabels):
                part = sub[sub["trip"] == trip]
                n_pos = int((part[class_col] == "positive").sum())
                n_neg = int((part[class_col] == "negative").sum())
                n_tested = n_pos + n_neg
                if n_tested == 0:
                    continue
                pct = n_pos / n_tested * 100.0
                lo, hi = wilson_ci(n_pos, n_tested)
                xs.append(xlab)
                ys.append(round(pct, 1))
                hovers.append(trip_hover(trip, stype, pct, n_pos, n_tested, lo, hi, test_label=label))
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=stype,
                    marker=dict(size=6, color=palette[i % len(palette)]),
                    line=dict(color=palette[i % len(palette)], width=1.6),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hovers,
                    showlegend=(pi == 0),
                    legendgroup=stype,
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_yaxes(title_text="Percent positive", range=[0, 100], row=r + 1, col=c + 1)
    fig.update_layout(title=title)
    if len(trips) > 16:
        fig.update_xaxes(tickangle=-55)
    return apply_chart_style(fig, height=max(520, 280 * rows))


def fig_map(df, result_col, result_label, title, hide_untested=True):
    """Map of collection points, colored by result; marker size follows titer for serology."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for the current filters")
        return apply_chart_style(fig)

    pts = df[df["coord_ok"]].copy()
    class_col = f"__class__{result_col}"
    titer_col = f"__titer__{result_col}"
    pts["status"] = pts[class_col].map({
        "positive": "Positive",
        "negative": "Negative",
        "not_tested": "Not tested / excluded",
    }).fillna("Not tested / excluded")
    if hide_untested:
        pts = pts[pts["status"] != "Not tested / excluded"]
    if pts.empty:
        fig = go.Figure()
        fig.update_layout(title="No usable coordinates for the current filters")
        return apply_chart_style(fig)

    pts["marker_size"] = 9.0
    if titer_col in pts.columns:
        titers = pts[titer_col]
        has_titer = titers.notna()
        if has_titer.any():
            pts.loc[has_titer, "marker_size"] = 10.0 + 5.0 * (
                titers[has_titer].clip(lower=50, upper=12800).map(lambda v: math.log10(float(v) / 50.0))
            )
    pts["titer_label"] = pts[titer_col].map(lambda v: f"1:{int(v):,}" if pd.notna(v) else "—") if titer_col in pts.columns else "—"
    pts["date_label"] = pts["collection_date"].map(lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else "—")
    hover = pts.copy()
    hover["hover_trip"] = pts["trip"].astype(str)
    hover["hover_project"] = pts["project"].astype(str)
    hover["hover_type"] = pts["sample_type"].astype(str)
    hover["hover_province"] = pts["province"].astype(str)
    hover["hover_district"] = pts["district"].astype(str)
    hover["hover_tumbon"] = pts["tumbon"].astype(str)

    fig = px.scatter_map(
        hover,
        lat="lat",
        lon="lon",
        color="status",
        color_discrete_map=COLOR_STATUS,
        size="marker_size",
        size_max=22,
        custom_data=[
            "hover_trip", "hover_project", "hover_type",
            "hover_province", "hover_district", "hover_tumbon",
            "date_label", "titer_label",
        ],
        zoom=5,
        height=620,
    )
    fig.update_traces(hovertemplate=(
        "Trip=%{customdata[0]}<br>Project=%{customdata[1]}<br>Sample type=%{customdata[2]}"
        "<br>Province=%{customdata[3]} · District=%{customdata[4]} · Subdistrict=%{customdata[5]}"
        "<br>Date=%{customdata[6]}<br>Titer=%{customdata[7]}"
        "<br>lat=%{lat:.5f}, lon=%{lon:.5f}<extra>%{fullData.name}</extra>"
    ))
    fig.update_layout(
        title=title + f" — {result_label}",
        map_style="open-street-map",
        map=dict(
            center=dict(lat=float(pts["lat"].median()), lon=float(pts["lon"].median())),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend_title="Result",
    )
    return apply_chart_style(fig, height=640)


def fig_titer(df, serology_columns, result_lookup, title, include_negative=True):
    """Titer distribution for STG/MTG/SFG, faceted by test and colored by sample type."""
    fig = go.Figure()
    if df is None or df.empty or not serology_columns:
        fig.update_layout(title="Select a serology test (STG / MTG / SFG) to view the titer distribution")
        return apply_chart_style(fig)

    parts = []
    for col in serology_columns:
        label = result_lookup.get(col, col)
        titer_col = f"__titer__{col}"
        class_col = f"__class__{col}"
        if titer_col not in df.columns:
            continue
        base = df[["sample_type", titer_col, class_col]].copy()
        has_titer = base[titer_col].notna()
        if has_titer.any():
            rec = base.loc[has_titer, ["sample_type", titer_col]].rename(columns={"sample_type": "Sample type"})
            rec["Test"] = label
            rec["titer"] = rec[titer_col].astype(int)
            rec["Level"] = rec["titer"].map(lambda v: f"1:{int(v):,}")
            parts.append(rec[["Sample type", "Test", "titer", "Level"]])
        if include_negative:
            rec_n = base.loc[~has_titer & (base[class_col] == "negative"), ["sample_type"]].rename(
                columns={"sample_type": "Sample type"}
            )
            if not rec_n.empty:
                rec_n["Test"] = label
                rec_n["titer"] = 0
                rec_n["Level"] = "N (negative)"
                parts.append(rec_n[["Sample type", "Test", "titer", "Level"]])
    if not parts:
        fig.update_layout(title="No titer values for the current filters")
        return apply_chart_style(fig)

    tab = pd.concat(parts, ignore_index=True)
    levels = []
    if include_negative and (tab["titer"] == 0).any():
        levels.append("N (negative)")
    titer_vals = sorted({int(v) for v in tab["titer"] if v > 0})
    levels.extend([f"1:{v:,}" for v in titer_vals])
    types = sorted(tab["Sample type"].unique(), key=sample_type_sort_key)
    pathogens = list(dict.fromkeys(tab["Test"].tolist()))
    counts = (
        tab.groupby(["Test", "Sample type", "Level"])
        .size().reset_index(name="Count")
    )

    fig = px.bar(
        counts,
        x="Level",
        y="Count",
        color="Sample type",
        facet_row="Test" if len(pathogens) > 1 else None,
        category_orders={"Level": levels, "Sample type": types, "Test": pathogens},
        barmode="group",
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_xaxes(title_text="Titer")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(legend_title="Sample type")
    fig.update_traces(hovertemplate="Level=%{x}<br>Count=%{y:,}<br>Sample type=%{fullData.name}<extra></extra>")
    height = 420 if len(pathogens) == 1 else 280 * len(pathogens) + 80
    return apply_chart_style(fig, height=height)


PREVIEW_COLUMN_LABELS = {
    "id": "Original code",
    "trip": "Trip",
    "project": "Project",
    "main_project": "Main project",
    "sample_type": "Sample type",
    "collection_date": "Collection date",
    "province": "Province",
    "district": "District",
    "tumbon": "Subdistrict",
    "lat": "Latitude",
    "lon": "Longitude",
}


def raw_preview(df, result_columns, extra_cols, preview_rows=200):
    """Filtered raw-data table with readable English column headers."""
    cols = []
    for c in extra_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    for col, _label in result_columns:
        if col in df.columns and col not in cols:
            cols.append(col)
    show = df[cols].copy()
    if "collection_date" in show.columns:
        show["collection_date"] = show["collection_date"].map(
            lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else ""
        )
    rename = dict(PREVIEW_COLUMN_LABELS)
    for col, label in result_columns:
        rename[col] = label
    show = show.rename(columns=rename)
    return show.head(int(preview_rows)), show


def build_dashboard_views(
    df, result_columns, result_lookup, serology_columns, caption, n_total,
    hide_untested=True, group_counts_by="sample_type", map_col=None, map_label=None,
    include_titer_neg=True, preview_rows=200, extra_cols=None,
):
    """Compute every dashboard section from the filtered data."""
    kpis = compute_kpis(df, result_columns)
    summary = summarize_many(df, result_columns, hide_untested_types=hide_untested)
    trip_table = summarize_by_trip(df, result_columns, hide_untested_types=hide_untested)
    html = None
    if map_col is None and result_columns:
        map_col, map_label = result_columns[0]
    sero_selected = [(c, result_lookup.get(c, c)) for c in serology_columns if c in {x[0] for x in result_columns}]
    extra = extra_cols or []
    preview, full = raw_preview(df, result_columns, extra, preview_rows=preview_rows)
    figs = {
        "prevalence": fig_prevalence(summary, f"Percent positive by sample type, with 95% CI — {caption}"),
        "counts": fig_stacked_counts(
            df, result_columns,
            title=("Positive / negative / not tested counts by trip — " if group_counts_by == "trip"
                   else "Positive / negative / not tested counts by sample type — ") + caption,
            group_by=group_counts_by,
            hide_untested_types=False if group_counts_by == "trip" else hide_untested,
        ),
        "trip": fig_by_trip(
            df, result_columns,
            f"Percent positive by trip (ordered by first collection date) — {caption}",
            hide_untested_types=hide_untested,
        ),
        "map": fig_map(
            df, map_col, map_label or result_lookup.get(map_col, map_col),
            f"Sample collection map — {caption}",
            hide_untested=hide_untested,
        ),
        "titer": fig_titer(
            df, [c for c, _ in sero_selected], result_lookup,
            f"Serology titer distribution — {caption}",
            include_negative=include_titer_neg,
        ),
    }
    return {
        "kpis": kpis,
        "summary": summary,
        "trip_table": trip_table,
        "kpi_html": html,
        "figs": figs,
        "preview": preview,
        "full": full,
    }


# ----- ค่าตั้งต้นของชุดข้อมูล (ใช้ชุดเดียวกับโน้ตบุ๊ก) -----
SPREADSHEET_ID = "1RrvPRPPyF3ssW1t4DYeC3zjLmeA5u4cTzcsxRH0VxyM"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit"
SHEET_NAME = "Data for analysis"
SHEET_GID = "590337954"

COL_SAMPLE_TYPE = "Type of samples"
COL_TRIP = "Trip"
COL_PROJECT = "Project"
COL_MAIN_PROJECT = "Main Project"
COL_PROVINCE = "Province"
COL_DISTRICT = "District"
COL_TUMBON = "Tumbon"
COL_LATITUDE = "Latitude"
COL_LONGITUDE = "Longitude"
COL_DATE = "Collection date"
COL_ID = "Original code"

RESULT_COLUMNS = [
    ("Sero-STG2", "STG (serology)"),
    ("Sero-MT2", "MTG (serology)"),
    ("Sero-SFG2", "SFG (serology)"),
    ("Scrub typhus 47 kDa qPCR", "Orientia 47 kDa (qPCR)"),
    ("Scrub typhus TraD qPCR", "Orientia TraD (qPCR)"),
    ("Scrub typhus O16S qPCR", "Orientia 16S (qPCR)"),
    ("Rickettsia 17kDa qPCR", "Rickettsia 17 kDa (qPCR)"),
    ("SFTSV qPCR", "SFTSV (qPCR)"),
    ("SFTSV ONT", "SFTSV (ONT)"),
    ("SFTSV MAGPIX", "SFTSV (MAGPIX)"),
    ("HPS RT-PCR", "Hantavirus HPS (RT-PCR)"),
    ("Lepto qRT-PCR", "Leptospira (qRT-PCR)"),
    ("Lepto culture", "Leptospira (culture)"),
    ("Bartonella qPCR", "Bartonella (qPCR)"),
]
RESULT_LOOKUP = {col: label for col, label in RESULT_COLUMNS}

SEROLOGY_COLUMNS = ["Sero-STG2", "Sero-MT2", "Sero-SFG2"]

POSITIVE_TEXT = ["P", "Pos", "Positive", "POS"]
NEGATIVE_TEXT = ["N", "Neg", "Negative", "NEG"]
NOT_TESTED_TEXT = [
    "no sample", "na", "n/a", "not tested",
    "waiting", "waiting results", "out of stock",
]
TITER_CUTOFF = 50.0
TREAT_SUSPECTED_AS = "not_tested"
HIDE_UNTESTED_TYPES = True
TABLE_PREVIEW_ROWS = 200


def key_columns():
    return [
        COL_SAMPLE_TYPE, COL_TRIP, COL_PROJECT, COL_MAIN_PROJECT,
        COL_PROVINCE, COL_LATITUDE, COL_LONGITUDE, COL_DATE, COL_ID,
    ] + [col for col, _label in RESULT_COLUMNS]


def google_csv_urls():
    """ลิงก์ส่งออก CSV ถ้าชีตแชร์แบบเปิดลิงก์ได้ (anyone with the link can view)"""
    from urllib.parse import quote
    sheet = quote(SHEET_NAME)
    return [
        f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet}",
        f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={SHEET_GID}",
    ]


def _finalize_raw(df):
    if df is None or df.empty:
        raise ValueError("No data found in the selected file or sheet")
    df = df.copy()
    df.columns = unique_headers(list(df.columns))
    df = df.dropna(how="all")
    df, n_drop = drop_blank_rows(df, key_columns())
    return df, n_drop


def read_raw_from_path(path, sheet_name=SHEET_NAME):
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path, engine="openpyxl")
        use = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(xl, sheet_name=use)
    else:
        df = pd.read_csv(path)
    return _finalize_raw(df)


def read_raw_from_bytes(filename, data, sheet_name=SHEET_NAME):
    import io
    name = (filename or "data").lower()
    bio = io.BytesIO(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xl = pd.ExcelFile(bio, engine="openpyxl")
        use = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(xl, sheet_name=use)
    else:
        df = pd.read_csv(bio)
    return _finalize_raw(df)


def read_raw_from_google():
    last_error = None
    for url in google_csv_urls():
        try:
            df = pd.read_csv(url)
            if df is None or df.shape[1] < 5:
                raise ValueError("The CSV exported from the sheet has too few columns")
            cleaned, n_drop = _finalize_raw(df)
            return cleaned, n_drop, url
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "Could not read the Google Sheet via a public link. "
        "Share the sheet as Anyone with the link can view, or upload a file instead. "
        f"({last_error})"
    )


def prepare_dataset(raw, n_drop=0):
    """จำแนกผลบวก/ลบ และเตรียมคอลัมน์สำหรับกรองและกราฟ"""
    prepared, warnings = prepare_frame(
        raw,
        COL_SAMPLE_TYPE,
        COL_TRIP,
        COL_PROJECT,
        RESULT_COLUMNS,
        TITER_CUTOFF,
        POSITIVE_TEXT,
        NEGATIVE_TEXT,
        NOT_TESTED_TEXT,
        TREAT_SUSPECTED_AS,
        col_main_project=COL_MAIN_PROJECT,
        col_province=COL_PROVINCE,
        col_district=COL_DISTRICT,
        col_tumbon=COL_TUMBON,
        col_lat=COL_LATITUDE,
        col_lon=COL_LONGITUDE,
        col_date=COL_DATE,
        col_id=COL_ID if COL_ID in raw.columns else None,
    )
    return {
        "prepared": prepared,
        "warnings": warnings,
        "n_drop": int(n_drop),
        "n_rows": int(len(prepared)),
    }
