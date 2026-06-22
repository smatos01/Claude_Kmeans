import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Builder & Profiler",
    page_icon="🔵",
    layout="wide",
)

# ── dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f1117; color: #e0e0e0; }
  .block-container { padding-top: 2rem; }
  .metric-card {
    background: #1a1d27;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .segment-header {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .segment-sub {
    font-size: 1rem;
    color: #9ca3af;
    margin-bottom: 16px;
  }
  .chip-over  { background:#1a3a1a; color:#4ade80; border-radius:4px; padding:2px 8px; font-weight:600; }
  .chip-under { background:#3a1a1a; color:#f87171; border-radius:4px; padding:2px 8px; font-weight:600; }
  .chip-par   { background:#1e2130; color:#9ca3af; border-radius:4px; padding:2px 8px; font-weight:600; }
  h1, h2, h3 { color: #f0f0f0; }
  .stDataFrame { background: #1a1d27; }
  div[data-testid="stExpander"] { background: #1a1d27; border: 1px solid #2d3147; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

SAFE_COLORS = px.colors.qualitative.Safe

# ── helpers ───────────────────────────────────────────────────────────────────

def index_chip(idx: float) -> str:
    if idx >= 110:
        return f'<span class="chip-over">🟢 {idx:.0f} ▲</span>'
    elif idx <= 90:
        return f'<span class="chip-under">🔴 {idx:.0f} ▼</span>'
    else:
        return f'<span class="chip-par">⬜ {idx:.0f}</span>'


def plotly_dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="#2d3147", zerolinecolor="#2d3147"),
        yaxis=dict(gridcolor="#2d3147", zerolinecolor="#2d3147"),
    )
    return fig


def segment_color(label: int) -> str:
    return SAFE_COLORS[label % len(SAFE_COLORS)]

# ── main app ──────────────────────────────────────────────────────────────────

st.title("K-Means Builder & Profiler")
st.markdown("Upload a semicolon-delimited CSV, run K-Means end-to-end, and explore segment profiles.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.header("1 — Upload")
uploaded = st.file_uploader("Drop your SCV (.csv) file here", type=["csv"])

if not uploaded:
    st.stop()

try:
    raw_df = pd.read_csv(uploaded, sep=";")
    # if semicolon produced only 1 column the file is likely comma-delimited
    if len(raw_df.columns) == 1:
        uploaded.seek(0)
        raw_df = pd.read_csv(uploaded, sep=",")
        detected_sep = ","
    else:
        detected_sep = ";"
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

st.caption(f"Detected delimiter: `{'semicolon' if detected_sep == ';' else 'comma'}`")

st.subheader("Preview (first 5 rows)")
st.dataframe(raw_df.head(), use_container_width=True)

col_summary = pd.DataFrame({
    "Column": raw_df.columns,
    "dtype": raw_df.dtypes.values,
    "Unique values": [raw_df[c].nunique() for c in raw_df.columns],
    "Missing": [raw_df[c].isna().sum() for c in raw_df.columns],
})
st.subheader("Column summary")
st.dataframe(col_summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLEANING & COLUMN FILTERING
# ─────────────────────────────────────────────────────────────────────────────
st.header("2 — Column Filtering & Cleaning")

df = raw_df.copy()

# drop rows with missing values
n_before = len(df)
df = df.dropna()
n_dropped_rows = n_before - len(df)
if n_dropped_rows:
    st.info(f"Removed **{n_dropped_rows}** row(s) with missing values ({n_before} → {len(df)} rows).")

if df.empty:
    st.error("No rows remain after removing missing values.")
    st.stop()

# auto-filter columns
removed_cols = {}
kept_cols = []
for col in df.columns:
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    n_unique = df[col].nunique()
    if not is_numeric and n_unique > 5:
        removed_cols[col] = f"Non-numeric with {n_unique} unique values (> 5) — high cardinality"
    else:
        kept_cols.append(col)

# drop zero-variance columns from kept_cols
zero_var = []
for col in list(kept_cols):
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 1:
        kept_cols.remove(col)
        zero_var.append(col)
        removed_cols[col] = "Zero variance — identical value in all rows"

if removed_cols:
    with st.expander(f"Columns removed automatically ({len(removed_cols)})", expanded=True):
        st.dataframe(
            pd.DataFrame({"Column": list(removed_cols.keys()), "Reason": list(removed_cols.values())}),
            use_container_width=True,
        )

if not kept_cols:
    st.error("No usable columns found. Check that your file has numeric or low-cardinality categorical columns.")
    st.stop()

if len(kept_cols) == 1:
    st.warning(f"Only 1 column remains (`{kept_cols[0]}`). PCA may not be meaningful. You can still proceed.")

# user checklist
st.subheader("Select columns to include in clustering")
st.caption("Unticked columns are excluded from the model but remain available for profiling.")

selected_cols = []
cols_per_row = 4
col_widgets = st.columns(cols_per_row)
for i, col in enumerate(kept_cols):
    with col_widgets[i % cols_per_row]:
        checked = st.checkbox(col, value=True, key=f"col_chk_{col}")
        if checked:
            selected_cols.append(col)

if not selected_cols:
    st.error("Select at least one column to proceed.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
st.header("3 — Configure")

k_val = st.number_input("Number of clusters (K)", min_value=2, max_value=20, value=3, step=1)

show_elbow = st.checkbox("Show elbow chart before running")

run_btn = st.button("▶  Run K-Means", type="primary")

if show_elbow and run_btn is False:
    # compute elbow on current selected cols without running full pipeline
    pass  # computed below after encoding

# Build modelling df from selected cols only
model_input_df = df[selected_cols].copy()

# identify categorical vs numeric in selection
cat_cols = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(model_input_df[c])]
num_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(model_input_df[c])]

# one-hot encode
cat_map: dict[str, list[str]] = {}
if cat_cols:
    dummies = pd.get_dummies(model_input_df[cat_cols], prefix_sep="_", drop_first=False)
    for orig in cat_cols:
        cat_map[orig] = [c for c in dummies.columns if c.startswith(orig + "_")]
    model_input_df = pd.concat([model_input_df[num_cols], dummies], axis=1)
else:
    model_input_df = model_input_df[num_cols]

# scale
scaler = MinMaxScaler()
scaled_arr = scaler.fit_transform(model_input_df)

# elbow chart
if show_elbow:
    k_range = range(2, min(11, len(df)))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(scaled_arr)
        inertias.append(km.inertia_)
    fig_elbow = go.Figure(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                     line=dict(color="#60a5fa"), marker=dict(color="#60a5fa", size=8)))
    fig_elbow.update_layout(title="Elbow Chart", xaxis_title="K", yaxis_title="Inertia")
    plotly_dark_layout(fig_elbow)
    st.plotly_chart(fig_elbow, use_container_width=True)

if not run_btn:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
if k_val >= len(df):
    st.error(f"K ({k_val}) must be smaller than the number of rows ({len(df)}).")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PCA
# ─────────────────────────────────────────────────────────────────────────────
n_features = scaled_arr.shape[1]
max_components = min(n_features, len(df))
pca_full = PCA(n_components=max_components, random_state=42)
pca_full.fit(scaled_arr)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.searchsorted(cum_var, 0.90) + 1)
n_components = min(n_components, max_components)

if n_components < max_components:
    st.info(f"PCA: selected **{n_components}** components explaining **{cum_var[n_components-1]*100:.1f}%** of variance (≥ 90% threshold).")
else:
    st.warning(f"PCA capped at **{n_components}** components (all available). Explains **{cum_var[n_components-1]*100:.1f}%** of variance.")

var_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_components)],
    "Variance Explained (%)": [f"{v*100:.2f}%" for v in pca_full.explained_variance_ratio_[:n_components]],
})
var_df.loc[len(var_df)] = ["Total", f"{cum_var[n_components-1]*100:.2f}%"]
st.dataframe(var_df, use_container_width=True, hide_index=True)

pca = PCA(n_components=n_components, random_state=42)
pca_data = pca.fit_transform(scaled_arr)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
km = KMeans(n_clusters=int(k_val), init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(pca_data)

result_df = df.copy()
result_df["_segment"] = labels

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 — Results")

# segment overview
seg_counts = result_df["_segment"].value_counts().sort_index()
overview = pd.DataFrame({
    "Segment": [f"Segment {i+1}" for i in seg_counts.index],
    "Size (n)": seg_counts.values,
    "% of Total": [f"{v/len(result_df)*100:.1f}%" for v in seg_counts.values],
})
st.subheader("Segment Overview")
st.dataframe(overview, use_container_width=True, hide_index=True)

# PCA scatter
pc_df = pd.DataFrame(pca_data[:, :2], columns=["PC1", "PC2"])
pc_df["Segment"] = [f"Segment {l+1}" for l in labels]
pc_df["_label"] = labels

color_map = {f"Segment {i+1}": SAFE_COLORS[i % len(SAFE_COLORS)] for i in range(int(k_val))}

fig_scatter = px.scatter(
    pc_df, x="PC1", y="PC2", color="Segment",
    color_discrete_map=color_map,
    title="PCA Scatter (PC1 vs PC2)",
    opacity=0.7,
)
plotly_dark_layout(fig_scatter)
st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SEGMENT PROFILES
# ─────────────────────────────────────────────────────────────────────────────
st.header("5 — Segment Profiles")

# all original numeric columns (including those excluded from clustering)
orig_num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# all original categorical columns (including excluded ones, ≤5 unique)
orig_cat_cols = [c for c in df.columns
                 if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 5]

dataset_num_means = {c: result_df[c].mean() for c in orig_num_cols}
dataset_cat_dist: dict[str, dict] = {}
for c in orig_cat_cols:
    dataset_cat_dist[c] = (result_df[c].value_counts() / len(result_df)).to_dict()

tab_labels = [f"Segment {i+1}" for i in range(int(k_val))]
tabs = st.tabs(tab_labels)

for seg_idx, tab in enumerate(tabs):
    with tab:
        seg_df = result_df[result_df["_segment"] == seg_idx]
        color_hex = SAFE_COLORS[seg_idx % len(SAFE_COLORS)]

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="segment-header" style="color:{color_hex}">Segment {seg_idx+1}</div>'
            f'<div class="segment-sub">n = {len(seg_df):,} &nbsp;·&nbsp; {len(seg_df)/len(result_df)*100:.1f}% of total</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # numeric profile
        if orig_num_cols:
            st.markdown("#### Numeric Features")
            rows = []
            for col in orig_num_cols:
                seg_mean = seg_df[col].mean()
                ds_mean = dataset_num_means[col]
                idx = (seg_mean / ds_mean * 100) if ds_mean != 0 else 0
                rows.append({
                    "Feature": col,
                    "Segment avg": f"{seg_mean:.2f}",
                    "Dataset avg": f"{ds_mean:.2f}",
                    "Index": idx,
                    "Signal": index_chip(idx),
                })
            num_df = pd.DataFrame(rows)
            st.write(
                num_df[["Feature", "Segment avg", "Dataset avg", "Signal"]]
                .to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )

        # categorical profile
        if orig_cat_cols:
            st.markdown("#### Categorical Features")
            for col in orig_cat_cols:
                st.markdown(f"**{col}**")
                cat_rows = []
                for cat_val, ds_pct in dataset_cat_dist[col].items():
                    seg_count = (seg_df[col] == cat_val).sum()
                    seg_pct = seg_count / len(seg_df) if len(seg_df) else 0
                    idx = (seg_pct / ds_pct * 100) if ds_pct != 0 else 0
                    cat_rows.append({
                        "Category": f"{col} = {cat_val}",
                        "Seg count": int(seg_count),
                        "Seg %": f"{seg_pct*100:.1f}%",
                        "Dataset %": f"{ds_pct*100:.1f}%",
                        "Index": idx,
                        "Signal": index_chip(idx),
                    })
                cat_df = pd.DataFrame(cat_rows)
                st.write(
                    cat_df[["Category", "Seg count", "Seg %", "Dataset %", "Signal"]]
                    .to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
