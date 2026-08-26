import re
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vessel Weight Predictor", page_icon="⚙️", layout="wide")

# Column layout for the "Vessel Database (2)" sheet: same as the Rev1 template,
# with two trailing columns (Volume, Area) added after Folder.
# 3 header rows (category / sub-label / units), data starts on row 5.
COLUMN_NAMES = [
    'equip_no', 'qty', 'description', 'pfd_no', 'location', 'capacity',
    'duty_kw', 'absorbed_kw', 'design_press', 'design_temp_max', 'design_temp_min',
    'oper_press', 'oper_temp', 'length_tt', 'diameter_id',
    'wt_unit_dry', 'wt_unit_oper', 'wt_tot_dry', 'wt_tot_oper', 'wt_test',
    'material', 'orientation', 'unit_cost', 'sub_total', 'remarks',
    'project', 'year', 'folder', 'volume_given', 'area_given'
]
SHEET_NAME = 'Vessel Database (2)'

# --- Required inputs: dimensions drive the geometry; design conditions kept for reference ---
DIMENSION_FEATURES = ['length_num', 'diameter_num']
DESIGN_CONDITION_FEATURES = ['design_press_num', 'design_temp_max_num', 'design_temp_min_num']
CATEGORICAL_FEATURES = ['material_category', 'orientation_clean']

# --- Outputs: weight only for now. Unit cost was removed — a leave-one-out
# accuracy check showed the cost model (volume-based curve x material
# factor) had a median error of ~70% and was unreliable, so it's disabled
# until a better approach is found (see conversation for the diagnosis).
WEIGHT_TARGETS = ['wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num']
TARGET_COLS = WEIGHT_TARGETS
WEIGHT_LABELS = {
    'wt_unit_dry_num': 'Unit Dry',
    'wt_unit_oper_num': 'Unit Operating',
    'wt_test_num': 'Test',
}

REFERENCE_COLS = ['project', 'year']
AUTO_OPTION = "Auto (recommend from history)"


# =====================================================================
# CLEANING HELPERS
# =====================================================================
def first_num(x):
    """Pull the first number out of a cell like '3,600 ' or '14000'."""
    if pd.isna(x):
        return None
    s = str(x).replace(',', '')
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else None


def categorize_material(s):
    """Keep the material text exactly as entered — only tidy whitespace/line breaks."""
    if pd.isna(s) or str(s).strip() == '':
        return 'Unknown'
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def normalize_orientation(s):
    if pd.isna(s):
        return 'Unknown'
    s = str(s).strip().upper()
    if s.startswith('V'):
        return 'Vertical'
    if s.startswith('H'):
        return 'Horizontal'
    return 'Unknown'


def compute_geometry(length_mm, diameter_mm):
    """Cylinder-with-flat-ends volume (m3) and surface area (m2) from length
    and internal diameter in mm. Matches the workbook's Volume/Area columns
    to within rounding, verified against the source data."""
    L = length_mm / 1000.0
    D = diameter_mm / 1000.0
    volume = (math.pi / 4) * D ** 2 * L
    area = math.pi * D * L + (math.pi / 2) * D ** 2
    return volume, area


@st.cache_data
def load_and_clean(file):
    raw = pd.read_excel(file, sheet_name=SHEET_NAME, header=None, skiprows=4)
    raw = raw.iloc[:, 1:1 + len(COLUMN_NAMES)]
    raw.columns = COLUMN_NAMES

    df = raw.copy()
    df['length_num'] = df['length_tt'].apply(first_num)
    df['diameter_num'] = df['diameter_id'].apply(first_num)
    df['design_press_num'] = df['design_press'].apply(first_num)
    df['design_temp_max_num'] = df['design_temp_max'].apply(first_num)
    df['design_temp_min_num'] = df['design_temp_min'].apply(first_num)
    df['wt_unit_dry_num'] = df['wt_unit_dry'].apply(first_num)
    df['wt_unit_oper_num'] = df['wt_unit_oper'].apply(first_num)
    df['wt_test_num'] = df['wt_test'].apply(first_num)
    df['unit_cost_num'] = df['unit_cost'].apply(first_num)  # kept for reference display only
    df['material_category'] = df['material'].apply(categorize_material)
    df['orientation_clean'] = df['orientation'].apply(normalize_orientation)

    # Volume/Area are always (re)computed from length & diameter, rather than
    # trusting the workbook's stored Volume/Area cells, since a few rows have
    # those cells blank — this keeps every row usable and self-consistent.
    geo = df.apply(
        lambda r: compute_geometry(r['length_num'], r['diameter_num'])
        if pd.notna(r['length_num']) and pd.notna(r['diameter_num']) else (None, None),
        axis=1, result_type='expand'
    )
    df['volume_m3'] = geo[0]
    df['area_m2'] = geo[1]

    return df


DISPLAY_COLS = (['equip_no', 'description', 'length_num', 'diameter_num',
                  'volume_m3', 'area_m2'] + DESIGN_CONDITION_FEATURES
                 + CATEGORICAL_FEATURES + WEIGHT_TARGETS + ['unit_cost_num'] + REFERENCE_COLS)

REQUIRED_COLS = DIMENSION_FEATURES + TARGET_COLS


def interpolate(two_refs: pd.DataFrame, query_volume: float, target_col: str):
    """Linear interpolation (or extrapolation, if the query volume falls
    outside the 2 references' range) of target_col along the Volume axis."""
    v_a, v_b = two_refs['volume_m3'].iloc[0], two_refs['volume_m3'].iloc[1]
    t_a, t_b = two_refs[target_col].iloc[0], two_refs[target_col].iloc[1]
    if v_b == v_a:
        return (t_a + t_b) / 2, 0.5
    frac = (query_volume - v_a) / (v_b - v_a)
    value = t_a + frac * (t_b - t_a)
    return value, frac


# =====================================================================
# PAGE
# =====================================================================
st.title("⚙️ Vessel Weight Predictor")
st.caption(
    "Computes Volume & Area from your entered Length and Internal Diameter, finds the "
    "2 closest historical vessels by combined Volume + Area similarity, and linearly "
    "interpolates weight between them. Cost prediction is temporarily disabled — a "
    "validation check showed it was unreliable (median error ~70%); it will return once "
    "a better approach is worked out."
)

uploaded = st.file_uploader("Upload vessel database (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info(
        f"Upload the vessel database Excel file to begin. Expects a sheet named "
        f"'{SHEET_NAME}' with 3 header rows followed by data rows, matching the current template."
    )
    st.stop()

try:
    with st.spinner("Reading and cleaning the workbook..."):
        df = load_and_clean(uploaded)
except ValueError:
    st.error(f"Couldn't find sheet '{SHEET_NAME}' in this workbook. Check the sheet name and try again.")
    st.stop()

st.success(f"Loaded {len(df)} equipment rows from '{SHEET_NAME}'.")

with st.expander("Preview cleaned data (Volume & Area computed from Length/Diameter)"):
    st.dataframe(df[DISPLAY_COLS], use_container_width=True)

clean = df.dropna(subset=REQUIRED_COLS + ['volume_m3']).copy()
n_usable = len(clean)
st.caption(
    f"{n_usable} of {len(df)} rows had complete data (length, diameter, and all 3 weight "
    f"figures present) and can be used as interpolation references."
)

if n_usable < 2:
    st.warning(
        "Need at least 2 complete historical rows to interpolate between. "
        "Check that length, diameter, and weight are filled in for more rows."
    )
    st.stop()

st.divider()

# =====================================================================
# PREDICTION FORM
# =====================================================================
st.subheader("Predict weight for a new vessel")

material_options = [AUTO_OPTION] + sorted(clean['material_category'].unique())
orientation_options = [AUTO_OPTION] + sorted(clean['orientation_clean'].unique())

with st.form("prediction_form"):
    st.markdown("**Dimensions** (used to compute Volume & Area, and to find the closest references)")
    g1, g2 = st.columns(2)
    with g1:
        user_length = st.number_input("Length T/T (mm)", min_value=1.0, value=6000.0, step=100.0)
    with g2:
        user_diameter = st.number_input("Internal diameter (mm)", min_value=1.0, value=2000.0, step=50.0)

    st.markdown("**Design conditions** (recorded for reference)")
    d1, d2, d3 = st.columns(3)
    with d1:
        user_press = st.number_input("Design pressure (barg)", value=20.0, step=1.0)
    with d2:
        user_temp_max = st.number_input("Design temperature — Max (°C)", value=100.0, step=5.0)
    with d3:
        user_temp_min = st.number_input("Design temperature — Min (°C)", value=0.0, step=5.0)

    st.markdown("**Construction (optional — leave on Auto to get a recommendation)**")
    c1, c2 = st.columns(2)
    with c1:
        user_material = st.selectbox("Material category", material_options)
    with c2:
        user_orientation = st.selectbox("Orientation", orientation_options)

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

if submitted:
    query_volume, query_area = compute_geometry(user_length, user_diameter)
    st.session_state['prediction_ctx'] = dict(
        query_volume=query_volume, query_area=query_area,
        user_material=user_material, user_orientation=user_orientation,
    )

if 'prediction_ctx' in st.session_state:
    ctx = st.session_state['prediction_ctx']
    query_volume, query_area = ctx['query_volume'], ctx['query_area']
    user_material, user_orientation = ctx['user_material'], ctx['user_orientation']

    st.markdown("**Computed geometry**")
    gc1, gc2 = st.columns(2)
    gc1.metric("Volume", f"{query_volume:.2f} m³")
    gc2.metric("Area", f"{query_area:.2f} m²")

    # --- Rank ALL usable historical vessels by combined Volume + Area distance ---
    vol_std = clean['volume_m3'].std() or 1.0
    area_std = clean['area_m2'].std() or 1.0
    ranked = clean.copy()
    ranked['distance'] = np.sqrt(
        ((ranked['volume_m3'] - query_volume) / vol_std) ** 2
        + ((ranked['area_m2'] - query_area) / area_std) ** 2
    )
    ranked = ranked.sort_values('distance').reset_index(drop=True)

    # --- Let the user override which 2 vessels to interpolate between ---
    # Defaults to the closest match by shape, but geometry alone can't know
    # that two vessels belong to the same design family/service — so the
    # engineer gets the final say on which pair is actually appropriate.
    def _label(i):
        r = ranked.iloc[i]
        return (f"{r['equip_no']} — Vol {r['volume_m3']:.2f} m³, "
                f"Area {r['area_m2']:.2f} m² (shape distance {r['distance']:.3f})")

    st.markdown("**Reference vessels for interpolation**")
    selected_idx = st.multiselect(
        "Choose exactly 2 (defaults to the closest match by Volume + Area — override if a "
        "different pair from the same design family is more appropriate)",
        options=list(range(len(ranked))),
        default=[0, 1],
        format_func=_label,
        max_selections=2,
        key='ref_override',
    )

    if len(selected_idx) != 2:
        st.info("Select exactly 2 reference vessels above to see the interpolated prediction.")
        st.stop()

    two_refs = ranked.iloc[selected_idx].sort_values('volume_m3').reset_index(drop=True)
    v_min, v_max = two_refs['volume_m3'].min(), two_refs['volume_m3'].max()
    is_true_interpolation = v_min <= query_volume <= v_max

    if is_true_interpolation:
        st.success(
            f"Query volume falls between the 2 chosen reference vessels — "
            f"interpolating between **{two_refs.iloc[0]['equip_no']}** and "
            f"**{two_refs.iloc[1]['equip_no']}**."
        )
    else:
        st.warning(
            f"Query volume falls outside the range of **{two_refs.iloc[0]['equip_no']}** "
            f"and **{two_refs.iloc[1]['equip_no']}** — the calculation below is an "
            f"extrapolation beyond these 2 references, not a true interpolation."
        )

    # --- Resolve optional material/orientation from the 2 references (for display only) ---
    material_is_auto = user_material == AUTO_OPTION
    orientation_is_auto = user_orientation == AUTO_OPTION
    recommended_material = two_refs['material_category'].mode().iloc[0]
    recommended_orientation = two_refs['orientation_clean'].mode().iloc[0]
    resolved_material = recommended_material if material_is_auto else user_material
    resolved_orientation = recommended_orientation if orientation_is_auto else user_orientation

    if material_is_auto or orientation_is_auto:
        st.markdown("**Recommended construction**")
        rc1, rc2 = st.columns(2)
        if material_is_auto:
            rc1.metric("Recommended Material", resolved_material)
        if orientation_is_auto:
            rc2.metric("Recommended Orientation", resolved_orientation)
        st.caption("Based on the 2 chosen reference vessels.")

    # --- Interpolate each weight output along the Volume axis ---
    final_values, fracs = {}, {}
    for target in TARGET_COLS:
        value, frac = interpolate(two_refs, query_volume, target)
        final_values[target] = value
        fracs[target] = frac

    st.markdown("**Predicted weight breakdown**")
    w1, w2, w3 = st.columns(3)
    for col, target in zip([w1, w2, w3], WEIGHT_TARGETS):
        col.metric(WEIGHT_LABELS[target], f"{final_values[target]:.2f} MT")

    with st.expander("See the 2 reference vessels and interpolation detail"):
        detail = pd.DataFrame({
            "Output": [WEIGHT_LABELS[t] for t in WEIGHT_TARGETS],
            f"{two_refs.iloc[0]['equip_no']} (Vol {v_min:.1f} m³)":
                [f"{two_refs.iloc[0][t]:.2f}" for t in WEIGHT_TARGETS],
            f"{two_refs.iloc[1]['equip_no']} (Vol {v_max:.1f} m³)":
                [f"{two_refs.iloc[1][t]:.2f}" for t in WEIGHT_TARGETS],
            "Interpolated (Your vessel)":
                [f"{final_values[t]:.2f}" for t in WEIGHT_TARGETS],
        })
        st.table(detail)
        st.caption(
            f"Interpolation fraction along the Volume axis: {fracs['wt_unit_dry_num']:.2f} "
            f"(0.0 = matches the smaller vessel exactly, 1.0 = matches the larger vessel exactly; "
            f"outside [0, 1] means extrapolation)."
        )

    st.markdown("**The 2 reference vessels used** (historical cost shown for context only — not used in the calculation)")
    st.dataframe(
        two_refs[['equip_no', 'description', 'length_num', 'diameter_num', 'volume_m3', 'area_m2',
                   'material_category', 'orientation_clean']
                 + WEIGHT_TARGETS + ['unit_cost_num'] + REFERENCE_COLS + ['distance']],
        use_container_width=True
    )