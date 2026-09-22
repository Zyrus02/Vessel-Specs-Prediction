# =====================================================================
# 1. IMPORTS & SETUP
# These are the "toolboxes" the application uses to run.
# =====================================================================
import math              # Used for basic math like Pi (for circular geometry)
import re                # 'Regular Expressions' - used to find numbers hidden inside text
import numpy as np       # Used for advanced math, like calculating the "shape distance" (square roots, etc.)
import pandas as pd      # The "accountant" tool. Reads Excel files and organizes data into tables (DataFrames)
import streamlit as st   # The web builder. Turns this python script into an interactive website
import matplotlib.pyplot as plt # The graphing tool used to draw the cost vs size charts

# Sets up the web page title and tells it to use the full width of the screen
st.set_page_config(page_title="Engineering Specs Predictor", page_icon="📐", layout="wide")

# A standard option we use in dropdown menus later
AUTO_OPTION = "Auto (recommend from history)"

# =====================================================================
# 2. MECHANICAL (VESSELS) CONFIGURATION
# Defines how to read the Vessel Excel sheet and what columns to expect.
# =====================================================================
COLUMN_NAMES = [
    'equip_no', 'qty', 'description', 'pfd_no', 'location', 'capacity',
    'duty_kw', 'absorbed_kw', 'design_press', 'design_temp_max', 'design_temp_min',
    'oper_press', 'oper_temp', 'length_tt', 'diameter_id',
    'wt_unit_dry', 'wt_unit_oper', 'wt_tot_dry', 'wt_tot_oper', 'wt_test',
    'material', 'orientation', 'unit_cost', 'sub_total', 'remarks',
    'project', 'year', 'folder', 'volume_given', 'area_given'
]
SHEET_NAME = 'Vessel Database (2)'

# Grouping columns together so it's easier to reference them later
DIMENSION_FEATURES = ['length_num', 'diameter_num']
DESIGN_CONDITION_FEATURES = ['design_press_num', 'design_temp_max_num', 'design_temp_min_num']
CATEGORICAL_FEATURES = ['material_category', 'orientation_clean']
WEIGHT_TARGETS = ['wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num']

WEIGHT_LABELS = {
    'wt_unit_dry_num': 'Unit Dry',
    'wt_unit_oper_num': 'Unit Operating',
    'wt_test_num': 'Test',
}
REFERENCE_COLS = ['project', 'year']
REQUIRED_COLS = DIMENSION_FEATURES + WEIGHT_TARGETS + ['unit_cost_num']
DISPLAY_COLS = (['equip_no', 'description', 'length_num', 'diameter_num',
                  'volume_m3', 'area_m2'] + DESIGN_CONDITION_FEATURES
                 + CATEGORICAL_FEATURES + WEIGHT_TARGETS + ['unit_cost_num'] + REFERENCE_COLS)

# =====================================================================
# 3. MECHANICAL DATA CLEANERS
# Excel data is often messy. These functions clean the data so the computer can do math on it.
# =====================================================================
def first_num(x):
    """Takes a messy cell like ' 1,500 kg ' and extracts just the number: 1500.0"""
    if pd.isna(x): return None
    s = str(x).replace(',', '') # Remove commas so thousands don't break the math
    m = re.search(r'-?\d+\.?\d*', s) # Search for the first valid number
    return float(m.group()) if m else None

def categorize_material(s):
    """Removes weird extra spaces so 'Carbon  Steel' becomes 'Carbon Steel'."""
    if pd.isna(s) or str(s).strip() == '': return 'Unknown'
    return re.sub(r'\s+', ' ', str(s).strip())

def normalize_orientation(s):
    """Forces orientation to be either 'Vertical', 'Horizontal', or 'Unknown'."""
    if pd.isna(s): return 'Unknown'
    s = str(s).strip().upper()
    if s.startswith('V'): return 'Vertical'
    if s.startswith('H'): return 'Horizontal'
    return 'Unknown'

def compute_geometry(length_mm, diameter_mm):
    """Uses basic cylinder math to calculate Volume (m3) and Surface Area (m2)."""
    L = length_mm / 1000.0 # Convert mm to meters
    D = diameter_mm / 1000.0
    volume = (math.pi / 4) * D ** 2 * L
    area = math.pi * D * L + (math.pi / 2) * D ** 2
    return volume, area

@st.cache_data # Tells Streamlit to remember this step so it doesn't reload the Excel file every time you click a button
def load_and_clean(file):
    """Loads the Mechanical Excel file and runs all the cleaning functions above on it."""
    # Skip the first 4 rows because they are just titles/headers in your Excel file
    raw = pd.read_excel(file, sheet_name=SHEET_NAME, header=None, skiprows=4)
    raw = raw.iloc[:, 1:1 + len(COLUMN_NAMES)]
    raw.columns = COLUMN_NAMES

    df = raw.copy()
    # Apply the 'first_num' cleaner to all numerical columns
    df['length_num'] = df['length_tt'].apply(first_num)
    df['diameter_num'] = df['diameter_id'].apply(first_num)
    df['design_press_num'] = df['design_press'].apply(first_num)
    df['design_temp_max_num'] = df['design_temp_max'].apply(first_num)
    df['design_temp_min_num'] = df['design_temp_min'].apply(first_num)
    df['wt_unit_dry_num'] = df['wt_unit_dry'].apply(first_num)
    df['wt_unit_oper_num'] = df['wt_unit_oper'].apply(first_num)
    df['wt_test_num'] = df['wt_test'].apply(first_num)
    df['unit_cost_num'] = df['unit_cost'].apply(first_num)
    
    # Apply text cleaners
    df['material_category'] = df['material'].apply(categorize_material)
    df['orientation_clean'] = df['orientation'].apply(normalize_orientation)

    # Calculate actual volume and area for every row in the historical database
    geo = df.apply(
        lambda r: compute_geometry(r['length_num'], r['diameter_num'])
        if pd.notna(r['length_num']) and pd.notna(r['diameter_num']) else (None, None),
        axis=1, result_type='expand'
    )
    df['volume_m3'] = geo[0]
    df['area_m2'] = geo[1]

    return df

# =====================================================================
# 4. PIPING (VALVES) CONFIG & HELPERS
# Similar to Mechanical, but tailored for the Valves Excel file.
# =====================================================================
VALVE_SHEET_NAME = 'OVERALL DATABASE'
VALVE_COLUMNS = [
    'item_no', 'year', 'project', 'valve_type', 'subtype', 'size_1_mm', 'size_2_mm',
    'bore', 'rating', 'end_connection', 'material', 'unit_weight_kg', 'unit_cost_rm'
]

def clean_valve_rating(val):
    """Fixes inconsistencies in how ratings are typed (e.g. making 'API5000' standardized)."""
    if pd.isna(val): return 'Unknown'
    s = str(val).strip().upper()
    return 'API 5000' if s == 'API5000' else s

def clean_numeric(val):
    """Cleans numbers, but also handles cases where cells just have a dash '-' meaning empty."""
    if pd.isna(val) or str(val).strip() == '-': return None
    s = str(val).replace(',', '')
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else None

@st.cache_data
def load_and_clean_valves(file):
    """Loads and cleans the Valve Excel file."""
    raw = pd.read_excel(file, sheet_name=VALVE_SHEET_NAME, header=2) # Headers start on row 3
    raw.columns = VALVE_COLUMNS
    
    df = raw.copy()
    df['size_num'] = df['size_1_mm'].apply(clean_numeric)
    df['weight_num'] = df['unit_weight_kg'].apply(clean_numeric)
    df['cost_num'] = df['unit_cost_rm'].apply(clean_numeric)
    df['rating_clean'] = df['rating'].apply(clean_valve_rating)
    df['type_clean'] = df['valve_type'].astype(str).str.strip().str.upper()
    df['material_clean'] = df['material'].astype(str).str.strip().str.upper()
    
    # Throw away rows that are missing the size or cost, because we can't predict with them
    return df.dropna(subset=['size_num', 'cost_num'])

# =====================================================================
# 4.5 MECHANICAL (TANKS) CONFIG & HELPERS
# =====================================================================
TANK_SHEET_NAME = 'Sheet1 (2)'

def compute_geometry_tank(length_mm, width_mm, height_mm):
    """Uses rectangular prism math to calculate Volume (m3) and Surface Area (m2)."""
    L = length_mm / 1000.0
    W = width_mm / 1000.0
    H = height_mm / 1000.0
    volume = L * W * H
    area = 2 * (L * W + L * H + W * H)
    return volume, area

@st.cache_data
def load_and_clean_tanks(file):
    """Loads and cleans the highlighted tank columns via absolute index skipping messy headers."""
    raw = pd.read_excel(file, sheet_name=TANK_SHEET_NAME, header=None, skiprows=3)
    df = raw.copy()
    
    df['equip_no'] = df[1]
    df['description'] = df[3]
    df['length_num'] = df[14].apply(first_num)
    df['width_num'] = df[15].apply(first_num)
    df['height_num'] = df[16].apply(first_num)
    df['wt_unit_dry_num'] = df[17].apply(first_num)
    df['wt_unit_oper_num'] = df[18].apply(first_num)
    df['wt_test_num'] = df[21].apply(first_num)
    df['material_category'] = df[22].apply(categorize_material)
    df['unit_cost_num'] = df[25].apply(first_num)
    df['project'] = df[28]
    df['year'] = df[29]
    
    geo = df.apply(
        lambda r: compute_geometry_tank(r['length_num'], r['width_num'], r['height_num'])
        if pd.notna(r['length_num']) and pd.notna(r['width_num']) and pd.notna(r['height_num']) else (None, None),
        axis=1, result_type='expand'
    )
    df['volume_m3'] = geo[0]
    df['area_m2'] = geo[1]

    return df

# =====================================================================
# 4.7 MECHANICAL (FILTERS) CONFIG & HELPERS 
# =====================================================================
FILTER_SHEET_NAME = 'Sheet1 (2)'

def clean_filter_dimensions(val):
    """Extracts numbers from messy filter dimensions like '330 (OD)' or '2,000 S/F'."""
    if pd.isna(val) or str(val).strip() == '-': return None
    s = str(val).replace(',', '')
    m = re.search(r'-?\d+\.?\d*', s)
    return float(m.group()) if m else None

@st.cache_data
def load_and_clean_filters(file):
    """Loads and cleans the filter columns via absolute index skipping messy headers."""
    raw = pd.read_excel(file, sheet_name=FILTER_SHEET_NAME, header=None, skiprows=4)
    df = raw.copy()
    
    df['equip_no'] = df[1]
    df['description'] = df[3]
    
    # Capacity handling
    df['capacity_raw'] = df[6]
    df['capacity_num'] = pd.to_numeric(df['capacity_raw'], errors='coerce')
    
    # Design Conditions
    df['design_press'] = df[10]
    df['design_temp_max'] = df[11]
    df['design_temp_min'] = df[12]
    
    # Dimensions (Col 16 is Width/ID, 17 is Height)
    df['diameter_raw'] = df[16] # Keep raw string for UI display (e.g. '330 (OD)')
    df['height_raw'] = df[17]   # Keep raw string for UI display (e.g. '1,260 T/T')
    
    df['diameter_num'] = df[16].apply(clean_filter_dimensions)
    df['height_num'] = df[17].apply(clean_filter_dimensions)
    
    # Weights and Costs
    df['wt_unit_dry_num'] = df[18].apply(first_num)
    df['wt_unit_oper_num'] = df[19].apply(first_num)
    df['wt_test_num'] = df[22].apply(first_num)
    
    df['material_category'] = df[23].apply(categorize_material)
    df['unit_cost_num'] = df[26].apply(first_num)
    
    df['project'] = df[29]
    df['year'] = df[30]

    return df.dropna(subset=['capacity_num', 'material_category', 'unit_cost_num'])

# =====================================================================
# 4.9 MECHANICAL (LAUNCHER) CONFIG & HELPERS
# =====================================================================
LAUNCHER_SHEET_NAME = 'Sheet1 (2)'

LAUNCHER_COL_MAP = {
    'equip_no':          1,    # EQUIPMENT NO. (BED)
    'description':       3,    # DESCRIPTION
    'design_press':      9,    # DESIGN CONDITIONS — PRESS. (barg)
    'design_temp_max':   10,   # DESIGN CONDITIONS — TEMP. Max (oC)
    'design_temp_min':   11,   # DESIGN CONDITIONS — TEMP. Min (oC)
    'major_length':      14,   # DIMENSIONS/UNIT — MAJOR LENGTH (mm)
    'minor_length':      15,   # DIMENSIONS/UNIT — MINOR LENGTH (mm)
    'major_id':          16,   # DIMENSIONS/UNIT — MAJOR ID (mm)
    'minor_id':          17,   # DIMENSIONS/UNIT — MINOR ID (mm)
    'wt_unit_dry':       19,   # WEIGHT — UNIT DRY (MT)
    'wt_unit_oper':      20,   # WEIGHT — UNIT OPER. (MT)
    'wt_test':           23,   # WEIGHT — TEST (MT)
    'material':          24,   # MATERIAL
    'unit_cost':         26,   # UNIT COST (MYR)
    'project':           29,   # NAME OF PROJECT
    'year':              30,   # YEAR
}

def compute_geometry_launcher(major_length_mm, major_id_mm, minor_length_mm, minor_id_mm):
    """Adds up the Major and Minor cylindrical sections into one total Volume (m3) and Area (m2)."""
    major_volume, major_area = compute_geometry(major_length_mm, major_id_mm)
    minor_volume, minor_area = compute_geometry(minor_length_mm, minor_id_mm)
    total_volume = major_volume + minor_volume
    total_area = major_area + minor_area
    return total_volume, total_area, major_volume, major_area, minor_volume, minor_area

@st.cache_data
def load_and_clean_launchers(file):
    """Loads and cleans the Launcher workbook using the column index map above."""
    raw = pd.read_excel(file, sheet_name=LAUNCHER_SHEET_NAME, header=None, skiprows=4)
    df = raw.copy()

    df['equip_no'] = df[LAUNCHER_COL_MAP['equip_no']]
    df['description'] = df[LAUNCHER_COL_MAP['description']]

    df['design_press_num'] = df[LAUNCHER_COL_MAP['design_press']].apply(first_num)
    df['design_temp_max_num'] = df[LAUNCHER_COL_MAP['design_temp_max']].apply(first_num)
    df['design_temp_min_num'] = df[LAUNCHER_COL_MAP['design_temp_min']].apply(first_num)

    df['major_length_num'] = df[LAUNCHER_COL_MAP['major_length']].apply(first_num)
    df['major_id_num'] = df[LAUNCHER_COL_MAP['major_id']].apply(first_num)
    df['minor_length_num'] = df[LAUNCHER_COL_MAP['minor_length']].apply(first_num)
    df['minor_id_num'] = df[LAUNCHER_COL_MAP['minor_id']].apply(first_num)

    df['wt_unit_dry_num'] = df[LAUNCHER_COL_MAP['wt_unit_dry']].apply(first_num)
    df['wt_unit_oper_num'] = df[LAUNCHER_COL_MAP['wt_unit_oper']].apply(first_num)
    df['wt_test_num'] = df[LAUNCHER_COL_MAP['wt_test']].apply(first_num)

    df['material_category'] = df[LAUNCHER_COL_MAP['material']].apply(categorize_material)
    df['unit_cost_num'] = df[LAUNCHER_COL_MAP['unit_cost']].apply(first_num)

    df['project'] = df[LAUNCHER_COL_MAP['project']]
    df['year'] = df[LAUNCHER_COL_MAP['year']]

    # Calculate combined Major + Minor volume/area for every historical row
    geo = df.apply(
        lambda r: compute_geometry_launcher(
            r['major_length_num'], r['major_id_num'], r['minor_length_num'], r['minor_id_num']
        ) if pd.notna(r['major_length_num']) and pd.notna(r['major_id_num'])
           and pd.notna(r['minor_length_num']) and pd.notna(r['minor_id_num']) else (None, None, None, None, None, None),
        axis=1, result_type='expand'
    )
    df['volume_m3'] = geo[0]
    df['area_m2'] = geo[1]
    df['major_volume_m3'] = geo[2]
    df['major_area_m2'] = geo[3]
    df['minor_volume_m3'] = geo[4]
    df['minor_area_m2'] = geo[5]

    # Require essential fields for interpolation
    req_cols = ['major_length_num', 'major_id_num', 'wt_unit_dry_num', 'unit_cost_num', 'volume_m3', 'area_m2']
    return df.dropna(subset=req_cols)

# =====================================================================
# 5. SHARED MATH: LINEAR INTERPOLATION
# The core prediction brain used by both Mechanical and Piping.
# =====================================================================
def interpolate_by_axis(two_refs: pd.DataFrame, query_axis_val: float, target_col: str, axis_col: str):
    """
    Draws a straight mathematical line between 2 known historical points, 
    then finds exactly where your new target sits on that line.
    """
    # Sort the two reference vessels from smallest to largest
    refs_sorted = two_refs.sort_values(axis_col).reset_index(drop=True)
    
    # Grab the X values (e.g., Area) and Y values (e.g., Cost) for both historical vessels
    a_val, b_val = refs_sorted[axis_col].iloc[0], refs_sorted[axis_col].iloc[1]
    t_a, t_b = refs_sorted[target_col].iloc[0], refs_sorted[target_col].iloc[1]
    
    # Safety check: if both historical vessels are the exact same size, just average their costs
    if b_val == a_val:
        return (t_a + t_b) / 2.0, 0.5
        
    # Calculate the exact percentage distance the new vessel is between the two historical ones
    frac = (query_axis_val - a_val) / (b_val - a_val)
    
    # Apply that percentage to the Cost or Weight to get the final prediction
    value = t_a + frac * (t_b - t_a)
    return value, frac

# =====================================================================
# 6. PAGE BUILDER: MECHANICAL (VESSELS)
# =====================================================================
def mechanical_page():
    st.title("⚙️ Vessel Weight & Cost Predictor")
    st.caption("Computes Volume & Area... (Operating Weight dynamically calculated by Density. Cost interpolated by Area/Weight.)")

    # Creates a file upload button in the sidebar
    uploaded = st.file_uploader("Upload vessel database (.xlsx)", type=["xlsx"], key="mech_up")

    # Stop the app until a file is uploaded
    if uploaded is None:
        st.info(f"Upload the vessel database Excel file to begin. Expects sheet '{SHEET_NAME}'.")
        st.stop()

    # Load data
    try:
        with st.spinner("Reading and cleaning the workbook..."):
            df = load_and_clean(uploaded)
    except ValueError:
        st.error(f"Couldn't find sheet '{SHEET_NAME}'. Check sheet name and try again.")
        st.stop()

    st.success(f"Loaded {len(df)} equipment rows from '{SHEET_NAME}'.")

    with st.expander("Preview cleaned data"):
        st.dataframe(df[DISPLAY_COLS], use_container_width=True)

    # Filter out rows that don't have enough data to be useful
    clean = df.dropna(subset=REQUIRED_COLS + ['volume_m3', 'area_m2']).copy()
    if len(clean) < 2:
        st.warning("Need at least 2 complete historical rows to interpolate between.")
        st.stop()

    st.divider()
    st.subheader("Predict specs & cost for a new vessel")

    # 6A. THE INPUT FORM
    material_options = [AUTO_OPTION] + sorted(clean['material_category'].astype(str).unique())
    orientation_options = [AUTO_OPTION] + sorted(clean['orientation_clean'].astype(str).unique())

    with st.form("prediction_form"):
        st.markdown("**Dimensions** (drives shape matching and geometry calculation)")
        g1, g2 = st.columns(2)
        with g1: user_length = st.number_input("Length T/T (mm)", min_value=1.0, value=6000.0, step=100.0)
        with g2: user_diameter = st.number_input("Internal diameter (mm)", min_value=1.0, value=2000.0, step=50.0)

        st.markdown("**Costing Material Selection**")
        cost_mat = st.radio("Select Vessel Metallurgy for Costing", options=["Carbon Steel (CS)", "Stainless Steel (SS)"], horizontal=True)

        st.markdown("**Design conditions & Content** (informational / design checks)")
        d1, d2, d3, d4 = st.columns(4)
        with d1: user_press = st.number_input("Design pressure (barg)", value=20.0, step=1.0)
        with d2: user_temp_max = st.number_input("Design temperature — Max (°C)", value=100.0, step=5.0)
        with d3: user_temp_min = st.number_input("Design temperature — Min (°C)", value=0.0, step=5.0)
        with d4: content_density = st.number_input("Content Density (kg/m³)", value=1000.0, step=50.0, help="e.g. Water is ~1000 kg/m³. Used to calculate Operating Weight assuming 80% full.")

        st.markdown("**Construction Filter**")
        c1, c2 = st.columns(2)
        with c1: user_material = st.selectbox("Historical Material Category (Strict Filter)", material_options)
        with c2: user_orientation = st.selectbox("Orientation (Informational)", orientation_options)

        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    # 6B. THE MATCHING LOGIC (KNN / Shape Distance)
    if submitted:
        query_volume, query_area = compute_geometry(user_length, user_diameter)
        st.session_state['mech_ctx'] = dict(
            query_volume=query_volume, 
            query_area=query_area, 
            cost_mat=cost_mat,
            content_density=content_density,
            user_material=user_material 
        )

    if 'mech_ctx' in st.session_state:
        ctx = st.session_state['mech_ctx']
        query_volume, query_area = ctx['query_volume'], ctx['query_area']
        cost_mat = ctx['cost_mat']
        content_density = ctx.get('content_density', 1000.0)
        hist_mat = ctx.get('user_material', AUTO_OPTION)

        st.markdown("**Computed Geometry**")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Total Vessel Volume", f"{query_volume:.2f} m³")
        
        content_volume = 0.8 * query_volume
        gc2.metric(
            "Content Volume (80% Full)", 
            f"{content_volume:.2f} m³", 
            help=f"Calculation: 0.8 × {query_volume:.2f} m³ (Total Volume)"
        )
        
        gc3.metric("Surface Area of Vessel", f"{query_area:.2f} m²")

        # ==========================================================
        # NEW LOGIC: Filter Historical Data by User Material Selection
        # ==========================================================
        working_df = clean.copy()
        if hist_mat != AUTO_OPTION:
            working_df = working_df[working_df['material_category'] == hist_mat].copy()
            if len(working_df) < 2:
                st.error(f"Not enough historical data found for material '{hist_mat}'. We need at least 2 references to interpolate. Please select '{AUTO_OPTION}' or a different material.")
                st.stop()

        vol_std = working_df['volume_m3'].std() or 1.0
        area_std = working_df['area_m2'].std() or 1.0
        
        ranked = working_df.copy()
        ranked['distance'] = np.sqrt(
            ((ranked['volume_m3'] - query_volume) / vol_std) ** 2
            + ((ranked['area_m2'] - query_area) / area_std) ** 2
        )
        ranked = ranked.sort_values('distance').reset_index(drop=True)

        best_idx_1 = 0  
        best_idx_2 = 1  
        
        if len(ranked) > 1:
            vol_1 = ranked.iloc[best_idx_1]['volume_m3']
            for i in range(1, len(ranked)):
                vol_i = ranked.iloc[i]['volume_m3']
                if (vol_1 <= query_volume <= vol_i) or (vol_i <= query_volume <= vol_1):
                    best_idx_2 = i
                    break  

        default_selections = [best_idx_1, best_idx_2] if len(ranked) > 1 else [0]

        # FIX: Force Streamlit to overwrite cached selections 
        if submitted:
            st.session_state["mech_refs"] = default_selections

        def _label(i):
            r = ranked.iloc[i]
            return (f"{r['equip_no']} — {r['material_category']} — Vol {r['volume_m3']:.2f} m³, Area {r['area_m2']:.2f} m² (Dist {r['distance']:.3f})")

        st.markdown("**Reference vessels for interpolation**")
        selected_idx = st.multiselect(
            f"Choose exactly 2 reference vessels (Filtered by: {hist_mat}):", 
            options=list(range(len(ranked))), 
            default=default_selections, 
            format_func=_label, 
            max_selections=2,
            key="mech_refs"
        )

        if len(selected_idx) != 2:
            st.info("Select exactly 2 reference vessels above to view predictions.")
            st.stop()

        two_refs = ranked.iloc[selected_idx].copy()
        
        v_min, v_max = two_refs['volume_m3'].min(), two_refs['volume_m3'].max()
        a_min, a_max = two_refs['area_m2'].min(), two_refs['area_m2'].max()
        is_extrapolating_area = not (a_min <= query_area <= a_max)

        if (v_min <= query_volume <= v_max) and not is_extrapolating_area:
            st.success("Query falls inside the reference bounds for both Volume and Area.")
        else:
            st.warning("Query falls outside reference ranges for Volume or Area; output may reflect linear extrapolation.")

        # 6C. CALCULATING THE PREDICTIONS
        final_weights = {}
        for target in ['wt_unit_dry_num', 'wt_test_num']:
            val, _ = interpolate_by_axis(two_refs, query_volume, target, 'volume_m3')
            final_weights[target] = val
            
        query_weight = final_weights['wt_unit_dry_num']

        content_weight_kg = content_density * (0.8 * query_volume)
        content_weight_mt = content_weight_kg / 1000.0
        final_weights['wt_unit_oper_num'] = query_weight + content_weight_mt

        base_cost_area, _ = interpolate_by_axis(two_refs, query_area, 'unit_cost_num', 'area_m2')
        base_cost_weight, _ = interpolate_by_axis(two_refs, query_weight, 'unit_cost_num', 'wt_unit_dry_num')
        
        cost_multiplier = 4.0 if "Stainless Steel" in cost_mat else 1.0
        final_unit_cost_area = max(0.0, base_cost_area * cost_multiplier) 
        final_unit_cost_weight = max(0.0, base_cost_weight * cost_multiplier)

        # 6D. DISPLAYING THE RESULTS
        st.markdown("**Predicted Weight Breakdown**")
        w1, w2, w3 = st.columns(3)
        w1.metric("Unit Dry (Interpolated)", f"{final_weights['wt_unit_dry_num']:.2f} MT")
        w2.metric("Unit Operating (Calculated)", f"{final_weights['wt_unit_oper_num']:.2f} MT", 
                  help=f"Dry Wt ({final_weights['wt_unit_dry_num']:.2f} MT) + Content ({content_weight_mt:.2f} MT)")
        w3.metric("Test Weight (Interpolated)", f"{final_weights['wt_test_num']:.2f} MT")

        st.markdown("**Predicted Equipment Cost Comparison**")
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Final Cost (Based on Area)", f"MYR {final_unit_cost_area:,.2f}")
        c_col2.metric("Final Cost (Based on Weight)", f"MYR {final_unit_cost_weight:,.2f}")
        c_col3.metric("Material Factor Applied", f"{cost_multiplier:.1f}x ({cost_mat})")

        # 6E. DRAWING THE GRAPHS
        st.markdown("---")
        st.markdown("**Interpolation Visualizations: Area vs. Dry Weight**")
        
        y_refs = two_refs['unit_cost_num'].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x_refs_area = two_refs['area_m2'].values
        x_min_area = min(x_refs_area[0], x_refs_area[1], query_area) * 0.98
        x_max_area = max(x_refs_area[0], x_refs_area[1], query_area) * 1.02
        if x_refs_area[1] != x_refs_area[0]:
            slope_area = (y_refs[1] - y_refs[0]) / (x_refs_area[1] - x_refs_area[0])
            ax1.plot(np.linspace(x_min_area, x_max_area, 10), slope_area * np.linspace(x_min_area, x_max_area, 10) + (y_refs[0] - slope_area * x_refs_area[0]), color='gray', linestyle='--', alpha=0.7)
            
        ax1.scatter(x_refs_area, y_refs, color='blue', s=100, zorder=5, label='References') 
        ax1.scatter([query_area], [base_cost_area], color='red', s=250, marker='*', zorder=6, label='New Vessel') 
        ax1.set_xlabel("Surface Area (m²)", fontweight='bold')
        ax1.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax1.set_title("Cost Interpolation by Area")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax1.legend()

        x_refs_wt = two_refs['wt_unit_dry_num'].values
        x_min_wt = min(x_refs_wt[0], x_refs_wt[1], query_weight) * 0.98
        x_max_wt = max(x_refs_wt[0], x_refs_wt[1], query_weight) * 1.02
        if x_refs_wt[1] != x_refs_wt[0]:
            slope_wt = (y_refs[1] - y_refs[0]) / (x_refs_wt[1] - x_refs_wt[0])
            
            # ---> FIX APPLIED HERE: Changed x_min_w to x_min_wt and x_max_w to x_max_wt
            ax2.plot(np.linspace(x_min_wt, x_max_wt, 10), slope_wt * np.linspace(x_min_wt, x_max_wt, 10) + (y_refs[0] - slope_wt * x_refs_wt[0]), color='gray', linestyle='--', alpha=0.7)
            
        ax2.scatter(x_refs_wt, y_refs, color='blue', s=100, zorder=5, label='References')
        ax2.scatter([query_weight], [base_cost_weight], color='red', s=250, marker='*', zorder=6, label='New Vessel')
        ax2.set_xlabel("Dry Weight (MT)", fontweight='bold')
        ax2.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax2.set_title("Cost Interpolation by Dry Weight")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax2.legend()
        
        plt.tight_layout() 
        st.pyplot(fig) 

        with st.expander("See calculation details and reference comparisons"):
            v_ref_a, v_ref_b = two_refs.iloc[0], two_refs.iloc[1]
            proj_a = v_ref_a['project'] if pd.notna(v_ref_a['project']) else "Unknown Project"
            proj_b = v_ref_b['project'] if pd.notna(v_ref_b['project']) else "Unknown Project"
            
            detail_data = {
                "Metric": ["Area Basis", "Vol Basis", "Dry Wt (MT)", "Oper Wt (MT) - Calculated", "Test Wt (MT)", "Hist. Cost (MYR)", "Final Cost - Area (MYR)", "Final Cost - Wt (MYR)"],
                f"Ref 1 ({v_ref_a['equip_no']} - {proj_a})": [
                    f"{v_ref_a['area_m2']:.1f}", f"{v_ref_a['volume_m3']:.1f}", f"{v_ref_a['wt_unit_dry_num']:.2f}",
                    f"{v_ref_a['wt_unit_oper_num']:.2f}", f"{v_ref_a['wt_test_num']:.2f}", f"{v_ref_a['unit_cost_num']:,.0f}", "-", "-"
                ],
                f"Ref 2 ({v_ref_b['equip_no']} - {proj_b})": [
                    f"{v_ref_b['area_m2']:.1f}", f"{v_ref_b['volume_m3']:.1f}", f"{v_ref_b['wt_unit_dry_num']:.2f}",
                    f"{v_ref_b['wt_unit_oper_num']:.2f}", f"{v_ref_b['wt_test_num']:.2f}", f"{v_ref_b['unit_cost_num']:,.0f}", "-", "-"
                ],
                "Interpolated Output": [
                    f"{query_area:.1f}", f"{query_volume:.1f}", f"{final_weights['wt_unit_dry_num']:.2f}",
                    f"{final_weights['wt_unit_oper_num']:.2f}", f"{final_weights['wt_test_num']:.2f}", "-",
                    f"{final_unit_cost_area:,.2f}", f"{final_unit_cost_weight:,.2f}"
                ]
            }
            st.table(pd.DataFrame(detail_data))

        st.markdown("---")
        st.markdown("### 📋 Historical Project & Reference Details")
        st.caption("Detailed database records for the reference vessels used in this calculation.")
        
        final_display_cols = DISPLAY_COLS + ['distance']
        st.dataframe(two_refs[final_display_cols], use_container_width=True)

# =====================================================================
# 7. PAGE BUILDER: PIPING (VALVES)
# =====================================================================
def piping_page():
    st.title("🔧 Valve Weight & Cost Predictor")
    st.caption("Predicts valve weight and cost by filtering historical data for exact matches...")

    uploaded = st.file_uploader("Upload valve database (.xlsx)", type=["xlsx"], key="valve_up")

    if uploaded is None:
        st.info(f"Upload the valve database Excel file to begin. Expects sheet '{VALVE_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the piping workbook..."):
            df = load_and_clean_valves(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    with st.expander("Preview cleaned data"):
        display_cols = ['item_no', 'type_clean', 'rating_clean', 'material_clean', 'size_num', 'weight_num', 'cost_num']
        st.dataframe(df[display_cols].head(100), use_container_width=True)

    st.divider()
    st.subheader("Predict specs & cost for a new valve")

    valve_types = sorted(df['type_clean'].unique())
    valve_materials = sorted(df['material_clean'].unique())
    valve_ratings = sorted(df['rating_clean'].unique())

    with st.form("valve_prediction_form"):
        st.markdown("**Valve Specifications** (used for exact matching)")
        c1, c2, c3 = st.columns(3)
        with c1: user_type = st.selectbox("Valve Type", valve_types)
        with c2: user_rating = st.selectbox("Pressure Rating", valve_ratings)
        with c3: user_material = st.selectbox("Material Specification", valve_materials)
        
        st.markdown("**Target Dimension**")
        user_size = st.number_input("Nominal Size (mm)", min_value=15.0, value=100.0, step=25.0)
            
        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    if submitted:
        filtered = df[
            (df['type_clean'] == user_type) & 
            (df['rating_clean'] == user_rating) & 
            (df['material_clean'] == user_material)
        ].copy()

        if len(filtered) == 0:
            st.error(f"No historical data found for {user_type} valves in {user_material} at Rating {user_rating}.")
            st.stop()

        filtered['size_diff'] = abs(filtered['size_num'] - user_size)
        filtered = filtered.sort_values(['size_diff', 'year'], ascending=[True, False])
        unique_sizes = filtered.drop_duplicates(subset=['size_num']).reset_index(drop=True)
        
        if len(unique_sizes) >= 2:
            best_idx_1, best_idx_2 = 0, 1
            size_1 = unique_sizes.iloc[best_idx_1]['size_num']
            for i in range(1, len(unique_sizes)):
                size_i = unique_sizes.iloc[i]['size_num']
                if (size_1 <= user_size <= size_i) or (size_i <= user_size <= size_1):
                    best_idx_2 = i
                    break
            closest_sizes = unique_sizes.iloc[[best_idx_1, best_idx_2]]
        else:
            closest_sizes = unique_sizes.head(2)

        if len(closest_sizes) < 2:
            st.warning("Only found 1 distinct size for these specs. Extrapolation is impossible; displaying exact historical match instead.")
            final_cost = closest_sizes.iloc[0]['cost_num']
            final_weight = closest_sizes.iloc[0]['weight_num']
            is_interpolated = False
        else:
            final_cost, _ = interpolate_by_axis(closest_sizes, user_size, 'cost_num', 'size_num')
            if closest_sizes['weight_num'].isna().any():
                final_weight = None
            else:
                final_weight, _ = interpolate_by_axis(closest_sizes, user_size, 'weight_num', 'size_num')
            is_interpolated = True

        st.markdown("**Predicted Output**")
        out1, out2, out3 = st.columns(3)
        out1.metric("Predicted Cost", f"RM {max(0.0, final_cost):,.2f}")
        out2.metric("Predicted Weight", f"{final_weight:.1f} kg" if pd.notna(final_weight) else "N/A")
        out3.metric("Calculation Method", "Interpolation" if is_interpolated else "Direct Historical Match")

        if is_interpolated:
            st.markdown("---")
            st.markdown("**Interpolation Visualization (Size vs. Cost)**")
            
            refs_sorted = closest_sizes.sort_values('size_num')
            x_refs = refs_sorted['size_num'].values
            y_refs = refs_sorted['cost_num'].values

            fig, ax = plt.subplots(figsize=(10, 5))
            
            x_min_plot = min(x_refs[0], x_refs[1], user_size) * 0.8
            x_max_plot = max(x_refs[0], x_refs[1], user_size) * 1.2
            
            if x_refs[1] != x_refs[0]:
                slope = (y_refs[1] - y_refs[0]) / (x_refs[1] - x_refs[0])
                intercept = y_refs[0] - slope * x_refs[0]
                x_line = np.linspace(x_min_plot, x_max_plot, 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='gray', linestyle='--', alpha=0.7, label='Interpolation Trendline')
            
            ax.scatter(x_refs, y_refs, color='blue', s=100, zorder=5, label='Historical References')
            ax.scatter([user_size], [final_cost], color='red', s=250, marker='*', zorder=6, label='Predicted Cost')
            
            ax.set_xlabel("Nominal Size (mm)", fontweight='bold')
            ax.set_ylabel("Unit Cost (RM)", fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.legend()
            st.pyplot(fig)
            
        st.markdown("---")
        st.markdown("### 📋 Historical Project & Reference Details")
        st.caption("Detailed database records for the reference valves used in this calculation.")
        
        display_cols_piping = ['item_no', 'project', 'year', 'valve_type', 'material_clean', 'rating_clean', 'size_num', 'weight_num', 'cost_num']
        st.dataframe(closest_sizes[display_cols_piping], use_container_width=True)

# =====================================================================
# 7.5 PAGE BUILDER: TANKS
# =====================================================================
def tank_page():
    st.title("🛢️ Tank Weight & Cost Predictor")
    st.caption("Computes Rectangular Volume & Area... (Operating Weight dynamically calculated by Density. Cost interpolated strictly by Area/Weight using historical matches.)")

    uploaded = st.file_uploader("Upload tank database (.xlsx)", type=["xlsx"], key="tank_up")

    if uploaded is None:
        st.info(f"Upload the tank database Excel file to begin. Expects sheet '{TANK_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the workbook..."):
            df = load_and_clean_tanks(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    tank_required_cols = ['length_num', 'width_num', 'height_num', 'wt_unit_dry_num', 'wt_test_num', 'unit_cost_num']
    tank_display_cols = ['equip_no', 'description', 'length_num', 'width_num', 'height_num', 
                         'volume_m3', 'area_m2', 'material_category', 
                         'wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num', 'unit_cost_num', 'project', 'year']

    with st.expander("Preview cleaned data"):
        st.dataframe(df[tank_display_cols], use_container_width=True)

    clean = df.dropna(subset=tank_required_cols + ['volume_m3', 'area_m2']).copy()
    if len(clean) < 2:
        st.warning("Need at least 2 complete historical rows to interpolate between.")
        st.stop()

    st.divider()
    st.subheader("Predict specs & cost for a new tank")

    material_options = [AUTO_OPTION] + sorted(clean['material_category'].astype(str).unique())

    with st.form("tank_prediction_form"):
        st.markdown("**Dimensions (Rectangular Prisms)** (drives shape matching and geometry calculation)")
        g1, g2, g3 = st.columns(3)
        with g1: user_length = st.number_input("Length (mm)", min_value=1.0, value=3000.0, step=100.0)
        with g2: user_width = st.number_input("Width (mm)", min_value=1.0, value=2000.0, step=100.0)
        with g3: user_height = st.number_input("Height (mm)", min_value=1.0, value=2000.0, step=100.0)

        st.markdown("**Material & Content**")
        c1, c2 = st.columns(2)
        with c1: user_material = st.selectbox("Material Specification (Strict Filter)", material_options, key="tank_hist_mat")
        with c2: content_density = st.number_input("Content Density (kg/m³)", value=1000.0, step=50.0, help="e.g. Water is ~1000 kg/m³. Used to calculate Operating Weight assuming 80% full.")

        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    if submitted:
        query_volume, query_area = compute_geometry_tank(user_length, user_width, user_height)
        st.session_state['tank_ctx'] = dict(
            query_volume=query_volume, 
            query_area=query_area, 
            content_density=content_density,
            user_material=user_material 
        )

    if 'tank_ctx' in st.session_state:
        ctx = st.session_state['tank_ctx']
        query_volume, query_area = ctx['query_volume'], ctx['query_area']
        content_density = ctx.get('content_density', 1000.0)
        hist_mat = ctx.get('user_material', AUTO_OPTION)

        st.markdown("**Computed Geometry**")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Total Tank Volume", f"{query_volume:.2f} m³")
        gc2.metric("Content Volume (80% Full)", f"{0.8 * query_volume:.2f} m³")
        gc3.metric("Surface Area of Tank", f"{query_area:.2f} m²")

        working_df = clean.copy()
        if hist_mat != AUTO_OPTION:
            working_df = working_df[working_df['material_category'] == hist_mat].copy()
            if len(working_df) < 2:
                st.error(f"Not enough historical data found for material '{hist_mat}'. We need at least 2 references to interpolate. Please select '{AUTO_OPTION}' or a different material.")
                st.stop()

        vol_std, area_std = working_df['volume_m3'].std() or 1.0, working_df['area_m2'].std() or 1.0
        
        ranked = working_df.copy()
        ranked['distance'] = np.sqrt(((ranked['volume_m3'] - query_volume) / vol_std) ** 2 + ((ranked['area_m2'] - query_area) / area_std) ** 2)
        ranked = ranked.sort_values('distance').reset_index(drop=True)

        best_idx_1, best_idx_2 = 0, 1
        if len(ranked) > 1:
            vol_1 = ranked.iloc[best_idx_1]['volume_m3']
            for i in range(1, len(ranked)):
                vol_i = ranked.iloc[i]['volume_m3']
                if (vol_1 <= query_volume <= vol_i) or (vol_i <= query_volume <= vol_1):
                    best_idx_2 = i
                    break

        default_selections = [best_idx_1, best_idx_2] if len(ranked) > 1 else [0]
        
        if submitted:
            st.session_state["tank_refs"] = default_selections

        selected_idx = st.multiselect(
            f"Choose exactly 2 reference tanks (Filtered by: {hist_mat}):", 
            options=list(range(len(ranked))), 
            default=default_selections, 
            format_func=lambda i: f"{ranked.iloc[i]['equip_no']} — {ranked.iloc[i]['material_category']} — Vol {ranked.iloc[i]['volume_m3']:.2f} m³, Area {ranked.iloc[i]['area_m2']:.2f} m² (Dist {ranked.iloc[i]['distance']:.3f})", 
            max_selections=2, 
            key="tank_refs"
        )

        if len(selected_idx) != 2:
            st.info("Select exactly 2 reference tanks above to view predictions.")
            st.stop()

        two_refs = ranked.iloc[selected_idx].copy()
        v_min, v_max = two_refs['volume_m3'].min(), two_refs['volume_m3'].max()
        a_min, a_max = two_refs['area_m2'].min(), two_refs['area_m2'].max()

        if (v_min <= query_volume <= v_max) and (a_min <= query_area <= a_max):
            st.success("Query falls inside the reference bounds for both Volume and Area.")
        else:
            st.warning("Query falls outside reference ranges for Volume or Area; output may reflect linear extrapolation.")

        final_weights = {}
        for target in ['wt_unit_dry_num', 'wt_test_num']:
            final_weights[target], _ = interpolate_by_axis(two_refs, query_volume, target, 'volume_m3')
            
        query_weight = final_weights['wt_unit_dry_num']
        content_weight_mt = (content_density * (0.8 * query_volume)) / 1000.0
        final_weights['wt_unit_oper_num'] = query_weight + content_weight_mt

        base_cost_area, _ = interpolate_by_axis(two_refs, query_area, 'unit_cost_num', 'area_m2')
        base_cost_weight, _ = interpolate_by_axis(two_refs, query_weight, 'unit_cost_num', 'wt_unit_dry_num')
        
        final_unit_cost_area = max(0.0, base_cost_area)
        final_unit_cost_weight = max(0.0, base_cost_weight)

        st.markdown("**Predicted Weight Breakdown**")
        w1, w2, w3 = st.columns(3)
        w1.metric("Unit Dry (Interpolated)", f"{final_weights['wt_unit_dry_num']:.2f} MT")
        w2.metric("Unit Operating (Calculated)", f"{final_weights['wt_unit_oper_num']:.2f} MT", help=f"Dry Wt + Content ({content_weight_mt:.2f} MT)")
        w3.metric("Test Weight (Interpolated)", f"{final_weights['wt_test_num']:.2f} MT")

        st.markdown("**Predicted Equipment Cost Comparison**")
        c_col1, c_col2 = st.columns(2)
        c_col1.metric("Final Cost (Based on Area)", f"MYR {final_unit_cost_area:,.2f}")
        c_col2.metric("Final Cost (Based on Weight)", f"MYR {final_unit_cost_weight:,.2f}")

        st.markdown("---")
        st.markdown("**Interpolation Visualizations: Area vs. Dry Weight**")
        y_refs = two_refs['unit_cost_num'].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x_refs_area = two_refs['area_m2'].values
        x_min_a, x_max_a = min(x_refs_area[0], x_refs_area[1], query_area) * 0.98, max(x_refs_area[0], x_refs_area[1], query_area) * 1.02
        if x_refs_area[1] != x_refs_area[0]:
            slope_area = (y_refs[1] - y_refs[0]) / (x_refs_area[1] - x_refs_area[0])
            ax1.plot(np.linspace(x_min_a, x_max_a, 10), slope_area * np.linspace(x_min_a, x_max_a, 10) + (y_refs[0] - slope_area * x_refs_area[0]), color='gray', linestyle='--', alpha=0.7)
        ax1.scatter(x_refs_area, y_refs, color='blue', s=100, zorder=5, label='References')
        ax1.scatter([query_area], [base_cost_area], color='red', s=250, marker='*', zorder=6, label='New Tank')
        ax1.set_xlabel("Surface Area (m²)", fontweight='bold')
        ax1.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax1.set_title("Cost Interpolation by Area")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax1.legend()

        x_refs_wt = two_refs['wt_unit_dry_num'].values
        x_min_w, x_max_w = min(x_refs_wt[0], x_refs_wt[1], query_weight) * 0.98, max(x_refs_wt[0], x_refs_wt[1], query_weight) * 1.02
        if x_refs_wt[1] != x_refs_wt[0]:
            slope_wt = (y_refs[1] - y_refs[0]) / (x_refs_wt[1] - x_refs_wt[0])
            ax2.plot(np.linspace(x_min_w, x_max_w, 10), slope_wt * np.linspace(x_min_w, x_max_w, 10) + (y_refs[0] - slope_wt * x_refs_wt[0]), color='gray', linestyle='--', alpha=0.7)
        ax2.scatter(x_refs_wt, y_refs, color='blue', s=100, zorder=5, label='References')
        ax2.scatter([query_weight], [base_cost_weight], color='red', s=250, marker='*', zorder=6, label='New Tank')
        ax2.set_xlabel("Dry Weight (MT)", fontweight='bold')
        ax2.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax2.set_title("Cost Interpolation by Dry Weight")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📋 Historical Project & Reference Details")
        st.dataframe(two_refs[tank_display_cols + ['distance']], use_container_width=True)

# =====================================================================
# 7.7 PAGE BUILDER: MECHANICAL (FILTERS)
# =====================================================================
def filter_page():
    st.title("Filter Spec & Cost Lookup")
    st.caption("Strict historical lookup. Matches based on Material and closest matching (rounded up) Capacity. No interpolation.")

    uploaded = st.file_uploader("Upload filter database (.xlsx)", type=["xlsx"], key="filter_up")

    if uploaded is None:
        st.info(f"Upload the filter database Excel file to begin. Expects sheet '{FILTER_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the filter workbook..."):
            df = load_and_clean_filters(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    display_cols = ['equip_no', 'description', 'capacity_num', 'material_category', 
                    'design_press', 'design_temp_max', 'design_temp_min',
                    'diameter_raw', 'height_raw', 'wt_unit_dry_num', 'wt_unit_oper_num', 
                    'wt_test_num', 'unit_cost_num', 'project', 'year']

    with st.expander("Preview cleaned data"):
        st.dataframe(df[display_cols], use_container_width=True)

    st.divider()
    st.subheader("Lookup specs & cost for a new filter")

    material_options = sorted(df['material_category'].astype(str).unique())
    
    # Inject 'SS' as a placeholder if it doesn't exist in the historical data
    if 'SS' not in material_options:
        material_options.append('SS')
        material_options = sorted(material_options)

    with st.form("filter_prediction_form"):
        st.markdown("**Search Parameters**")
        c1, c2 = st.columns(2)
        with c1: user_capacity = st.number_input("Selection Capacity (m³)", min_value=1.0, value=10.0, step=1.0)
        with c2: user_material = st.selectbox("Selection Material", material_options)

        submitted = st.form_submit_button("Lookup Existing Data", use_container_width=True, type="primary")

    if submitted:
        # Special check for SS placeholder
        if user_material == 'SS' and len(df[df['material_category'] == 'SS']) == 0:
            st.warning("⚠️ Stainless Steel (SS) filters are currently unavailable in the historical database. Please select another material.")
            st.stop()
            
        # Step 1: Filter by exact material
        mat_filtered = df[df['material_category'] == user_material].copy()
        
        if len(mat_filtered) == 0:
            st.error(f"No historical data found for '{user_material}' filters.")
            st.stop()
            
        # Step 2: Capacity Round-up Logic
        valid_capacities = mat_filtered[mat_filtered['capacity_num'] >= user_capacity]
        
        if len(valid_capacities) == 0:
            st.warning(f"Requested capacity ({user_capacity} m³) exceeds our maximum historical data for {user_material}. Displaying the largest available matching filters.")
            max_cap = mat_filtered['capacity_num'].max()
            final_matches = mat_filtered[mat_filtered['capacity_num'] == max_cap]
        else:
            target_capacity = valid_capacities['capacity_num'].min()
            final_matches = mat_filtered[mat_filtered['capacity_num'] == target_capacity]

            if target_capacity > user_capacity:
                st.info(f"Requested capacity: **{user_capacity} m³**. Rounded up to nearest available historical size: **{target_capacity} m³**.")
            else:
                st.success(f"Found exact capacity match for **{target_capacity} m³**.")

        # Display the results
        st.markdown("### 🎯 Exact Historical Matches")
        
        res_df = final_matches[display_cols].copy()
        res_df.columns = ['Equip No.', 'Description', 'Capacity (m³)', 'Material', 
                          'Design Press. (barg)', 'Design Temp Max (°C)', 'Design Temp Min (°C)',
                          'Diameter/Width', 'Height', 'Dry Wt (MT)', 'Oper. Wt (MT)', 
                          'Test Wt (MT)', 'Unit Cost (MYR)', 'Project', 'Year']
        
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        
        if len(final_matches) > 1:
            st.markdown(f"*Note: Found {len(final_matches)} historical filters with identical capacity and material. Averages shown below.*")
            
        avg_cost = final_matches['unit_cost_num'].mean()
        avg_dry = final_matches['wt_unit_dry_num'].mean()
        avg_oper = final_matches['wt_unit_oper_num'].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Historical Cost", f"MYR {avg_cost:,.2f}")
        c2.metric("Average Dry Weight", f"{avg_dry:,.2f} MT")
        c3.metric("Average Operating Weight", f"{avg_oper:,.2f} MT")



# =====================================================================
# 7.9 PAGE BUILDER: MECHANICAL (LAUNCHER)
# =====================================================================
def launcher_page():
    st.title("Launcher/Receiver Weight & Cost Predictor")
    st.caption("Computes total Volume & Area from Major + Minor cylindrical sections. "
               "Design conditions are informational. Weights & Cost interpolated from 2 chosen historical references.")

    uploaded = st.file_uploader("Upload launcher database (.xlsx)", type=["xlsx"], key="launcher_up")

    if uploaded is None:
        st.info(f"Upload the launcher database Excel file to begin. Expects sheet '{LAUNCHER_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the workbook..."):
            df = load_and_clean_launchers(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    launcher_required_cols = ['major_length_num', 'major_id_num', 'minor_length_num', 'minor_id_num',
                               'wt_unit_dry_num', 'wt_test_num', 'unit_cost_num']
    launcher_display_cols = ['equip_no', 'description', 'design_press_num', 'design_temp_max_num', 'design_temp_min_num',
                              'major_length_num', 'major_id_num', 'minor_length_num', 'minor_id_num',
                              'volume_m3', 'area_m2', 'material_category',
                              'wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num', 'unit_cost_num',
                              'project', 'year']

    with st.expander("Preview cleaned data"):
        st.dataframe(df[launcher_display_cols], use_container_width=True)

    clean = df.dropna(subset=launcher_required_cols + ['volume_m3', 'area_m2']).copy()
    if len(clean) < 2:
        st.warning("Need at least 2 complete historical rows to interpolate between.")
        st.stop()

    st.divider()
    st.subheader("Predict specs & cost for a new launcher")

    material_options = [AUTO_OPTION] + sorted(clean['material_category'].astype(str).unique())

    with st.form("launcher_prediction_form"):
        st.markdown("**Design Conditions**")
        d1, d2, d3 = st.columns(3)
        with d1: user_press = st.number_input("Design pressure (barg)", value=20.0, step=1.0)
        with d2: user_temp_max = st.number_input("Design temperature — Max (°C)", value=100.0, step=5.0)
        with d3: user_temp_min = st.number_input("Design temperature — Min (°C)", value=0.0, step=5.0)

        st.markdown("**Dimensions — Major Section** (drives shape matching and geometry calculation)")
        g1, g2 = st.columns(2)
        with g1: user_major_length = st.number_input("Major length (mm)", min_value=1.0, value=6000.0, step=100.0)
        with g2: user_major_id = st.number_input("Major internal diameter (mm)", min_value=1.0, value=900.0, step=50.0)

        st.markdown("**Dimensions — Minor Section**")
        g3, g4 = st.columns(2)
        with g3: user_minor_length = st.number_input("Minor length (mm)", min_value=1.0, value=2000.0, step=100.0)
        with g4: user_minor_id = st.number_input("Minor internal diameter (mm)", min_value=1.0, value=400.0, step=50.0)

        st.markdown("**Construction Filter**")
        user_material = st.selectbox("Historical Material Category (Strict Filter)", material_options, key="launcher_hist_mat")

        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    if submitted:
        total_volume, total_area, major_volume, major_area, minor_volume, minor_area = compute_geometry_launcher(
            user_major_length, user_major_id, user_minor_length, user_minor_id
        )
        st.session_state['launcher_ctx'] = dict(
            query_volume=total_volume,
            query_area=total_area,
            major_volume=major_volume,
            major_area=major_area,
            minor_volume=minor_volume,
            minor_area=minor_area,
            user_press=user_press,
            user_temp_max=user_temp_max,
            user_temp_min=user_temp_min,
            user_material=user_material,
        )

    if 'launcher_ctx' in st.session_state:
        ctx = st.session_state['launcher_ctx']
        query_volume, query_area = ctx['query_volume'], ctx['query_area']
        hist_mat = ctx.get('user_material', AUTO_OPTION)

        st.markdown("**Computed Geometry (Major + Minor)**")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Major Section", f"{ctx['major_volume']:.3f} m³ / {ctx['major_area']:.3f} m²")
        gc2.metric("Minor Section", f"{ctx['minor_volume']:.3f} m³ / {ctx['minor_area']:.3f} m²")
        gc3.metric("Total (Major + Minor)", f"{query_volume:.3f} m³ / {query_area:.3f} m²")

        working_df = clean.copy()
        if hist_mat != AUTO_OPTION:
            working_df = working_df[working_df['material_category'] == hist_mat].copy()
            if len(working_df) < 2:
                st.error(f"Not enough historical data found for material '{hist_mat}'. We need at least 2 references to interpolate. Please select '{AUTO_OPTION}' or a different material.")
                st.stop()

        vol_std, area_std = working_df['volume_m3'].std() or 1.0, working_df['area_m2'].std() or 1.0

        ranked = working_df.copy()
        ranked['distance'] = np.sqrt(
            ((ranked['volume_m3'] - query_volume) / vol_std) ** 2
            + ((ranked['area_m2'] - query_area) / area_std) ** 2
        )
        ranked = ranked.sort_values('distance').reset_index(drop=True)

        best_idx_1, best_idx_2 = 0, 1
        if len(ranked) > 1:
            vol_1 = ranked.iloc[best_idx_1]['volume_m3']
            for i in range(1, len(ranked)):
                vol_i = ranked.iloc[i]['volume_m3']
                if (vol_1 <= query_volume <= vol_i) or (vol_i <= query_volume <= vol_1):
                    best_idx_2 = i
                    break

        default_selections = [best_idx_1, best_idx_2] if len(ranked) > 1 else [0]

        if submitted:
            st.session_state["launcher_refs"] = default_selections

        selected_idx = st.multiselect(
            f"Choose exactly 2 reference launchers (Filtered by: {hist_mat}):",
            options=list(range(len(ranked))),
            default=default_selections,
            format_func=lambda i: (f"{ranked.iloc[i]['equip_no']} — {ranked.iloc[i]['material_category']} — "
                                    f"Vol {ranked.iloc[i]['volume_m3']:.2f} m³, Area {ranked.iloc[i]['area_m2']:.2f} m² "
                                    f"(Dist {ranked.iloc[i]['distance']:.3f})"),
            max_selections=2,
            key="launcher_refs"
        )

        if len(selected_idx) != 2:
            st.info("Select exactly 2 reference launchers above to view predictions.")
            st.stop()

        two_refs = ranked.iloc[selected_idx].copy()
        v_min, v_max = two_refs['volume_m3'].min(), two_refs['volume_m3'].max()
        a_min, a_max = two_refs['area_m2'].min(), two_refs['area_m2'].max()

        if (v_min <= query_volume <= v_max) and (a_min <= query_area <= a_max):
            st.success("Query falls inside the reference bounds for both Volume and Area.")
        else:
            st.warning("Query falls outside reference ranges for Volume or Area; output may reflect linear extrapolation.")

        # Unit Dry, Unit Operating and Test weights are all interpolated directly by Volume
        final_weights = {}
        for target in ['wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num']:
            final_weights[target], _ = interpolate_by_axis(two_refs, query_volume, target, 'volume_m3')

        query_weight = final_weights['wt_unit_dry_num']

        base_cost_area, _ = interpolate_by_axis(two_refs, query_area, 'unit_cost_num', 'area_m2')
        base_cost_weight, _ = interpolate_by_axis(two_refs, query_weight, 'unit_cost_num', 'wt_unit_dry_num')

        final_unit_cost_area = max(0.0, base_cost_area)
        final_unit_cost_weight = max(0.0, base_cost_weight)

        st.markdown("**Predicted Weight Breakdown**")
        w1, w2, w3 = st.columns(3)
        w1.metric("Unit Dry (Interpolated)", f"{final_weights['wt_unit_dry_num']:.2f} MT")
        w2.metric("Unit Operating (Interpolated)", f"{final_weights['wt_unit_oper_num']:.2f} MT")
        w3.metric("Test Weight (Interpolated)", f"{final_weights['wt_test_num']:.2f} MT")

        st.markdown("**Predicted Equipment Cost Comparison**")
        c_col1, c_col2 = st.columns(2)
        c_col1.metric("Final Cost (Based on Area)", f"MYR {final_unit_cost_area:,.2f}")
        c_col2.metric("Final Cost (Based on Weight)", f"MYR {final_unit_cost_weight:,.2f}")

        st.markdown("---")
        st.markdown("**Interpolation Visualizations: Area vs. Dry Weight**")
        y_refs = two_refs['unit_cost_num'].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x_refs_area = two_refs['area_m2'].values
        x_min_a = min(x_refs_area[0], x_refs_area[1], query_area) * 0.98
        x_max_a = max(x_refs_area[0], x_refs_area[1], query_area) * 1.02
        if x_refs_area[1] != x_refs_area[0]:
            slope_area = (y_refs[1] - y_refs[0]) / (x_refs_area[1] - x_refs_area[0])
            ax1.plot(np.linspace(x_min_a, x_max_a, 10), slope_area * np.linspace(x_min_a, x_max_a, 10) + (y_refs[0] - slope_area * x_refs_area[0]), color='gray', linestyle='--', alpha=0.7)
        ax1.scatter(x_refs_area, y_refs, color='blue', s=100, zorder=5, label='References')
        ax1.scatter([query_area], [base_cost_area], color='red', s=250, marker='*', zorder=6, label='New Launcher')
        ax1.set_xlabel("Surface Area (m²)", fontweight='bold')
        ax1.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax1.set_title("Cost Interpolation by Area")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax1.legend()

        x_refs_wt = two_refs['wt_unit_dry_num'].values
        x_min_w = min(x_refs_wt[0], x_refs_wt[1], query_weight) * 0.98
        x_max_w = max(x_refs_wt[0], x_refs_wt[1], query_weight) * 1.02
        if x_refs_wt[1] != x_refs_wt[0]:
            slope_wt = (y_refs[1] - y_refs[0]) / (x_refs_wt[1] - x_refs_wt[0])
            ax2.plot(np.linspace(x_min_w, x_max_w, 10), slope_wt * np.linspace(x_min_w, x_max_w, 10) + (y_refs[0] - slope_wt * x_refs_wt[0]), color='gray', linestyle='--', alpha=0.7)
        ax2.scatter(x_refs_wt, y_refs, color='blue', s=100, zorder=5, label='References')
        ax2.scatter([query_weight], [base_cost_weight], color='red', s=250, marker='*', zorder=6, label='New Launcher')
        ax2.set_xlabel("Dry Weight (MT)", fontweight='bold')
        ax2.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax2.set_title("Cost Interpolation by Dry Weight")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📋 Historical Project & Reference Details")
        st.dataframe(two_refs[launcher_display_cols + ['distance']], use_container_width=True)

# =====================================================================
# 8. NAVIGATION MENU
# =====================================================================
PAGES = {
    "Mechanical (Vessel)": mechanical_page, 
    "Mechanical (Tank)": tank_page,
    "Mechanical (Filter)": filter_page,
    "Mechanical (Launcher)": launcher_page,
    "Piping (Valve)": piping_page,         
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Discipline", list(PAGES.keys()), key="nav_selection")
PAGES[selected_page]()
