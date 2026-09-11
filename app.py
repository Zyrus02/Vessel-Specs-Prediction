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
# 6. PAGE BUILDER: MECHANICAL
# This creates the UI and logic for the Custom Vessel Predictor page.
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
    material_options = [AUTO_OPTION] + sorted(clean['material_category'].unique())
    orientation_options = [AUTO_OPTION] + sorted(clean['orientation_clean'].unique())

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
        # NEW INPUT: User sets the fluid/content density to calculate Operating Weight
        with d4: content_density = st.number_input("Content Density (kg/m³)", value=1000.0, step=50.0, help="e.g. Water is ~1000 kg/m³. Used to calculate Operating Weight assuming 80% full.")

        st.markdown("**Construction (optional)**")
        c1, c2 = st.columns(2)
        with c1: user_material = st.selectbox("Historical Material Category", material_options)
        with c2: user_orientation = st.selectbox("Orientation", orientation_options)

        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    # 6B. THE MATCHING LOGIC (KNN / Shape Distance)
    if submitted:
        # Step 1: Calculate the Volume and Area of the new tank
        query_volume, query_area = compute_geometry(user_length, user_diameter)
        
        # Save these into memory (including density) so they don't disappear if the user clicks something else
        st.session_state['mech_ctx'] = dict(
            query_volume=query_volume, 
            query_area=query_area, 
            cost_mat=cost_mat,
            content_density=content_density
        )

    if 'mech_ctx' in st.session_state:
        ctx = st.session_state['mech_ctx']
        query_volume, query_area = ctx['query_volume'], ctx['query_area']
        cost_mat = ctx['cost_mat']
        content_density = ctx.get('content_density', 1000.0)

        st.markdown("**Computed Geometry**")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Total Vessel Volume", f"{query_volume:.2f} m³")
        
        # Calculate 80% of the volume for the content
        content_volume = 0.8 * query_volume
        gc2.metric(
            "Content Volume (80% Full)", 
            f"{content_volume:.2f} m³", 
            help=f"Calculation: 0.8 × {query_volume:.2f} m³ (Total Volume)"
        )
        
        gc3.metric("Surface Area of Vessel", f"{query_area:.2f} m²")

        # Step 2: Calculate Standard Deviation to put Volume and Area on an equal playing field
        vol_std = clean['volume_m3'].std() or 1.0
        area_std = clean['area_m2'].std() or 1.0
        
        # Step 3: Calculate the "Shape Distance" using the Euclidean distance formula (a^2 + b^2 = c^2)
        ranked = clean.copy()
        ranked['distance'] = np.sqrt(
            ((ranked['volume_m3'] - query_volume) / vol_std) ** 2
            + ((ranked['area_m2'] - query_area) / area_std) ** 2
        )
        
        # Sort from most similar (smallest distance) to least similar
        ranked = ranked.sort_values('distance').reset_index(drop=True)

        def _label(i):
            r = ranked.iloc[i]
            return (f"{r['equip_no']} — Vol {r['volume_m3']:.2f} m³, Area {r['area_m2']:.2f} m², Cost {r['unit_cost_num']:,.0f} MYR (Dist {r['distance']:.3f})")

        st.markdown("**Reference vessels for interpolation**")
        # Let the user choose which of the top matches to use (defaults to the best 2)
        selected_idx = st.multiselect("Choose exactly 2 reference vessels:", options=list(range(len(ranked))), default=[0, 1], format_func=_label, max_selections=2)

        if len(selected_idx) != 2:
            st.info("Select exactly 2 reference vessels above to view predictions.")
            st.stop()

        two_refs = ranked.iloc[selected_idx].copy()
        
        # Check if the new vessel is smaller or larger than BOTH historical vessels. If so, it warns about "extrapolation"
        v_min, v_max = two_refs['volume_m3'].min(), two_refs['volume_m3'].max()
        a_min, a_max = two_refs['area_m2'].min(), two_refs['area_m2'].max()
        is_extrapolating_area = not (a_min <= query_area <= a_max)

        if (v_min <= query_volume <= v_max) and not is_extrapolating_area:
            st.success("Query falls inside the reference bounds for both Volume and Area.")
        else:
            st.warning("Query falls outside reference ranges for Volume or Area; output may reflect linear extrapolation.")

        # 6C. CALCULATING THE PREDICTIONS
        final_weights = {}
        
        # Interpolate ONLY Dry Weight and Test Weight
        for target in ['wt_unit_dry_num', 'wt_test_num']:
            val, _ = interpolate_by_axis(two_refs, query_volume, target, 'volume_m3')
            final_weights[target] = val
            
        query_weight = final_weights['wt_unit_dry_num']

        # DYNAMIC CALCULATION: Operating Weight
        # Formula: Density * (0.8 * Volume) to get kg, then divide by 1000 for Metric Tons
        content_weight_kg = content_density * (0.8 * query_volume)
        content_weight_mt = content_weight_kg / 1000.0
        
        # Sum the interpolated Dry Weight and the calculated Content Weight
        final_weights['wt_unit_oper_num'] = query_weight + content_weight_mt

        # Predict Cost by interpolating both Area and Weight (to compare them)
        base_cost_area, _ = interpolate_by_axis(two_refs, query_area, 'unit_cost_num', 'area_m2')
        base_cost_weight, _ = interpolate_by_axis(two_refs, query_weight, 'unit_cost_num', 'wt_unit_dry_num')
        
        # Apply the material pricing multiplier
        cost_multiplier = 4.0 if "Stainless Steel" in cost_mat else 1.0
        final_unit_cost_area = max(0.0, base_cost_area * cost_multiplier) # max(0.0) ensures prices never drop below zero
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
        
        # Creates a side-by-side graph layout (1 row, 2 columns)
        y_refs = two_refs['unit_cost_num'].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Draw Area Graph (Left side) ---
        x_refs_area = two_refs['area_m2'].values
        x_min_area = min(x_refs_area[0], x_refs_area[1], query_area) * 0.98
        x_max_area = max(x_refs_area[0], x_refs_area[1], query_area) * 1.02
        if x_refs_area[1] != x_refs_area[0]:
            # Calculate the slope of the line (Rise over Run)
            slope_area = (y_refs[1] - y_refs[0]) / (x_refs_area[1] - x_refs_area[0])
            # Draw the dotted trendline
            ax1.plot(np.linspace(x_min_area, x_max_area, 10), slope_area * np.linspace(x_min_area, x_max_area, 10) + (y_refs[0] - slope_area * x_refs_area[0]), color='gray', linestyle='--', alpha=0.7)
            
        ax1.scatter(x_refs_area, y_refs, color='blue', s=100, zorder=5, label='References') # Plot historical points
        ax1.scatter([query_area], [base_cost_area], color='red', s=250, marker='*', zorder=6, label='New Vessel') # Plot predicted point
        ax1.set_xlabel("Surface Area (m²)", fontweight='bold')
        ax1.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax1.set_title("Cost Interpolation by Area")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax1.legend()

        # --- Draw Weight Graph (Right side) ---
        x_refs_wt = two_refs['wt_unit_dry_num'].values
        x_min_wt = min(x_refs_wt[0], x_refs_wt[1], query_weight) * 0.98
        x_max_wt = max(x_refs_wt[0], x_refs_wt[1], query_weight) * 1.02
        if x_refs_wt[1] != x_refs_wt[0]:
            slope_wt = (y_refs[1] - y_refs[0]) / (x_refs_wt[1] - x_refs_wt[0])
            ax2.plot(np.linspace(x_min_wt, x_max_wt, 10), slope_wt * np.linspace(x_min_wt, x_max_wt, 10) + (y_refs[0] - slope_wt * x_refs_wt[0]), color='gray', linestyle='--', alpha=0.7)
            
        ax2.scatter(x_refs_wt, y_refs, color='blue', s=100, zorder=5, label='References')
        ax2.scatter([query_weight], [base_cost_weight], color='red', s=250, marker='*', zorder=6, label='New Vessel')
        ax2.set_xlabel("Dry Weight (MT)", fontweight='bold')
        ax2.set_ylabel("Base Cost (MYR)", fontweight='bold')
        ax2.set_title("Cost Interpolation by Dry Weight")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax2.legend()
        
        plt.tight_layout() # Makes sure graphs don't overlap
        st.pyplot(fig) # Tells Streamlit to display the graph

        # Show raw data at the bottom in a drop-down expander
        with st.expander("See calculation details and reference comparisons"):
            v_ref_a, v_ref_b = two_refs.iloc[0], two_refs.iloc[1]
            detail_data = {
                "Metric": ["Area Basis", "Vol Basis", "Dry Wt (MT)", "Oper Wt (MT) - Calculated", "Test Wt (MT)", "Hist. Cost (MYR)", "Final Cost - Area (MYR)", "Final Cost - Wt (MYR)"],
                f"Ref 1 ({v_ref_a['equip_no']})": [
                    f"{v_ref_a['area_m2']:.1f}", f"{v_ref_a['volume_m3']:.1f}", f"{v_ref_a['wt_unit_dry_num']:.2f}",
                    f"{v_ref_a['wt_unit_oper_num']:.2f}", f"{v_ref_a['wt_test_num']:.2f}", f"{v_ref_a['unit_cost_num']:,.0f}", "-", "-"
                ],
                f"Ref 2 ({v_ref_b['equip_no']})": [
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

# =====================================================================
# 7. PAGE BUILDER: PIPING
# This creates the UI and logic for the Standard Valve Predictor page.
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

    # Grab all unique options for the dropdown menus
    valve_types = sorted(df['type_clean'].unique())
    valve_materials = sorted(df['material_clean'].unique())
    valve_ratings = sorted(df['rating_clean'].unique())

    # 7A. THE INPUT FORM
    with st.form("valve_prediction_form"):
        st.markdown("**Valve Specifications** (used for exact matching)")
        c1, c2, c3 = st.columns(3)
        with c1: user_type = st.selectbox("Valve Type", valve_types)
        with c2: user_rating = st.selectbox("Pressure Rating", valve_ratings)
        with c3: user_material = st.selectbox("Material Specification", valve_materials)
        
        st.markdown("**Target Dimension**")
        user_size = st.number_input("Nominal Size (mm)", min_value=15.0, value=100.0, step=25.0)
            
        submitted = st.form_submit_button("Predict Weight & Cost", use_container_width=True, type="primary")

    # 7B. MATCHING LOGIC (Exact Filtering)
    if submitted:
        # Instead of shape distance, we filter the database to find ONLY identical types/ratings/materials
        filtered = df[
            (df['type_clean'] == user_type) & 
            (df['rating_clean'] == user_rating) & 
            (df['material_clean'] == user_material)
        ].copy()

        if len(filtered) == 0:
            st.error(f"No historical data found for {user_type} valves in {user_material} at Rating {user_rating}.")
            st.stop()

        # Calculate how close the historical sizes are to the user's requested size
        filtered['size_diff'] = abs(filtered['size_num'] - user_size)
        
        # Sort so the closest sizes are at the top. If there's a tie, put the newest 'year' at the top
        filtered = filtered.sort_values(['size_diff', 'year'], ascending=[True, False])
        
        # Grab exactly two distinct sizes (drop duplicates) to draw our interpolation line
        closest_sizes = filtered.drop_duplicates(subset=['size_num']).head(2)

        # 7C. CALCULATING PREDICTIONS
        if len(closest_sizes) < 2:
            st.warning("Only found 1 distinct size for these specs. Extrapolation is impossible; displaying exact historical match instead.")
            final_cost = closest_sizes.iloc[0]['cost_num']
            final_weight = closest_sizes.iloc[0]['weight_num']
            is_interpolated = False
        else:
            final_cost, _ = interpolate_by_axis(closest_sizes, user_size, 'cost_num', 'size_num')
            # Handle missing weight data gracefully
            if closest_sizes['weight_num'].isna().any():
                final_weight = None
            else:
                final_weight, _ = interpolate_by_axis(closest_sizes, user_size, 'weight_num', 'size_num')
            is_interpolated = True

        final_cost = max(0.0, final_cost)

        # 7D. DISPLAYING RESULTS
        st.markdown("**Predicted Output**")
        out1, out2, out3 = st.columns(3)
        out1.metric("Predicted Cost", f"RM {final_cost:,.2f}")
        if final_weight is not None and not pd.isna(final_weight):
            out2.metric("Predicted Weight", f"{final_weight:.1f} kg")
        else:
            out2.metric("Predicted Weight", "N/A")
        out3.metric("Calculation Method", "Interpolation" if is_interpolated else "Direct Historical Match")

        # 7E. DRAWING THE GRAPH
        if is_interpolated:
            st.markdown("---")
            st.markdown("**Interpolation Visualization (Size vs. Cost)**")
            
            refs_sorted = closest_sizes.sort_values('size_num')
            x_refs = refs_sorted['size_num'].values
            y_refs = refs_sorted['cost_num'].values

            fig, ax = plt.subplots(figsize=(10, 5))
            
            x_min_plot = min(x_refs[0], x_refs[1], user_size) * 0.8
            x_max_plot = max(x_refs[0], x_refs[1], user_size) * 1.2
            
            # Draw the dotted trendline
            if x_refs[1] != x_refs[0]:
                slope = (y_refs[1] - y_refs[0]) / (x_refs[1] - x_refs[0])
                intercept = y_refs[0] - slope * x_refs[0]
                x_line = np.linspace(x_min_plot, x_max_plot, 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='gray', linestyle='--', alpha=0.7, label='Interpolation Trendline')
            
            # Plot the dots
            ax.scatter(x_refs, y_refs, color='blue', s=100, zorder=5, label='Historical References')
            ax.scatter([user_size], [final_cost], color='red', s=250, marker='*', zorder=6, label='Predicted Cost')
            
            # Formatting the graph
            ax.set_xlabel("Nominal Size (mm)", fontweight='bold')
            ax.set_ylabel("Unit Cost (RM)", fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.legend()
            st.pyplot(fig)
            st.markdown("---")

        with st.expander("See calculation details and filtered references"):
            display_cols = ['item_no', 'year', 'project', 'size_num', 'rating_clean', 'weight_num', 'cost_num']
            st.dataframe(closest_sizes[display_cols], use_container_width=True)

# =====================================================================
# 8. NAVIGATION MENU
# This acts as a switchboard. It maps the names on the sidebar to the functions above.
# =====================================================================
PAGES = {
    "Mechanical": mechanical_page, # Links to the Vessel predictor
    "Piping": piping_page,         # Links to the Valve predictor
}

st.sidebar.title("Navigation")
# Create radio buttons in the sidebar
selected_page = st.sidebar.radio("Discipline", list(PAGES.keys()), key="nav_selection")

# Run whichever function the user selected
PAGES[selected_page]()
