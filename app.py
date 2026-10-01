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
# 4.8 MECHANICAL (HEAT EXCHANGER) CONFIG & HELPERS
# Input (blue-highlighted in the source workbook): Type of Heat Exchanger + Capacity (kW)
# Output (green-highlighted): Unit Cost, Sub Total (Total Cost), Unit Operating Weight
# =====================================================================
HX_SHEET_NAME = 'Sheet1 (2)'

# Column positions (0-indexed, after skiprows=4) matching
# "Overall_Heat_Exchanger_Database_Rev1.xlsx" / sheet 'Sheet1 (2)'.
HX_COL_MAP = {
    'equip_no':          1,    # EQUIPMENT NO. (BED)
    'qty':               2,    # QTY
    'description':       3,    # DESCRIPTION
    'hx_type':           6,    # TYPE OF HEAT EXCHANGER  (blue input)
    'capacity':          7,    # CAPACITY (kW)           (blue input)
    'power_unit_duty':   8,    # POWER/UNIT — DUTY (kW)
    'design_press':      10,   # DESIGN CONDITIONS — PRESS. (barg)
    'design_temp_max':   11,   # DESIGN CONDITIONS — TEMP. Max (oC)
    'design_temp_min':   12,   # DESIGN CONDITIONS — TEMP. Min (oC)
    'length':            15,   # DIMENSIONS/UNIT — LENGTH (mm)
    'width_id':          16,   # DIMENSIONS/UNIT — WIDTH/ID (mm)
    'height':            17,   # DIMENSIONS/UNIT — HEIGHT (mm)
    'wt_unit_dry':       18,   # WEIGHT — UNIT DRY (MT)
    'wt_unit_oper':      19,   # WEIGHT — UNIT OPER. (MT)   (green output)
    'wt_tot_dry':        20,   # WEIGHT — TOT. DRY (MT)
    'wt_tot_oper':       21,   # WEIGHT — TOT. OPER. (MT)
    'wt_test':           22,   # WEIGHT — TEST (MT)
    'material':          23,   # MATERIAL
    'orientation':       24,   # ORIENTATION (V/H)
    'unit_cost':         25,   # UNIT COST (MYR)           (green output)
    'sub_total':         26,   # SUB TOTAL (MYR) — "Total Cost" (green output)
    'project':           28,   # NAME OF PROJECT
    'year':              29,   # YEAR
}

@st.cache_data
def load_and_clean_heat_exchangers(file):
    """Loads and cleans the Heat Exchanger workbook using the column index map above."""
    raw = pd.read_excel(file, sheet_name=HX_SHEET_NAME, header=None, skiprows=4)
    df = raw.copy()

    df['equip_no'] = df[HX_COL_MAP['equip_no']]
    df['description'] = df[HX_COL_MAP['description']]
    df['hx_type'] = df[HX_COL_MAP['hx_type']].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) and str(x).strip() not in ('', '-') else 'Unknown')

    # Keep the raw capacity string for display (e.g. "17396 (total)"), and a cleaned
    # numeric version (17396.0) for matching/ranking/rounding.
    df['capacity_raw'] = df[HX_COL_MAP['capacity']]
    df['capacity_num'] = df[HX_COL_MAP['capacity']].apply(first_num)

    df['design_press'] = df[HX_COL_MAP['design_press']]
    df['design_temp_max'] = df[HX_COL_MAP['design_temp_max']]
    df['design_temp_min'] = df[HX_COL_MAP['design_temp_min']]

    # Dimensions kept as raw strings for display (often messy, e.g. "406.4 (OD)" or "Shell: 320/ 260")
    df['length_raw'] = df[HX_COL_MAP['length']]
    df['width_id_raw'] = df[HX_COL_MAP['width_id']]
    df['height_raw'] = df[HX_COL_MAP['height']]
    df['qty'] = df[HX_COL_MAP['qty']].apply(first_num)
    # Kept exactly as-is (no cleaning/rounding) since these often carry qualifying text
    # like "2 x 50% config." or "1 bay" / "(3 x 33%) 11.0 (each fan motor)".
    df['qty_raw'] = df[HX_COL_MAP['qty']]
    df['power_unit_duty_raw'] = df[HX_COL_MAP['power_unit_duty']]

    df['wt_unit_dry_num'] = df[HX_COL_MAP['wt_unit_dry']].apply(first_num)
    df['wt_unit_oper_num'] = df[HX_COL_MAP['wt_unit_oper']].apply(first_num)
    df['wt_tot_dry_num'] = df[HX_COL_MAP['wt_tot_dry']].apply(first_num)
    df['wt_tot_oper_num'] = df[HX_COL_MAP['wt_tot_oper']].apply(first_num)
    df['wt_test_num'] = df[HX_COL_MAP['wt_test']].apply(first_num)

    df['material_category'] = df[HX_COL_MAP['material']].apply(categorize_material)
    df['orientation'] = df[HX_COL_MAP['orientation']].apply(normalize_orientation)

    df['unit_cost_num'] = df[HX_COL_MAP['unit_cost']].apply(first_num)
    df['sub_total_num'] = df[HX_COL_MAP['sub_total']].apply(first_num)

    df['project'] = df[HX_COL_MAP['project']]
    df['year'] = df[HX_COL_MAP['year']]

    return df.dropna(subset=['capacity_num', 'hx_type', 'unit_cost_num'])

# =====================================================================
# 4.85 MECHANICAL (TOTE TANK) CONFIG & HELPERS
# Input (blue-highlighted): Capacity (m3) + Selection Material
# Output (green-highlighted): Unit Cost (MYR)
# Match logic: filter to capacity >= input (exact or higher), then narrow to
# only the LATEST year among that filtered subset, and show every row that
# still qualifies (no single nearest-neighbor pick).
# =====================================================================
TOTE_SHEET_NAME = 'Sheet1 (2)'

# Column positions (0-indexed, after skiprows=4) matching
# "Overall_Tote_Tank_Database_Rev1.xlsx" / sheet 'Sheet1 (2)'.
TOTE_COL_MAP = {
    'equip_no':          1,    # EQUIPMENT NO. (BED)
    'qty':               2,    # QTY
    'description':       3,    # DESCRIPTION
    'capacity':          6,    # CAPACITY (m3)              (blue input)
    'power_unit_duty':   7,    # POWER/UNIT — DUTY (kW)
    'power_unit_abs':    8,    # POWER/UNIT — ABSORBED (kW)
    'design_press':      9,    # DESIGN CONDITIONS — PRESS.
    'design_temp_max':   10,   # DESIGN CONDITIONS — TEMP. Max (oC)
    'design_temp_min':   11,   # DESIGN CONDITIONS — TEMP. Min (oC)
    'oper_press':        12,   # OPERATING CONDITIONS — PRESS.
    'oper_temp':         13,   # OPERATING CONDITIONS — TEMP.
    'length':            14,   # DIMENSIONS/UNIT — LENGTH (mm)
    'width_id':          15,   # DIMENSIONS/UNIT — WIDTH/ID (mm)
    'height':            16,   # DIMENSIONS/UNIT — HEIGHT (mm)
    'wt_unit_dry':       17,   # WEIGHT — UNIT DRY (MT)
    'wt_unit_oper':      18,   # WEIGHT — UNIT OPER. (MT)
    'wt_tot_dry':        19,   # WEIGHT — TOT. DRY (MT)
    'wt_tot_oper':       20,   # WEIGHT — TOT. OPER. (MT)
    'wt_test':           21,   # WEIGHT — TEST (MT)
    'material':          22,   # SELECTION MATERIAL         (blue input)
    'material_detail':   23,   # MATERIAL DETAIL
    'orientation':       24,   # ORIENTATION (V/H)
    'unit_cost':         25,   # UNIT COST (MYR)            (green output)
    'sub_total':         26,   # SUB TOTAL (MYR)
    'project':           28,   # NAME OF PROJECT
    'year':              29,   # YEAR
}

@st.cache_data
def load_and_clean_tote_tanks(file):
    """Loads and cleans the Tote Tank workbook using the column index map above."""
    raw = pd.read_excel(file, sheet_name=TOTE_SHEET_NAME, header=None, skiprows=4)
    df = raw.copy()

    df['equip_no'] = df[TOTE_COL_MAP['equip_no']]
    df['qty_raw'] = df[TOTE_COL_MAP['qty']]
    df['description'] = df[TOTE_COL_MAP['description']]

    df['capacity_raw'] = df[TOTE_COL_MAP['capacity']]
    df['capacity_num'] = df[TOTE_COL_MAP['capacity']].apply(first_num)

    df['power_unit_duty_raw'] = df[TOTE_COL_MAP['power_unit_duty']]
    df['power_unit_abs_raw'] = df[TOTE_COL_MAP['power_unit_abs']]

    df['design_press'] = df[TOTE_COL_MAP['design_press']]
    df['design_temp_max'] = df[TOTE_COL_MAP['design_temp_max']]
    df['design_temp_min'] = df[TOTE_COL_MAP['design_temp_min']]
    df['oper_press'] = df[TOTE_COL_MAP['oper_press']]
    df['oper_temp'] = df[TOTE_COL_MAP['oper_temp']]

    df['length_raw'] = df[TOTE_COL_MAP['length']]
    df['width_id_raw'] = df[TOTE_COL_MAP['width_id']]
    df['height_raw'] = df[TOTE_COL_MAP['height']]

    df['wt_unit_dry_num'] = df[TOTE_COL_MAP['wt_unit_dry']].apply(first_num)
    df['wt_unit_oper_num'] = df[TOTE_COL_MAP['wt_unit_oper']].apply(first_num)
    df['wt_tot_dry_num'] = df[TOTE_COL_MAP['wt_tot_dry']].apply(first_num)
    df['wt_tot_oper_num'] = df[TOTE_COL_MAP['wt_tot_oper']].apply(first_num)
    df['wt_test_num'] = df[TOTE_COL_MAP['wt_test']].apply(first_num)

    df['material_category'] = df[TOTE_COL_MAP['material']].apply(categorize_material)
    df['material_detail'] = df[TOTE_COL_MAP['material_detail']]
    df['orientation'] = df[TOTE_COL_MAP['orientation']].apply(normalize_orientation)

    df['unit_cost_num'] = df[TOTE_COL_MAP['unit_cost']].apply(first_num)
    df['sub_total_num'] = df[TOTE_COL_MAP['sub_total']].apply(first_num)

    df['project'] = df[TOTE_COL_MAP['project']]
    df['year_num'] = df[TOTE_COL_MAP['year']].apply(first_num)

    return df.dropna(subset=['capacity_num', 'material_category', 'unit_cost_num', 'year_num'])

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
    st.title("Tank Weight & Cost Predictor")
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
# 7.8 PAGE BUILDER: MECHANICAL (HEAT EXCHANGER)
# =====================================================================
def heat_exchanger_page():
    st.title("Heat Exchanger Spec & Cost Lookup")
    st.caption("Strict historical lookup. Matches based on Type of Heat Exchanger and the closest matching Capacity "
               "(rounded to the nearest value, not just rounded up). No interpolation — the 2 closest historical "
               "references are shown, with the nearer one recommended.")

    uploaded = st.file_uploader("Upload heat exchanger database (.xlsx)", type=["xlsx"], key="hx_up")

    if uploaded is None:
        st.info(f"Upload the heat exchanger database Excel file to begin. Expects sheet '{HX_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the heat exchanger workbook..."):
            df = load_and_clean_heat_exchangers(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    display_cols = ['equip_no', 'qty_raw', 'description', 'hx_type', 'capacity_raw', 'power_unit_duty_raw',
                     'design_press', 'design_temp_max', 'design_temp_min',
                     'length_raw', 'width_id_raw', 'height_raw',
                     'wt_unit_dry_num', 'wt_unit_oper_num', 'wt_test_num',
                     'material_category', 'orientation',
                     'unit_cost_num', 'sub_total_num', 'project', 'year']
    display_headers = ['Equip No.', 'Quantity', 'Description', 'Type', 'Capacity (kW)', 'Power/Unit (kW)',
                        'Design Press. (barg)', 'Design Temp Max (°C)', 'Design Temp Min (°C)',
                        'Length', 'Width/ID', 'Height',
                        'Dry Wt (MT)', 'Oper. Wt (MT)', 'Test Wt (MT)',
                        'Material', 'Orientation',
                        'Unit Cost (MYR)', 'Total Cost (MYR)', 'Project', 'Year']

    with st.expander("Preview cleaned data"):
        st.dataframe(df[display_cols], use_container_width=True)

    st.divider()
    st.subheader("Lookup specs & cost for a new heat exchanger")

    type_options = sorted(df['hx_type'].astype(str).unique())

    # Placeholder types for future reference — not yet present in the historical database
    PLACEHOLDER_HX_TYPES = ['PLATE', 'OTHER']
    for placeholder in PLACEHOLDER_HX_TYPES:
        if placeholder not in type_options:
            type_options.append(placeholder)
    type_options = sorted(type_options)

    with st.form("hx_prediction_form"):
        st.markdown("**Search Parameters**")
        c1, c2 = st.columns(2)
        with c1: user_capacity = st.number_input("Selection Capacity (kW)", min_value=1.0, value=1000.0, step=50.0)
        with c2: user_type = st.selectbox("Type of Heat Exchanger", type_options)

        submitted = st.form_submit_button("Lookup Existing Data", use_container_width=True, type="primary")

    if submitted:
        # Special check for PLATE / OTHER placeholders
        if user_type in PLACEHOLDER_HX_TYPES and len(df[df['hx_type'] == user_type]) == 0:
            st.warning(f"⚠️ '{user_type}' heat exchangers are currently unavailable in the historical database. "
                       f"Please select another type.")
            st.stop()

        # Step 1: Filter by exact Type match
        type_filtered = df[df['hx_type'] == user_type].copy()

        if len(type_filtered) == 0:
            st.error(f"No historical data found for '{user_type}' heat exchangers.")
            st.stop()

        # Step 2: Compare the input capacity AGAINST the historical database by bracketing —
        # find the historical record immediately BELOW and the one immediately ABOVE the
        # input, so the comparison happens *in between* two real historical data points
        # (rather than just picking the 2 absolute-nearest rows). The recommendation still
        # always favors the bigger (upper) value, consistent with the original convention.
        type_filtered = type_filtered.sort_values('capacity_num').reset_index(drop=True)

        lower_matches = type_filtered[type_filtered['capacity_num'] <= user_capacity]
        upper_matches = type_filtered[type_filtered['capacity_num'] >= user_capacity]

        lower_row = lower_matches.iloc[-1] if len(lower_matches) > 0 else None
        upper_row = upper_matches.iloc[0] if len(upper_matches) > 0 else None

        exact_match = (lower_row is not None and upper_row is not None
                       and lower_row['capacity_num'] == upper_row['capacity_num'])

        if exact_match:
            two_refs = type_filtered.loc[[lower_row.name]].copy()
            recommended_name = lower_row.name
            st.success(f"Exact historical match found at **{lower_row['capacity_num']:,.1f} kW** "
                       f"(Equip No. {lower_row['equip_no']}).")
        elif lower_row is not None and upper_row is not None:
            # Input falls strictly BETWEEN two historical capacities
            two_refs = type_filtered.loc[[lower_row.name, upper_row.name]].copy()
            recommended_name = upper_row.name
            st.info(f"Requested capacity **{user_capacity:,.1f} kW** falls between historical capacities "
                    f"**{lower_row['capacity_num']:,.1f} kW** ({lower_row['equip_no']}) and "
                    f"**{upper_row['capacity_num']:,.1f} kW** ({upper_row['equip_no']}).")
        elif upper_row is not None:
            # Input is smaller than every historical record — no lower reference exists
            two_refs = upper_matches.iloc[:2].copy()
            recommended_name = upper_row.name
            st.warning(f"Requested capacity **{user_capacity:,.1f} kW** is smaller than every historical record "
                       f"for '{user_type}'. No smaller reference exists — recommending the closest bigger value.")
        else:
            # Input is bigger than every historical record — no bigger reference exists
            two_refs = lower_matches.iloc[-2:].copy()
            recommended_name = lower_row.name
            st.warning(f"Requested capacity **{user_capacity:,.1f} kW** exceeds every historical record "
                       f"for '{user_type}'. No bigger value is available — recommending the largest historical "
                       f"reference instead.")

        n_refs = len(two_refs)
        recommended = type_filtered.loc[recommended_name]

        # Display the results
        st.markdown("### 🎯 Historical Matches (Compared Between Database Values)")
        if n_refs < 2:
            st.caption("Only 1 historical record exists for this Type — showing the single available reference.")

        res_df = two_refs[display_cols].copy()
        res_df.columns = display_headers
        res_df.insert(0, 'Match', ["✅ Recommended (Bigger)" if idx == recommended_name else "Reference"
                                    for idx in two_refs.index])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Below the table: show Unit Cost, Total Cost, Unit Operating Weight for both references,
        # with the recommended (bigger-value) one flagged.
        st.markdown("### Recommended Output (Bigger Value Highlighted)")

        ref_cols = st.columns(n_refs)
        for i, (idx, row) in enumerate(two_refs.iterrows()):
            with ref_cols[i]:
                label = "✅ Recommended (Bigger Value)" if idx == recommended_name else "Reference"
                st.markdown(f"**{label}**")
                st.caption(f"{row['equip_no']} — {row['capacity_raw']} kW")
                st.metric("Unit Cost (MYR)", f"{row['unit_cost_num']:,.2f}")
                st.metric("Total Cost (MYR)", f"{row['sub_total_num']:,.2f}" if pd.notna(row['sub_total_num']) else "N/A")
                st.metric("Unit Dry Weight (MT)", f"{row['wt_unit_dry_num']:,.2f}" if pd.notna(row['wt_unit_dry_num']) else "N/A")
                st.metric("Unit Operating Weight (MT)", f"{row['wt_unit_oper_num']:,.2f}" if pd.notna(row['wt_unit_oper_num']) else "N/A")

        st.success(f"**Recommendation:** Use the values from **{recommended['equip_no']}** "
                   f"at **{recommended['capacity_num']:,.1f} kW** — the bigger of the two compared values.")

# =====================================================================
# 7.85 PAGE BUILDER: MECHANICAL (TOTE TANK)
# =====================================================================
def tote_tank_page():
    st.title("Tote Tank Spec & Cost Lookup")
    st.caption("Historical lookup filtered by Capacity (exact or higher only) and Material, then narrowed down "
               "to the LATEST project year among the qualifying records. Every row that still meets both "
               "conditions is shown — no single nearest-neighbor pick.")

    uploaded = st.file_uploader("Upload tote tank database (.xlsx)", type=["xlsx"], key="tote_up")

    if uploaded is None:
        st.info(f"Upload the tote tank database Excel file to begin. Expects sheet '{TOTE_SHEET_NAME}'.")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the tote tank workbook..."):
            df = load_and_clean_tote_tanks(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    display_cols = ['equip_no', 'qty_raw', 'description', 'capacity_raw',
                     'power_unit_duty_raw', 'power_unit_abs_raw',
                     'design_press', 'design_temp_max', 'design_temp_min',
                     'oper_press', 'oper_temp',
                     'length_raw', 'width_id_raw', 'height_raw',
                     'wt_unit_dry_num', 'wt_unit_oper_num', 'wt_tot_dry_num', 'wt_tot_oper_num', 'wt_test_num',
                     'material_category', 'material_detail', 'orientation',
                     'unit_cost_num', 'sub_total_num', 'project', 'year_num']
    display_headers = ['Equip No.', 'Quantity', 'Description', 'Capacity (m³)',
                        'Power/Unit Duty (kW)', 'Power/Unit Absorbed (kW)',
                        'Design Press.', 'Design Temp Max (°C)', 'Design Temp Min (°C)',
                        'Oper. Press.', 'Oper. Temp (°C)',
                        'Length', 'Width/ID', 'Height',
                        'Dry Wt (MT)', 'Oper. Wt (MT)', 'Tot. Dry Wt (MT)', 'Tot. Oper. Wt (MT)', 'Test Wt (MT)',
                        'Material', 'Material Detail', 'Orientation',
                        'Unit Cost (MYR)', 'Total Cost (MYR)', 'Project', 'Year']

    with st.expander("Preview cleaned data"):
        st.dataframe(df[display_cols], use_container_width=True)

    st.divider()
    st.subheader("Lookup specs & cost for a new tote tank")

    material_options = sorted(df['material_category'].astype(str).unique())

    # Placeholder material for future reference — not yet present in the historical database
    PLACEHOLDER_TOTE_MATERIALS = ['CS']
    for placeholder in PLACEHOLDER_TOTE_MATERIALS:
        if placeholder not in material_options:
            material_options.append(placeholder)
    material_options = [AUTO_OPTION] + sorted(material_options)

    with st.form("tote_prediction_form"):
        st.markdown("**Search Parameters**")
        c1, c2 = st.columns(2)
        with c1: user_capacity = st.number_input("Minimum Capacity (m³)", min_value=0.1, value=2.0, step=0.5)
        with c2: user_material = st.selectbox("Selection Material", material_options)

        submitted = st.form_submit_button("Lookup Existing Data", use_container_width=True, type="primary")

    if submitted:
        # Special check for placeholder materials (e.g. CS) with no historical data yet
        if user_material in PLACEHOLDER_TOTE_MATERIALS and len(df[df['material_category'] == user_material]) == 0:
            st.warning(f"⚠️ '{user_material}' tote tanks are currently unavailable in the historical database. "
                       f"Please select another material.")
            st.stop()

        # Condition 1: exact or higher capacity value
        capacity_filtered = df[df['capacity_num'] >= user_capacity].copy()

        # Optional material filter (blue input)
        if user_material != AUTO_OPTION:
            capacity_filtered = capacity_filtered[capacity_filtered['material_category'] == user_material]

        if len(capacity_filtered) == 0:
            st.error(f"No historical tote tanks found with capacity ≥ {user_capacity:,.2f} m³"
                      + (f" and material '{user_material}'." if user_material != AUTO_OPTION else "."))
            st.stop()

        # Condition 2: narrow down to the LATEST year among the capacity-filtered subset
        latest_year = capacity_filtered['year_num'].max()
        final_matches = capacity_filtered[capacity_filtered['year_num'] == latest_year].copy()
        final_matches = final_matches.sort_values('capacity_num').reset_index(drop=True)

        st.info(f"Filtered to capacity ≥ **{user_capacity:,.2f} m³**"
                + (f", material **{user_material}**" if user_material != AUTO_OPTION else "")
                + f", then narrowed to the latest project year found: **{int(latest_year)}**. "
                f"**{len(final_matches)}** historical record(s) match all conditions.")

        st.markdown("### 🎯 Historical Matches (All Qualifying Records)")
        res_df = final_matches[display_cols].copy()
        res_df.columns = display_headers
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.markdown("### 💰 Output Summary — Unit Cost (MYR)")
        s1, s2, s3 = st.columns(3)
        s1.metric("Lowest Unit Cost", f"{final_matches['unit_cost_num'].min():,.2f}")
        s2.metric("Highest Unit Cost", f"{final_matches['unit_cost_num'].max():,.2f}")
        s3.metric("Average Unit Cost", f"{final_matches['unit_cost_num'].mean():,.2f}")

        st.markdown("### ⚖️ Output Summary — Unit Dry Weight & Unit Test Weight (MT)")
        weight_df = final_matches[['equip_no', 'wt_unit_dry_num', 'wt_test_num']].copy()
        weight_df.columns = ['Equip No.', 'Unit Dry Weight (MT)', 'Unit Test Weight (MT)']
        st.dataframe(weight_df, use_container_width=True, hide_index=True)

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

        st.markdown("**Computed Geometry (Volume / Area)**")
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
# 7.95 PACKAGE (MECHANICAL PACKAGES) CONFIG & HELPERS
# Input  (blue-highlighted): Capacity + List of Package
# Output (green-highlighted): Power, Design Conditions, Dimensions,
#                             Weights, Material, Unit Cost, Year
# Multi-skid rule: rows whose "LIST OF PACKAGE" cell is MERGED in Excel are
# ONE package made of several skids. Each skid keeps its own dimensions /
# weights / power / material. Cost is handled by how Excel stores it:
#   - Unit Cost merged across the skids  -> ONE package cost (shared)
#   - Unit Cost NOT merged (own value per skid) -> per-skid costs, summed
# So a future multi-skid package with different cost per skid works with no
# code change: just merge the package name/capacity cells and leave cost unmerged.
# =====================================================================
PACKAGE_SHEET_NAME = 'MAIN'
PACKAGE_FIRST_DATA_ROW = 5
PACKAGE_COLS = {
    'equip_no': 'B', 'qty': 'C', 'description': 'D', 'pfd': 'E', 'location': 'F',
    'package': 'G', 'capacity': 'H', 'unit': 'I',
    'power_duty': 'J', 'power_abs': 'K',
    'design_press': 'L', 'design_temp_max': 'M', 'design_temp_min': 'N',
    'oper_press': 'O', 'oper_temp': 'P',
    'length': 'Q', 'width': 'R', 'height': 'S',
    'wt_unit_dry': 'T', 'wt_unit_oper': 'U', 'wt_tot_dry': 'V', 'wt_tot_oper': 'W', 'wt_test': 'X',
    'material': 'Y', 'orientation': 'Z', 'unit_cost': 'AA', 'sub_total': 'AB',
    'remarks': 'AC', 'project': 'AD', 'year': 'AE', 'folder': 'AF',
}

@st.cache_data
def load_and_clean_packages(file):
    """Reads the Package workbook; groups merged rows into one package with N skids."""
    from openpyxl import load_workbook
    from openpyxl.utils import column_index_from_string
    ws = load_workbook(file, data_only=True)[PACKAGE_SHEET_NAME]
    idx = {k: column_index_from_string(v) for k, v in PACKAGE_COLS.items()}

    # Map every merged cell -> (top-left row, bottom row) so blank merged cells can be resolved
    merged = {}
    for rng in ws.merged_cells.ranges:
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                merged[(r, c)] = (rng.min_row, rng.min_col, rng.max_row - rng.min_row + 1)

    def val(r, key):
        c = idx[key]
        if (r, c) in merged:
            tr, tc, _ = merged[(r, c)]
            return ws.cell(tr, tc).value
        return ws.cell(r, c).value

    def span(r, key):
        """(top_row, n_rows) if this cell is merged vertically, else (r, 1)."""
        m = merged.get((r, idx[key]))
        return (m[0], m[2]) if m else (r, 1)

    rows, group_of = [], {}
    for r in range(PACKAGE_FIRST_DATA_ROW, ws.max_row + 1):
        if ws.cell(r, idx['equip_no']).value is None and val(r, 'package') is None:
            continue
        top, n = span(r, 'package')
        gid = top if n > 1 else r                      # merged package cell = same package
        cost_top, cost_n = span(r, 'unit_cost')
        rec = {k: val(r, k) if k in ('package', 'capacity', 'unit') else ws.cell(r, idx[k]).value
               for k in idx}
        rec['group_id'] = gid
        rec['excel_row'] = r
        rec['cost_shared'] = cost_n > 1               # cost merged across skids
        rec['unit_cost_raw'] = val(r, 'unit_cost')    # top-left value if merged
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['skid_no'] = df.groupby('group_id').cumcount() + 1
    df['n_skids'] = df.groupby('group_id')['group_id'].transform('size')
    df['package_clean'] = df['package'].apply(lambda s: str(s).strip().upper() if pd.notna(s) else None)
    df['capacity_num'] = df['capacity'].apply(first_num)
    df['unit_clean'] = df['unit'].apply(lambda s: str(s).strip() if pd.notna(s) else 'Unknown')
    df['year_num'] = df['year'].apply(first_num)
    df['qty_num'] = df['qty'].apply(first_num)
    df['unit_cost_num'] = df['unit_cost_raw'].apply(first_num)
    df['sub_total_num'] = df['sub_total'].apply(first_num)
    for k in ['power_duty', 'power_abs', 'design_press', 'design_temp_max', 'design_temp_min',
              'length', 'width', 'height', 'wt_unit_dry', 'wt_unit_oper', 'wt_tot_dry', 'wt_tot_oper', 'wt_test']:
        df[k + '_num'] = df[k].apply(first_num)
    df['material_clean'] = df['material'].apply(categorize_material)
    # Package cost: shared cost counted once; separate per-skid costs are summed
    def pkg_cost(g):
        return g['unit_cost_num'].iloc[0] if g['cost_shared'].iloc[0] else g['unit_cost_num'].sum(min_count=1)
    costs = df.groupby('group_id').apply(pkg_cost).rename('package_cost')
    df = df.merge(costs, left_on='group_id', right_index=True)
    return df

PACKAGE_LIST_SHEET = 'LIST PACKAGE'

@st.cache_data
def load_package_list(file):
    """Reads the master 'LIST PACKAGE' sheet -> ordered, de-duplicated (package, unit) pairs."""
    raw = pd.read_excel(file, sheet_name=PACKAGE_LIST_SHEET, header=0)
    raw = raw.iloc[:, 1:3]
    raw.columns = ['package', 'unit']
    raw = raw.dropna(subset=['package'])
    raw['package'] = raw['package'].astype(str).str.strip().str.upper()
    raw['unit'] = raw['unit'].apply(lambda u: str(u).strip() if pd.notna(u) else 'Unknown')
    return raw.drop_duplicates().reset_index(drop=True)


def _norm_unit(u):
    """Small spelling differences (m3/h vs m3/hr) must not split a package into two units."""
    return re.sub(r'\s+', '', str(u).strip().lower()).replace('m3/hr', 'm3/h')


# =====================================================================
# 7.95 PAGE BUILDER: PACKAGE
# Match logic (same idea as Heat Exchanger): compare the requested capacity against the
# historical database and show only the 2 closest references that BRACKET it (one at/below,
# one at/above). The bigger one is the recommendation.
# =====================================================================
def select_package_references(heads: pd.DataFrame, user_capacity: float):
    """
    heads = one row per package record (skid 1 of each package).
    Returns (list_of_group_ids_to_show, recommended_group_id, status).
      status: 'exact' | 'between' | 'below_all' | 'above_all'
    If several records share the same capacity, only the LATEST year is kept for that capacity.
    """
    g = heads.sort_values(['capacity_num', 'year_num', 'excel_row'], ascending=[True, False, True])
    g = g.drop_duplicates('capacity_num', keep='first').reset_index(drop=True)

    lower_m = g[g['capacity_num'] <= user_capacity]
    upper_m = g[g['capacity_num'] >= user_capacity]
    lower = lower_m.iloc[-1] if len(lower_m) else None
    upper = upper_m.iloc[0] if len(upper_m) else None

    if lower is not None and upper is not None and lower['capacity_num'] == upper['capacity_num']:
        return [lower['group_id']], lower['group_id'], 'exact'
    if lower is not None and upper is not None:
        return [lower['group_id'], upper['group_id']], upper['group_id'], 'between'
    if upper is not None:      # requested capacity is smaller than everything in the database
        return upper_m['group_id'].iloc[:2].tolist(), upper['group_id'], 'below_all'
    # requested capacity is bigger than everything in the database
    return lower_m['group_id'].iloc[-2:].tolist(), lower['group_id'], 'above_all'


def _pfmt(v, d=2):
    return f"{v:,.{d}f}" if pd.notna(v) else "n/a"


def _wsum(g, col):
    """Sum of a weight column across skids (NaN if no skid has a value)."""
    s = pd.to_numeric(g[col], errors='coerce').sum(min_count=1)
    return s


def _package_totals(g):
    """Package-level totals used in both the comparison table and the detail section."""
    head = g.iloc[0]
    n = int(head['n_skids'])
    qty = head['qty_num'] if pd.notna(head['qty_num']) else 1
    if n == 1:
        if pd.notna(head['sub_total_num']):
            cost = head['sub_total_num']                       # SUB TOTAL as recorded in the database
        else:
            cost = head['package_cost'] * qty if pd.notna(head['package_cost']) else float('nan')
    else:
        cost = head['package_cost']   # multi-skid: shared cost, or sum of per-skid costs
    return dict(n=n, qty=qty, cost=cost,
                dry=_wsum(g, 'wt_unit_dry_num'), oper=_wsum(g, 'wt_unit_oper_num'), test=_wsum(g, 'wt_test_num'))


# (label, dataframe column, was green-highlighted in the Excel database, money?)
DB_FIELDS = [
    ('Equip No.', 'equip_no', True, False), ('Qty', 'qty', True, False), ('Description', 'description', True, False),
    ('PFD / UFD No.', 'pfd', False, False), ('Location', 'location', False, False),
    ('List of Package', 'package', False, False), ('Capacity', 'capacity', False, False), ('Unit', 'unit', False, False),
    ('Power/Unit Duty (kW)', 'power_duty', True, False), ('Power/Unit Absorbed (kW)', 'power_abs', True, False),
    ('Design Press. (barg)', 'design_press', True, False), ('Design Temp Max (°C)', 'design_temp_max', True, False),
    ('Design Temp Min (°C)', 'design_temp_min', True, False),
    ('Oper. Press. (barg)', 'oper_press', False, False), ('Oper. Temp (°C)', 'oper_temp', False, False),
    ('Length (mm)', 'length', True, False), ('Width/ID (mm)', 'width', True, False), ('Height (mm)', 'height', True, False),
    ('Unit Dry (MT)', 'wt_unit_dry', False, False), ('Unit Oper. (MT)', 'wt_unit_oper', False, False),
    ('Tot. Dry (MT)', 'wt_tot_dry', False, False), ('Tot. Oper. (MT)', 'wt_tot_oper', False, False),
    ('Test (MT)', 'wt_test', False, False),
    ('Material', 'material', True, False), ('Orientation', 'orientation', False, False),
    ('Unit Cost (MYR)', 'unit_cost', True, True), ('Sub Total (MYR)', 'sub_total', False, True),
    ('Remarks', 'remarks', False, False), ('Project', 'project', False, False),
    ('Year', 'year', True, False), ('Folder', 'folder', False, False),
]


def _disp(v, money=False):
    """Show a database cell as recorded (text stays text, numbers tidied, blanks empty)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,.2f}" if money else (f"{int(v)}" if float(v).is_integer() else f"{round(v, 4):g}")
    return str(v).strip()


def _db_values(r):
    """{label: value} for ALL database columns of one skid row."""
    out = {}
    for label, key, _, money in DB_FIELDS:
        v = r[key]
        if key in ('unit_cost', 'sub_total') and bool(r['cost_shared']) and int(r['skid_no']) > 1:
            v = '(shared with Skid 1)'      # merged cell in Excel: one cost for the whole package
        out[label] = _disp(v, money)
    return out


def _render_skid_metrics(r):
    """Weights, then dimensions, of one skid."""
    w1, w2, w3 = st.columns(3)
    w1.metric("Unit Dry (MT)", _pfmt(r['wt_unit_dry_num']))
    w2.metric("Unit Operating (MT)", _pfmt(r['wt_unit_oper_num']))
    w3.metric("Test (MT)", _pfmt(r['wt_test_num']))
    st.markdown("**📏 Dimensions (mm)**")
    d1, d2, d3 = st.columns(3)
    d1.metric("Length", _pfmt(r['length_num'], 0))
    d2.metric("Width / ID", _pfmt(r['width_num'], 0))
    d3.metric("Height", _pfmt(r['height_num'], 0))


def _render_package_metrics(g, heading):
    head = g.iloc[0]
    t = _package_totals(g)
    n = t['n']
    st.markdown(f"### {heading}")
    st.caption(f"{head['equip_no']} — {head['description']} · {head['capacity_num']:g} {head['unit_clean']} · "
               f"{head['project']} ({int(head['year_num'])})" + (f" · {n} skids" if n > 1 else ""))

    if n == 1:
        _render_skid_metrics(head)
        k1, k2, k3 = st.columns(3)
        k1.metric("Unit Cost (MYR)", _pfmt(head['package_cost']))
        k2.metric("Quantity", f"{t['qty']:g}")
        k3.metric("Total Cost (MYR)", _pfmt(t['cost']))
        return

    # Multi-skid: Skid 1, Skid 2, ... each with its weights and dimensions, then the package total
    for _, r in g.iterrows():
        st.markdown(f"#### Skid {int(r['skid_no'])} — {r['equip_no']} · {r['description']}")
        _render_skid_metrics(r)
        if not head['cost_shared']:
            st.metric(f"Skid {int(r['skid_no'])} Unit Cost (MYR)", _pfmt(r['unit_cost_num']))
        st.markdown("---")

    st.markdown("#### 📦 Package Total")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Package Total Cost (MYR)", _pfmt(t['cost']))
    p2.metric("Total Unit Dry (MT)", _pfmt(t['dry']))
    p3.metric("Total Unit Operating (MT)", _pfmt(t['oper']))
    p4.metric("Total Test (MT)", _pfmt(t['test']))
    st.caption(("One cost covers all skids (merged cell in Excel). " if head['cost_shared']
                else "Each skid has its own unit cost; package cost is their sum. ")
               + "Package weights are the sum of the skids' unit weights. Quantity is not applied to multi-skid packages.")


def package_page():
    st.title("Package Spec & Cost Lookup")
    st.caption("Filtered by List of Package, then the requested Capacity is compared against the historical "
               "database. The 2 closest references (one at/below, one at/above) are shown and the bigger one is "
               "recommended. Packages made of several skids (merged rows in Excel, e.g. AGRU) are shown "
               "with one row per skid, followed by a package total row.")

    uploaded = st.file_uploader("Upload package database (.xlsx)", type=["xlsx"], key="pkg_up")
    if uploaded is None:
        st.info(f"Upload the package database Excel file to begin. Expects sheet '{PACKAGE_SHEET_NAME}'.")
        st.stop()
    try:
        with st.spinner("Reading and cleaning the package workbook..."):
            df = load_and_clean_packages(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    for c in ['length', 'width', 'height', 'power_duty', 'power_abs', 'design_press',
              'design_temp_max', 'design_temp_min', 'wt_unit_dry', 'wt_unit_oper', 'wt_test']:
        if c + '_num' not in df.columns:
            df[c + '_num'] = None

    with st.expander("Preview cleaned data"):
        st.dataframe(df[['package_clean', 'capacity_num', 'unit_clean', 'group_id', 'skid_no', 'n_skids',
                         'cost_shared', 'unit_cost_num', 'package_cost', 'year_num', 'equip_no', 'description']],
                     use_container_width=True)

    # The LIST PACKAGE sheet is the reference for the dropdown (package + its unit).
    # If the sheet is missing, fall back to the packages/units found in the data.
    try:
        uploaded.seek(0)
        pkg_list = load_package_list(uploaded)
    except Exception:
        pkg_list = (df.dropna(subset=['package_clean'])[['package_clean', 'unit_clean']]
                    .rename(columns={'package_clean': 'package', 'unit_clean': 'unit'})
                    .drop_duplicates().sort_values(['package', 'unit']).reset_index(drop=True))
        st.warning(f"Sheet '{PACKAGE_LIST_SHEET}' not found — using the packages found in '{PACKAGE_SHEET_NAME}' instead.")

    # When the list gives a package ONE unit, that unit is used for its database records
    # (this also fixes records whose UNIT cell was typed wrongly, e.g. A-6800 "CHEMICAL INJECTION").
    single_unit = pkg_list.groupby('package')['unit'].agg(lambda x: x.iloc[0] if x.nunique() == 1 else None).dropna()
    df['unit_clean'] = [single_unit.get(pk, u) for pk, u in zip(df['package_clean'], df['unit_clean'])]
    df['unit_key'] = df['unit_clean'].apply(_norm_unit)

    # A package record is usable if its first skid has package, capacity and year
    ok_groups = df[df['skid_no'] == 1].dropna(subset=['package_clean', 'capacity_num', 'year_num'])['group_id']
    usable = df[df['group_id'].isin(ok_groups)]
    st.divider()
    st.subheader("Lookup specs & cost for a new package")

    with st.form("package_prediction_form"):
        st.markdown("**Search Parameters**")
        pkg_options = {f"{p} ({u})": (p, u) for p, u in pkg_list.itertuples(index=False, name=None)}
        c1, c2 = st.columns(2)
        with c1:
            pkg_choice = st.selectbox("List of Package (Capacity Unit)", list(pkg_options.keys()))
        with c2:
            user_capacity = st.number_input("Required Capacity", min_value=0.0, value=10.0, step=1.0)
        submitted = st.form_submit_button("Lookup Existing Data", use_container_width=True, type="primary")

    if not submitted:
        return

    user_pkg, user_unit = pkg_options[pkg_choice]
    m = usable[(usable['package_clean'] == user_pkg) & (usable['unit_key'] == _norm_unit(user_unit))]
    if m.empty:
        st.error(f"No historical '{user_pkg}' record with a capacity in {user_unit} exists in the database yet.")
        st.stop()

    ids, rec_id, status = select_package_references(m[m['skid_no'] == 1], user_capacity)
    unit_lbl = user_unit
    recs = {gid: m[m['group_id'] == gid].sort_values('skid_no') for gid in ids}
    cap_of = lambda gid: recs[gid].iloc[0]['capacity_num']
    eq_of = lambda gid: recs[gid].iloc[0]['equip_no']

    if status == 'exact':
        st.success(f"Exact historical match found at **{cap_of(rec_id):,.2f} {unit_lbl}** (Equip No. {eq_of(rec_id)}).")
    elif status == 'between':
        lo, hi = ids
        st.info(f"Requested capacity **{user_capacity:,.2f} {unit_lbl}** falls between historical capacities "
                f"**{cap_of(lo):,.2f}** ({eq_of(lo)}) and **{cap_of(hi):,.2f}** ({eq_of(hi)}).")
    elif status == 'below_all':
        st.warning(f"Requested capacity **{user_capacity:,.2f} {unit_lbl}** is smaller than every historical record for "
                   f"'{user_pkg}'. No smaller reference exists — recommending the closest bigger value.")
    else:
        st.warning(f"Requested capacity **{user_capacity:,.2f} {unit_lbl}** exceeds every historical record for "
                   f"'{user_pkg}'. No bigger value is available — recommending the largest historical reference instead.")
    if len(ids) < 2 and status != 'exact':
        st.caption("Only 1 historical record exists for this package — showing the single available reference.")

    labels = {gid: ("✅ Recommended (Bigger)" if gid == rec_id else "Reference") for gid in ids}

    # Reference table: ALL database columns for each reference (one row per skid)
    st.markdown("### 🎯 Reference Table")
    st.caption("All columns of the database are shown.")
    rows = []
    for gid in ids:
        g = recs[gid]
        for _, r in g.iterrows():
            row = {'Match': labels[gid]}
            if int(r['n_skids']) > 1:
                row['Skid'] = str(int(r['skid_no']))
            row.update(_db_values(r))
            rows.append(row)
        if int(g.iloc[0]['n_skids']) > 1:      # multi-skid: add the package total
            t = _package_totals(g)
            rows.append({'Match': labels[gid], 'Skid': 'Total', 'Equip No.': 'PACKAGE TOTAL',
                         'Unit Dry (MT)': _disp(t['dry']), 'Unit Oper. (MT)': _disp(t['oper']),
                         'Tot. Dry (MT)': _disp(_wsum(g, 'wt_tot_dry_num')),
                         'Tot. Oper. (MT)': _disp(_wsum(g, 'wt_tot_oper_num')),
                         'Test (MT)': _disp(t['test']),
                         'Sub Total (MYR)': _disp(t['cost'], money=True)})
    ref_df = pd.DataFrame(rows).fillna('')
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    # Key values (weights, dimensions, cost) for each reference
    for gid in ids:
        st.divider()
        _render_package_metrics(recs[gid], labels[gid])

    st.divider()
    st.success(f"**Recommendation:** Use the values from **{eq_of(rec_id)}** at **{cap_of(rec_id):,.2f} {unit_lbl}**"
               + (" — the bigger of the two compared values." if len(ids) > 1 else "."))

# =====================================================================
# 7.97 PUMP (DISCIPLINE) CONFIG & HELPERS
# Inputs : Type of Pump (list comes from Sheet2) + Flowrate CAPACITY (m3/h) + dP CAPACITY (bar)
# Outputs: (a) Flowrate comparison  -> closest lower & closest higher reference, bigger one proceeds
#          (b) dP comparison        -> done among the rows that proceeded from (a), bigger one is recommended
# =====================================================================
PUMP_DATA_SHEET = 'Sheet1'      # pump database
PUMP_TYPE_SHEET = 'Sheet2'      # master list of pump types (reference list)

# Column positions (0-indexed, after skiprows=4) matching "Overall_Pump_Database_Rev1.xlsx" / 'Sheet1'
PUMP_COL_MAP = {
    'equip_no':     1,    # EQUIPMENT NO. (BED)
    'qty':          2,    # QTY
    'description':  3,    # DESCRIPTION
    'pump_type':    6,    # Type of Pump
    'flow':         7,    # Flowrate CAPACITY (m3/h)  (input)
    'dp':           8,    # dP CAPACITY (bar)         (input)
    'detail_cap':   9,    # Detail CAPACITY
    'power_duty':   10,   # POWER/UNIT - DUTY (kW)
    'power_abs':    11,   # POWER/UNIT - ABSORBED (kW)
    'design_press': 12,   # DESIGN CONDITIONS - PRESS. (barg)
    'design_tmax':  13,   # DESIGN CONDITIONS - TEMP. Max (oC)
    'design_tmin':  14,   # DESIGN CONDITIONS - TEMP. Min (oC)
    'length':       17,   # DIMENSIONS/UNIT - LENGTH (mm)
    'width_id':     18,   # DIMENSIONS/UNIT - WIDTH/ID (mm)
    'height':       19,   # DIMENSIONS/UNIT - HEIGHT (mm)
    'wt_dry':       20,   # UNIT DRY (MT)
    'wt_oper':      21,   # UNIT OPER. (MT)
    'wt_test':      24,   # TEST (MT)
    'material':     25,   # MATERIAL
    'orientation':  26,   # ORIENTATION
    'unit_cost':    27,   # UNIT COST (MYR)
    'remarks':      29,   # REMARKS
    'project':      30,   # NAME OF PROJECT
    'year':         31,   # YEAR
}

# (dataframe column, header shown in the output table) - exactly the requested output columns
PUMP_DISPLAY = [
    ('equip_no',     'EQUIPMENT NO. (BED)'),
    ('qty',          'QTY'),
    ('description',  'DESCRIPTION'),
    ('detail_cap',   'Detail CAPACITY'),
    ('power_duty',   'POWER/UNIT - Duty (kW)'),
    ('power_abs',    'POWER/UNIT - Absorbed (kW)'),
    ('design_press', 'DESIGN PRESS. (barg)'),
    ('design_tmax',  'DESIGN TEMP. Max (°C)'),
    ('design_tmin',  'DESIGN TEMP. Min (°C)'),
    ('length',       'LENGTH (mm)'),
    ('width_id',     'WIDTH/ID (mm)'),
    ('height',       'HEIGHT (mm)'),
    ('wt_dry',       'UNIT DRY (MT)'),
    ('wt_oper',      'UNIT OPER (MT)'),
    ('wt_test',      'TEST (MT)'),
    ('material',     'MATERIAL'),
    ('orientation',  'ORIENTATION'),
    ('unit_cost',    'UNIT COST (MYR)'),
    ('remarks',      'REMARKS'),
    ('project',      'NAME OF PROJECT'),
    ('year',         'YEAR'),
]

def _pump_norm(s):
    """Collapses repeated spaces/new-lines so 'Pump -  API 675' and 'Pump - API 675' match."""
    return re.sub(r'\s+', ' ', str(s).strip()) if pd.notna(s) else ''

@st.cache_data
def load_and_clean_pumps(file):
    """Loads the pump database (Sheet1) and the master pump-type list (Sheet2)."""
    raw = pd.read_excel(file, sheet_name=PUMP_DATA_SHEET, header=None, skiprows=4)
    df = pd.DataFrame({k: raw[i] for k, i in PUMP_COL_MAP.items()})

    df['pump_type'] = df['pump_type'].apply(_pump_norm)
    df['flow_num'] = df['flow'].apply(first_num)
    df['dp_num'] = df['dp'].apply(first_num)
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    # Keep only real pump rows (must have a type, a flowrate and a dP to be comparable)
    df = df[(df['pump_type'] != '') & df['flow_num'].notna() & df['dp_num'].notna()].reset_index(drop=True)

    types_raw = pd.read_excel(file, sheet_name=PUMP_TYPE_SHEET, header=None)
    master_types = [_pump_norm(t) for t in types_raw[1].dropna()
                    if _pump_norm(t) and _pump_norm(t).lower() != 'type of pump']
    return df, master_types

def select_pump_references(type_df: pd.DataFrame, user_flow: float, user_dp: float):
    """
    Step a) Flowrate: find the closest LOWER and closest HIGHER flowrate reference.
            The bigger (higher) flowrate proceeds to step b.
    Step b) dP: among the rows at the proceeding flowrate, find the closest LOWER and
            closest HIGHER dP. The bigger (higher) dP is the recommendation.
    Returns a dict with the rows/indices of every stage plus plain-language notes.
    """
    notes = []
    flows = sorted(type_df['flow_num'].unique())

    exact_flow = user_flow in flows
    lower_flows = [f for f in flows if f < user_flow]
    upper_flows = [f for f in flows if f > user_flow]

    if exact_flow:
        lower_flow = upper_flow = user_flow
        notes.append(f"Exact flowrate match at {user_flow:g} m³/h.")
    else:
        lower_flow = lower_flows[-1] if lower_flows else None
        upper_flow = upper_flows[0] if upper_flows else None

    if upper_flow is not None:
        proceed_flow = upper_flow
        if lower_flow is None:
            notes.append(f"Flowrate {user_flow:g} m³/h is below every record for this type - no lower reference exists. "
                         f"Proceeding with the closest bigger flowrate ({upper_flow:g} m³/h).")
        elif not exact_flow:
            notes.append(f"Flowrate {user_flow:g} m³/h falls between {lower_flow:g} and {upper_flow:g} m³/h. "
                         f"The bigger value ({upper_flow:g} m³/h) proceeds to the dP comparison.")
    else:
        proceed_flow = lower_flow
        notes.append(f"Flowrate {user_flow:g} m³/h exceeds every record for this type - no bigger reference exists. "
                     f"Proceeding with the largest available flowrate ({lower_flow:g} m³/h).")

    lower_flow_rows = type_df[type_df['flow_num'] == lower_flow] if lower_flow is not None else type_df.iloc[0:0]
    upper_flow_rows = type_df[type_df['flow_num'] == upper_flow] if upper_flow is not None else type_df.iloc[0:0]
    proceed_rows = type_df[type_df['flow_num'] == proceed_flow]

    # ---- Step b: dP comparison among the rows that proceeded from step a ----
    dps = sorted(proceed_rows['dp_num'].unique())
    exact_dp = user_dp in dps
    lower_dps = [d for d in dps if d < user_dp]
    upper_dps = [d for d in dps if d > user_dp]

    if exact_dp:
        lower_dp = upper_dp = user_dp
        notes.append(f"Exact dP match at {user_dp:g} bar.")
    else:
        lower_dp = lower_dps[-1] if lower_dps else None
        upper_dp = upper_dps[0] if upper_dps else None

    if upper_dp is not None:
        rec_dp = upper_dp
        if len(dps) == 1 and not exact_dp:
            notes.append(f"Only one dP value ({upper_dp:g} bar) exists at {proceed_flow:g} m³/h, which is higher than the "
                         f"requested {user_dp:g} bar - it is taken as the reference.")
        elif lower_dp is None:
            notes.append(f"dP {user_dp:g} bar is below every record at {proceed_flow:g} m³/h - no lower reference exists. "
                         f"Recommending the closest bigger dP ({upper_dp:g} bar).")
        elif not exact_dp:
            notes.append(f"dP {user_dp:g} bar falls between {lower_dp:g} and {upper_dp:g} bar at {proceed_flow:g} m³/h. "
                         f"The bigger value ({upper_dp:g} bar) is recommended.")
    else:
        rec_dp = lower_dp
        notes.append(f"dP {user_dp:g} bar exceeds every record at {proceed_flow:g} m³/h - no bigger reference exists. "
                     f"Recommending the largest available dP ({lower_dp:g} bar); this is BELOW the requested duty, "
                     f"so an engineering check is required before using it.")

    lower_dp_rows = proceed_rows[proceed_rows['dp_num'] == lower_dp] if lower_dp is not None else proceed_rows.iloc[0:0]
    upper_dp_rows = proceed_rows[proceed_rows['dp_num'] == upper_dp] if upper_dp is not None else proceed_rows.iloc[0:0]
    rec_candidates = proceed_rows[proceed_rows['dp_num'] == rec_dp]
    # If several historical rows share the same flow & dP, take the most recent project year
    rec_row = rec_candidates.sort_values('year_num', ascending=False, na_position='last').iloc[0]

    return dict(notes=notes, lower_flow=lower_flow, upper_flow=upper_flow, proceed_flow=proceed_flow,
                lower_flow_rows=lower_flow_rows, upper_flow_rows=upper_flow_rows,
                lower_dp=lower_dp, upper_dp=upper_dp, rec_dp=rec_dp,
                lower_dp_rows=lower_dp_rows, upper_dp_rows=upper_dp_rows,
                rec_row=rec_row, n_same_spec=len(rec_candidates))

def _pump_table(rows: pd.DataFrame, match_labels=None):
    """Builds a display table with ALL requested reference columns (text-safe for Streamlit)."""
    out = pd.DataFrame({hdr: rows[col] for col, hdr in PUMP_DISPLAY})
    out = out.apply(lambda c: c.map(lambda v: '-' if pd.isna(v) else
                                    (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))))
    if match_labels is not None:
        out.insert(0, 'Match', match_labels)
    return out

# =====================================================================
# 7.97 PAGE BUILDER: PUMP
# =====================================================================
def pump_page():
    st.title("Pump Spec & Cost Lookup")
    st.caption("Strict historical lookup (no interpolation). Select the Type of Pump, Flowrate and dP. "
               "Step a) the flowrate is compared against the closest lower and closest higher references - the bigger "
               "value proceeds. Step b) the dP is compared among the references from step a) - the bigger value "
               "is recommended.")

    uploaded = st.file_uploader("Upload pump database (.xlsx)", type=["xlsx"], key="pump_up")
    if uploaded is None:
        st.info(f"Upload the pump database Excel file to begin. Expects sheets '{PUMP_DATA_SHEET}' (data) "
                f"and '{PUMP_TYPE_SHEET}' (pump type list).")
        st.stop()

    try:
        with st.spinner("Reading and cleaning the pump workbook..."):
            df, master_types = load_and_clean_pumps(uploaded)
    except Exception as e:
        st.error(f"Error processing workbook: {e}")
        st.stop()

    with st.expander("Preview cleaned data"):
        st.dataframe(_pump_table(df), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Lookup specs & cost for a new pump")

    # Type list comes from Sheet2; add any type that exists in the data but not in the list
    type_options = list(master_types)
    for t in sorted(df['pump_type'].unique()):
        if t not in type_options:
            type_options.append(t)

    with st.form("pump_prediction_form"):
        st.markdown("**Search Parameters**")
        user_type = st.selectbox("Type of Pump", type_options)
        c1, c2 = st.columns(2)
        with c1: user_flow = st.number_input("Flowrate CAPACITY (m³/h)", min_value=0.0, value=10.0, step=1.0, format="%.3f")
        with c2: user_dp = st.number_input("dP CAPACITY (bar)", min_value=0.0, value=5.0, step=0.5, format="%.2f")
        submitted = st.form_submit_button("Lookup Existing Data", use_container_width=True, type="primary")

    if not submitted:
        return

    type_df = df[df['pump_type'] == user_type].copy()
    if len(type_df) == 0:
        st.warning(f"⚠️ '{user_type}' is unavailable at the moment - there is no historical data for this pump type. "
                   f"Please select another type.")
        st.stop()

    res = select_pump_references(type_df, user_flow, user_dp)

    # ---------------- Step a ----------------
    st.markdown("### Flowrate comparison (closest lower & higher reference)")
    for n in res['notes']:
        st.info(n)

    a_rows, a_labels = [], []
    if res['lower_flow'] is not None and res['lower_flow'] != res['proceed_flow']:
        a_rows.append(res['lower_flow_rows']); a_labels += [f"Lower flow ({res['lower_flow']:g} m³/h)"] * len(res['lower_flow_rows'])
    a_rows.append(res['upper_flow_rows'] if res['upper_flow'] is not None else res['lower_flow_rows'])
    a_labels += [f"✅ Bigger flow ({res['proceed_flow']:g} m³/h) → proceeds to dP"] * len(a_rows[-1])

    # ---------------- Step b ----------------
    st.markdown(f"### dP comparison at {res['proceed_flow']:g} m³/h")
    b_rows, b_labels = [], []
    if res['lower_dp'] is not None and res['lower_dp'] != res['rec_dp']:
        b_rows.append(res['lower_dp_rows']); b_labels += [f"Lower dP ({res['lower_dp']:g} bar)"] * len(res['lower_dp_rows'])
    rec_rows = res['upper_dp_rows'] if res['upper_dp'] is not None else res['lower_dp_rows']
    b_rows.append(rec_rows); b_labels += [f"✅ Bigger dP ({res['rec_dp']:g} bar) → recommended"] * len(rec_rows)
    st.dataframe(_pump_table(pd.concat(b_rows), b_labels), use_container_width=True, hide_index=True)

    # ---------------- Recommendation ----------------
    rec = res['rec_row']
    st.markdown("### 🎯 Recommended Pump Specification")
    rec_df = rec.to_frame().T
    st.dataframe(_pump_table(rec_df), use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unit Cost (MYR)", f"{rec['unit_cost']:,.2f}" if isinstance(rec['unit_cost'], (int, float)) and pd.notna(rec['unit_cost']) else str(rec['unit_cost']))
    m2.metric("Unit Dry (MT)", str(rec['wt_dry']))
    m3.metric("Unit Oper. (MT)", str(rec['wt_oper']))
    m4.metric("Test (MT)", str(rec['wt_test']))

    # ---------------- Step c: compare ALL rows similar to the recommended pump ----------------
    similar = type_df[type_df['flow_num'] == rec['flow_num']].copy()
    if len(similar) > 1:
        st.markdown(f"### All similar pumps at {rec['flow_num']:g} m³/h ({len(similar)} records)")
        st.caption("Every historical pump of this type with the same flowrate as the recommended pump, "
                   "sorted by dP, so the specs, weights and costs can be compared side by side.")
        similar = similar.sort_values(['dp_num', 'year_num'], ascending=[True, False])
        rec_cost = pd.to_numeric(rec['unit_cost'], errors='coerce')

        def _label(r):
            if r.name == rec.name:
                return "✅ Recommended"
            if r['dp_num'] == rec['dp_num']:
                return "Same flow & dP"
            return "Higher dP" if r['dp_num'] > rec['dp_num'] else "Lower dP"

        cmp_df = _pump_table(similar, [_label(r) for _, r in similar.iterrows()])
        cmp_df.insert(1, 'Flowrate (m³/h)', similar['flow_num'].map(lambda v: f"{v:g}").values)
        cmp_df.insert(2, 'dP (bar)', similar['dp_num'].map(lambda v: f"{v:g}").values)
        costs = pd.to_numeric(similar['unit_cost'], errors='coerce')
        cmp_df['Cost vs Recommended'] = [
            '-' if pd.isna(c) or pd.isna(rec_cost) or rec_cost == 0 else f"{(c - rec_cost) / rec_cost * 100:+.1f}%"
            for c in costs]
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        n_same = int((similar['dp_num'] == rec['dp_num']).sum())
        if n_same > 1:
            st.caption(f"{n_same} records share both this flowrate and dP; the most recent project year is "
                       f"marked as recommended.")

    st.success(f"**Recommendation:** Use the values from **{rec['equip_no']}** ({rec['project']}, {rec['year']}) - "
               f"{rec['flow_num']:g} m³/h @ {rec['dp_num']:g} bar - the bigger of the compared values for "
               f"your input of {user_flow:g} m³/h @ {user_dp:g} bar.")

# =====================================================================
# 8. NAVIGATION MENU
# =====================================================================
PAGES = {
    "Mechanical (Vessel)": mechanical_page, 
    "Mechanical (Tank)": tank_page,
    "Mechanical (Filter)": filter_page,
    "Mechanical (Heat Exchanger)": heat_exchanger_page,
    "Mechanical (Tote Tank)": tote_tank_page,
    "Mechanical (Launcher)": launcher_page,
    "Package": package_page,
    "Pump": pump_page,
    "Piping (Valve)": piping_page,         
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Discipline", list(PAGES.keys()), key="nav_selection")
PAGES[selected_page]()