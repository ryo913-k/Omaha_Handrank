import streamlit as st
# 設定は一番最初
st.set_page_config(page_title="Omaha Hand Analyzer", layout="wide")

# カスタムCSS
st.markdown("""
<style>
    ul[data-testid="stSelectboxVirtualDropdown"] { z-index: 99999 !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    /* Draw Pattern Box */
    .draw-box {
        border: 1px solid #ddd; border-left: 5px solid #ccc;
        background-color: #fff; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .draw-info { flex-grow: 1; }
    .draw-title { font-weight: bold; font-size: 15px; color: #333; }
    .draw-sub { font-size: 12px; color: #666; margin-top: 2px;}
    .draw-stat { text-align: right; min-width: 80px; }
    .draw-count { font-size: 14px; font-weight:bold; color: #333; }
    .draw-pct { font-size: 11px; color: #888; }
</style>
""", unsafe_allow_html=True)

# Imports
from data_loader import load_plo_data, load_flo8_data
from plo_logic import render_plo_preflop, render_plo_postflop
from flo8_logic import render_flo8

# Initialize Session State
if 'plo_input' not in st.session_state: st.session_state.plo_input = "As Ks Jd Th"
if 'flo8_input' not in st.session_state: st.session_state.flo8_input = "Ad Ah 2s 3d"
if 'plo_input_text' not in st.session_state: st.session_state.plo_input_text = st.session_state.plo_input
if 'flo8_input_text' not in st.session_state: st.session_state.flo8_input_text = st.session_state.flo8_input
if 'pf_board' not in st.session_state: st.session_state.pf_board = ""
if 'pf_board_text' not in st.session_state: st.session_state.pf_board_text = ""

# Load Data
df_plo = load_plo_data()
df_flo8 = load_flo8_data()

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "PLO Board Analyzer", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# Main Routing
if game_mode == "PLO (High Only)":
    render_plo_preflop(df_plo)

elif game_mode == "PLO Board Analyzer":
    render_plo_postflop(df_plo)

elif game_mode == "FLO8 (Hi/Lo)":
    render_flo8(df_flo8)

elif game_mode == "Guide":
    st.header("📖 Guide")
    st.markdown("""
    **Omaha Hand Analyzer**
    
    * **PLO (High Only)**: プリフロップのハンド強度、エクイティ分布、フィルタリング分析。
    * **PLO Board Analyzer**: 指定したボードにおけるレンジ分析。ドローの強さや分布を可視化。
    * **FLO8 (Hi/Lo)**: ハイローハンドのHutchinsonポイントとスクープ率。
    """)

# Footer
with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")