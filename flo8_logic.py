import streamlit as st
from utils import normalize_input_text, render_hand_html, render_card_selector, set_input_callback
from heuristics import calculate_flo8_heuristic

def render_flo8(df_flo8):
    with st.sidebar:
        with st.expander("1. 🔍 Hand Rank", expanded=True):
            if df_flo8 is not None:
                c_rk8_1, c_rk8_2 = st.columns([1,2])
                with c_rk8_1:
                    srk8 = st.number_input("Rank", 1, len(df_flo8), 1, key="prk_flo8", label_visibility="collapsed")
                with c_rk8_2:
                    fr8 = df_flo8[df_flo8['rank']==srk8]
                    if not fr8.empty:
                        r8_found = fr8.iloc[0]
                        if st.button("Analyze", key="bcp_flo8"):
                             set_input_callback('flo8_input', r8_found['hand'])
                             st.rerun()
                    else: st.write("-")
                if not fr8.empty:
                    st.caption(f"**{r8_found['hand']}** (Top {r8_found['pct_total']:.2f}%)")
            else: st.write("Data not loaded")

    st.header("⚖️ FLO8 Strategy")
    render_card_selector('flo8_input')
    inp8_raw = st.text_input("FLO8 Hand (Text)", key='flo8_input_text')
    
    if inp8_raw != st.session_state.flo8_input: st.session_state.flo8_input = inp8_raw

    i8 = normalize_input_text(st.session_state.flo8_input)
    if i8: st.markdown(render_hand_html(" ".join(i8)), unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        if len(i8)==4:
            sc, dt = calculate_flo8_heuristic(" ".join(i8))
            st.metric("Hutchinson Points", sc, help="Target: 20+ to play")
            st.bar_chart(dt)
        else: st.info("Enter 4 cards.")

    with c2:
        if df_flo8 is not None and len(i8)==4:
            r8 = df_flo8[df_flo8["card_set"]==frozenset(i8)]
            if not r8.empty:
                rr = r8.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Scoop %", f"{rr['scoop_pct']:.1f}%")
                m2.metric("High Eq", f"{rr['high_equity']:.1f}%")
                m3.metric("Low Eq", f"{rr['low_equity']:.1f}%")
                st.caption(f"Rank: #{rr['rank']} (Top {rr['pct_total']:.1f}%)")
            else: st.warning("Not found.")