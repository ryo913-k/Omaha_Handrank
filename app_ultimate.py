import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
import random
from collections import Counter, defaultdict
from itertools import combinations
from heuristics import calculate_flo8_heuristic

# ==========================================
# 1. Config & Styles
# ==========================================
st.set_page_config(page_title="Omaha Hand Analyzer", layout="wide")

st.markdown("""
<style>
    /* マルチセレクトのドロップダウン調整 */
    ul[data-testid="stSelectboxVirtualDropdown"] { z-index: 99999 !important; }
    /* サイドバーの余白調整 */
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    /* スマホUI調整 */
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    
    /* Draw Pattern Box (Board Analyzer用) */
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

# ==========================================
# 2. Helper Classes & Functions
# ==========================================
class SimpleCard:
    def __init__(self, card_str):
        if not card_str:
            self.rank = -1; self.suit = ''; return
        rank_char = card_str[:-1].upper()
        ranks = "23456789TJQKA"
        self.rank = ranks.index(rank_char) if rank_char in ranks else -1
        self.suit = card_str[-1].lower()
    def __repr__(self): return f"{self.rank}{self.suit}"

def normalize_input_text(text):
    if not text: return []
    text = unicodedata.normalize('NFKC', text)
    parts = text.split()
    cleaned = []
    for p in parts:
        if len(p) >= 2: cleaned.append(p[:-1].upper() + p[-1].lower())
    return cleaned

def render_hand_html(hand_str, size=45):
    if not hand_str: return ""
    cards = hand_str.split()
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    color_map = {'s': 'black', 'h': '#d32f2f', 'd': '#1976d2', 'c': '#388e3c'}
    
    html = "<div style='display:flex; gap:6px; margin-bottom:10px; flex-wrap: wrap;'>"
    for c in cards:
        if len(c) < 2: continue
        rank = c[:-1]
        suit = c[-1].lower()
        symbol = suit_map.get(suit, suit)
        color = color_map.get(suit, 'black')
        
        style = (
            f"width:{size}px; height:{size*1.35}px; background-color:white; "
            f"border:1px solid #bbb; border-radius:5px; "
            f"display:flex; justify-content:center; align-items:center; "
            f"font-size:{size*0.45}px; font-weight:bold; color:{color}; "
            f"box-shadow:2px 2px 4px rgba(0,0,0,0.1);"
        )
        html += f"<div style='{style}'>{rank}{symbol}</div>"
    html += "</div>"
    return html

def get_hand_tags(hand_str):
    try: cards = [SimpleCard(s) for s in hand_str.split()]
    except: return []
    tags = []
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    rc = {r: ranks.count(r) for r in ranks}
    pairs = [r for r, c in rc.items() if c == 2]
    if 12 in pairs: tags.append("AA")
    if 11 in pairs: tags.append("KK")
    if 10 in pairs: tags.append("QQ")
    if len(pairs)==2: tags.append("Double Pair")
    elif len(pairs)==1: tags.append("Single Pair")
    elif len(set(ranks))==4: tags.append("No Pair")

    sc = {s: suits.count(s) for s in suits}
    s_dist = sorted(sc.values(), reverse=True) + [0]*(4-len(sc))
    is_ds = (s_dist[0]==2 and s_dist[1]==2)
    is_mono = (s_dist[0]==4)
    is_ss = (s_dist[0]>=2 and not is_ds and not is_mono)
    
    if is_ds: tags.append("Double Suited")
    if s_dist[0]==1: tags.append("Rainbow")
    if is_mono: tags.append("Monotone")
    if is_ss: tags.append("Single Suited")
    
    if len(set(ranks))==4:
        ur = sorted(list(set(ranks)), reverse=True)
        gaps = [ur[i]-ur[i+1] for i in range(3)]
        if gaps==[1,1,1]: tags.append("Perfect Rundown")
        elif sum(gaps)==5: tags.append("Double Gap Rundown")
        if min(ranks)>=8: tags.append("Broadway")
    return tags

def set_input_callback(target_key, value):
    st.session_state[target_key] = value
    st.session_state[f"{target_key}_text"] = value
    # リセット処理
    for s in ['s','h','d','c']:
        ms_key = f"ms_{s}_{target_key}"
        if ms_key in st.session_state:
            st.session_state[ms_key] = []

# ==========================================
# 3. Postflop Logic (Board Analyzer)
# ==========================================
def eval_5card_score(cards):
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = (len(set(suits)) == 1)
    uniq = sorted(list(set(ranks)))
    is_str = False; str_high = -1
    if len(uniq) == 5:
        if uniq[4]-uniq[0] == 4: is_str = True; str_high = uniq[4]
        elif uniq == [0,1,2,3,12]: is_str = True; str_high = 3
    
    rc = Counter(ranks); counts = sorted(rc.values(), reverse=True)
    
    if is_str and is_flush: return 900 + str_high
    if counts[0] == 4: return 800 + ranks[0]
    if counts[0] == 3 and counts[1] == 2: return 700 + [k for k,v in rc.items() if v==3][0]
    if is_flush: return 600 + ranks[0]
    if is_str: return 500 + str_high
    if counts[0] == 3: return 400 + [k for k,v in rc.items() if v==3][0]
    if counts[0] == 2 and counts[1] == 2: return 300 + max([k for k,v in rc.items() if v==2])
    if counts[0] == 2: return 200 + [k for k,v in rc.items() if v==2][0]
    return 100 + ranks[0]

def get_best_score_strict(hand_cards, board_cards):
    best = -1
    for h in combinations(hand_cards, 2):
        for b in combinations(board_cards, 3):
            score = eval_5card_score(list(h)+list(b))
            if score > best: best = score
    return best

def precompute_nut_scores(board_cards):
    nut_scores = {}
    deck_ranks = range(13)
    for r in deck_ranks:
        sim_board_ranks = [c.rank for c in board_cards] + [r]
        br_set = set(sim_board_ranks)
        best_str = -1
        for top in range(12, 2, -1):
            needed = {top, top-1, top-2, top-3, top-4}
            if top == 3: needed = {0,1,2,3,12}
            if len(needed.intersection(br_set)) >= 3:
                best_str = 500 + top
                break
        nut_scores[r] = best_str if best_str != -1 else 0
    return nut_scores

def evaluate_hand_and_draws(hand_cards, board_cards, nut_scores_map):
    best_score = get_best_score_strict(hand_cards, board_cards)
    
    cat_map = {9:"StrFlush",8:"Quads",7:"FullHouse",6:"Flush",5:"Straight",4:"Set/Trips",3:"TwoPair",2:"Pair",1:"HighCard"}
    made_cat_val = best_score // 100
    made_label = cat_map.get(made_cat_val, "HighCard")
    
    if len(board_cards) >= 5:
        return made_label, "", {'Str':{'tot':0,'nut':0}, 'Fls':{'tot':0,'nut':0}}, 0
    
    suits = ['s','h','d','c']
    deck_ranks = range(13)
    used = set((c.rank, c.suit) for c in hand_cards + board_cards)
    
    outs = {'Str':{'tot':0,'nut':0}, 'Fls':{'tot':0,'nut':0}}
    
    # Flush Outs
    flush_draw_type = None
    for s in suits:
        h_c = sum(1 for c in hand_cards if c.suit==s)
        b_c = sum(1 for c in board_cards if c.suit==s)
        if h_c >= 2 and b_c == 2:
            rem = 13 - h_c - b_c
            outs['Fls']['tot'] = rem
            h_ranks = [c.rank for c in hand_cards if c.suit==s]
            b_ranks = [c.rank for c in board_cards if c.suit==s]
            max_b = max(b_ranks) if b_ranks else -1
            is_nut = False
            if 12 in h_ranks: is_nut = True
            elif 11 in h_ranks and max_b==12: is_nut = True
            elif 10 in h_ranks and 12 in b_ranks and 11 in b_ranks: is_nut = True
            if is_nut: 
                outs['Fls']['nut'] = rem
                flush_draw_type = "Nut FD"
            else:
                flush_draw_type = "FD"
            break
            
    # Straight Outs
    curr_str_score = 0
    if made_cat_val == 5: curr_str_score = best_score
    
    for r in deck_ranks:
        if all((r, s) in used for s in suits): continue
        test_board = board_cards + [SimpleCard(f"{'23456789TJQKA'[r]}s")]
        sim_score = get_best_score_strict(hand_cards, test_board)
        
        if 500 <= sim_score < 600:
            if sim_score > curr_str_score:
                av_suits = sum(1 for s in suits if (r,s) not in used)
                outs['Str']['tot'] += av_suits
                nut_limit = nut_scores_map.get(r, 999)
                if sim_score >= nut_limit:
                    outs['Str']['nut'] += av_suits

    draw_list = []
    if flush_draw_type: draw_list.append(flush_draw_type)
    str_tot = outs['Str']['tot']
    str_nut = outs['Str']['nut']
    
    if str_tot > 0:
        label = f"Wrap({str_tot})" if str_tot >= 9 else f"Str({str_tot})"
        if str_nut > 0: label += f" [{str_nut}N]"
        draw_list.append(label)
        
    draw_str = " + ".join(draw_list) if draw_list else "No Draw"
    total_outs_val = outs['Str']['tot'] + outs['Fls']['tot']
    
    return made_label, draw_str, outs, total_outs_val

# ==========================================
# 4. Data Loading
# ==========================================
@st.cache_data
def load_plo_data(csv_path="plo_detailed_ranking.zip"):
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        df["nut_equity"] = (df["win_SF"] + df["win_Quads"] + df["win_FH"] + df["win_Flush"] + df["win_Straight"])
        df["nut_quality"] = (df["nut_equity"] / df["equity"]).fillna(0)
        df["rank"] = df["equity"].rank(ascending=False, method='first').astype(int)
        df["pct"] = (df["rank"] / len(df)) * 100
        df["tags"] = df["hand"].apply(get_hand_tags)
        def get_max_rank(hand_str):
            try: return "23456789TJQKA"[max([SimpleCard(s).rank for s in hand_str.split()])]
            except: return "?"
        df["top_rank"] = df["hand"].apply(get_max_rank)
        return df.sort_values("rank")
    except: return None

@st.cache_data
def load_flo8_data(csv_path="flo8_ranking.csv"):
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        df["rank"] = df["equity"].rank(ascending=False, method='first').astype(int)
        df["pct_total"] = (df["rank"] / len(df)) * 100
        return df
    except: return None

# ==========================================
# 5. UI Components
# ==========================================
def render_card_selector(session_key):
    with st.expander("🃏 Open Card Selector (by Suit)", expanded=False):
        ranks_list = list("AKQJT98765432")
        c_s, c_h, c_d, c_c = st.columns(4)
        
        with c_s:
            st.markdown("**♠ Spades**")
            sel_s = st.multiselect("Spades", ranks_list, key=f"ms_s_{session_key}", label_visibility="collapsed")
        with c_h:
            st.markdown("**:red[♥ Hearts]**")
            sel_h = st.multiselect("Hearts", ranks_list, key=f"ms_h_{session_key}", label_visibility="collapsed")
        with c_d:
            st.markdown("**:blue[♦ Diamonds]**")
            sel_d = st.multiselect("Diamonds", ranks_list, key=f"ms_d_{session_key}", label_visibility="collapsed")
        with c_c:
            st.markdown("**:green[♣ Clubs]**")
            sel_c = st.multiselect("Clubs", ranks_list, key=f"ms_c_{session_key}", label_visibility="collapsed")

        collected = [f"{r}s" for r in sel_s] + [f"{r}h" for r in sel_h] + [f"{r}d" for r in sel_d] + [f"{r}c" for r in sel_c]

        if len(collected) == 4:
            final_hand = " ".join(collected)
            if st.session_state.get(session_key) != final_hand:
                set_input_callback(session_key, final_hand)
                st.rerun()
            return collected
        elif len(collected) > 0:
            st.caption(f"Selected: {len(collected)}/4 cards.")
    return []

# ==========================================
# 6. Main Application Logic
# ==========================================
st.title("🃏 Omaha Hand Analyzer")

# Init Session
for k in ['plo_input', 'flo8_input', 'p1_fixed', 'p2_fixed', 'pf_board']:
    if k not in st.session_state: st.session_state[k] = ""
    tk = f"{k}_text"
    if tk not in st.session_state: st.session_state[tk] = ""

if st.session_state.plo_input == "": st.session_state.plo_input = "As Ks Jd Th"
if st.session_state.flo8_input == "": st.session_state.flo8_input = "Ad Ah 2s 3d"

df_plo = load_plo_data()
df_flo8 = load_flo8_data()

with st.sidebar:
    st.title("Navigation")
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "PLO Board Analyzer", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# ==========================================
# MODE: PLO (Rich Features Restored)
# ==========================================
if game_mode == "PLO (High Only)":
    if df_plo is None:
        st.warning("Data loading failed.")
    else:
        # Variables
        ranks_opt = list("AKQJT98765432")
        avail_tags = ["AA","KK","QQ","Double Pair","Double Suited","Single Suited","A-High Suit","Rainbow","Monotone","Broadway","Perfect Rundown","Double Gap Rundown"]
        
        with st.sidebar:
            with st.expander("1. ⚙️ Scenario", expanded=False):
                spr = st.select_slider("Stack Depth / SPR", ["Short","Medium","Deep","Very Deep"], value="Medium")
                nw = 0.0 if "Short" in spr else 0.3 if "Medium" in spr else 0.6 if "Deep" in spr else 0.8
                st.caption(f"Nut Weight: {nw*100:.0f}%")

            with st.expander("2. 🔍 Hand Rank", expanded=False):
                c_rk1, c_rk2 = st.columns([1,2])
                with c_rk1:
                    srk = st.number_input("Rank", 1, len(df_plo), 1, key="prk_plo", label_visibility="collapsed")
                with c_rk2:
                    fr = df_plo[df_plo['rank']==srk]
                    if not fr.empty:
                        r = fr.iloc[0]
                        if st.button("Analyze", key="bcp_plo"):
                             set_input_callback('plo_input', r['hand']); st.rerun()
                    else: st.write("-")
                if not fr.empty: st.caption(f"**{r['hand']}** (Top {r['pct']:.2f}%)")

            with st.expander("3. 🎨 Highlights", expanded=False):
                hl_tags_1 = st.multiselect("Group 1 (🔴 Red)", avail_tags, key="hl1")
                hl_tags_2 = st.multiselect("Group 2 (🔵 Blue)", avail_tags, key="hl2")
                hl_tags_3 = st.multiselect("Group 3 (🟢 Green)", avail_tags, key="hl3")

            with st.expander("4. 🏷️ Filter", expanded=True):
                sel_top = st.multiselect("Top Rank", ranks_opt)
                inc_tags = st.multiselect("Include", avail_tags)
                exc_tags = st.multiselect("Exclude", avail_tags)

            st.divider()
            d_limit = st.slider("List Limit", 5, 100, 20, 5)

            filtered_df = None
            if sel_top or inc_tags or exc_tags:
                tmp = df_plo
                if sel_top: tmp = tmp[tmp["top_rank"].isin(sel_top)]
                if inc_tags or exc_tags:
                    iset, eset = set(inc_tags), set(exc_tags)
                    tmp = tmp[tmp["tags"].apply(lambda t: iset.issubset(set(t)) and eset.isdisjoint(set(t)))]
                filtered_df = tmp

            st.markdown(f"**Results (Top {d_limit})**")
            if filtered_df is not None:
                if not filtered_df.empty:
                    th = filtered_df.head(d_limit)
                    hset1 = set(hl_tags_1); hset2 = set(hl_tags_2); hset3 = set(hl_tags_3)
                    for _, r in th.iterrows():
                        rtags = set(r['tags'])
                        prefix = ""
                        if hl_tags_1 and hset1.issubset(rtags): prefix += "🔴"
                        if hl_tags_2 and hset2.issubset(rtags): prefix += "🔵"
                        if hl_tags_3 and hset3.issubset(rtags): prefix += "🟢"
                        lbl = f"{prefix} {r['hand']} (#{r['rank']})"
                        if st.button(lbl, key=f"s_{r['rank']}"):
                            set_input_callback('plo_input', r['hand']); st.rerun()
                    st.caption(f"Found: {len(filtered_df):,}")
                else: st.write("No hands found.")
            elif not (sel_top or inc_tags or exc_tags): st.write("(No filters)")

        st.header("🔥 PLO Strategy")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.subheader("🔍 Hand Input")
            render_card_selector('plo_input')
            inp_raw = st.text_input("Enter Hand (Text)", key='plo_input_text')
            if inp_raw != st.session_state.plo_input: st.session_state.plo_input = inp_raw
            inp = normalize_input_text(st.session_state.plo_input)
            if inp: st.markdown(render_hand_html(" ".join(inp)), unsafe_allow_html=True)
            
            if len(inp)==4:
                res = df_plo[df_plo["card_set"]==frozenset(inp)]
                if not res.empty:
                    row = res.iloc[0]
                    eq = row["equity"] * 100; ne = row["nut_equity"] * 100
                    sc = (eq*(1-nw)) + ((row["nut_quality"]*100)*nw)
                    m1,m2,m3 = st.columns(3)
                    m1.metric("Power Score", f"{sc:.1f}")
                    m2.metric("Raw Equity", f"{eq:.1f}%")
                    m3.metric("Nut Equity", f"{ne:.1f}%")
                    st.write("🏷️ " + " ".join([f"`{t}`" for t in row['tags']]))
                    st.caption(f"Rank: {int(row['rank']):,} (Top {row['pct']:.1f}%)")
                else: st.warning("Hand not found.")

        with c2:
            if 'row' in locals():
                st.subheader("📊 Win Distribution")
                w_sf = row["win_SF"]; w_qd = row["win_Quads"]; w_fh = row["win_FH"]
                w_fl = row["win_Flush"]; w_st = row["win_Straight"]
                w_wk = max(0, row["equity"] - (w_sf+w_qd+w_fh+w_fl+w_st))
                lse = 1.0 - row["equity"]
                sizes = [w_st, w_fl, w_sf+w_qd+w_fh, w_wk, lse]
                labels = ['Straight+', 'Flush', 'FullHouse+', 'Pair (Fragile)', 'Lose']
                colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FFC107', '#EEEEEE']
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                pdata = [(s,l,c) for s,l,c in zip(sizes, labels, colors) if s > 0.001]
                if pdata:
                    ps, pl, pc = zip(*pdata)
                    w, _, _ = ax1.pie(ps, autopct='%1.1f%%', startangle=90, colors=pc)
                    ax1.legend(w, pl, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    st.pyplot(fig1)

        if 'row' in locals():
            st.divider()
            cc1, cc2 = st.columns(2)
            with cc1:
                st.subheader("📈 Equity Curve")
                seek_pct = st.slider("🔍 Seek Hand Strength (Top X%)", 0.0, 100.0, 10.0, 0.1)
                s_idx = int(len(df_plo) * (seek_pct / 100))
                if s_idx >= len(df_plo): s_idx = len(df_plo) - 1
                s_row = df_plo.iloc[s_idx]
                st.info(f"**Top {seek_pct:.1f}% Boundary**")
                sk1, sk2 = st.columns([3, 1])
                with sk1:
                    st.markdown(render_hand_html(s_row['hand']), unsafe_allow_html=True)
                    st.caption(f"Eq: {s_row['equity']*100:.1f}%")
                with sk2:
                    st.button("Analyze", key=f"b_seek_plo_{seek_pct}", on_click=set_input_callback, args=('plo_input', s_row['hand']))
                scurve = df_plo.iloc[::200, :]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.plot(scurve["pct"], scurve["equity"], c="#cccccc", label="All")
                ax3.scatter(row["pct"], row["equity"], c="red", s=150, marker='*', zorder=10, label="You")
                ax3.scatter(s_row["pct"], s_row["equity"], c="blue", s=80, zorder=9, label="Seek")
                ax3.axvline(x=seek_pct, color="blue", ls=":", alpha=0.5)
                ax3.set_xlabel("Top X%"); ax3.set_ylabel("Equity")
                st.pyplot(fig3)

            with cc2:
                st.subheader("🌌 Equity Scatter")
                cmode = st.radio("Scatter", ["Mode A", "Mode B"], horizontal=True, label_visibility="collapsed")
                st.caption("Mode A: Eq vs Quality / Mode B: Eq vs Nut Eq")
                use_auto_zoom = st.checkbox("🔍 Auto Zoom", value=True)
                
                @st.cache_data
                def get_bg(df): return df.sample(3000, random_state=42).copy()
                bg = get_bg(df_plo)

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                def gxy(d, m): return d["equity"], (d["nut_quality"] if "Mode A" in m else d["nut_equity"])
                bx, by = gxy(bg, cmode)
                mx, my = gxy(pd.DataFrame([row]), cmode); mx, my = mx.iloc[0], my.iloc[0]
                cbg = bg["nut_quality"] if "Mode A" in cmode else (1.0-(bx-by))
                ax2.scatter(bx, by, c=cbg, cmap="coolwarm_r", s=10, alpha=0.1, label='Others')
                if "Mode B" in cmode: ax2.plot([0,1],[0,1], ls="--", c="gray", alpha=0.5)

                xmin, xmax, ymin, ymax = mx, mx, my, my
                focused = False
                if filtered_df is not None and not filtered_df.empty:
                    fdf = filtered_df.sample(2000, random_state=42) if len(filtered_df)>2000 else filtered_df
                    fx, fy = gxy(fdf, cmode)
                    ax2.scatter(fx, fy, fc='none', ec='gold', s=30, label='Filtered')
                    xmin, xmax = min(xmin, fx.min()), max(xmax, fx.max())
                    ymin, ymax = min(ymin, fy.min()), max(ymax, fy.max())
                    focused = True
                
                groups = [(hl_tags_1, 'crimson', 'Grp1'), (hl_tags_2, 'dodgerblue', 'Grp2'), (hl_tags_3, 'limegreen', 'Grp3')]
                for tags, color, lbl_prefix in groups:
                    if tags:
                        src = filtered_df if filtered_df is not None else df_plo
                        ht = set(tags)
                        mask = src["tags"].apply(lambda t: ht.issubset(set(t)))
                        hdf = src[mask]
                        if not hdf.empty:
                            hdf_s = hdf.sample(2000, random_state=42) if len(hdf)>2000 else hdf
                            hx, hy = gxy(hdf_s, cmode)
                            ax2.scatter(hx, hy, fc='none', ec=color, s=50, lw=1.5, label=lbl_prefix)
                            xmin, xmax = min(xmin, hx.min()), max(xmax, hx.max())
                            ymin, ymax = min(ymin, hy.min()), max(ymax, hy.max())
                            focused = True

                ax2.scatter(mx, my, c='black', s=150, marker='*', ec='white', zorder=10, label='You')
                if use_auto_zoom:
                    if not focused: xmin, xmax, ymin, ymax = bx.min(), bx.max(), by.min(), by.max()
                    if xmax == xmin: xmin -= 0.1; xmax += 0.1
                    if ymax == ymin: ymin -= 0.1; ymax += 0.1
                    x_span = xmax - xmin; y_span = ymax - ymin
                    if x_span < 0.15: d=(0.15-x_span)/2; xmin-=d; xmax+=d
                    if y_span < 0.15: d=(0.15-y_span)/2; ymin-=d; ymax+=d
                    m=0.05
                    ax2.set_xlim(max(0, xmin-m), min(1, xmax+m))
                    ax2.set_ylim(max(0, ymin-m), min(1, ymax+m))
                else: ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)
                ax2.grid(True, ls='--', alpha=0.3)
                ax2.legend(fontsize=8, loc='upper left')
                st.pyplot(fig2)

# ==========================
# PLO BOARD ANALYZER
# ==========================
elif game_mode == "PLO Board Analyzer":
    st.header("📊 PLO Board Analyzer")
    if df_plo is not None:
        with st.sidebar:
            st.markdown("### Range Settings")
            p1_val = st.slider("Player 1 (Blue) Top %", 5, 100, 15, 5)
            p2_val = st.slider("Player 2 (Red) Top %", 5, 100, 50, 5)
        
        st.subheader("1. Board Input")
        c1,c2,c3 = st.columns(3)
        deck = [f"{r}{s}" for r in "AKQJT98765432" for s in "shdc"]
        if c1.button("🎲 Flop"): set_input_callback('pf_board', " ".join(random.sample(deck, 3))); st.rerun()
        if c2.button("🎲 Turn"): set_input_callback('pf_board', " ".join(random.sample(deck, 4))); st.rerun()
        if c3.button("🎲 River"): set_input_callback('pf_board', " ".join(random.sample(deck, 5))); st.rerun()
        
        b_cards = normalize_input_text(st.text_input("Board Cards", key='pf_board_text'))
        if b_cards: st.markdown(render_hand_html(" ".join(b_cards), 50), unsafe_allow_html=True)
        
        if st.button("🚀 Analyze", type="primary"):
            if len(b_cards)<3: st.error("Need 3+ cards")
            else:
                with st.spinner("Analyzing..."):
                    def get_sample(val):
                        limit = int(len(df_plo)*(val/100))
                        sub = df_plo.iloc[:limit]
                        return sub["hand"].sample(2000).tolist() if len(sub)>2000 else sub["hand"].tolist()
                    
                    p1h = get_sample(p1_val); p2h = get_sample(p2_val)
                    b_objs = [SimpleCard(c) for c in b_cards]
                    is_river = (len(b_cards)>=5)
                    nut_scores_map = precompute_nut_scores(b_objs)
                    
                    def analyze_range(h_list):
                        made_stats = defaultdict(int)
                        patterns = defaultdict(lambda: {'count':0, 'ex':""})
                        outs_agg = {'Str':{'tot':0,'nut':0}, 'Fls':{'tot':0,'nut':0}}
                        
                        for h_str in h_list:
                            h_objs = [SimpleCard(c) for c in h_str.split()]
                            made, draw, outs, tot_outs = evaluate_hand_and_draws(h_objs, b_objs, nut_scores_map)
                            made_stats[made] += 1
                            if not is_river:
                                outs_agg['Str']['tot'] += outs['Str']['tot']
                                outs_agg['Str']['nut'] += outs['Str']['nut']
                                outs_agg['Fls']['tot'] += outs['Fls']['tot']
                                outs_agg['Fls']['nut'] += outs['Fls']['nut']
                                if draw != "No Draw":
                                    key = (draw, tot_outs)
                                    patterns[key]['count'] += 1
                                    patterns[key]['ex'] = h_str
                        return made_stats, outs_agg, patterns

                    p1_made, p1_outs, p1_pat = analyze_range(p1h)
                    p2_made, p2_outs, p2_pat = analyze_range(p2h)
                    
                    st.divider()
                    c_g1, c_g2 = st.columns(2)
                    with c_g1:
                        st.write("##### Made Hands")
                        cats = ["StrFlush","Quads","FullHouse","Flush","Straight","Set/Trips","TwoPair","Pair","HighCard"]
                        fig1, ax1 = plt.subplots(figsize=(5,4))
                        y = np.arange(len(cats))
                        p1_v = [p1_made.get(c,0)/len(p1h)*100 for c in cats]
                        p2_v = [p2_made.get(c,0)/len(p2h)*100 for c in cats]
                        ax1.barh(y+0.2, p1_v, 0.4, label='P1', color='dodgerblue')
                        ax1.barh(y-0.2, p2_v, 0.4, label='P2', color='crimson')
                        ax1.set_yticks(y); ax1.set_yticklabels(cats); ax1.legend()
                        st.pyplot(fig1)
                    
                    with c_g2:
                        if not is_river:
                            st.write("##### Avg Outs (Nut vs Non-Nut)")
                            cats = ['Str', 'Fls']
                            fig2, ax2 = plt.subplots(figsize=(5,4))
                            x = np.arange(2)
                            p1_t = [p1_outs[k]['tot']/len(p1h) for k in cats]
                            p1_n = [p1_outs[k]['nut']/len(p1h) for k in cats]
                            p1_w = [t-n for t,n in zip(p1_t, p1_n)]
                            p2_t = [p2_outs[k]['tot']/len(p2h) for k in cats]
                            p2_n = [p2_outs[k]['nut']/len(p2h) for k in cats]
                            p2_w = [t-n for t,n in zip(p2_t, p2_n)]
                            ax2.bar(x-0.2, p1_n, 0.4, color='#1565C0', label='P1 Nut')
                            ax2.bar(x-0.2, p1_w, 0.4, bottom=p1_n, color='#90CAF9')
                            ax2.bar(x+0.2, p2_n, 0.4, color='#C62828', label='P2 Nut')
                            ax2.bar(x+0.2, p2_w, 0.4, bottom=p2_n, color='#EF9A9A')
                            ax2.set_xticks(x); ax2.set_xticklabels(cats); ax2.legend()
                            st.pyplot(fig2)
                    
                    if not is_river:
                        st.divider()
                        st.subheader("🏆 Top Draw Patterns")
                        max_disp = st.slider("Max Items", 5, 30, 10)
                        
                        def show_patterns(pat_dict, total, title, color):
                            st.markdown(f"**{title}**")
                            lst = []
                            for (lbl, outs), dat in pat_dict.items():
                                is_nut_fd = "Nut FD" in lbl
                                lst.append({'lbl':lbl, 'outs':outs, 'cnt':dat['count'], 'ex':dat['ex'], 'nfd':is_nut_fd})
                            lst.sort(key=lambda x: (x['nfd'], x['outs']), reverse=True)
                            
                            for p in lst[:max_disp]:
                                pct = p['cnt']/total*100
                                border_col = "#d32f2f" if p['nfd'] else color
                                st.markdown(f"""
                                <div class="draw-box" style="border-left: 5px solid {border_col}">
                                    <div class="draw-info">
                                        <div class="draw-title">{p['lbl']}</div>
                                        <div class="draw-sub">{render_hand_html(p['ex'], 20)}</div>
                                    </div>
                                    <div class="draw-stat">
                                        <div class="draw-count">{pct:.1f}%</div>
                                        <div class="draw-pct">({p['cnt']})</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        c_p1, c_p2 = st.columns(2)
                        with c_p1: show_patterns(p1_pat, len(p1h), "Player 1 Draws", "#2196F3")
                        with c_p2: show_patterns(p2_pat, len(p2h), "Player 2 Draws", "#999")

# ==========================================
# MODE: FLO8 (Rich Features Restored)
# ==========================================
elif game_mode == "FLO8 (Hi/Lo)":
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

elif game_mode == "Guide":
    st.header("📖 Guide")
    st.markdown("All features (PLO Rich, PLO Board Analyzer, FLO8 Rich) active.")

with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
