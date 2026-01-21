import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
import random
from collections import Counter, defaultdict
from heuristics import calculate_flo8_heuristic

# ==========================================
# 1. Config & Styles
# ==========================================
st.set_page_config(page_title="Omaha Hand Analyzer", layout="wide")

st.markdown("""
<style>
    /* スマホUI調整 */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    /* カード表示用 */
    .card-display {
        display: flex; gap: 5px; justify-content: center; margin-bottom: 10px;
    }
    .card-box {
        width: 45px; height: 60px; background: white; border: 1px solid #ccc;
        border-radius: 5px; display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 18px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
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
    if not hand_str: return "<div style='height:60px; color:#aaa; display:flex; align-items:center;'>No cards selected</div>"
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
    widget_key = f"{target_key}_text"
    if widget_key in st.session_state:
        st.session_state[widget_key] = value
    # セレクター一時状態のリセット
    temp_key = f"sel_temp_{target_key}"
    if temp_key in st.session_state:
        st.session_state[temp_key] = []

# ==========================================
# 3. Postflop Logic (Outs Calculation)
# ==========================================
def get_best_straight_rank(ranks):
    uniq = sorted(list(set(ranks)))
    if len(uniq) < 5: return -1
    best = -1
    for i in range(len(uniq)-4):
        if uniq[i+4] - uniq[i] == 4:
            best = max(best, uniq[i+4])
    if {0,1,2,3,12}.issubset(set(uniq)): best = max(best, 3)
    return best

def calculate_detailed_outs(hand_cards, board_cards):
    """
    ストレート、フラッシュ、フルハウスのアウツ数とナッツ判定を行う
    """
    deck_ranks = range(13)
    suits = ['s','h','d','c']
    used_cards = set((c.rank, c.suit) for c in hand_cards + board_cards)
    
    outs = {
        'Straight': {'total': 0, 'nut': 0},
        'Flush': {'total': 0, 'nut': 0},
        'FullHouse': {'total': 0, 'nut': 0}
    }
    
    # Pre-check current state
    current_all = hand_cards + board_cards
    current_ranks = [c.rank for c in current_all]
    has_flush = False
    for s in suits:
        if sum(1 for c in hand_cards if c.suit==s)>=2 and sum(1 for c in board_cards if c.suit==s)>=3: has_flush = True
    has_fh = any(current_ranks.count(r)>=3 for r in current_ranks) and len(set(current_ranks)) < len(current_ranks) # Simplified FH check logic
    
    # Iterate all unseen cards (approx 45 cards)
    # This is fast enough for single hand analysis
    for r in deck_ranks:
        for s in suits:
            if (r, s) in used_cards: continue
            
            # --- Simulation Context ---
            sim_board = board_cards + [SimpleCard(f"{'23456789TJQKA'[r]}{s}")]
            sim_all = hand_cards + sim_board
            sim_ranks = [c.rank for c in sim_all]
            
            # 1. Flush Check
            is_flush_now = False
            flush_nut = False
            for st in suits:
                h_c = sum(1 for c in hand_cards if c.suit==st)
                b_c = sum(1 for c in sim_board if c.suit==st)
                if h_c >= 2 and b_c >= 3:
                    is_flush_now = True
                    # Nut check (Ace high flush?)
                    hand_s_ranks = [c.rank for c in hand_cards if c.suit==st]
                    if 12 in hand_s_ranks: flush_nut = True # Simplification: Ace is nut
                    # Technically needs to check if board blocks higher, but Ace is good approx
            
            if is_flush_now and not has_flush:
                outs['Flush']['total'] += 1
                if flush_nut: outs['Flush']['nut'] += 1
            
            # 2. Full House Check
            # Logic: If we hit a set or trips that pairs the board
            is_fh_now = False
            rc = Counter(sim_ranks)
            threes = [k for k,v in rc.items() if v>=3]
            pairs = [k for k,v in rc.items() if v>=2]
            if threes and (len(pairs) >= 2 or len(threes) >= 2): is_fh_now = True
            
            if is_fh_now and not has_fh and not is_flush_now:
                outs['FullHouse']['total'] += 1
                # FH Nut check is complex, treat top set as nut approx
                # If board pair is high, our FH is strong
                outs['FullHouse']['nut'] += 1 # Optimistic
                
            # 3. Straight Check
            # Only count if not Flush and not FH (Hierarchy)
            if not is_flush_now and not is_fh_now:
                my_str = get_best_straight_rank(sim_ranks)
                if my_str != -1:
                    # It's a straight out
                    outs['Straight']['total'] += 1
                    
                    # Nut Straight Check
                    board_plus_r = [c.rank for c in board_cards] + [r]
                    # Calc max possible straight with this board+r
                    br_set = set(board_plus_r)
                    possible_max = -1
                    # Check all straights
                    for top in range(4, 13): # 6-high to A-high
                        needed = {top, top-1, top-2, top-3, top-4}
                        if len(needed - br_set) <= 2: possible_max = top
                    if len({0,1,2,3,12} - br_set) <= 2: possible_max = max(possible_max, 3)
                    
                    if my_str >= possible_max:
                        outs['Straight']['nut'] += 1

    return outs

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
            try:
                cards = [SimpleCard(s) for s in hand_str.split()]
                max_card = max(cards, key=lambda c: c.rank)
                return "23456789TJQKA"[max_card.rank]
            except: return "?"
        df["top_rank"] = df["hand"].apply(get_max_rank)
        return df.sort_values("rank")
    except FileNotFoundError: return None

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
# 5. UI Components (New Button Selector)
# ==========================================
def render_card_selector_buttons(session_key):
    # 一時保存用のキー
    temp_key = f"sel_temp_{session_key}"
    if temp_key not in st.session_state: st.session_state[temp_key] = []
    
    current = st.session_state[temp_key]
    
    with st.expander("🃏 Open Card Selector (Tap Buttons)", expanded=True):
        # 現在の選択状態表示
        st.write("Selected:")
        if current:
            st.markdown(render_hand_html(" ".join(current)), unsafe_allow_html=True)
        else:
            st.caption("No cards selected")
            
        c_act1, c_act2 = st.columns(2)
        if c_act1.button("Clear", key=f"clr_{session_key}"):
            st.session_state[temp_key] = []
            st.rerun()
        if c_act2.button("Apply / Close", key=f"apl_{session_key}", type="primary"):
            if len(current) > 0:
                set_input_callback(session_key, " ".join(current))
            st.rerun()

        st.divider()
        
        # タブでスート切り替え
        tabs = st.tabs(["♠ Spades", "♥ Hearts", "♦ Diamonds", "♣ Clubs"])
        ranks = list("AKQJT98765432")
        suits = ['s', 'h', 'd', 'c']
        suit_icons = ['♠', '♥', '♦', '♣']
        
        for i, tab in enumerate(tabs):
            with tab:
                cols = st.columns(4)
                for r_idx, r in enumerate(ranks):
                    card_str = f"{r}{suits[i]}"
                    disabled = (card_str in current) or (len(current) >= 5) # Max 5 for board
                    
                    if cols[r_idx % 4].button(f"{r}{suit_icons[i]}", key=f"btn_{session_key}_{card_str}", disabled=disabled):
                        current.append(card_str)
                        st.session_state[temp_key] = current
                        st.rerun()

# ==========================================
# 6. Main Logic
# ==========================================
st.title("🃏 Omaha Hand Analyzer")

# Init
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
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "Postflop Range", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# -----------------
# PLO MODE
# -----------------
if game_mode == "PLO (High Only)":
    if df_plo is not None:
        with st.sidebar:
            with st.expander("1. ⚙️ Scenario"):
                spr = st.select_slider("Stack Depth", ["Short","Medium","Deep","Very Deep"], value="Medium")
                nw = 0.0 if "Short" in spr else 0.3 if "Medium" in spr else 0.6 if "Deep" in spr else 0.8
                st.caption(f"Nut Weight: {nw*100:.0f}%")
            
            with st.expander("2. 🔍 Hand Rank"):
                c1,c2=st.columns([1,2])
                srk=c1.number_input("Rank",1,len(df_plo),1,label_visibility="collapsed")
                if c2.button("Analyze"):
                    h = df_plo[df_plo['rank']==srk].iloc[0]['hand']
                    set_input_callback('plo_input', h); st.rerun()

            with st.expander("3. 🏷️ Filter"):
                sel_top = st.multiselect("Top Rank", list("AKQJT98765432"))
                inc_tags = st.multiselect("Include", ["AA","KK","Double Suited","Rundown"])
                
            d_limit = st.slider("List Limit", 5, 50, 10)
            
            # Filtering
            filtered_df = df_plo
            if sel_top: filtered_df = filtered_df[filtered_df["top_rank"].isin(sel_top)]
            if inc_tags: 
                iset = set(inc_tags)
                filtered_df = filtered_df[filtered_df["tags"].apply(lambda t: iset.issubset(set(t)))]
            
            st.markdown(f"**Results (Top {d_limit})**")
            for _, r in filtered_df.head(d_limit).iterrows():
                if st.button(f"{r['hand']} (#{r['rank']})", key=f"l_{r['rank']}"):
                    set_input_callback('plo_input', r['hand']); st.rerun()

        st.header("🔥 PLO Strategy")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.subheader("🔍 Hand Input")
            render_card_selector_buttons('plo_input')
            
            inp_raw = st.text_input("Enter Hand", key='plo_input_text')
            if inp_raw != st.session_state.plo_input: st.session_state.plo_input = inp_raw
            
            inp = normalize_input_text(st.session_state.plo_input)
            if inp: st.markdown(render_hand_html(" ".join(inp)), unsafe_allow_html=True)
            
            if len(inp)==4:
                res = df_plo[df_plo["card_set"]==frozenset(inp)]
                if not res.empty:
                    row = res.iloc[0]
                    eq = row["equity"]*100; ne = row["nut_equity"]*100
                    sc = (eq*(1-nw)) + ((row["nut_quality"]*100)*nw)
                    m1,m2,m3=st.columns(3)
                    m1.metric("Power Score",f"{sc:.1f}"); m2.metric("Equity",f"{eq:.1f}%"); m3.metric("Nut Eq",f"{ne:.1f}%")
                    st.write("🏷️ " + " ".join([f"`{t}`" for t in row['tags']]))
                    st.caption(f"Rank: {int(row['rank']):,} (Top {row['pct']:.1f}%)")

        with c2:
            if 'row' in locals():
                st.subheader("📊 Win Dist")
                fig1, ax1 = plt.subplots(figsize=(4,3))
                sizes = [row['win_Straight'], row['win_Flush'], row['win_FH']+row['win_Quads'], 1.0-row['equity']]
                labels = ['Str', 'Fls', 'FH+', 'Lose']
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50','#2196F3','#9C27B0','#EEEEEE'])
                st.pyplot(fig1)
        
        if 'row' in locals():
            st.divider()
            st.subheader("📈 Equity Curve")
            seek = st.slider("Seek Strength (Top %)", 0.0, 100.0, 10.0)
            idx = int(len(df_plo)*(seek/100))
            s_row = df_plo.iloc[idx] if idx<len(df_plo) else df_plo.iloc[-1]
            st.caption(f"Top {seek}% Hand: {s_row['hand']} ({s_row['equity']*100:.1f}%)")
            fig3, ax3 = plt.subplots(figsize=(6,3))
            ax3.plot(df_plo.iloc[::200]["pct"], df_plo.iloc[::200]["equity"], c="#ccc")
            ax3.scatter(row["pct"], row["equity"], c="red", s=100, zorder=5, label="You")
            ax3.axvline(seek, c="blue", ls=":")
            st.pyplot(fig3)

# -----------------
# POSTFLOP RANGE MODE
# -----------------
elif game_mode == "Postflop Range":
    st.header("📊 Postflop Range Analysis")
    if df_plo is not None:
        with st.sidebar:
            st.markdown("### 🎲 Board")
            if st.button("🎲 Random Board"):
                deck = []
                for r in "AKQJT98765432":
                    for s in "shdc": deck.append(f"{r}{s}")
                rb = random.sample(deck, 3)
                st.session_state.pf_board = " ".join(rb)
                st.rerun()

            st.divider()
            st.markdown("### Player 1 (Blue)")
            p1_mode = st.selectbox("Type", ["Top %", "Fixed"], key="p1t")
            if "Top" in p1_mode: p1_val = st.select_slider("Top %", list(range(5,105,5)), value=15, key="p1v")
            else: p1_fix = st.text_input("Hand", key="p1f")

            st.markdown("### Player 2 (Red)")
            p2_mode = st.selectbox("Type", ["Top %", "Fixed"], key="p2t")
            if "Top" in p2_mode: p2_val = st.select_slider("Top %", list(range(5,105,5)), value=50, key="p2v")
            else: p2_fix = st.text_input("Hand", key="p2f")

        st.subheader("1. Board Input")
        render_card_selector_buttons('pf_board')
        
        inp_board = st.text_input("Board Cards", key='pf_board_text')
        if inp_board != st.session_state.pf_board: st.session_state.pf_board = inp_board
        
        b_cards = normalize_input_text(st.session_state.pf_board)
        if b_cards: st.markdown(render_hand_html(" ".join(b_cards), size=50), unsafe_allow_html=True)

        if st.button("🚀 Analyze Range Hits", type="primary"):
            if len(b_cards)<3: st.error("Need 3+ cards")
            else:
                with st.spinner("Simulating 5,000 hands..."):
                    # Get Hands
                    def get_range(mode, val, fix, df):
                        if "Fixed" in mode:
                            h = normalize_input_text(fix)
                            return [" ".join(h)] if len(h)==4 else []
                        else:
                            limit = int(len(df)*(val/100))
                            sub = df.iloc[:limit]
                            return sub["hand"].sample(5000).tolist() if len(sub)>5000 else sub["hand"].tolist()

                    p1h = get_range(p1_mode, p1_val if "Top" in p1_mode else 0, st.session_state.get('p1f',""), df_plo)
                    p2h = get_range(p2_mode, p2_val if "Top" in p2_mode else 0, st.session_state.get('p2f',""), df_plo)

                    if p1h and p2h:
                        # Analysis Logic
                        b_objs = [SimpleCard(c) for c in b_cards]
                        
                        def analyze(h_list):
                            stats = defaultdict(float) # Avg outs
                            counts = defaultdict(int) # Hits count
                            total = len(h_list)
                            
                            cats_agg = {'Str':0, 'Fls':0, 'FH':0}
                            nut_agg = {'Str':0, 'Fls':0, 'FH':0}
                            
                            for h_str in h_list:
                                h_objs = [SimpleCard(c) for c in h_str.split()]
                                outs = calculate_detailed_outs(h_objs, b_objs)
                                
                                # Summing outs
                                for k in ['Straight','Flush','FullHouse']:
                                    cats_agg[k] += outs[k]['total']
                                    nut_agg[k] += outs[k]['nut']
                            
                            return {k: v/total for k,v in cats_agg.items()}, {k: v/total for k,v in nut_agg.items()}

                        p1_avg, p1_nut = analyze(p1h)
                        p2_avg, p2_nut = analyze(p2h)

                        st.divider()
                        st.subheader("🎯 Average Outs Analysis")
                        
                        # Graph
                        labels = ["Straight", "Flush", "Full House"]
                        keys = ['Str', 'Fls', 'FH']
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        x = np.arange(len(labels))
                        w = 0.35
                        
                        # P1 Bars
                        p1_tot = [p1_avg[k] for k in keys]
                        p1_n = [p1_nut[k] for k in keys]
                        p1_weak = [t-n for t,n in zip(p1_tot, p1_n)]
                        
                        ax.bar(x - w/2, p1_n, w, label='P1 Nut Outs', color='#1976D2')
                        ax.bar(x - w/2, p1_weak, w, bottom=p1_n, label='P1 Non-Nut', color='#90CAF9')
                        
                        # P2 Bars
                        p2_tot = [p2_avg[k] for k in keys]
                        p2_n = [p2_nut[k] for k in keys]
                        p2_weak = [t-n for t,n in zip(p2_tot, p2_n)]
                        
                        ax.bar(x + w/2, p2_n, w, label='P2 Nut Outs', color='#D32F2F')
                        ax.bar(x + w/2, p2_weak, w, bottom=p2_n, label='P2 Non-Nut', color='#EF9A9A')
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels)
                        ax.set_ylabel("Avg Outs Count")
                        ax.set_title("Range vs Range: Outs Comparison")
                        ax.legend()
                        ax.grid(axis='y', ls='--', alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        st.caption("※棒グラフの濃い色が『ナッツ級アウツ』、薄い色が『それ以外のアウツ』の平均枚数です。")

# -----------------
# FLO8 & Guide (Simplified for space)
# -----------------
elif game_mode == "FLO8 (Hi/Lo)":
    st.header("⚖️ FLO8 Strategy")
    render_card_selector_buttons('flo8_input')
    i8 = normalize_input_text(st.text_input("Hand", key='flo8_input_text'))
    if i8:
        st.markdown(render_hand_html(" ".join(i8)), unsafe_allow_html=True)
        if len(i8)==4:
            sc, dt = calculate_flo8_heuristic(" ".join(i8))
            st.metric("Hutchinson Pts", sc)
            st.bar_chart(dt)

elif game_mode == "Guide":
    st.header("📖 Guide")
    st.markdown("### Update Info")
    st.markdown("- **Card Selector**: Button type for mobile.")
    st.markdown("- **Postflop**: Added 'Random Board' and detailed Outs Graph (Nut vs Non-Nut).")

with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
