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
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    
    /* Draw Pattern Box */
    .draw-box {
        border: 1px solid #ddd; border-left: 5px solid #2196F3;
        background-color: #fff; padding: 10px; margin-bottom: 8px; border-radius: 4px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .draw-title { font-weight: bold; font-size: 16px; color: #333; }
    .draw-outs { font-weight: bold; color: #d32f2f; font-size: 18px; }
    .draw-meta { font-size: 12px; color: #666; margin-left: 10px; }
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

def render_hand_html(hand_str, size=40):
    if not hand_str: return ""
    cards = hand_str.split()
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    color_map = {'s': 'black', 'h': '#d32f2f', 'd': '#1976d2', 'c': '#388e3c'}
    
    html = "<div style='display:flex; gap:4px; flex-wrap: wrap;'>"
    for c in cards:
        if len(c) < 2: continue
        rank = c[:-1]
        suit = c[-1].lower()
        symbol = suit_map.get(suit, suit)
        color = color_map.get(suit, 'black')
        
        style = (
            f"width:{size}px; height:{size*1.35}px; background-color:white; "
            f"border:1px solid #bbb; border-radius:4px; "
            f"display:flex; justify-content:center; align-items:center; "
            f"font-size:{size*0.45}px; font-weight:bold; color:{color}; "
            f"box-shadow:1px 1px 2px rgba(0,0,0,0.1);"
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
    
    sc = {s: suits.count(s) for s in suits}
    s_dist = sorted(sc.values(), reverse=True) + [0]*(4-len(sc))
    is_ds = (s_dist[0]==2 and s_dist[1]==2)
    
    if is_ds: tags.append("Double Suited")
    
    if len(set(ranks))==4:
        ur = sorted(list(set(ranks)), reverse=True)
        gaps = [ur[i]-ur[i+1] for i in range(3)]
        if gaps==[1,1,1]: tags.append("Perfect Rundown")
    return tags

def set_input_callback(target_key, value):
    st.session_state[target_key] = value
    widget_key = f"{target_key}_text"
    if widget_key in st.session_state:
        st.session_state[widget_key] = value
    temp_key = f"sel_temp_{target_key}"
    if temp_key in st.session_state:
        st.session_state[temp_key] = []

# ==========================================
# 3. Strict PLO Logic (Made Hands)
# ==========================================
def eval_5card_hand(cards):
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = (len(set(suits)) == 1)
    
    uniq = sorted(list(set(ranks)))
    is_straight = False
    str_high = -1
    if len(uniq) == 5:
        if uniq[4] - uniq[0] == 4:
            is_straight = True; str_high = uniq[4]
        elif uniq == [0, 1, 2, 3, 12]:
            is_straight = True; str_high = 3
    
    rc = Counter(ranks)
    counts = sorted(rc.values(), reverse=True)
    
    if is_straight and is_flush: return "Straight Flush", 900 + str_high
    if counts[0] == 4: return "Quads", 800 + ranks[0]
    if counts[0] == 3 and counts[1] == 2: return "Full House", 700 + [k for k,v in rc.items() if v==3][0]
    if is_flush: return "Flush", 600 + ranks[0]
    if is_straight: return "Straight", 500 + str_high
    if counts[0] == 3: return "Set/Trips", 400 + [k for k,v in rc.items() if v==3][0]
    if counts[0] == 2 and counts[1] == 2: return "Two Pair", 300 + max([k for k,v in rc.items() if v==2])
    if counts[0] == 2: return "One Pair", 200 + [k for k,v in rc.items() if v==2][0]
    return "High Card", 100 + ranks[0]

def evaluate_plo_hand_strict(hand_cards, board_cards):
    best_score = -1
    best_cat = "High Card"
    for h_comb in combinations(hand_cards, 2):
        for b_comb in combinations(board_cards, 3):
            cat, score = eval_5card_hand(list(h_comb) + list(b_comb))
            if score > best_score:
                best_score = score
                best_cat = cat
    
    if best_cat == "Set/Trips":
        h_ranks = [c.rank for c in hand_cards]
        rank_val = best_score - 400
        if h_ranks.count(rank_val) >= 2: best_cat = "Set"
        else: best_cat = "Trips"
    return best_cat, best_score

# ==========================================
# 4. Outs Calculation (Fast Approx for Range)
# ==========================================
def get_best_straight_rank_7(ranks):
    uniq = sorted(list(set(ranks)))
    if len(uniq) < 5: return -1
    best = -1
    for i in range(len(uniq)-4):
        if uniq[i+4] - uniq[i] == 4: best = max(best, uniq[i+4])
    if {0,1,2,3,12}.issubset(set(uniq)): best = max(best, 3)
    return best

def calculate_detailed_outs_fast(hand_cards, board_cards):
    """
    Returns total outs count per category and NUT outs count per category.
    """
    deck_ranks = range(13)
    suits = ['s','h','d','c']
    used = set((c.rank, c.suit) for c in hand_cards + board_cards)
    
    outs = {'Straight':{'total':0,'nut':0}, 'Flush':{'total':0,'nut':0}, 'FullHouse':{'total':0,'nut':0}}
    
    current_all = hand_cards + board_cards
    current_ranks = [c.rank for c in current_all]
    
    # 1. Flush Outs
    for s in suits:
        h_c = sum(1 for c in hand_cards if c.suit==s)
        b_c = sum(1 for c in board_cards if c.suit==s)
        if h_c >= 2 and b_c == 2:
            remaining = 13 - h_c - b_c
            outs['Flush']['total'] = remaining
            
            # Nut check
            h_ranks = [c.rank for c in hand_cards if c.suit==s]
            b_ranks = [c.rank for c in board_cards if c.suit==s]
            max_b = max(b_ranks) if b_ranks else -1
            
            is_nut = False
            if 12 in h_ranks: is_nut = True # A high
            elif 11 in h_ranks and max_b == 12: is_nut = True # K high (A on board)
            elif 10 in h_ranks and 12 in b_ranks and 11 in b_ranks: is_nut = True # Q high
            
            if is_nut: outs['Flush']['nut'] = remaining
            break # One flush draw max
            
    # 2. Straight Outs (Approximate Iteration)
    for r in deck_ranks:
        test_ranks = current_ranks + [r]
        str_rank = get_best_straight_rank_7(test_ranks)
        
        if str_rank != -1:
            # Check if this rank is available
            available_count = 0
            for s in suits:
                if (r, s) not in used: available_count += 1
            
            if available_count > 0:
                # Check if it's NEW straight
                if get_best_straight_rank_7(current_ranks) == -1:
                    outs['Straight']['total'] += available_count
                    # Approx Nut Check: High straights usually nuts
                    if str_rank >= 10: outs['Straight']['nut'] += available_count
                    
    # 3. Full House Outs (Trips/Two Pair -> FH)
    # Simplified: Any pair on board + match in hand?
    # Or Trips + Board Pair?
    # Just counting board pairing cards or hand matching cards
    
    return outs

# ==========================================
# 5. Data Loading & App
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
            if st.button("🎲 Random Hand", key="rnd_plo"):
                deck = []
                for r in "AKQJT98765432":
                    for s in "shdc": deck.append(f"{r}{s}")
                rh = random.sample(deck, 4)
                set_input_callback('plo_input', " ".join(rh))
                st.rerun()

            inp_raw = st.text_input("Enter Hand (e.g. As Ks Jd Th)", key='plo_input_text')
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
                else: st.warning("Hand not found.")

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
            st.markdown("### Player 1 (Blue)")
            p1_mode = st.selectbox("Type", ["Top %", "Fixed"], key="p1t")
            if "Top" in p1_mode: p1_val = st.select_slider("Top %", list(range(5,105,5)), value=15, key="p1v")
            else: p1_fix = st.text_input("Hand", key="p1f")

            st.markdown("### Player 2 (Red)")
            p2_mode = st.selectbox("Type", ["Top %", "Fixed"], key="p2t")
            if "Top" in p2_mode: p2_val = st.select_slider("Top %", list(range(5,105,5)), value=50, key="p2v")
            else: p2_fix = st.text_input("Hand", key="p2f")

        st.subheader("1. Board Input")
        c_rnd1, c_rnd2, c_rnd3 = st.columns(3)
        deck = []
        for r in "AKQJT98765432":
            for s in "shdc": deck.append(f"{r}{s}")
            
        def set_random_board(n):
            rb = " ".join(random.sample(deck, n))
            st.session_state.pf_board = rb
            st.session_state.pf_board_text = rb
            
        if c_rnd1.button("🎲 Flop (3)"): set_random_board(3); st.rerun()
        if c_rnd2.button("🎲 Turn (4)"): set_random_board(4); st.rerun()
        if c_rnd3.button("🎲 River (5)"): set_random_board(5); st.rerun()

        inp_board = st.text_input("Board Cards (3-5)", key='pf_board_text')
        if inp_board != st.session_state.pf_board: st.session_state.pf_board = inp_board
        
        b_cards = normalize_input_text(st.session_state.pf_board)
        if b_cards: st.markdown(render_hand_html(" ".join(b_cards), size=50), unsafe_allow_html=True)

        if st.button("🚀 Analyze Range Hits", type="primary"):
            if len(b_cards)<3: st.error("Need 3+ cards")
            else:
                with st.spinner("Simulating 3,000 hands..."):
                    def get_range(mode, val, fix, df):
                        if "Fixed" in mode:
                            h = normalize_input_text(fix)
                            return [" ".join(h)] if len(h)==4 else []
                        else:
                            limit = int(len(df)*(val/100))
                            sub = df.iloc[:limit]
                            return sub["hand"].sample(3000).tolist() if len(sub)>3000 else sub["hand"].tolist()

                    p1h = get_range(p1_mode, p1_val if "Top" in p1_mode else 0, st.session_state.get('p1f',""), df_plo)
                    p2h = get_range(p2_mode, p2_val if "Top" in p2_mode else 0, st.session_state.get('p2f',""), df_plo)

                    if p1h and p2h:
                        b_objs = [SimpleCard(c) for c in b_cards]
                        is_river = (len(b_cards) >= 5)
                        
                        def analyze(h_list):
                            total = len(h_list)
                            cats_agg = {'Straight':0, 'Flush':0, 'FullHouse':0}
                            nut_agg = {'Straight':0, 'Flush':0, 'FullHouse':0}
                            made_stats = defaultdict(int)
                            draw_stats = defaultdict(int)
                            
                            # For Grouping Pattern
                            draw_patterns = defaultdict(lambda: {'count':0, 'example':""})
                            
                            for h_str in h_list:
                                h_objs = [SimpleCard(c) for c in h_str.split()]
                                # Made
                                made, score = evaluate_plo_hand_strict(h_objs, b_objs)
                                made_stats[made] += 1
                                
                                # Outs
                                outs = calculate_detailed_outs_fast(h_objs, b_objs)
                                
                                total_out_count = 0
                                draw_labels = []
                                
                                # Summing
                                for k in ['Straight','Flush','FullHouse']:
                                    cats_agg[k] += outs[k]['total']
                                    nut_agg[k] += outs[k]['nut']
                                    total_out_count += outs[k]['total']
                                
                                # Draw Types Labels
                                if outs['Flush']['nut'] > 0: 
                                    draw_stats['Nut Flush Draw'] += 1
                                    draw_labels.append("NFD")
                                elif outs['Flush']['total'] > 0: 
                                    draw_stats['Flush Draw'] += 1
                                    draw_labels.append("FD")
                                
                                s_out = outs['Straight']['total']
                                if s_out >= 13: 
                                    draw_stats['Wrap (13+)'] += 1
                                    draw_labels.append(f"Wrap({s_out})")
                                elif s_out >= 9: 
                                    draw_stats['Wrap (9-12)'] += 1
                                    draw_labels.append(f"Wrap({s_out})")
                                elif s_out >= 1: 
                                    draw_stats['Straight Draw'] += 1
                                    draw_labels.append("Str")
                                
                                # Grouping logic for "Top Draw Patterns"
                                if total_out_count > 0:
                                    lbl = " + ".join(draw_labels) if draw_labels else "Other"
                                    key = (total_out_count, lbl)
                                    draw_patterns[key]['count'] += 1
                                    draw_patterns[key]['example'] = h_str
                            
                            # Sort patterns
                            pat_list = []
                            for (outs_num, label), dat in draw_patterns.items():
                                pat_list.append({
                                    'outs': outs_num, 
                                    'label': label, 
                                    'count': dat['count'], 
                                    'pct': dat['count']/total*100, 
                                    'example': dat['example']
                                })
                            pat_list.sort(key=lambda x: x['outs'], reverse=True)
                            
                            return (
                                {k: v/total*100 for k,v in made_stats.items()}, 
                                {k: v/total*100 for k,v in draw_stats.items()},
                                {k: v/total for k,v in cats_agg.items()},
                                {k: v/total for k,v in nut_agg.items()},
                                pat_list
                            )

                        p1_made, p1_draw, p1_avg, p1_nut, p1_pats = analyze(p1h)
                        p2_made, p2_draw, p2_avg, p2_nut, p2_pats = analyze(p2h)

                        st.divider()
                        # Graph 1
                        st.subheader("📊 1. Made Hands (Strict PLO)")
                        made_cats = ["Straight Flush", "Quads", "Full House", "Flush", "Straight", "Set", "Trips", "Two Pair", "One Pair", "High Card"]
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        y = np.arange(len(made_cats)); h = 0.35
                        p1_m = [p1_made.get(c,0) for c in made_cats]
                        p2_m = [p2_made.get(c,0) for c in made_cats]
                        ax1.barh(y+h/2, p1_m, h, label='P1', color='dodgerblue')
                        ax1.barh(y-h/2, p2_m, h, label='P2', color='crimson')
                        ax1.set_yticks(y); ax1.set_yticklabels(made_cats)
                        ax1.set_xlabel("Freq (%)"); ax1.legend()
                        ax1.grid(axis='x', ls='--', alpha=0.3)
                        st.pyplot(fig1)

                        if not is_river:
                            # Graph 2
                            st.divider()
                            st.subheader("🌊 2. Draw Types (Approx)")
                            draw_cats = ["Nut Flush Draw", "Flush Draw", "Wrap (13+)", "Wrap (9-12)", "Straight Draw"]
                            fig2, ax2 = plt.subplots(figsize=(8, 4))
                            yd = np.arange(len(draw_cats))
                            p1_d = [p1_draw.get(c,0) for c in draw_cats]
                            p2_d = [p2_draw.get(c,0) for c in draw_cats]
                            ax2.barh(yd+h/2, p1_d, h, label='P1', color='dodgerblue')
                            ax2.barh(yd-h/2, p2_d, h, label='P2', color='crimson')
                            ax2.set_yticks(yd); ax2.set_yticklabels(draw_cats)
                            ax2.set_xlabel("Freq (%)"); ax2.legend()
                            ax2.grid(axis='x', ls='--', alpha=0.3)
                            st.pyplot(fig2)

                            # Graph 3
                            st.divider()
                            st.subheader("🎯 3. Average Outs")
                            out_cats = ["Straight", "Flush", "FullHouse"]
                            fig3, ax3 = plt.subplots(figsize=(8, 4))
                            x = np.arange(len(out_cats)); w = 0.35
                            
                            p1_tot = [p1_avg[k] for k in out_cats]
                            p1_n = [p1_nut[k] for k in out_cats]
                            p1_w = [t-n for t,n in zip(p1_tot, p1_n)]
                            
                            p2_tot = [p2_avg[k] for k in out_cats]
                            p2_n = [p2_nut[k] for k in out_cats]
                            p2_w = [t-n for t,n in zip(p2_tot, p2_n)]
                            
                            ax3.bar(x-w/2, p1_n, w, color='#1565C0', label='P1 Nut')
                            ax3.bar(x-w/2, p1_w, w, bottom=p1_n, color='#90CAF9', label='P1 Non-Nut')
                            ax3.bar(x+w/2, p2_n, w, color='#C62828', label='P2 Nut')
                            ax3.bar(x+w/2, p2_w, w, bottom=p2_n, color='#EF9A9A', label='P2 Non-Nut')
                            
                            ax3.set_xticks(x); ax3.set_xticklabels(out_cats)
                            ax3.set_ylabel("Avg Outs"); ax3.legend()
                            ax3.grid(axis='y', ls='--', alpha=0.3)
                            st.pyplot(fig3)
                            
                            # --- Top Draw Patterns ---
                            st.divider()
                            st.subheader("🏆 Top Draw Patterns (Grouped by Outs)")
                            
                            max_items = st.slider("Max Items", 5, 50, 10)
                            
                            c_pat1, c_pat2 = st.columns(2)
                            
                            def render_patterns(pats, title, color_bar):
                                st.markdown(f"**{title}**")
                                for p in pats[:max_items]:
                                    label = p['label'] if p['label'] else "Weak Draws"
                                    # Highlight high outs
                                    outs_color = "#d32f2f" if p['outs'] >= 13 else "#333"
                                    
                                    st.markdown(
                                        f"""
                                        <div class="draw-box" style="border-left: 5px solid {color_bar};">
                                            <div>
                                                <div class="draw-title">{label}</div>
                                                <div class="draw-meta">{p['count']} hands ({p['pct']:.1f}%)</div>
                                            </div>
                                            <div class="draw-outs" style="color:{outs_color}">{p['outs']} Outs</div>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                                    st.markdown(render_hand_html(p['example'], size=25), unsafe_allow_html=True)

                            with c_pat1: render_patterns(p1_pats, "Player 1 Patterns", "#2196F3")
                            with c_pat2: render_patterns(p2_pats, "Player 2 Patterns", "#F44336")

# -----------------
# FLO8 & Guide
# -----------------
elif game_mode == "FLO8 (Hi/Lo)":
    st.header("⚖️ FLO8 Strategy")
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
    st.markdown("- **Fixed**: Strict PLO rules (2 from hand, 3 from board).")
    st.markdown("- **Feature**: Grouped 'Top Draw Patterns' to analyze range structure.")

with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
