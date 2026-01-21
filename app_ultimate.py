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

def render_hand_html(hand_str, size=40):
    if not hand_str: return ""
    cards = hand_str.split()
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    color_map = {'s': 'black', 'h': '#d32f2f', 'd': '#1976d2', 'c': '#388e3c'}
    html = "<div style='display:flex; gap:4px; flex-wrap: wrap;'>"
    for c in cards:
        if len(c) < 2: continue
        rank = c[:-1]; suit = c[-1].lower()
        symbol = suit_map.get(suit, suit); color = color_map.get(suit, 'black')
        style = (f"width:{size}px; height:{size*1.35}px; background-color:white; border:1px solid #bbb; "
                 f"border-radius:4px; display:flex; justify-content:center; align-items:center; "
                 f"font-size:{size*0.45}px; font-weight:bold; color:{color}; box-shadow:1px 1px 2px rgba(0,0,0,0.1);")
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
    
    sc = {s: suits.count(s) for s in suits}
    s_dist = sorted(sc.values(), reverse=True) + [0]*(4-len(sc))
    is_ds = (s_dist[0]==2 and s_dist[1]==2)
    is_mono = (s_dist[0]==4)
    if is_ds: tags.append("Double Suited")
    if s_dist[0]==1: tags.append("Rainbow")
    if is_mono: tags.append("Monotone")
    
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

# ==========================================
# 3. Logic: Strict Evaluation & Outs
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
# 5. UI & Main
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
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "Postflop Range", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# ==========================
# PLO PREFLOP (RESTORED)
# ==========================
if game_mode == "PLO (High Only)":
    if df_plo is not None:
        with st.sidebar:
            with st.expander("1. ⚙️ Scenario", expanded=False):
                spr = st.select_slider("Stack Depth", ["Short","Medium","Deep","Very Deep"], value="Medium")
                nw = 0.0 if "Short" in spr else 0.3 if "Medium" in spr else 0.6 if "Deep" in spr else 0.8
                st.caption(f"Nut Weight: {nw*100:.0f}%")
            
            with st.expander("2. 🔍 Hand Rank", expanded=False):
                c1,c2=st.columns([1,2])
                srk=c1.number_input("Rank",1,len(df_plo),1,label_visibility="collapsed")
                if c2.button("Analyze"): set_input_callback('plo_input', df_plo.iloc[srk-1]['hand']); st.rerun()

            with st.expander("3. 🎨 Highlights", expanded=False):
                hl_tags_1 = st.multiselect("Group 1 (🔴 Red)", ["AA","KK","Double Suited"], key="hl1")
                hl_tags_2 = st.multiselect("Group 2 (🔵 Blue)", ["Rundown","Double Pair"], key="hl2")
                hl_tags_3 = st.multiselect("Group 3 (🟢 Green)", ["Single Suited","Monotone"], key="hl3")

            with st.expander("4. 🏷️ Filter", expanded=True):
                sel_top = st.multiselect("Top Rank", list("AKQJT98765432"))
                inc_tags = st.multiselect("Include", ["AA","KK","Double Suited","Rundown"])
                exc_tags = st.multiselect("Exclude", ["Monotone", "Rainbow"])
            
            d_limit = st.slider("List Limit", 5, 50, 10)
            f_df = df_plo
            if sel_top: f_df = f_df[f_df["top_rank"].isin(sel_top)]
            if inc_tags: f_df = f_df[f_df["tags"].apply(lambda t: set(inc_tags).issubset(set(t)))]
            if exc_tags: f_df = f_df[f_df["tags"].apply(lambda t: set(exc_tags).isdisjoint(set(t)))]
            
            st.markdown(f"**Results (Top {d_limit})**")
            for _, r in f_df.head(d_limit).iterrows():
                if st.button(f"{r['hand']} (#{r['rank']})", key=f"l_{r['rank']}"):
                    set_input_callback('plo_input', r['hand']); st.rerun()

        st.header("🔥 PLO Strategy")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.subheader("🔍 Hand Input")
            if st.button("🎲 Random"):
                deck = [f"{r}{s}" for r in "AKQJT98765432" for s in "shdc"]
                set_input_callback('plo_input', " ".join(random.sample(deck, 4))); st.rerun()
            inp = normalize_input_text(st.text_input("Hand", key='plo_input_text'))
            if inp: st.markdown(render_hand_html(" ".join(inp)), unsafe_allow_html=True)
            
            if len(inp)==4:
                res = df_plo[df_plo["card_set"]==frozenset(inp)]
                if not res.empty:
                    row = res.iloc[0]
                    eq=row["equity"]*100; ne=row["nut_equity"]*100
                    sc=(eq*(1-nw)) + ((row["nut_quality"]*100)*nw)
                    m1,m2,m3=st.columns(3)
                    m1.metric("Score",f"{sc:.1f}"); m2.metric("Eq",f"{eq:.1f}%"); m3.metric("Nut Eq",f"{ne:.1f}%")
                    st.write("🏷️ " + " ".join([f"`{t}`" for t in row['tags']]))
                    st.caption(f"Rank: {int(row['rank']):,} (Top {row['pct']:.1f}%)")

        with c2:
            if 'row' in locals():
                st.subheader("📊 Win Dist")
                fig1, ax1 = plt.subplots(figsize=(4,3))
                sizes = [row['win_Straight'], row['win_Flush'], row['win_FH']+row['win_Quads'], 1.0-row['equity']]
                ax1.pie(sizes, labels=['Str', 'Fls', 'FH+', 'Lose'], autopct='%1.1f%%', colors=['#4CAF50','#2196F3','#9C27B0','#EEEEEE'])
                st.pyplot(fig1)
        
        if 'row' in locals():
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("📈 Equity Curve")
                seek = st.slider("Seek %", 0.0, 100.0, 10.0)
                idx = int(len(df_plo)*(seek/100)); s_row = df_plo.iloc[idx] if idx<len(df_plo) else df_plo.iloc[-1]
                st.caption(f"Top {seek}%: {s_row['hand']}")
                fig3, ax3 = plt.subplots(figsize=(5,3))
                ax3.plot(df_plo.iloc[::300]["pct"], df_plo.iloc[::300]["equity"], c="#ccc")
                ax3.scatter(row["pct"], row["equity"], c="red", s=100, label="You")
                ax3.axvline(seek, c="blue", ls=":")
                st.pyplot(fig3)
            with c4:
                st.subheader("🌌 Scatter")
                use_zoom = st.checkbox("Auto Zoom", value=True)
                fig2, ax2 = plt.subplots(figsize=(5,3))
                bg=df_plo.sample(2000)
                ax2.scatter(bg["equity"], bg["nut_equity"], c='#eee', s=10)
                
                # Highlights
                for tags, col in [(hl_tags_1,'crimson'), (hl_tags_2,'dodgerblue'), (hl_tags_3,'limegreen')]:
                    if tags:
                        hset = set(tags)
                        sub = df_plo[df_plo["tags"].apply(lambda t: hset.issubset(set(t)))].sample(min(1000, len(df_plo)))
                        ax2.scatter(sub["equity"], sub["nut_equity"], fc='none', ec=col, s=30)
                
                ax2.scatter(row["equity"], row["nut_equity"], c='red', s=100, marker='*', zorder=10)
                
                if use_zoom:
                    ax2.set_xlim(max(0, row['equity']-0.15), min(1, row['equity']+0.15))
                    ax2.set_ylim(max(0, row['nut_equity']-0.15), min(1, row['nut_equity']+0.15))
                st.pyplot(fig2)

# ==========================
# POSTFLOP RANGE (Fixed)
# ==========================
elif game_mode == "Postflop Range":
    st.header("📊 Postflop Range Analysis")
    if df_plo is not None:
        with st.sidebar:
            st.markdown("### Range Settings")
            p1_val = st.slider("Player 1 (Blue) Top %", 5, 100, 15, 5)
            p2_val = st.slider("Player 2 (Red) Top %", 5, 100, 50, 5)
        
        st.subheader("1. Board Input")
        c1,c2,c3 = st.columns(3)
        # Fix: Direct list creation to avoid None display bug
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

# ==========================
# FLO8 (RESTORED)
# ==========================
elif game_mode == "FLO8 (Hi/Lo)":
    st.header("⚖️ FLO8 Strategy")
    with st.sidebar:
        with st.expander("1. 🔍 Rank Search", expanded=True):
            if df_flo8 is not None:
                c1,c2=st.columns([1,2])
                rk=c1.number_input("Rank",1,len(df_flo8),1,label_visibility="collapsed")
                if c2.button("Analyze"): set_input_callback('flo8_input', df_flo8.iloc[rk-1]['hand']); st.rerun()
                
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("🎲 Random Hand", key="rnd_flo8"):
            deck=[f"{r}{s}" for r in "AKQJT98765432" for s in "shdc"]
            set_input_callback('flo8_input', " ".join(random.sample(deck, 4))); st.rerun()
        i8 = normalize_input_text(st.text_input("Hand", key='flo8_input_text'))
        if i8: st.markdown(render_hand_html(" ".join(i8)), unsafe_allow_html=True)
    
    with c2:
        if i8 and len(i8)==4:
            sc, dt = calculate_flo8_heuristic(" ".join(i8))
            st.metric("Hutchinson Points", sc, help="20+ points recommended")
            st.bar_chart(dt)
            if df_flo8 is not None:
                r = df_flo8[df_flo8["card_set"]==frozenset(i8)]
                if not r.empty:
                    rr = r.iloc[0]
                    cc1,cc2,cc3 = st.columns(3)
                    cc1.metric("Scoop %", f"{rr['scoop_pct']:.1f}%")
                    cc2.metric("High Eq", f"{rr['high_equity']:.1f}%")
                    cc3.metric("Low Eq", f"{rr['low_equity']:.1f}%")
                    st.caption(f"Rank: #{rr['rank']} (Top {rr['pct_total']:.1f}%)")

elif game_mode == "Guide":
    st.header("📖 Guide")
    st.markdown("PLO & FLO8 detailed features restored. Postflop bug fixed.")

with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
