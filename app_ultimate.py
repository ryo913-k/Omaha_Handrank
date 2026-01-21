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
    /* UI調整 */
    ul[data-testid="stSelectboxVirtualDropdown"] { z-index: 99999 !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
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
    
    html = "<div style='display:flex; gap:4px; margin-bottom:5px; flex-wrap: wrap;'>"
    for c in cards:
        if len(c) < 2: continue
        rank = c[:-1]
        suit = c[-1].lower()
        symbol = suit_map.get(suit, suit)
        color = color_map.get(suit, 'black')
        
        style = (
            f"width:{size}px; height:{size*1.4}px; background-color:white; "
            f"border:1px solid #bbb; border-radius:4px; "
            f"display:flex; justify-content:center; align-items:center; "
            f"font-size:{size*0.45}px; font-weight:bold; color:{color}; "
            f"box-shadow:1px 1px 3px rgba(0,0,0,0.1);"
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
    for suit in ['s', 'h', 'd', 'c']:
        ms_key = f"ms_{suit}_{target_key}"
        if ms_key in st.session_state:
            st.session_state[ms_key] = []

# ==========================================
# 3. Postflop Evaluator (Detailed)
# ==========================================
def get_best_straight_rank(ranks):
    """
    与えられたランクのリストから最強のストレートのハイカードランクを返す。
    ストレートがない場合は -1
    A=12, 2=0, ..., 5=3. Wheel(A2345)=3 (5-high)
    """
    uniq = sorted(list(set(ranks)))
    if len(uniq) < 5: return -1
    
    best = -1
    # Regular straights
    for i in range(len(uniq)-4):
        if uniq[i+4] - uniq[i] == 4:
            best = max(best, uniq[i+4])
            
    # Wheel (A,2,3,4,5) -> ranks [0,1,2,3, ..., 12]
    if {0,1,2,3,12}.issubset(set(uniq)):
        best = max(best, 3) # 5-high
        
    return best

def calculate_straight_outs(hand_cards, board_cards):
    """
    ストレートアウツの詳細計算
    - フラッシュになるアウツは除外
    - ナッツアウツかどうか判定
    """
    deck_ranks = range(13) # 2~A
    suits = ['s','h','d','c']
    
    # 既に使われているカード
    used_cards = set((c.rank, c.suit) for c in hand_cards + board_cards)
    
    total_outs_count = 0
    nut_outs_count = 0
    
    # Check flush potential on board to exclude flush cards
    # Boardが3枚以上ある前提
    board_flush_suit = None
    sc = Counter([c.suit for c in board_cards])
    # Boardに3枚以上あるスーツがあれば、そのスーツのアウツは警戒対象
    # ただし「フラッシュ完成はアウツとしない」なので、Handと合わせて5枚になるなら除外
    
    # 全ての未知のカード(Rank, Suit)についてループするのは重いので、
    # Rankごとに判定してからSuitでフィルタする
    
    # 1. 現状のベストストレート（比較用：リバーでボードが変わるため、仮想ナッツ判定に必要）
    # ナッツ判定： 「このランクが落ちた時、ボード+そのカードで作れる最強ストレート」 vs 「自分のハンド+ボード+そのカード」
    
    outs_ranks = []
    
    for r in deck_ranks:
        # このランクがアウツになりうるか？
        # 自分のハンド＋ボード＋ランクr でストレートができるか
        test_hand_ranks = [c.rank for c in hand_cards] + [c.rank for c in board_cards] + [r]
        my_straight = get_best_straight_rank(test_hand_ranks)
        
        if my_straight != -1:
            # ストレート完成。これが「アウツ」として有効か、スートごとにチェック
            
            # このランクrを持つカード4枚のうち、まだデッキにあるもの
            for s in suits:
                if (r, s) in used_cards: continue
                
                # Check Flush: Hand + Board + (r,s)
                # フラッシュ完成なら除外
                temp_all = hand_cards + board_cards + [SimpleCard(f"X{s}")] # Dummy suit
                # (SimpleCardのコンストラクタ簡略化のためロジックで判定)
                
                # フラッシュチェック
                is_flush = False
                suit_count = Counter([c.suit for c in hand_cards] + [c.suit for c in board_cards] + [s])
                if suit_count[s] >= 5: is_flush = True
                
                if is_flush:
                    continue # フラッシュになるカードはストレートアウツとしてカウントしない
                
                # ここまで来れば「純粋なストレートアウツ」
                total_outs_count += 1
                
                # ナッツ判定
                # ボード + このカード(r) で可能な最強ストレート（仮想敵）
                # 敵は任意の2枚を持てる -> ボード3枚+r1枚+敵2枚
                # 簡易判定: 「ボード+r」に含まれるランクだけで構成されうるストレートの上限ではなく、
                # 「ボード+r」に対して、任意の2枚を持って作れる最強のストレートと、自分のストレートを比較
                
                board_plus_r = [c.rank for c in board_cards] + [r]
                
                # 仮想ナッツの計算（重いが、正確性のため）
                # 効率化: ナッツは常に 「ボード+r」のランク構成によって決まる
                # (ボード+r)に合う 最強の2枚の組み合わせを探すのは大変。
                # しかし、Omahaのナッツは「ボード+r」の隙間を埋めるか、上につけるか。
                # 実用的な近似: get_best_straight_rank は既にハンド2枚制限をしていない（簡易版）ため、
                # 全ランク(0..12)から、「ボード+r」と組み合わせてできる最強ランクを計算すればよい
                
                possible_max = -1
                # 敵が持ちうる最強ハンド（任意の2枚）をシミュレートするのは重すぎる
                # -> 代替: 自分のストレートランクが、理論上の天井(Aハイ)に近いか、あるいはボード構成上最強か
                
                # Solver級の厳密なナッツ判定は計算量が爆発するため、ここでは
                # 「自分のストレートランク」が「ボード+r を使って作れる理論上の最強ストレート」と一致するか確認
                
                # 理論上最強: ボード+r のランクに加え、任意の2ランクを追加して作れるMax
                # これは計算ロジックでカバー可能
                # ボード+r のランク集合
                br_set = set(board_plus_r)
                
                # 考えられる全てのストレート(A-5 ... T-A)について、
                # そのストレートを構成する5枚のうち、3枚以上が br_set に含まれていれば、
                # 残り2枚を敵が持っている可能性があるため、そのストレートは成立しうる。
                
                the_nut_rank = -1
                # Check 5-high to A-high (3 to 12)
                # Wheel (A,2,3,4,5) -> rank 3
                if len({0,1,2}.intersection(br_set)) + len({12}.intersection(br_set)) >= (3 if r in [0,1,2,12] else 3): 
                     # 厳密には r が入ることで条件を満たすか？
                     # rは既に br_setに入っている
                     pass
                
                # 簡易ロジック:
                # 1. 可能なストレートの形（5連続）を全てリストアップ
                # 2. その形を作るのに必要な「不足カード」が2枚以下であるものを抽出
                # 3. その中で最強のランクを持つものが「ナッツ」
                
                straights = []
                # Regular 2-6 (4) to T-A (12)
                for top in range(4, 13): # 4(6high) to 12(Ahigh)
                    needed = {top, top-1, top-2, top-3, top-4}
                    missing = len(needed - br_set)
                    if missing <= 2:
                        straights.append(top)
                # Wheel (5-high)
                needed_wheel = {0,1,2,3,12}
                if len(needed_wheel - br_set) <= 2:
                    straights.append(3)
                
                if straights:
                    the_nut_rank = max(straights)
                
                if my_straight >= the_nut_rank:
                    nut_outs_count += 1

    return total_outs_count, nut_outs_count

def evaluate_hits_detailed(hand_cards, board_cards):
    """
    役判定＋ストレート詳細分析
    """
    all_cards = hand_cards + board_cards
    ranks = [c.rank for c in all_cards]
    
    # --- Made Hands (Existing Logic) ---
    made = "High Card"
    
    # Flush
    is_flush = False
    for s in ['s','h','d','c']:
        h_cnt = sum(1 for c in hand_cards if c.suit == s)
        b_cnt = sum(1 for c in board_cards if c.suit == s)
        if h_cnt >= 2 and b_cnt >= 3: is_flush = True; break
            
    # Straight
    uniq_ranks = sorted(list(set(ranks)))
    is_straight = False
    if get_best_straight_rank(ranks) != -1: is_straight = True

    # Pairs/Sets
    rc = Counter(ranks)
    is_quads = any(c == 4 for c in rc.values())
    is_fh = (any(c == 3 for c in rc.values()) and any(c >= 2 for k,c in rc.items() if rc[k]!=3 or list(rc.values()).count(3)>1))
    
    h_rc = Counter([c.rank for c in hand_cards])
    b_rc = Counter([c.rank for c in board_cards])
    
    has_set = any(h_rc[r] == 2 and b_rc[r] == 1 for r in h_rc)
    has_trips = any(h_rc[r] == 1 and b_rc[r] == 2 for r in h_rc)
    pair_count = sum(1 for c in rc.values() if c >= 2)
    
    if is_quads: made = "Quads"
    elif is_fh: made = "Full House"
    elif is_flush: made = "Flush"
    elif is_straight: made = "Straight"
    elif has_set: made = "Set"
    elif has_trips: made = "Trips"
    elif pair_count >= 2: made = "Two Pair"
    elif pair_count == 1:
        board_ranks = [c.rank for c in board_cards]
        max_b = max(board_ranks) if board_ranks else -1
        if max_b in [c.rank for c in hand_cards]: made = "Top Pair"
        elif any(c.rank > max_b for c in hand_cards if h_rc[c.rank]==2): made = "Overpair"
        else: made = "Weak Pair"
    
    # --- Draws Analysis ---
    draws = []
    str_outs = 0
    str_nut_outs = 0
    
    if len(board_cards) <= 4: # Flop or Turn
        # 1. Flush Draw
        for s in ['s','h','d','c']:
            h_cnt = sum(1 for c in hand_cards if c.suit == s)
            b_cnt = sum(1 for c in board_cards if c.suit == s)
            if h_cnt >= 2 and b_cnt == 2:
                # Nut Check
                hand_max_s = max([c.rank for c in hand_cards if c.suit==s]) if h_cnt>0 else -1
                if hand_max_s == 12: draws.append("Nut Flush Draw")
                else: draws.append("Flush Draw")
        
        # 2. Straight Draw (Outs Calculation)
        # 既にストレート完成している場合はドローとは呼ばない（さらに上を目指す場合はあるがここでは省略）
        if made not in ["Straight", "Flush", "Full House", "Quads"]:
            t_outs, n_outs = calculate_straight_outs(hand_cards, board_cards)
            str_outs = t_outs
            str_nut_outs = n_outs
            
            if t_outs >= 13: draws.append("Wrap (13+ outs)")
            elif t_outs >= 9: draws.append("Wrap (9-12 outs)")
            elif t_outs == 8: draws.append("OESD (8 outs)")
            elif t_outs >= 1: draws.append("Gutshot (1-7 outs)")

    return made, draws, str_outs, str_nut_outs

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
    except FileNotFoundError:
        return None

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
                st.session_state[session_key] = final_hand
                st.session_state[f"{session_key}_text"] = final_hand
                st.rerun()
            return collected
        elif len(collected) > 0:
            st.caption(f"Selected: {len(collected)}/4 cards.")
        
    return []

# ==========================================
# 6. Main Application Logic
# ==========================================
st.title("🃏 Omaha Hand Analyzer")

for k in ['plo_input', 'flo8_input', 'p1_fixed', 'p2_fixed', 'pf_board']:
    if k not in st.session_state: st.session_state[k] = ""
if st.session_state.plo_input == "": st.session_state.plo_input = "As Ks Jd Th"
if st.session_state.flo8_input == "": st.session_state.flo8_input = "Ad Ah 2s 3d"

for k in ['plo_input', 'flo8_input', 'p1_fixed', 'p2_fixed', 'pf_board']:
    tk = f"{k}_text"
    if tk not in st.session_state: st.session_state[tk] = st.session_state[k]

df_plo = load_plo_data()
df_flo8 = load_flo8_data()

with st.sidebar:
    st.title("Navigation")
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "Postflop Range", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# ==========================================
# MODE: PLO
# ==========================================
if game_mode == "PLO (High Only)":
    if df_plo is None:
        st.warning("Data loading failed. Please upload 'plo_detailed_ranking.zip'.")
    else:
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
                             set_input_callback('plo_input', r['hand'])
                             st.rerun()
                    else: st.write("-")
                if not fr.empty:
                    st.caption(f"**{r['hand']}** (Top {r['pct']:.2f}%)")

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
                            set_input_callback('plo_input', r['hand'])
                            st.rerun()
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
                
                groups = [(hl_tags_1, 'crimson', 'Grp1(Red)'), (hl_tags_2, 'dodgerblue', 'Grp2(Blu)'), (hl_tags_3, 'limegreen', 'Grp3(Grn)')]
                for tags, color, lbl_prefix in groups:
                    if tags:
                        src = filtered_df if filtered_df is not None else df_plo
                        ht = set(tags)
                        mask = src["tags"].apply(lambda t: ht.issubset(set(t)))
                        hdf = src[mask]
                        if not hdf.empty:
                            hdf_s = hdf.sample(2000, random_state=42) if len(hdf)>2000 else hdf
                            hx, hy = gxy(hdf_s, cmode)
                            label_text = f"{lbl_prefix}: {','.join(tags)[:10]}.."
                            ax2.scatter(hx, hy, fc='none', ec=color, s=50, lw=1.5, label=label_text)
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

# ==========================================
# MODE: Postflop Range
# ==========================================
elif game_mode == "Postflop Range":
    st.header("📊 Postflop Range Analysis")
    
    if df_plo is None:
        st.warning("DB not loaded.")
    else:
        with st.sidebar:
            st.markdown("### Player Settings")
            st.markdown("**Player 1**")
            p1_mode = st.selectbox("P1 Type", ["Top % Range", "Fixed Hand"], key="p1_type")
            if "Range" in p1_mode:
                p1_range_val = st.select_slider("P1 Top %", options=list(range(5, 105, 5)), value=15)
            else:
                p1_fixed = st.text_input("P1 Hand", key="p1_fixed_text")
                if p1_fixed != st.session_state.p1_fixed: st.session_state.p1_fixed = p1_fixed

            st.divider()
            st.markdown("**Player 2**")
            p2_mode = st.selectbox("P2 Type", ["Top % Range", "Fixed Hand"], key="p2_type")
            if "Range" in p2_mode:
                p2_range_val = st.select_slider("P2 Top %", options=list(range(5, 105, 5)), value=50)
            else:
                p2_fixed = st.text_input("P2 Hand", key="p2_fixed_text")
                if p2_fixed != st.session_state.p2_fixed: st.session_state.p2_fixed = p2_fixed

        st.subheader("1. Board Input")
        render_card_selector('pf_board')
        
        pf_board_raw = st.text_input("Board Cards (3-5 cards)", key='pf_board_text', placeholder="e.g. Ks 7d 2c")
        if pf_board_raw != st.session_state.pf_board: st.session_state.pf_board = pf_board_raw
        
        board_cards = normalize_input_text(st.session_state.pf_board)
        if board_cards: st.markdown(render_hand_html(" ".join(board_cards), size=50), unsafe_allow_html=True)

        if st.button("🚀 Analyze Range Hits", type="primary"):
            if len(board_cards) < 3:
                st.error("Please enter at least 3 board cards.")
            else:
                with st.spinner("Analyzing Ranges (Simulating 5,000 hands)..."):
                    def get_hands_from_range(mode, range_val, fixed_val, df):
                        if "Fixed" in mode:
                            h = normalize_input_text(fixed_val)
                            return [" ".join(h)] if len(h)==4 else []
                        else:
                            limit_rank = int(len(df) * (range_val / 100))
                            sub = df.iloc[:limit_rank]
                            sample_size = 5000 # Increased for stability
                            if len(sub) > sample_size:
                                return sub["hand"].sample(sample_size).tolist()
                            return sub["hand"].tolist()

                    p1_hands = get_hands_from_range(p1_mode, p1_range_val if "Range" in p1_mode else "", st.session_state.p1_fixed, df_plo)
                    p2_hands = get_hands_from_range(p2_mode, p2_range_val if "Range" in p2_mode else "", st.session_state.p2_fixed, df_plo)

                    if not p1_hands or not p2_hands:
                        st.error("Invalid hand input.")
                    else:
                        def analyze_list_detailed(h_list, b_objs):
                            stats = defaultdict(int)
                            draws_stats = defaultdict(int)
                            total_outs_sum = 0
                            total_nut_outs_sum = 0
                            draw_hits = 0 # Hands that have draws
                            
                            total = len(h_list)
                            for h_str in h_list:
                                h_objs = [SimpleCard(c) for c in h_str.split()]
                                made, draws, str_outs, str_nut_outs = evaluate_hits_detailed(h_objs, b_objs)
                                stats[made] += 1
                                for d in draws: draws_stats[d] += 1
                                
                                if str_outs > 0:
                                    total_outs_sum += str_outs
                                    total_nut_outs_sum += str_nut_outs
                                    draw_hits += 1
                            
                            # Averages
                            avg_outs = total_outs_sum / draw_hits if draw_hits > 0 else 0
                            avg_nut_outs = total_nut_outs_sum / draw_hits if draw_hits > 0 else 0
                            
                            return (
                                {k: v/total*100 for k,v in stats.items()}, 
                                {k: v/total*100 for k,v in draws_stats.items()},
                                avg_outs,
                                avg_nut_outs
                            )

                        board_objs = [SimpleCard(c) for c in board_cards]
                        p1_made, p1_draw, p1_ao, p1_ano = analyze_list_detailed(p1_hands, board_objs)
                        p2_made, p2_draw, p2_ao, p2_ano = analyze_list_detailed(p2_hands, board_objs)

                        st.divider()
                        
                        # Metrics Row
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.write("**Player 1 Straight Stats (Avg)**")
                            c_a, c_b = st.columns(2)
                            c_a.metric("Avg Outs", f"{p1_ao:.1f}")
                            c_b.metric("Avg Nut Outs", f"{p1_ano:.1f}")
                        with col_m2:
                            st.write("**Player 2 Straight Stats (Avg)**")
                            c_a, c_b = st.columns(2)
                            c_a.metric("Avg Outs", f"{p2_ao:.1f}")
                            c_b.metric("Avg Nut Outs", f"{p2_ano:.1f}")

                        st.divider()
                        c_plot1, c_plot2 = st.columns(2)
                        
                        cats_made = ["Quads", "Full House", "Flush", "Straight", "Set", "Trips", "Two Pair", "Overpair", "Top Pair"]
                        cats_draw = ["Nut Flush Draw", "Flush Draw", "Wrap (13+ outs)", "Wrap (9-12 outs)", "OESD (8 outs)", "Gutshot (1-7 outs)"]

                        def plot_comp(cats, p1_data, p2_data, title):
                            p1_vals = [p1_data.get(c, 0) for c in cats]
                            p2_vals = [p2_data.get(c, 0) for c in cats]
                            fig, ax = plt.subplots(figsize=(5,4))
                            y = np.arange(len(cats))
                            h = 0.35
                            ax.barh(y + h/2, p1_vals, h, label='P1', color='dodgerblue')
                            ax.barh(y - h/2, p2_vals, h, label='P2', color='crimson')
                            ax.set_yticks(y)
                            ax.set_yticklabels(cats)
                            ax.set_xlabel("Freq (%)")
                            ax.set_title(title)
                            ax.legend()
                            ax.grid(axis='x', linestyle='--', alpha=0.5)
                            return fig

                        with c_plot1:
                            st.write("##### Made Hands")
                            st.pyplot(plot_comp(cats_made, p1_made, p2_made, "Made Hand Distribution"))
                        with c_plot2:
                            st.write("##### Draws (Flop/Turn)")
                            st.pyplot(plot_comp(cats_draw, p1_draw, p2_draw, "Draw Distribution"))

# ==========================================
# MODE: FLO8
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
    st.header("📖 Omaha Hand Analyzer 取扱説明書")
    
    st.markdown("""
    このツールは、**Pot Limit Omaha (PLO)** および **Fixed Limit Omaha Hi/Lo (FLO8)** のハンド強度を、
    数億回のシミュレーションデータと数学的モデルに基づいて精密に分析するアプリケーションです。
    """)

    st.divider()

    st.subheader("1. 画面の切り替え")
    st.info("サイドバーの一番上にある **[Game Mode]** でモードを切り替えます。")
    st.markdown("""
    - **🔥 PLO (High Only)**: 通常のオマハ（ハイのみ）。詳細な勝率データとナッツポテンシャル分析が可能です。
    - **📊 Postflop Range**: フロップ以降のレンジ分析。お互いのレンジがボードにどう絡んでいるかを可視化します。
    - **⚖️ FLO8 (Hi/Lo)**: ハイロー（エイトオアベター）。Hutchinsonポイントとスクープ率を表示します。
    """)

    st.divider()

    st.subheader("2. 🔥 PLO モードの機能")
    
    st.markdown("#### A. ハンド入力")
    st.write("2通りの方法でハンドを入力できます。")
    st.markdown("""
    1. **🃏 Open Card Selector**: スートごとに分かれたパネルから、クリックで4枚を選択します（スマホ対応）。
    2. **テキスト入力**: `As Ks Jd Th` のように直接入力します（大文字小文字区別なし）。
    """)
    
    st.markdown("#### B. 分析指標")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Power Score**: 勝率 + ナッツの作りやすさ + SPRを考慮した総合スコア。
        - **Raw Equity**: 単純なオールイン勝率。
        - **Nut Equity**: ナッツ級の役で勝つ確率。
        """)
    with col2:
        st.markdown("""
        - **Tags**: `Double Suited`, `Rundown` などの特徴判定。
        - **Global Rank**: 全ハンド中の順位（Top X%）。
        """)

    st.markdown("#### C. グラフ分析")
    st.markdown("""
    - **📊 Win Distribution (円グラフ)** どの役で勝つかを表示。
    - **📈 Equity Curve (順位曲線)** 全体の中での位置。**シークバー**で上位X%のハンドを逆引き可能。
    - **🌌 Equity Scatter (散布図)** 「勝率」と「ナッツ品質」のバランス。サイドバーで3色ハイライト設定可能。
    """)
    
    st.divider()
    
    st.subheader("3. 📊 Postflop Range モード")
    st.markdown("""
    **「自分のレンジ（例：上位15%）は、このフロップでどれくらい強いのか？」** を分析します。
    
    1. **Player設定**: P1とP2のレンジ（Top %）または固定ハンドを設定。
    2. **Board入力**: フロップ（3枚）〜リバー（5枚）を入力。
    3. **Analyze**: ボタンを押すと、レンジ内のハンドがボードにどう絡んでいるか（セット、フラッシュ、ドロー等）を棒グラフで比較表示します。
    """)

    st.divider()

    st.subheader("4. ⚖️ FLO8 モードの機能")
    st.markdown("""
    - **Hutchinson Points**: FLO8のハンド評価点（20点以上が目安）。
    - **Scoop %**: ハイ・ロー総取りの確率。
    """)
    
    st.success("Analysis powered by custom simulation engine.")

# サイドバー下部にクレジット表示
with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
