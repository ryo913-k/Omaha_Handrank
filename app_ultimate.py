import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
from heuristics import calculate_flo8_heuristic

# ==========================================
# 1. Config & Styles
# ==========================================
st.set_page_config(page_title="Omaha Ultimate Solver", layout="wide")

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

def normalize_input_text(text):
    if not text: return []
    text = unicodedata.normalize('NFKC', text)
    parts = text.split()
    cleaned = []
    for p in parts:
        if len(p) >= 2: cleaned.append(p[:-1].upper() + p[-1].lower())
    return cleaned

def render_hand_html(hand_str):
    if not hand_str: return ""
    cards = hand_str.split()
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    color_map = {'s': 'black', 'h': '#d32f2f', 'd': '#1976d2', 'c': '#388e3c'}
    
    html = "<div style='display:flex; gap:8px; margin-bottom:10px; flex-wrap: wrap;'>"
    for c in cards:
        if len(c) < 2: continue
        rank = c[:-1]
        suit = c[-1].lower()
        symbol = suit_map.get(suit, suit)
        color = color_map.get(suit, 'black')
        
        style = (
            f"width:45px; height:60px; background-color:white; "
            f"border:1px solid #bbb; border-radius:6px; "
            f"display:flex; justify-content:center; align-items:center; "
            f"font-size:20px; font-weight:bold; color:{color}; "
            f"box-shadow:2px 2px 5px rgba(0,0,0,0.1);"
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
    sv = sorted(sc.values(), reverse=True)
    s_dist = sv + [0]*(4-len(sv))
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

# 【修正】ボタン押下時にカードセレクターの状態もリセットする関数
def set_input_callback(target_key, value):
    # 1. メインの入力値を更新
    st.session_state[target_key] = value
    
    # 2. テキストボックスのウィジェット状態も同期
    widget_key = f"{target_key}_text"
    if widget_key in st.session_state:
        st.session_state[widget_key] = value
    
    # 3. 【重要】カードセレクター(Multiselect)の状態をクリアする
    # これをしないと、古い選択状態が残ってしまい、即座に上書きされてしまう
    for suit in ['s', 'h', 'd', 'c']:
        ms_key = f"ms_{suit}_{target_key}"
        if ms_key in st.session_state:
            st.session_state[ms_key] = []

# ==========================================
# 3. Data Loading
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
# 4. UI Components
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
# 5. Main Application Logic
# ==========================================
st.title("🃏 Omaha Ultimate Solver")

if 'plo_input' not in st.session_state: st.session_state.plo_input = "As Ks Jd Th"
if 'flo8_input' not in st.session_state: st.session_state.flo8_input = "Ad Ah 2s 3d"
if 'plo_input_text' not in st.session_state: st.session_state.plo_input_text = st.session_state.plo_input
if 'flo8_input_text' not in st.session_state: st.session_state.flo8_input_text = st.session_state.flo8_input

df_plo = load_plo_data()
df_flo8 = load_flo8_data()

with st.sidebar:
    st.title("Navigation")
    game_mode = st.radio("Game Mode", ["PLO (High Only)", "FLO8 (Hi/Lo)", "Guide"], label_visibility="collapsed")
    st.divider()

# ==========================================
# MODE: PLO
# ==========================================
if game_mode == "PLO (High Only)":
    if df_plo is None:
        st.warning("Data loading failed. Please upload 'plo_detailed_ranking.zip'.")
    else:
        # --- PLO Sidebar (Improved Layout) ---
        with st.sidebar:
            # 1. Filters (Collapsible, Default Open)
            with st.expander("1. 🏷️ Filter Conditions", expanded=True):
                ranks_opt = list("AKQJT98765432")
                sel_top = st.multiselect("Top Rank", ranks_opt)
                
                avail_tags = ["AA","KK","QQ","Double Pair","Double Suited","Single Suited","A-High Suit","Rainbow","Monotone","Broadway","Perfect Rundown","Double Gap Rundown"]
                inc_tags = st.multiselect("Include", avail_tags)
                exc_tags = st.multiselect("Exclude", avail_tags)

            # 2. Highlight (Collapsible, Default Closed)
            with st.expander("2. 🎨 Highlight Groups", expanded=False):
                hl_tags_1 = st.multiselect("Group 1 (🔴 Red)", avail_tags, key="hl1")
                hl_tags_2 = st.multiselect("Group 2 (🔵 Blue)", avail_tags, key="hl2")
                hl_tags_3 = st.multiselect("Group 3 (🟢 Green)", avail_tags, key="hl3")

            # 3. Results List
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

            st.divider()

            # 4. Rank Search
            with st.expander("3. 🔍 Rank Search", expanded=False):
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

            # 5. Scenario
            with st.expander("4. ⚙️ Scenario", expanded=False):
                spr = st.select_slider("Stack Depth / SPR", ["Short","Medium","Deep","Very Deep"], value="Medium")
                nw = 0.0 if "Short" in spr else 0.3 if "Medium" in spr else 0.6 if "Deep" in spr else 0.8
                st.caption(f"Nut Weight: {nw*100:.0f}%")

        # --- PLO MAIN ---
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
                    # Keyにseek_pctを含めることで、スライダー変更時にボタンを再生成し、引数を確実に更新する
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
                
                # Highlights 1-3
                groups = [(hl_tags_1, 'crimson', 'Grp1(Red)'), 
                          (hl_tags_2, 'dodgerblue', 'Grp2(Blu)'), 
                          (hl_tags_3, 'limegreen', 'Grp3(Grn)')]
                
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
                else:
                    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)
                
                ax2.grid(True, ls='--', alpha=0.3)
                ax2.legend(fontsize=8, loc='upper left')
                st.pyplot(fig2)

# ==========================================
# MODE: FLO8
# ==========================================
elif game_mode == "FLO8 (Hi/Lo)":
    with st.sidebar:
        # 1. Rank Search
        with st.expander("1. 🔍 Rank Search", expanded=True):
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
    st.header("📖 Omaha Ultimate Solver 取扱説明書")
    
    st.markdown("""
    このツールは、**Pot Limit Omaha (PLO)** および **Fixed Limit Omaha Hi/Lo (FLO8)** のハンド強度を、
    数億回のシミュレーションデータと数学的モデルに基づいて精密に分析するアプリケーションです。
    """)

    st.divider()

    st.subheader("1. 画面の切り替え")
    st.info("サイドバーの一番上にある **[Game Mode]** でモードを切り替えます。")
    st.markdown("""
    - **🔥 PLO (High Only)**: 通常のオマハ（ハイのみ）。詳細な勝率データとナッツポテンシャル分析が可能です。
    - **⚖️ FLO8 (Hi/Lo)**: ハイロー（エイトオアベター）。Hutchinsonポイントとスクープ率を表示します。
    """)

    st.divider()

    st.subheader("2. 🔥 PLO モードの機能")
    
    st.markdown("#### A. ハンド入力 (Hand Input)")
    st.write("2通りの方法でハンドを入力できます。")
    st.markdown("""
    1. **🃏 Open Card Selector**: スートごとに分かれたパネルから、クリックで4枚を選択します（スマホ対応）。
    2. **テキスト入力**: `As Ks Jd Th` のように直接入力します（大文字小文字区別なし）。
    """)
    
    st.markdown("#### B. 分析指標 (Metrics)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Power Score**: 勝率に加え、「ナッツの作りやすさ」と「SPR（スタック比）」を考慮した総合スコアです。
        - **Raw Equity**: 単純なオールイン勝率です。
        - **Nut Equity**: ナッツ級の役（フラッシュ、フルハウス以上）で勝つ確率です。
        """)
    with col2:
        st.markdown("""
        - **Tags**: ハンドの特徴（`Double Suited`, `Rundown` など）を自動判定します。
        - **Global Rank**: 全ハンド中の順位を表示します（例: Top 1.5%）。
        """)

    st.markdown("#### C. グラフ分析")
    st.markdown("""
    - **📊 Win Distribution (円グラフ)** 「どのような役で勝つか」の内訳です。ナッツ級（ストレート以上）が多いほど強いハンドと言えます。
      
    - **📈 Equity Curve (順位曲線)** あなたのハンドが全体の中でどの位置にいるかを可視化します。
      > **💡 シークバー機能**: グラフ上のバーを動かすと、「上位10%の境界線にあるハンドは何か？」などを逆引きして分析できます。

    - **🌌 Equity Scatter (散布図)** 「勝率(X軸)」と「ナッツ品質(Y軸)」のバランスを見ます。
      - 右上にあるほど最強です。
      - **Highlight機能**: サイドバーで設定した条件（赤・青・緑）のハンドがどこに分布しているかを確認できます。
    """)

    st.divider()

    st.subheader("3. ⚖️ FLO8 モードの機能")
    st.markdown("""
    FLO8は「ハイ」と「ロー」の両方を狙うゲームです。
    
    - **Hutchinson Points**:  
      FLO8の著名な評価システムです。一般的に **20ポイント以上** がプレイ可能なハンドとされています。
      - 棒グラフで「High Pair」「Low Potential」などの内訳を表示します。
      
    - **Scoop / Equity**:  
      - **Scoop %**: ハイとローの両方を取る（総取り）確率。最も重要な指標です。
      - **High/Low Eq**: それぞれのポットを獲得する確率です。
    """)

    st.divider()

    st.subheader("4. サイドバー機能 (便利ツール)")
    st.markdown("""
    - **🏷️ Filters (PLOのみ)** 「Aハイのみ」「ダブルスーテッドのみ」など、条件を絞ってランキングを表示します。
      
    - **🎨 Highlight Groups (PLOのみ)** 3つのグループ（🔴赤、🔵青、🟢緑）にそれぞれ違う条件を設定し、散布図上で色分け表示できます。
      
    - **🔍 Rank Search** 「1位のハンドは？」「1000位のハンドは？」など、順位からハンドを逆引きして分析ボタンでセットできます。
      
    - **⚙️ Scenario (SPR設定)** スタックの深さを設定します。
      - **Deep**: ナッツを作る能力（Nut Equity）を重視します。
      - **Short**: 単純な勝率（Raw Equity）を重視します。
    """)
    
    st.success("Analysis powered by custom simulation engine.")

# サイドバー下部にクレジット表示
with st.sidebar:
    st.markdown("---")
    st.markdown("© 2026 **Ryo** ([@Ryo_allin](https://x.com/Ryo_allin))")
