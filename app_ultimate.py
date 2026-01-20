import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
from heuristics import calculate_flo8_heuristic

# ページ設定
st.set_page_config(page_title="Omaha Ultimate Solver", layout="wide")

# ==========================================
# ユーティリティ
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
    
    has_A_suit = False
    if is_ds or is_ss or is_mono:
        for s, c in sc.items():
            if c>=2:
                if 12 in [card.rank for card in cards if card.suit==s]: has_A_suit=True
    if has_A_suit: tags.append("A-High Suit")

    if len(set(ranks))==4:
        ur = sorted(list(set(ranks)), reverse=True)
        gaps = [ur[i]-ur[i+1] for i in range(3)]
        if gaps==[1,1,1]: tags.append("Perfect Rundown")
        elif gaps==[2,1,1]: tags.append("Top Gap Rundown")
        elif gaps==[1,2,1]: tags.append("Mid Gap Rundown")
        elif gaps==[1,1,2]: tags.append("Bottom Gap Rundown")
        elif sum(gaps)==5: tags.append("Double Gap Rundown")
        if min(ranks)>=8: tags.append("Broadway")
    return tags

def set_plo_input(hand_str):
    st.session_state.plo_input = hand_str

# ==========================================
# データロード
# ==========================================
@st.cache_data
def load_plo_data_final(csv_path="plo_detailed_ranking.zip"):
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        
        df["nut_equity"] = (
            df["win_SF"] + 
            df["win_Quads"] + 
            df["win_FH"] + 
            df["win_Flush"] + 
            df["win_Straight"]
        )
        df["nut_quality"] = df["nut_equity"] / df["equity"]
        df["nut_quality"] = df["nut_quality"].fillna(0)
        
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
        
        df = df.sort_values("rank")
        return df
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
# UI
# ==========================================
st.title("🃏 Omaha Ultimate Solver")
st.caption("Strategic Analysis based on Win-Distribution & SPR")

if 'plo_input' not in st.session_state: st.session_state.plo_input = "As Ks Jd Th"
if 'flo8_input' not in st.session_state: st.session_state.flo8_input = "Ad Ah 2s 3d"

tab_plo, tab_flo8, tab_guide = st.tabs(["🔥 PLO (Detailed)", "⚖️ FLO8", "📖 Guide"])

with tab_plo:
    df_plo = load_plo_data_final()
    
    if df_plo is None:
        st.warning("Data loading failed. Please upload 'plo_detailed_ranking.zip'.")
    else:
        # Sidebar
        with st.sidebar:
            st.header("🏷️ Hand Filters")
            
            st.markdown("##### 🃏 Top Rank")
            ranks_opt = list("AKQJT98765432")
            sel_top = st.multiselect("Select Highest Rank", ranks_opt)
            st.divider()

            st.markdown("##### 🏷️ Tags")
            avail_tags = ["AA","KK","QQ","Double Pair","Double Suited","Single Suited","A-High Suit","Rainbow","Monotone","Broadway","Perfect Rundown","Top Gap Rundown","Mid Gap Rundown","Bottom Gap Rundown","Double Gap Rundown"]
            inc_tags = st.multiselect("✅ Include (AND)", avail_tags)
            exc_tags = st.multiselect("🚫 Exclude (NOT)", avail_tags)
            st.divider()
            
            st.markdown("##### 🎨 Highlight")
            high_tags = st.multiselect("Visual Highlight", avail_tags)
            st.divider()
            
            d_limit = st.slider("Display Limit", 5, 100, 20, 5)
            
            filtered_df = None
            if sel_top or inc_tags or exc_tags:
                tmp = df_plo
                if sel_top: tmp = tmp[tmp["top_rank"].isin(sel_top)]
                if inc_tags or exc_tags:
                    iset, eset = set(inc_tags), set(exc_tags)
                    tmp = tmp[tmp["tags"].apply(lambda t: iset.issubset(set(t)) and eset.isdisjoint(set(t)))]
                filtered_df = tmp

            st.write(f"Top {d_limit} Results:")
            if filtered_df is not None:
                if not filtered_df.empty:
                    th = filtered_df.head(d_limit)
                    hset = set(high_tags)
                    for _, r in th.iterrows():
                        lbl = f"{r['hand']} (#{r['rank']})"
                        if high_tags and hset.issubset(set(r['tags'])): lbl = f"🎨 {lbl}"
                        if st.button(lbl, key=f"s_{r['rank']}"):
                            st.session_state.plo_input = r['hand']; st.rerun()
                    st.caption(f"Found: {len(filtered_df):,}")
                else: st.write("No hands found.")
            elif not (sel_top or inc_tags or exc_tags): st.write("(No filters)")

        # Rank Search
        with st.expander("🔍 Rank Search"):
            c1, c2 = st.columns([1,3])
            with c1: srk = st.number_input("Rank", 1, len(df_plo), 1, key="prk")
            with c2:
                fr = df_plo[df_plo['rank']==srk]
                if not fr.empty:
                    r = fr.iloc[0]
                    st.markdown(f"**{r['hand']}** (Top {r['pct']:.2f}%) Tags: {' '.join(r['tags'])}")
                    if st.button("Analyze", key="bcp", on_click=set_plo_input, args=(r['hand'],)): pass

        st.divider()
        
        # Scenario
        with st.container():
            c_sc, _ = st.columns([1,2])
            with c_sc:
                spr = st.select_slider("Stack Depth / SPR", ["Short","Medium","Deep","Very Deep"], value="Medium")
                nw = 0.0 if "Short" in spr else 0.3 if "Medium" in spr else 0.6 if "Deep" in spr else 0.8

        st.divider()

        # Main Analysis
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.subheader("🔍 Hand Input")
            
            # --- カード選択 (Multiselect) ---
            # 全カードの選択肢リストを作成
            suits_disp = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
            all_cards_options = []
            for rank in "AKQJT98765432":
                for suit_code in "shdc":
                    # 表示ラベル: "A♠" / 値: "As"
                    label = f"{rank}{suits_disp[suit_code]}"
                    value = f"{rank}{suit_code}"
                    all_cards_options.append((label, value))
            
            # 選択肢ラベルのリスト
            options_labels = [opt[0] for opt in all_cards_options]
            
            # マルチセレクト表示
            selected_labels = st.multiselect(
                "🃏 Select Cards (Searchable)",
                options=options_labels,
                max_selections=4,
                placeholder="Choose 4 cards...",
                help="Type to search (e.g. 'As', 'K'). Select exactly 4 cards."
            )
            
            # 選択されたラベルを内部値(As, Kh...)に変換
            if len(selected_labels) == 4:
                selected_values = []
                for label in selected_labels:
                    # label ("A♠") から value ("As") を逆引き
                    val = next(opt[1] for opt in all_cards_options if opt[0] == label)
                    selected_values.append(val)
                
                # 自動的に入力欄を更新
                # 注意: multiselectの結果を優先して表示するため、テキスト入力を上書きするロジック
                current_visual_input = " ".join(selected_values)
                # 入力欄のデフォルト値を更新するためにsession_state操作はしない(ループするため)
                # 代わりに分析用の変数 `inp` をここで決定する
                inp = selected_values
                st.session_state.plo_input = current_visual_input # 同期
            else:
                # マルチセレクトが4枚未満なら、テキストボックスの入力を採用
                inp_raw = st.text_input("Enter Hand (Text)", key='plo_input')
                inp = normalize_input_text(inp_raw)

            # 現在のハンドを表示
            if inp:
                st.markdown(render_hand_html(" ".join(inp)), unsafe_allow_html=True)
            
            if len(inp)==4:
                res = df_plo[df_plo["card_set"]==frozenset(inp)]
                if not res.empty:
                    row = res.iloc[0]
                    eq = row["equity"] * 100
                    nq = row["nut_quality"]
                    ne = row["nut_equity"] * 100
                    sc = (eq*(1-nw)) + ((nq*100)*nw)
                    
                    m1,m2,m3 = st.columns(3)
                    m1.metric("Power Score", f"{sc:.1f}")
                    m2.metric("Raw Equity", f"{eq:.1f}%")
                    m3.metric("Nut Equity", f"{ne:.1f}%")
                    
                    st.write("🏷️ " + " ".join([f"`{t}`" for t in row['tags']]))
                    st.caption(f"Global Rank: {int(row['rank']):,} (Top {row['pct']:.1f}%)")
                else: st.warning("Hand not found.")
            elif len(selected_labels) > 0 and len(selected_labels) < 4:
                st.info(f"Select {4 - len(selected_labels)} more cards.")

        with c2:
            if 'row' in locals():
                st.subheader("📊 Win Distribution")
                w_sf = row["win_SF"]
                w_qd = row["win_Quads"]
                w_fh = row["win_FH"]
                w_fl = row["win_Flush"]
                w_st = row["win_Straight"]
                
                nut_sum = w_sf + w_qd + w_fh + w_fl + w_st
                w_wk = max(0, row["equity"] - nut_sum)
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
            
            # --- Equity Curve & Seek Bar ---
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
                    st.caption(f"Eq: {s_row['equity']*100:.1f}% | {' '.join(s_row['tags'])}")
                with sk2:
                    st.button("Analyze", key="b_seek", on_click=set_plo_input, args=(s_row['hand'],))
                
                scurve = df_plo.iloc[::200, :]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.plot(scurve["pct"], scurve["equity"], c="#cccccc", label="All")
                ax3.scatter(row["pct"], row["equity"], c="red", s=150, marker='*', zorder=10, label="You")
                ax3.scatter(s_row["pct"], s_row["equity"], c="blue", s=80, zorder=9, label="Seek")
                ax3.axvline(x=seek_pct, color="blue", ls=":", alpha=0.5)
                
                ax3.set_xlabel("Top X% of Hands"); ax3.set_ylabel("Equity")
                
                zoom_chk = st.checkbox("Zoom around Seek", False)
                if zoom_chk: ax3.set_xlim(max(0, seek_pct-10), min(100, seek_pct+10))
                else: ax3.set_xlim(0, 100)
                
                ax3.legend()
                ax3.grid(True, ls='--', alpha=0.3)
                st.pyplot(fig3)

            # --- Scatter Plot ---
            with cc2:
                st.subheader("🌌 Equity Scatter")

                cmode = st.radio("Scatter", ["Mode A", "Mode B"], horizontal=True, label_visibility="collapsed")
                st.caption("Mode A: Eq vs Quality / Mode B: Eq vs Nut Eq")
                azoom = st.checkbox("🔍 Auto Zoom", True)

                @st.cache_data
                def get_bg(df): return df.sample(3000, random_state=42).copy()
                bg = get_bg(df_plo)

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                def gxy(d, m): return d["equity"], (d["nut_quality"] if "Mode A" in m else d["nut_equity"])
                
                bx, by = gxy(bg, cmode)
                mx, my = gxy(pd.DataFrame([row]), cmode); mx, my = mx.iloc[0], my.iloc[0]

                cbg = bg["nut_quality"] if "Mode A" in cmode else (1.0-(bx-by))
                ax2.scatter(bx, by, c=cbg, cmap="coolwarm_r", s=10, alpha=0.1)
                if "Mode B" in cmode: ax2.plot([0,1],[0,1], ls="--", c="gray", alpha=0.5)

                xmin, xmax, ymin, ymax = mx, mx, my, my
                focused = False

                if filtered_df is not None and not filtered_df.empty:
                    fdf = filtered_df.sample(2000, random_state=42) if len(filtered_df)>2000 else filtered_df
                    fx, fy = gxy(fdf, cmode)
                    ax2.scatter(fx, fy, fc='none', ec='gold', s=30)
                    xmin, xmax = min(xmin, fx.min()), max(xmax, fx.max())
                    ymin, ymax = min(ymin, fy.min()), max(ymax, fy.max())
                    focused = True
                
                if high_tags:
                    ht = set(high_tags)
                    src = filtered_df if filtered_df is not None else df_plo
                    mask = src["tags"].apply(lambda t: ht.issubset(set(t)))
                    hdf_all = src[mask]
                    
                    if not hdf_all.empty:
                        hdf = hdf_all.sample(2000, random_state=42) if len(hdf_all)>2000 else hdf_all
                        hx, hy = gxy(hdf, cmode)
                        ax2.scatter(hx, hy, fc='none', ec='#FF00FF', s=60, lw=2)
                        xmin, xmax = min(xmin, hx.min()), max(xmax, hx.max())
                        ymin, ymax = min(ymin, hy.min()), max(ymax, hy.max())
                        focused = True

                ax2.scatter(mx, my, c='black', s=150, marker='*', ec='white', zorder=10)

                if azoom:
                    if not focused: xmin, xmax, ymin, ymax = bx.min(), bx.max(), by.min(), by.max()
                    
                    if xmax==xmin: xmin-=0.1; xmax+=0.1
                    if ymax==ymin: ymin-=0.1; ymax+=0.1
                    x_span, y_span = xmax-xmin, ymax-ymin
                    if x_span < 0.2: d=(0.2-x_span)/2; xmin-=d; xmax+=d
                    if y_span < 0.2: d=(0.2-y_span)/2; ymin-=d; ymax+=d
                    
                    margin=0.05
                    ax2.set_xlim(max(0, xmin-margin), min(1, xmax+margin))
                    ax2.set_ylim(max(0, ymin-margin), min(1, ymax+margin))
                else: ax2.set_xlim(0,1.05); ax2.set_ylim(0,1.05)
                
                ax2.grid(True, ls='--', alpha=0.3)
                st.pyplot(fig2)

with tab_flo8:
    df8 = load_flo8_data()
    if df8 is not None:
        c1, c2 = st.columns([1,2])
        with c1:
            i8 = normalize_input_text(st.text_input("FLO8 Hand", key='flo8_input'))
            if len(i8)==4:
                sc, dt = calculate_flo8_heuristic(" ".join(i8))
                st.metric("Points", sc)
                st.bar_chart(dt)
        with c2:
            if len(i8)==4:
                r8 = df8[df8["card_set"]==frozenset(i8)]
                if not r8.empty:
                    rr = r8.iloc[0]
                    st.metric("Scoop", f"{rr['scoop_pct']:.1f}%")
