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

def get_hand_tags(hand_str):
    try: cards = [SimpleCard(s) for s in hand_str.split()]
    except: return []
    tags = []
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # Pairs
    rc = {r: ranks.count(r) for r in ranks}
    pairs = [r for r, c in rc.items() if c == 2]
    if 12 in pairs: tags.append("AA")
    if 11 in pairs: tags.append("KK")
    if 10 in pairs: tags.append("QQ")
    if len(pairs)==2: tags.append("Double Pair")
    elif len(pairs)==1: tags.append("Single Pair")
    elif len(set(ranks))==4: tags.append("No Pair")

    # Suits
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

    # Rundowns
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

# ==========================================
# データロード
# ==========================================
@st.cache_data
def load_plo_data_v6(csv_path="plo_detailed_ranking.zip"): # バージョン更新
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        
        # Absolute Equity to Nut Equity
        df["nut_equity"] = (
            df["win_SF"] + 
            df["win_Quads"] + 
            df["win_FH"] + 
            df["win_Flush"] + 
            df["win_Straight"]
        )
        # 0除算回避
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
    df_plo = load_plo_data_v6()
    
    if df_plo is None:
        st.warning("Data loading failed. Please upload 'plo_detailed_ranking.zip'.")
    else:
        # Sidebar
        with st.sidebar:
            st.header("🏷️ Hand Filters")
            
            # Top Rank
            st.markdown("##### 🃏 Top Rank")
            ranks_opt = list("AKQJT98765432")
            sel_top = st.multiselect("Select Highest Rank", ranks_opt)
            st.divider()

            # Tags
            st.markdown("##### 🏷️ Tags")
            avail_tags = ["AA","KK","QQ","Double Pair","Double Suited","Single Suited","A-High Suit","Rainbow","Monotone","Broadway","Perfect Rundown","Top Gap Rundown","Mid Gap Rundown","Bottom Gap Rundown","Double Gap Rundown"]
            inc_tags = st.multiselect("✅ Include (AND)", avail_tags)
            exc_tags = st.multiselect("🚫 Exclude (NOT)", avail_tags)
            st.divider()
            
            # Highlight
            st.markdown("##### 🎨 Highlight")
            high_tags = st.multiselect("Visual Highlight", avail_tags)
            st.divider()
            
            d_limit = st.slider("Display Limit", 5, 100, 20, 5)
            
            # Filter Logic
            filtered_df = None
            if sel_top or inc_tags or exc_tags:
                tmp = df_plo
                if sel_top: tmp = tmp[tmp["top_rank"].isin(sel_top)]
                if inc_tags or exc_tags:
                    iset, eset = set(inc_tags), set(exc_tags)
                    tmp = tmp[tmp["tags"].apply(lambda t: iset.issubset(set(t)) and eset.isdisjoint(set(t)))]
                filtered_df = tmp

            # List
            st.write(f"Top {d_limit} Results:")
            if filtered_df is not None:
                if not filtered_df.empty:
                    # リスト表示は「トップランキング」が見たいのでheadのままでOK
                    th = filtered_df.head(d_limit)
                    hset = set(high_tags)
                    for _, r in th.iterrows():
                        lbl = f"{r['hand']} (#{r['rank']})"
                        if high_tags and hset.issubset(set(r['tags'])): lbl = f"🎨 {lbl}"
                        if st.button(lbl, key=f"s_{r['rank']}"):
                            st.session_state.plo_input = r['hand']; st.rerun()
                    st.caption(f"Found: {len(filtered_df):,} hands")
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
                    if st.button("Analyze", key="bcp"): st.session_state.plo_input=r['hand']; st.rerun()

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
            inp_raw = st.text_input("Enter Hand", key='plo_input')
            inp = normalize_input_text(inp_raw)
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
            with cc1:
                st.subheader("📈 Equity Curve")
                z20 = st.checkbox("🔍 Zoom Top 20%", False)
                scurve = df_plo.iloc[::200, :]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.plot(scurve["pct"], scurve["equity"], c="#cccccc")
                ax3.scatter(row["pct"], row["equity"], c="red", s=100, zorder=5)
                ax3.set_xlabel("Top X% of Hands"); ax3.set_ylabel("Equity")
                ax3.set_xlim(0, 20 if z20 else 100)
                ax3.grid(True, ls='--', alpha=0.3)
                st.pyplot(fig3)

            with cc2:
                # ==========================
                # Scatter Plot (修正箇所)
                # ==========================
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

                # 【修正1】フィルタ済みデータ (Gold) の間引き方を変更
                if filtered_df is not None and not filtered_df.empty:
                    # head(2000)ではなく、数が多ければランダムサンプリングする
                    if len(filtered_df) > 2000:
                        fdf = filtered_df.sample(n=2000, random_state=42)
                    else:
                        fdf = filtered_df
                        
                    fx, fy = gxy(fdf, cmode)
                    ax2.scatter(fx, fy, fc='none', ec='gold', s=30)
                    xmin, xmax = min(xmin, fx.min()), max(xmax, fx.max())
                    ymin, ymax = min(ymin, fy.min()), max(ymax, fy.max())
                    focused = True
                
                # 【修正2】ハイライトデータ (Magenta) の間引き方を変更
                if high_tags:
                    ht = set(high_tags)
                    # 検索対象: フィルタがあればそこから、なければ全体から
                    src = filtered_df if filtered_df is not None else df_plo
                    
                    # 全体からタグ検索 (applyは少し重いが正確さを優先)
                    # 以前の head(10000) 制限を撤廃し、全体から探す
                    mask = src["tags"].apply(lambda t: ht.issubset(set(t)))
                    hdf_all = src[mask]
                    
                    if not hdf_all.empty:
                        # ここでも数が多ければサンプリング
                        if len(hdf_all) > 2000:
                            hdf = hdf_all.sample(n=2000, random_state=42)
                        else:
                            hdf = hdf_all

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