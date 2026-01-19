import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
# import eval7 <-- 不要
from heuristics import calculate_flo8_heuristic

# ページ設定
st.set_page_config(page_title="Omaha Ultimate Solver", layout="wide")

# ==========================================
# ユーティリティ (SimpleCard & Tags)
# ==========================================
class SimpleCard:
    def __init__(self, card_str):
        if not card_str:
            self.rank = -1
            self.suit = ''
            return
        rank_char = card_str[:-1].upper()
        suit_char = card_str[-1].lower()
        ranks = "23456789TJQKA"
        if rank_char in ranks:
            self.rank = ranks.index(rank_char)
        else:
            self.rank = -1
        self.suit = suit_char

def normalize_input_text(text):
    if not text: return []
    text = unicodedata.normalize('NFKC', text)
    parts = text.split()
    cleaned_parts = []
    for p in parts:
        if len(p) >= 2:
            rank = p[:-1].upper()
            suit = p[-1].lower()
            cleaned_parts.append(rank + suit)
    return cleaned_parts

def get_hand_tags(hand_str):
    try:
        cards = [SimpleCard(s) for s in hand_str.split()]
    except:
        return []
    
    tags = []
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # Pairs
    rank_counts = {r: ranks.count(r) for r in ranks}
    pairs = [r for r, c in rank_counts.items() if c == 2]
    
    if 12 in pairs: tags.append("AA")
    if 11 in pairs: tags.append("KK")
    if 10 in pairs: tags.append("QQ")
    
    if len(pairs) == 2: tags.append("Double Pair")
    elif len(pairs) == 1: tags.append("Single Pair")
    elif len(set(ranks)) == 4: tags.append("No Pair")

    # Suits
    suit_counts = {s: suits.count(s) for s in suits}
    s_values = sorted(suit_counts.values(), reverse=True)
    s_dist = s_values + [0] * (4 - len(s_values))
    
    is_ds = (s_dist[0] == 2 and s_dist[1] == 2)
    is_monotone = (s_dist[0] == 4)
    is_ss = (s_dist[0] >= 2 and not is_ds and not is_monotone)
    is_rainbow = (s_dist[0] == 1)
    
    if is_ds: tags.append("Double Suited")
    if is_rainbow: tags.append("Rainbow")
    if is_monotone: tags.append("Monotone")
    if is_ss: tags.append("Single Suited")
    
    has_A_suit = False
    if is_ds or is_ss or is_monotone:
        for s, count in suit_counts.items():
            if count >= 2:
                suited_ranks = [c.rank for c in cards if c.suit == s]
                if 12 in suited_ranks: has_A_suit = True
    if has_A_suit: tags.append("A-High Suit")

    # Rundowns
    if len(set(ranks)) == 4:
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        gaps = [unique_ranks[i] - unique_ranks[i+1] for i in range(3)]
        
        if gaps == [1, 1, 1]: tags.append("Perfect Rundown")
        elif gaps == [2, 1, 1]: tags.append("Top Gap Rundown")
        elif gaps == [1, 2, 1]: tags.append("Mid Gap Rundown")
        elif gaps == [1, 1, 2]: tags.append("Bottom Gap Rundown")
        elif sum(gaps) == 5: tags.append("Double Gap Rundown")
        
        if min(ranks) >= 8: tags.append("Broadway")
            
    return tags

# ==========================================
# データロード (ロジック強化版)
# ==========================================
@st.cache_data
def load_plo_data_v4(csv_path="plo_detailed_ranking.zip"):
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        df["rank"] = df["equity"].rank(ascending=False, method='first').astype(int)
        df["pct"] = (df["rank"] / len(df)) * 100
        df["nut_quality"] = df["win_SF"] + df["win_Quads"] + df["win_FH"] + df["win_Flush"] + df["win_Straight"]
        df["nut_equity"] = df["equity"] * df["nut_quality"]
        df["tags"] = df["hand"].apply(get_hand_tags)
        
        # 【修正】並び順に依存せず、ハンド内の「最大ランク」を正確に抽出する
        def get_max_rank(hand_str):
            try:
                # 文字列 "As Kd..." を SimpleCard オブジェクトに変換
                cards = [SimpleCard(s) for s in hand_str.split()]
                # ランクの数値(0-12)が最大のものを探す
                max_card = max(cards, key=lambda c: c.rank)
                # そのカードのランク文字(A, K...)を復元する
                ranks_char = "23456789TJQKA"
                return ranks_char[max_card.rank]
            except:
                return "?"

        df["top_rank"] = df["hand"].apply(get_max_rank)
        
        df = df.sort_values("rank")
        return df
    except FileNotFoundError:
        return None

# ==========================================
# UI ロジック
# ==========================================
st.title("🃏 Omaha Ultimate Solver")
st.caption("Strategic Analysis based on Win-Distribution & SPR")

if 'plo_input' not in st.session_state:
    st.session_state.plo_input = "As Ks Jd Th"
if 'flo8_input' not in st.session_state:
    st.session_state.flo8_input = "Ad Ah 2s 3d"

tab_plo, tab_flo8, tab_guide = st.tabs(["🔥 PLO (Detailed)", "⚖️ FLO8 (Hi/Lo)", "📖 Guide"])

with tab_plo:
    # 新しいローダーを使用
    df_plo = load_plo_data_v4()
    
    if df_plo is None:
        st.warning("Data loading failed. Please upload 'plo_detailed_ranking.zip'.")
    else:
        # --- サイドバー ---
        with st.sidebar:
            st.header("🏷️ Hand Filters")
            
            # 1. Top Rank Filter (New!)
            st.markdown("##### 🃏 Top Rank")
            st.caption("ハンド内で最も強いランクを指定")
            rank_options = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
            selected_top_ranks = st.multiselect("Select Highest Rank", rank_options)
            
            st.divider()

            # 2. Tag Filter
            st.markdown("##### 🏷️ Tags")
            available_tags = [
                "AA", "KK", "QQ", "Double Pair", 
                "Double Suited", "Single Suited", "A-High Suit", "Rainbow", "Monotone",
                "Broadway", 
                "Perfect Rundown", "Top Gap Rundown", "Mid Gap Rundown", "Bottom Gap Rundown", "Double Gap Rundown"
            ]
            included_tags = st.multiselect("✅ Include (AND)", available_tags)
            excluded_tags = st.multiselect("🚫 Exclude (NOT)", available_tags)
            
            st.divider()
            
            # 3. Highlight
            st.markdown("##### 🎨 Highlight")
            highlight_tags = st.multiselect("Visual Highlight", available_tags)
            
            st.divider()
            display_limit = st.slider("Display Limit", 5, 100, 20, 5)
            
            # --- フィルタリングロジック ---
            filtered_df_all = None
            
            # 条件が一つでも設定されていたらフィルタリング開始
            if selected_top_ranks or included_tags or excluded_tags:
                
                # ベースは全データ
                temp_df = df_plo
                
                # 1. Top Rank Filter
                if selected_top_ranks:
                    temp_df = temp_df[temp_df["top_rank"].isin(selected_top_ranks)]
                
                # 2. Tag Filter
                if included_tags or excluded_tags:
                    inc_set = set(included_tags)
                    exc_set = set(excluded_tags)
                    def check_filter(t):
                        ht = set(t)
                        if included_tags and not inc_set.issubset(ht): return False
                        if excluded_tags and not exc_set.isdisjoint(ht): return False
                        return True
                    
                    temp_df = temp_df[temp_df["tags"].apply(check_filter)]
                
                filtered_df_all = temp_df

            # 結果表示
            st.write(f"Top {display_limit} Results:")
            
            if filtered_df_all is not None:
                if not filtered_df_all.empty:
                    top_hands = filtered_df_all.head(display_limit)
                    hl_set = set(highlight_tags)
                    
                    for _, r in top_hands.iterrows():
                        label = f"{r['hand']} (#{r['rank']})"
                        if highlight_tags and hl_set.issubset(set(r['tags'])):
                            label = f"🎨 {label}" 
                        
                        if st.button(label, key=f"side_{r['rank']}"):
                            st.session_state.plo_input = r['hand']
                            st.rerun()
                    
                    st.caption(f"Total matching: {len(filtered_df_all):,} hands")
                else:
                    st.write("No hands found.")
            elif not (selected_top_ranks or included_tags or excluded_tags):
                 st.write("(No filters applied)")

        # 1. 逆引き検索
        with st.expander("🔍 Rank Search"):
            c1, c2 = st.columns([1, 3])
            with c1:
                search_rank = st.number_input("Rank", 1, len(df_plo), 1, key="plo_rk")
            with c2:
                found_rows = df_plo[df_plo['rank'] == search_rank]
                if not found_rows.empty:
                    fr = found_rows.iloc[0]
                    t_str = " ".join([f"`{t}`" for t in fr['tags']])
                    st.markdown(f"**{fr['hand']}** (Top {fr['pct']:.2f}%)  \nTags: {t_str}")
                    if st.button("Analyze", key="btn_copy_plo"):
                        st.session_state.plo_input = fr['hand']
                        st.rerun()
                else: st.error("Not found.")

        st.divider()

        # 2. 設定エリア
        with st.container():
            c_set, _ = st.columns([1, 2])
            with c_set:
                stack_depth = st.select_slider(
                    "Stack Depth / SPR",
                    options=["Short (<20BB)", "Medium (50BB)", "Deep (100BB+)", "Very Deep (200BB+)"],
                    value="Medium (50BB)"
                )
                if "Short" in stack_depth: nut_weight = 0.0
                elif "Medium" in stack_depth: nut_weight = 0.3
                elif "Deep" in stack_depth: nut_weight = 0.6
                else: nut_weight = 0.8

        st.divider()

        # --- メイン分析エリア ---
        col1, col2 = st.columns([1, 1.3])
        
        with col1:
            st.subheader("🔍 Hand Input")
            raw = st.text_input("Enter Hand", key='plo_input')
            inp = normalize_input_text(raw)
            
            if len(inp) == 4:
                res = df_plo[df_plo["card_set"] == frozenset(inp)]
                if not res.empty:
                    row = res.iloc[0]
                    eq_val = row["equity"] * 100
                    nut_equity = row["nut_equity"] * 100
                    nut_quality = row["nut_quality"]
                    adj_score = (eq_val * (1 - nut_weight)) + ((nut_quality * 100) * nut_weight)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Power Score", f"{adj_score:.1f}", help="SPR Adjusted")
                    m2.metric("Raw Equity", f"{eq_val:.1f}%")
                    m3.metric("Nut Equity", f"{nut_equity:.1f}%")
                    
                    st.write("🏷️ Tags:")
                    st.write(" ".join([f"`{t}`" for t in row['tags']]))
                    st.caption(f"Global Rank: {int(row['rank']):,} (Top {row['pct']:.1f}%)")
                else:
                    st.warning("Hand not found.")
            
        with col2:
            if 'row' in locals():
                st.subheader("📊 Win Distribution")
                val_weak = max(0, row["equity"] - row["nut_equity"])
                val_lose = 1.0 - row["equity"]
                
                # Pie Chart
                sizes = [
                    row["win_Straight"]*row["equity"], 
                    row["win_Flush"]*row["equity"], 
                    (row["win_SF"]+row["win_Quads"]+row["win_FH"])*row["equity"], 
                    val_weak, 
                    val_lose
                ]
                labels = ['Straight+', 'Flush', 'FullHouse+', 'Pair (Fragile)', 'Lose']
                colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FFC107', '#EEEEEE']
                
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                wedges, _, _ = ax1.pie([s for s in sizes if s>0.001], autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                st.pyplot(fig1)

        if 'row' in locals():
            st.divider()
            c_chart1, c_chart2 = st.columns(2)
            
            # Equity Curve
            with c_chart1:
                st.subheader("📈 Equity Curve")
                zoom_curve = st.checkbox("🔍 Zoom Top 20%", value=False)
                
                sample_curve = df_plo.iloc[::200, :]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.plot(sample_curve["pct"], sample_curve["equity"], color="#cccccc")
                ax3.scatter(row["pct"], row["equity"], color="red", s=100, zorder=5)
                ax3.set_xlabel("Top X% of Hands"); ax3.set_ylabel("Equity")
                ax3.set_xlim(0, 20 if zoom_curve else 100)
                ax3.grid(True, ls='--', alpha=0.3)
                st.pyplot(fig3)

            # Scatter Plot
            with c_chart2:
                c_mode, c_zoom = st.columns([2, 1])
                with c_mode:
                    chart_mode = st.radio("Scatter", ["Mode A", "Mode B"], horizontal=True, label_visibility="collapsed")
                    st.caption("Mode A: Eq vs Quality / Mode B: Eq vs Nut Eq" )
                with c_zoom:
                    use_auto_zoom = st.checkbox("🔍 Auto Zoom", value=True)

                # Background
                @st.cache_data
                def get_bg_sample(df, n=3000): return df.sample(n=n, random_state=42).copy()
                bg_df = get_bg_sample(df_plo)

                # Prepare Plot Data
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                def get_xy(df, m):
                    x = df["equity"]
                    y = df["nut_quality"] if "Mode A" in m else df["nut_equity"]
                    return x, y

                bg_x, bg_y = get_xy(bg_df, chart_mode)
                my_x, my_y = get_xy(pd.DataFrame([row]), chart_mode)
                my_x, my_y = my_x.iloc[0], my_y.iloc[0]

                # Plot Background
                c_bg = bg_df["nut_quality"] if "Mode A" in chart_mode else (1.0 - (bg_x - bg_y))
                ax2.scatter(bg_x, bg_y, c=c_bg, cmap="coolwarm_r", s=10, alpha=0.1)
                if "Mode B" in chart_mode: ax2.plot([0,1],[0,1], ls="--", c="gray", alpha=0.5)

                # Zoom Range Init
                x_min, x_max, y_min, y_max = my_x, my_x, my_y, my_y
                has_focus = False

                # Filtered Group
                if filtered_df_all is not None and not filtered_df_all.empty:
                    f_df = filtered_df_all.head(2000)
                    f_x, f_y = get_xy(f_df, chart_mode)
                    ax2.scatter(f_x, f_y, facecolors='none', edgecolors='gold', s=30, lw=1)
                    x_min, x_max = min(x_min, f_x.min()), max(x_max, f_x.max())
                    y_min, y_max = min(y_min, f_y.min()), max(y_max, f_y.max())
                    has_focus = True
                
                # Highlight Group
                if highlight_tags:
                    hl_set = set(highlight_tags)
                    target = filtered_df_all if filtered_df_all is not None else df_plo
                    # Optimization: search from head(10000) if full scan is heavy
                    search_source = target if len(target) < 10000 else target.head(10000)
                    h_df = search_source[search_source["tags"].apply(lambda t: hl_set.issubset(set(t)))].head(2000)
                    
                    if not h_df.empty:
                        h_x, h_y = get_xy(h_df, chart_mode)
                        ax2.scatter(h_x, h_y, facecolors='none', edgecolors='#FF00FF', s=60, lw=2)
                        x_min, x_max = min(x_min, h_x.min()), max(x_max, h_x.max())
                        y_min, y_max = min(y_min, h_y.min()), max(y_max, h_y.max())
                        has_focus = True

                # Plot You
                ax2.scatter(my_x, my_y, c='black', s=150, marker='*', edgecolors='white', zorder=10)

                # Apply Zoom
                if use_auto_zoom:
                    if not has_focus: # If no filter/highlight, show background range
                        x_min, x_max = bg_x.min(), bg_x.max()
                        y_min, y_max = bg_y.min(), bg_y.max()
                    
                    # Prevent zero range
                    if x_max == x_min: x_min-=0.1; x_max+=0.1
                    if y_max == y_min: y_min-=0.1; y_max+=0.1
                    
                    margin = 0.05
                    ax2.set_xlim(max(0, x_min-margin), min(1, x_max+margin))
                    ax2.set_ylim(max(0, y_min-margin), min(1, y_max+margin))
                else:
                    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)
                
                ax2.set_xlabel("Raw Equity")
                ax2.set_ylabel("Nut Quality" if "Mode A" in chart_mode else "Nut Equity")
                ax2.grid(True, ls='--', alpha=0.3)
                st.pyplot(fig2)

with tab_flo8:
    # (省略: 前回のコードと同じ)
    df_flo8 = load_flo8_data()
    # ... (前回のFLO8ロジック) ...
    if df_flo8 is not None:
         # 簡易表示用
         st.write("FLO8 Module Loaded")
         raw8 = st.text_input("Enter FLO8 Hand", key='flo8_input')
         st.metric("Hutchinson Points", calculate_flo8_heuristic(normalize_input_text(raw8))[0])

with tab_guide:
    st.markdown("### Guide\nCheck the new **Top Rank Filter** in the sidebar!")