import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
# import eval7  <-- 削除 (これがエラーの原因)
from heuristics import calculate_flo8_heuristic

# ページ設定
st.set_page_config(page_title="Omaha Ultimate Solver", layout="wide")

# ==========================================
# ユーティリティ & タグ判定ロジック (eval7非依存版)
# ==========================================

# eval7の代わりに使う軽量クラス
class SimpleCard:
    def __init__(self, card_str):
        # "As", "Td", "9h" などを解析
        if not card_str:
            self.rank = -1
            self.suit = ''
            return
            
        rank_char = card_str[:-1].upper()
        suit_char = card_str[-1].lower()
        
        # ランクを数値に変換 (2=0, ... A=12) eval7準拠
        ranks = "23456789TJQKA"
        if rank_char in ranks:
            self.rank = ranks.index(rank_char)
        else:
            self.rank = -1 # Error
            
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
    """ハンド文字列からタグのリストを生成する (SimpleCard使用)"""
    try:
        # eval7.Card ではなく SimpleCard を使用
        cards = [SimpleCard(s) for s in hand_str.split()]
    except:
        return []
    
    tags = []
    # ランク降順ソート
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # 1. Pair Tags
    rank_counts = {r: ranks.count(r) for r in ranks}
    pairs = [r for r, c in rank_counts.items() if c == 2]
    
    if 12 in pairs: tags.append("AA")
    if 11 in pairs: tags.append("KK")
    if 10 in pairs: tags.append("QQ")
    
    if len(pairs) == 2:
        tags.append("Double Pair")
    elif len(pairs) == 1:
        tags.append("Single Pair")
    elif len(set(ranks)) == 4:
        tags.append("No Pair")

    # 2. Suitedness Tags
    suit_counts = {s: suits.count(s) for s in suits}
    s_values = sorted(suit_counts.values(), reverse=True)
    # 不足分を0埋めして4要素にする
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
                # そのスートを持つカードのランクを探す
                suited_ranks = [c.rank for c in cards if c.suit == s]
                if 12 in suited_ranks: # 12 = A
                    has_A_suit = True
    
    if has_A_suit:
        tags.append("A-High Suit")

    # 3. Structure Tags (Rundowns)
    if len(set(ranks)) == 4:
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        gaps = [unique_ranks[i] - unique_ranks[i+1] for i in range(3)]
        
        if gaps == [1, 1, 1]: tags.append("Perfect Rundown")
        elif gaps == [2, 1, 1]: tags.append("Top Gap Rundown")
        elif gaps == [1, 2, 1]: tags.append("Mid Gap Rundown")
        elif gaps == [1, 1, 2]: tags.append("Bottom Gap Rundown")
        elif sum(gaps) == 5: tags.append("Double Gap Rundown")
        
        if min(ranks) >= 8: # 8 = Ten (2=0... T=8)
            tags.append("Broadway")
            
    return tags

# ==========================================
# データロード
# ==========================================
@st.cache_data
def load_plo_data_v3(csv_path="plo_detailed_ranking.zip"): # zip対応
    try:
        # ZIPのままでもpandasは読めることが多いですが
        # エラーが出る場合は compression='zip' を明示
        df = pd.read_csv(csv_path)
        
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        df["rank"] = df["equity"].rank(ascending=False, method='first').astype(int)
        df["pct"] = (df["rank"] / len(df)) * 100
        df["nut_quality"] = df["win_SF"] + df["win_Quads"] + df["win_FH"] + df["win_Flush"] + df["win_Straight"]
        df["nut_equity"] = df["equity"] * df["nut_quality"]
        df["tags"] = df["hand"].apply(get_hand_tags)
        df = df.sort_values("rank")
        return df
    except FileNotFoundError:
        st.error(f"Error: {csv_path} not found. Please upload the data file.")
        return None

@st.cache_data
def load_flo8_data(csv_path="flo8_ranking.csv"): # こちらも必要ならzipに
    try:
        df = pd.read_csv(csv_path)
        df["card_set"] = df["hand"].apply(lambda x: frozenset(x.split()))
        df["rank"] = df["equity"].rank(ascending=False, method='first').astype(int)
        df["pct_total"] = (df["rank"] / len(df)) * 100
        df["rank_high"] = df["high_equity"].rank(ascending=False, method='first')
        df["pct_high"] = (df["rank_high"] / len(df)) * 100
        df["rank_low"] = df["low_equity"].rank(ascending=False, method='first')
        df["pct_low"] = (df["rank_low"] / len(df)) * 100
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
    df_plo = load_plo_data_v3()
    
    if df_plo is None:
        st.warning("Data loading failed.")
    else:
        # --- サイドバー ---
        with st.sidebar:
            st.header("🏷️ Tag Filter")
            
            available_tags = [
                "AA", "KK", "QQ", "Double Pair", 
                "Double Suited", "Single Suited", "A-High Suit", "Rainbow", "Monotone",
                "Broadway", 
                "Perfect Rundown", "Top Gap Rundown", "Mid Gap Rundown", "Bottom Gap Rundown", "Double Gap Rundown"
            ]
            
            included_tags = st.multiselect("✅ Include Tags (AND)", available_tags)
            excluded_tags = st.multiselect("🚫 Exclude Tags (NOT)", available_tags)
            
            st.divider()
            
            st.markdown("##### 🎨 Highlight Settings")
            highlight_tags = st.multiselect("Select Tags to Highlight", available_tags)
            
            st.divider()
            display_limit = st.slider("Display Limit", 5, 100, 20, 5)
            
            st.write(f"Top {display_limit} Results:")
            
            filtered_df_all = None
            if included_tags or excluded_tags:
                inc_set = set(included_tags)
                exc_set = set(excluded_tags)
                
                def check_filter(t):
                    ht = set(t)
                    if included_tags and not inc_set.issubset(ht): return False
                    if excluded_tags and not exc_set.isdisjoint(ht): return False
                    return True
                
                filtered_df_all = df_plo[df_plo["tags"].apply(check_filter)]
            
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
                    st.write("No hands found matching conditions.")
            elif included_tags or excluded_tags:
                 st.write("No hands found.")
            else:
                st.write("(Select tags to filter)")

        # 1. 逆引き検索
        with st.expander("🔍 Rank Search (順位からハンドを検索)"):
            c_srch1, c_srch2 = st.columns([1, 3])
            with c_srch1:
                max_rank = len(df_plo)
                search_rank = st.number_input("Rank", 1, max_rank, 1, step=1, key="plo_rank_search")
            with c_srch2:
                found_rows = df_plo[df_plo['rank'] == search_rank]
                if not found_rows.empty:
                    found_row = found_rows.iloc[0]
                    found_hand = found_row['hand']
                    tags_str = " ".join([f"`{t}`" for t in found_row['tags']])
                    st.markdown(f"**{found_hand}** (Top {found_row['pct']:.2f}%)  \nTags: {tags_str}")
                    if st.button("Copy to Analyzer", key="btn_copy_plo"):
                        st.session_state.plo_input = found_hand
                        st.rerun()
                else:
                    st.error(f"Rank {search_rank} not found.")

        st.divider()

        # 2. 設定エリア (SPR Slider)
        with st.container():
            col_set1, col_set2 = st.columns([1, 2])
            with col_set1:
                st.markdown("#### ⚙️ Scenario Setting")
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
                    nut_quality = row["nut_quality"]
                    nut_equity = row["nut_equity"] * 100
                    adjusted_score = (eq_val * (1 - nut_weight)) + ((nut_quality * 100) * nut_weight)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Power Score", f"{adjusted_score:.1f}", help="SPR調整後の総合評価")
                    m2.metric("Raw Equity", f"{eq_val:.1f}%")
                    m3.metric("Nut Equity", f"{nut_equity:.1f}%")
                    
                    st.write("🏷️ Tags:")
                    st.write(" ".join([f"`{t}`" for t in row['tags']]))

                    st.caption(f"**Global Rank:** {int(row['rank']):,} (Top {row['pct']:.1f}%)")
                    
                    with st.expander("See Formula"):
                        st.latex(rf"Score = (Equity \times {1-nut_weight:.1f}) + (NutQuality \times 100 \times {nut_weight:.1f})")

                    fragile_win_rate = eq_val - nut_equity
                    if "Deep" in stack_depth and fragile_win_rate > 20:
                        st.warning(f"⚠️ **Danger**: 勝率のうち {fragile_win_rate:.1f}% が「脆い勝ち」です。")
                    if nut_equity > 45:
                        st.success("💎 **Nutty Monster**: ナッツで勝てる確率が極めて高いハンドです。")
                else:
                    st.warning("Hand not found.")
            
        with col2:
            if 'row' in locals():
                st.subheader("📊 Win Distribution")
                val_weak_win = max(0, row["equity"] - row["nut_equity"])
                val_lose = 1.0 - row["equity"]
                
                p_sf = row["win_SF"] * row["equity"]
                p_quads = row["win_Quads"] * row["equity"]
                p_fh = row["win_FH"] * row["equity"]
                p_flush = row["win_Flush"] * row["equity"]
                p_straight = row["win_Straight"] * row["equity"]
                
                labels = ['Straight+', 'Flush', 'FullHouse+', 'Pair (Fragile)', 'Lose']
                sizes = [p_straight, p_flush, p_sf+p_quads+p_fh, val_weak_win, val_lose]
                colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FFC107', '#EEEEEE']
                explode = (0.05, 0.05, 0.05, 0, 0)

                fig1, ax1 = plt.subplots(figsize=(4, 3))
                plot_data = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0.001]
                if plot_data:
                    p_labels, p_sizes, p_colors, p_explode = zip(*plot_data)
                    wedges, texts, autotexts = ax1.pie(p_sizes, autopct='%1.1f%%', startangle=90, colors=p_colors, explode=p_explode, textprops={'fontsize': 8})
                    ax1.legend(wedges, p_labels, title="Outcome", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    st.pyplot(fig1)

        # --- 4. チャートエリア ---
        if 'row' in locals():
            st.divider()
            c_chart1, c_chart2 = st.columns(2)
            
            with c_chart1:
                # Equity Curve
                c_head, c_check = st.columns([3, 1])
                with c_head: st.subheader("📈 Equity Curve")
                with c_check:
                    zoom_curve = st.checkbox("🔍 Zoom Top 20%", value=False)
                
                sample_curve = df_plo.iloc[::200, :]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                
                ax3.plot(sample_curve["pct"], sample_curve["equity"], color="#cccccc", label="All Hands")
                ax3.scatter(row["pct"], row["equity"], color="red", s=100, zorder=5, label="You")
                
                ax3.set_xlabel("Top X% of Hands")
                ax3.set_ylabel("Equity")
                
                if zoom_curve:
                    ax3.set_xlim(0, 20)
                else:
                    ax3.set_xlim(0, 100)

                ax3.legend()
                ax3.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig3)

            with c_chart2:
                # Scatter Plot
                col_title, col_toggle = st.columns([2, 1])
                with col_title: 
                    chart_mode = st.radio(
                        "Scatter Mode",
                        ["Mode A: Efficiency", "Mode B: Abs. Power"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    if "Mode A" in chart_mode: st.caption("Mode A: Equity vs Quality")
                    else: st.caption("Mode B: Equity vs Nut Equity")
                    
                with col_toggle:
                    use_auto_zoom = st.checkbox("🔍 Auto Zoom", value=True)

                @st.cache_data
                def get_plo_scatter_background(df, n=3000):
                    return df.sample(n=n, random_state=42).copy()

                scatter_df_bg = get_plo_scatter_background(df_plo)

                plot_filtered_df = pd.DataFrame()
                if filtered_df_all is not None:
                    plot_filtered_df = filtered_df_all.head(2000)
                
                plot_highlight_df = pd.DataFrame()
                if highlight_tags:
                    hl_set = set(highlight_tags)
                    target_df = filtered_df_all if filtered_df_all is not None else df_plo
                    # ハイライト検索
                    if filtered_df_all is not None:
                         hl_mask = filtered_df_all["tags"].apply(lambda t: hl_set.issubset(set(t)))
                         plot_highlight_df = filtered_df_all[hl_mask].head(2000)
                    else:
                         # フィルタなし、ハイライトのみ (重いのでサンプルから)
                         temp_sample = df_plo.head(10000) 
                         hl_mask = temp_sample["tags"].apply(lambda t: hl_set.issubset(set(t)))
                         plot_highlight_df = temp_sample[hl_mask].head(2000)
                
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                
                def get_xy(df, mode):
                    if "Mode A" in mode:
                        return df["equity"], df["nut_quality"]
                    else:
                        return df["equity"], df["nut_equity"]

                bg_x, bg_y = get_xy(scatter_df_bg, chart_mode)
                my_x, my_y = (row["equity"], row["nut_quality"]) if "Mode A" in chart_mode else (row["equity"], row["nut_equity"])
                
                if "Mode A" in chart_mode:
                    ax2.set_ylabel("Nut Quality (0.0 - 1.0)")
                    c_bg = scatter_df_bg["nut_quality"]
                else:
                    ax2.set_ylabel("Nut Equity")
                    c_bg = 1.0 - (bg_x - bg_y) 
                    ax2.plot([0, 1], [0, 1], ls="--", c="gray", alpha=0.5)

                ax2.scatter(bg_x, bg_y, c=c_bg, cmap="coolwarm_r", s=10, alpha=0.1, label='Others')

                # Auto Zoom用範囲
                x_min, x_max = my_x, my_x
                y_min, y_max = my_y, my_y
                has_focus = False

                if not plot_filtered_df.empty:
                    f_x, f_y = get_xy(plot_filtered_df, chart_mode)
                    ax2.scatter(f_x, f_y, facecolors='none', edgecolors='gold', s=30, linewidth=1.0, label='Filtered')
                    x_min, x_max = min(x_min, f_x.min()), max(x_max, f_x.max())
                    y_min, y_max = min(y_min, f_y.min()), max(y_max, f_y.max())
                    has_focus = True

                if not plot_highlight_df.empty:
                    h_x, h_y = get_xy(plot_highlight_df, chart_mode)
                    ax2.scatter(h_x, h_y, facecolors='none', edgecolors='#FF00FF', s=60, linewidth=2.0, label='Highlighted', zorder=5)
                    x_min, x_max = min(x_min, h_x.min()), max(x_max, h_x.max())
                    y_min, y_max = min(y_min, h_y.min()), max(y_max, h_y.max())
                    has_focus = True

                ax2.scatter(my_x, my_y, c='black', s=150, marker='*', edgecolors='white', label='You', zorder=10)
                
                # 自分が範囲外にならないように
                x_min, x_max = min(x_min, my_x), max(x_max, my_x)
                y_min, y_max = min(y_min, my_y), max(y_max, my_y)

                if use_auto_zoom:
                    # フィルタもハイライトもない場合は背景全体を表示 (has_focusで判断)
                    if not has_focus:
                        x_min, x_max = bg_x.min(), bg_x.max()
                        y_min, y_max = bg_y.min(), bg_y.max()

                    min_span = 0.2
                    if (x_max - x_min) < min_span:
                        diff = (min_span - (x_max - x_min)) / 2
                        x_min -= diff; x_max += diff
                    if (y_max - y_min) < min_span:
                        diff = (min_span - (y_max - y_min)) / 2
                        y_min -= diff; y_max += diff
                    
                    margin = 0.05
                    ax2.set_xlim(max(0, x_min - margin), min(1, x_max + margin))
                    ax2.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
                else:
                    ax2.set_xlim(0, 1.05)
                    ax2.set_ylim(0, 1.05)

                ax2.set_xlabel("Raw Equity")
                ax2.legend(loc='upper left', fontsize=8)
                ax2.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig2)

with tab_flo8:
    st.header("FLO8 Strategy (Fixed Limit)")
    df_flo8 = load_flo8_data()
    if df_flo8 is None: st.warning("FLO8 data not loaded.")
    else:
        with st.expander("🔍 Rank Search"):
            c1, c2 = st.columns([1, 3])
            with c1:
                search_rank8 = st.number_input("Rank", 1, len(df_flo8), 1, key="f8_rk")
            with c2:
                fr8 = df_flo8.iloc[search_rank8-1]
                st.markdown(f"**{fr8['hand']}** (Top {fr8['pct_total']:.2f}%)")
                if st.button("Copy", key="cp8"):
                    st.session_state.flo8_input = fr8['hand']
                    st.rerun()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            raw8 = st.text_input("Enter Hand", key='flo8_input')
            inp8 = normalize_input_text(raw8)
            h_str8 = " ".join(inp8)
            score8, details8 = calculate_flo8_heuristic(h_str8)
            st.metric("Hutchinson Points", score8)
            st.bar_chart(details8)

        with col2:
            if len(inp8) == 4:
                res8 = df_flo8[df_flo8["card_set"] == frozenset(inp8)]
                if not res8.empty:
                    r8 = res8.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Scoop %", f"{r8['scoop_pct']:.1%}")
                    c2.metric("High Eq", f"{r8['high_equity']:.1%}")
                    c3.metric("Low Eq", f"{r8['low_equity']:.1%}")
                else: st.warning("Hand not found.")

with tab_guide:
    st.markdown("""
    ### 🏷️ Tag Definitions
    **Rundowns:** Perfect(No Gap), Top Gap, Mid Gap, Bottom Gap, Double Gap.
    **Pairs:** AA/KK/QQ, Double Pair, Single Pair.
    **Suits:** Double Suited, A-High Suit, Monotone.
    """)