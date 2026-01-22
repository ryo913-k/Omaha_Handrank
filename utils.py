import streamlit as st
import unicodedata

# ==========================================
# Common Classes & Helper Functions
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
        rank = c[:-1]; suit = c[-1].lower()
        symbol = suit_map.get(suit, suit); color = color_map.get(suit, 'black')
        style = (f"width:{size}px; height:{size*1.35}px; background-color:white; border:1px solid #bbb; "
                 f"border-radius:4px; display:flex; justify-content:center; align-items:center; "
                 f"font-size:{size*0.45}px; font-weight:bold; color:{color}; box-shadow:2px 2px 4px rgba(0,0,0,0.1);")
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
    for s in ['s','h','d','c']:
        ms_key = f"ms_{s}_{target_key}"
        if ms_key in st.session_state:
            st.session_state[ms_key] = []

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
                for s in ['s','h','d','c']:
                    ms_key = f"ms_{s}_{session_key}"
                    if ms_key in st.session_state: del st.session_state[ms_key]
                st.rerun()
            return collected
        elif len(collected) > 0:
            st.caption(f"Selected: {len(collected)}/4 cards.")
    return []