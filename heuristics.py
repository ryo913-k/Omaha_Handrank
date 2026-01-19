import eval7

def get_cards_objects(hand_str):
    try:
        return [eval7.Card(s) for s in hand_str.split()]
    except:
        return []

# --- PLO (High Only) 用ヒューリスティック ---
def calculate_plo_heuristic(hand_str):
    cards = get_cards_objects(hand_str)
    if len(cards) != 4: return 0, {}
    
    # 【修正】基準点を50.0に設定 (平均的なハンドの勝率)
    score = 50.0
    details = {"Base (Avg)": 50.0, "Structure": 0, "Suited": 0, "Pairs": 0}
    
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # --- 1. Pairs ---
    # AA=+16, KK=+13, QQ=+11, JJ=+9
    rank_counts = {r: ranks.count(r) for r in ranks}
    pairs = [r for r, c in rank_counts.items() if c == 2]
    
    pair_score = 0
    for p in pairs:
        if p == 12: pair_score += 15.7 # AA
        elif p == 11: pair_score += 12.6 # KK
        elif p == 10: pair_score += 10.5 # QQ
        elif p == 9: pair_score += 8.6   # JJ
        else: pair_score -= 0.4          # TT-22 (平均よりわずかに弱い)
    
    # ペナルティ
    if 3 in rank_counts.values(): pair_score -= 2.0
    if 4 in rank_counts.values(): pair_score -= 5.0
    
    score += pair_score
    details["Pairs"] += pair_score

    # --- 2. Suitedness ---
    s_counts = {s: suits.count(s) for s in suits}
    counts_list = sorted(list(s_counts.values()), reverse=True)
    suit_dist = counts_list + [0] * (4 - len(counts_list))
    
    suit_score = 0
    is_monotone = (suit_dist[0] == 4)
    is_ds = (suit_dist[0] == 2 and suit_dist[1] == 2)
    is_ss = (suit_dist[0] >= 2 and suit_dist[1] < 2 and not is_monotone)
    is_rainbow = (suit_dist[0] == 1)
    
    # A-High Suit check
    has_A_suit = False
    if is_ds or is_ss or is_monotone:
        for s, count in s_counts.items():
            if count >= 2:
                # そのスートを持つカードの中にAがあるか
                suited_ranks = [c.rank for c in cards if c.suit == s]
                if 12 in suited_ranks:
                    has_A_suit = True

    if is_ds:
        suit_score += 3.5 
        if has_A_suit: suit_score += 2.0 
    elif is_ss:
        if has_A_suit: suit_score += 4.0 
        else: suit_score -= 1.0 
    elif is_rainbow:
        suit_score -= 3.0
    elif is_monotone:
        suit_score -= 3.0

    score += suit_score
    details["Suited"] += suit_score

    # --- 3. Structure ---
    struct_score = 0
    unique_ranks = sorted(list(set(ranks)))
    
    if len(unique_ranks) == 4:
        gap = unique_ranks[-1] - unique_ranks[0]
        # 構造自体の平均的な強さ (データ上はマイナス補正)
        if gap == 3: struct_score -= 1.4 # Run
        elif gap == 4: struct_score -= 0.9 # 1-Gap
        elif gap == 5: struct_score -= 1.3 # 2-Gap
        else: struct_score -= 2.0 # Trash shape
        
        # ハイカードボーナス (構造の弱さをカードパワーで補う)
        # A=+1.5, K=+1.0, Q=+0.5
        for r in ranks:
            if r == 12: struct_score += 1.5
            elif r == 11: struct_score += 1.0
            elif r == 10: struct_score += 0.5
    else:
        # ペアがある場合などのハイカード加点
        for r in ranks:
            if r >= 10: struct_score += 0.5

    score += struct_score
    details["Structure"] += struct_score
    
    return score, details

# --- FLO8 (Fixed Limit Omaha 8) 用ヒューリスティック (変更なし) ---
def calculate_flo8_heuristic(hand_str):
    cards = get_cards_objects(hand_str)
    if len(cards) != 4: return 0, {}
    
    score = 0
    details = {"Suited": 0, "Pairs": 0, "Low": 0, "Connector": 0}
    ranks = [c.rank for c in cards]
    
    # 1. Suitedness
    suit_counts = {s: [] for s in range(4)}
    for c in cards: suit_counts[c.suit].append(c.rank)
    
    for s in suit_counts:
        rs = sorted(suit_counts[s], reverse=True)
        if len(rs) >= 2:
            hr = rs[0]
            if hr == 12: pts = 4 
            elif hr == 11: pts = 3 
            elif hr == 10: pts = 2.5 
            elif hr == 9: pts = 2 
            elif hr <= 5: pts = 2 
            else: pts = 1
            score += pts; details["Suited"] += pts

    # 2. Pairs
    rank_counts = {r: ranks.count(r) for r in ranks}
    for r, count in rank_counts.items():
        if count == 2:
            if r == 12: pts = 30 
            elif r == 11: pts = 16 
            elif r == 10: pts = 13 
            elif r == 9: pts = 10 
            else: pts = 4 
            score += pts; details["Pairs"] += pts
        elif count == 3: score += 2 
        elif count == 4: score -= 10 

    # 3. Low Potential
    has_A, has_2, has_3 = (12 in ranks), (0 in ranks), (1 in ranks)
    if has_A and has_2: score += 20; details["Low"] += 20
    elif has_A and has_3: score += 15; details["Low"] += 15
    elif 0 in ranks and 1 in ranks: score += 10; details["Low"] += 10
    elif has_A and (2 in ranks or 3 in ranks): score += 5; details["Low"] += 5

    # 4. Connector
    sorted_unique = sorted(list(set(ranks)))
    if len(sorted_unique) == 4:
        gap = sorted_unique[-1] - sorted_unique[0]
        if gap <= 4: score += 5; details["Connector"] += 5

    return score, details