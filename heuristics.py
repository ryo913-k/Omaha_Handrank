import collections

# ---------------------------------------------------------
# 簡易カードクラス (eval7を使わずにカードを処理するため)
# ---------------------------------------------------------
class SimpleCard:
    def __init__(self, card_str):
        if not card_str:
            self.rank = -1
            self.suit = ''
            return
            
        rank_char = card_str[:-1].upper()
        suit_char = card_str[-1].lower()
        
        # ランクを数値に変換 (2=0, ... A=12)
        ranks = "23456789TJQKA"
        if rank_char in ranks:
            self.rank = ranks.index(rank_char)
        else:
            self.rank = -1
            
        self.suit = suit_char
        self.raw = card_str

# ---------------------------------------------------------
# FLO8 (Hutchinson Point Count System) 計算ロジック
# ---------------------------------------------------------
def calculate_flo8_heuristic(hand_str):
    """
    Hutchinson Point Count Systemに基づき、
    FLO8におけるハンドの強さを評価する。
    """
    try:
        cards = [SimpleCard(s) for s in hand_str.split()]
    except:
        return 0, {"Error": 0}

    if len(cards) != 4:
        return 0, {"Invalid Hand": 0}

    points = 0.0
    details = {
        "Low Potential": 0.0,
        "High Pairs": 0.0,
        "Suited Bonus": 0.0,
        "Straight Potential": 0.0
    }

    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    
    # 1. Low Potential (A=20, 2=9, 3=6, 4=4, 5=2)
    # ペアになっているカードは、ローのセットとしては使えないため1枚分だけカウントするのが通例だが、
    # 厳密なHutchinsonでは「アンペアのローカード」を評価する。
    # ここでは簡易的に全てのローカードを加算し、ペア分は後で調整（あるいはそのまま評価）
    low_values = {12: 20, 0: 9, 1: 6, 2: 4, 3: 2} # 12=A, 0=2, 1=3...
    
    current_low_pts = 0
    # 重複を除く（A A 2 3 なら Aは1回だけカウント -> ローを作る能力）
    unique_ranks = set(ranks)
    for r in unique_ranks:
        if r in low_values:
            current_low_pts += low_values[r]
            
    points += current_low_pts
    details["Low Potential"] = current_low_pts

    # 2. High Pairs
    # AA=30, KK=16, QQ=13, JJ=10, TT=10, 99=9, 88=8
    # 77-44=4-7pts (Middle pairs often devalued in FLO8)
    # ここでは主要なペア得点を加算
    rank_counts = collections.Counter(ranks)
    pair_pts = 0
    
    for r, count in rank_counts.items():
        if count == 2:
            if r == 12: pair_pts += 30 # AA
            elif r == 11: pair_pts += 16 # KK
            elif r == 10: pair_pts += 13 # QQ
            elif r == 9: pair_pts += 10 # JJ
            elif r == 8: pair_pts += 10 # TT
            elif r == 7: pair_pts += 9 # 99
            elif r == 6: pair_pts += 8 # 88
            else: pair_pts += 4 # Small pairs
        elif count == 3:
            # Trips are bad in Omaha
            pair_pts -= 5 # Penalty roughly
    
    points += pair_pts
    details["High Pairs"] = pair_pts

    # 3. Suited Bonus
    # A-suited=4, K-suited=3, Q-suited=2.5, J-suited=2
    # 2枚以上同じスートがある場合のみ加算
    suit_counts = collections.Counter(suits)
    suited_pts = 0
    
    for s, count in suit_counts.items():
        if count >= 2:
            # そのスートを持つカードの中で一番高いランクを見る
            suited_cards_ranks = [c.rank for c in cards if c.suit == s]
            max_r = max(suited_cards_ranks)
            
            if max_r == 12: suited_pts += 4 # A-High Flush Draw
            elif max_r == 11: suited_pts += 3 # K-High
            elif max_r == 10: suited_pts += 2.5 # Q-High
            elif max_r == 9: suited_pts += 2 # J-High
            elif max_r <= 8: suited_pts += 1 # Low Flush Draw
            
            # Double Suited Bonus
            if count == 2 and len(suit_counts) == 2:
                # すでに加算されているので追加ボーナスは微調整
                pass
                
    points += suited_pts
    details["Suited Bonus"] = suited_pts

    # 4. Straight Potential (Connectivity)
    # 簡易判定: ギャップの少なさで加点
    # A,K,Q,J,T,9,8,7,6,5,4,3,2 (12..0)
    # 20点満点くらいで評価
    straight_pts = 0
    unique_sorted = sorted(list(unique_ranks), reverse=True)
    
    if len(unique_sorted) >= 2:
        gaps = 0
        connections = 0
        for i in range(len(unique_sorted)-1):
            diff = unique_sorted[i] - unique_sorted[i+1]
            if diff == 1:
                connections += 1
            elif diff == 2:
                gaps += 1
        
        # 簡易評価
        if connections == 3: straight_pts += 12 # Rundown
        elif connections == 2: straight_pts += 8
        elif connections == 1: straight_pts += 2
        
        # ギャップペナルティなしで加点のみ
        if gaps > 0: straight_pts += 1

    points += straight_pts
    details["Straight Potential"] = straight_pts

    return int(points), details