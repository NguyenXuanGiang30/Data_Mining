"""
association.py - Luật kết hợp / Apriori

Module này chịu trách nhiệm:
- Chạy thuật toán Apriori tìm frequent itemsets
- Sinh luật kết hợp (association rules)
- Lọc và diễn giải luật có ý nghĩa y học
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(df_binary, min_support=0.1, min_confidence=0.5, min_lift=1.0,
                metric="confidence", target_col="heart_disease"):
    """
    Chạy Apriori và sinh association rules.
    
    Parameters
    ----------
    df_binary : pd.DataFrame
        DataFrame đã rời rạc hoá (binary 0/1).
    min_support : float
    min_confidence : float
    min_lift : float
    metric : str
    target_col : str
        Cột target để lọc luật liên quan bệnh tim.
    
    Returns
    -------
    frequent_itemsets : pd.DataFrame
    rules : pd.DataFrame
    rules_heart : pd.DataFrame
        Luật liên quan đến bệnh tim.
    """
    print("=" * 60)
    print("🔗 ASSOCIATION RULE MINING (Apriori)")
    print("=" * 60)
    
    # 1. Tìm frequent itemsets
    print(f"\n--- Finding frequent itemsets (min_support={min_support})...")
    frequent_itemsets = apriori(df_binary, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    print(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
    
    # 2. Sinh association rules
    print(f"\n--- Generating rules (min_{metric}={min_confidence})...")
    if len(frequent_itemsets) == 0:
        print("⚠️  No frequent itemsets found! Try lowering min_support.")
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric=metric,
                              min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
    rules = rules[rules["lift"] >= min_lift]
    rules = rules.sort_values("lift", ascending=False)
    print(f"✅ Generated {len(rules)} rules (lift >= {min_lift})")
    
    # 3. Lọc luật liên quan bệnh tim
    rules_heart = rules[
        rules["consequents"].apply(lambda x: target_col in str(x))
    ].copy()
    rules_heart = rules_heart.sort_values("confidence", ascending=False)
    print(f"✅ {len(rules_heart)} rules related to '{target_col}'")
    
    return frequent_itemsets, rules, rules_heart


def format_rules_table(rules, top_n=20):
    """
    Format luật thành bảng dễ đọc.
    
    Returns
    -------
    pd.DataFrame
        Bảng gồm: antecedents, consequents, support, confidence, lift.
    """
    if len(rules) == 0:
        return pd.DataFrame()
    
    df = rules.head(top_n)[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    df["antecedents"] = df["antecedents"].apply(lambda x: ", ".join(list(x)))
    df["consequents"] = df["consequents"].apply(lambda x: ", ".join(list(x)))
    df["support"] = df["support"].round(4)
    df["confidence"] = df["confidence"].round(4)
    df["lift"] = df["lift"].round(4)
    
    return df.reset_index(drop=True)


def interpret_rules(rules_heart, top_n=10):
    """
    Diễn giải luật kết hợp theo ý nghĩa y học.
    
    Parameters
    ----------
    rules_heart : pd.DataFrame
        Luật liên quan bệnh tim.
    top_n : int
    
    Returns
    -------
    list of str
        Danh sách diễn giải.
    """
    interpretations = []
    
    for i, (_, row) in enumerate(rules_heart.head(top_n).iterrows()):
        antecedents = ", ".join(list(row["antecedents"]))
        confidence = row["confidence"]
        lift = row["lift"]
        support = row["support"]
        
        interpretation = (
            f"Luật {i+1}: Nếu {antecedents} → Nguy cơ bệnh tim\n"
            f"  • Confidence: {confidence:.1%} (≈ {confidence*100:.0f}% bệnh nhân với đặc điểm này bị bệnh tim)\n"
            f"  • Lift: {lift:.2f} (nguy cơ cao gấp {lift:.1f} lần so với trung bình)\n"
            f"  • Support: {support:.1%} (xuất hiện trong {support*100:.1f}% dataset)"
        )
        interpretations.append(interpretation)
    
    return interpretations
