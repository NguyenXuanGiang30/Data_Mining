"""
app.py - Heart Care AI  ·  Streamlit Demo
Dự đoán nguy cơ bệnh tim bằng Machine Learning

Chạy:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────
# Page config
# ──────────────────────────────────────
st.set_page_config(
    page_title="Heart Care AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Global ── */
        .stApp {
            background: linear-gradient(145deg, #0f1729 0%, #0b0f1a 50%, #070b14 100%);
            color: #e8ecf5;
        }

        /* ── Hero Banner ── */
        .hero-banner {
            background: linear-gradient(135deg, rgba(99,102,241,.25) 0%, rgba(236,72,153,.18) 100%);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 16px;
            padding: 1.4rem 1.8rem;
            margin-bottom: .6rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .hero-banner .hero-icon { font-size: 2.6rem; }
        .hero-banner h1 {
            margin: 0; font-size: 1.65rem; font-weight: 700;
            background: linear-gradient(90deg, #c7d2fe, #f9a8d4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-banner p { margin: .25rem 0 0; opacity: .85; font-size: .88rem; }

        /* ── Section headers ── */
        .section-hdr {
            display: flex; align-items: center; gap: .5rem;
            padding: .35rem 0 .25rem;
            font-weight: 600; font-size: .95rem;
            color: #a5b4fc;
            border-bottom: 1px solid rgba(165,180,252,.15);
            margin-bottom: .55rem;
        }

        /* ── Glass card ── */
        .glass-card {
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(255,255,255,.07);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: .7rem;
            backdrop-filter: blur(8px);
        }

        /* ── Result banner ── */
        .result-banner {
            border-radius: 14px;
            padding: 1rem 1.4rem;
            text-align: center;
            margin-bottom: .5rem;
        }
        .result-high {
            background: linear-gradient(135deg, rgba(239,68,68,.2), rgba(220,38,38,.12));
            border: 1px solid rgba(239,68,68,.3);
        }
        .result-low {
            background: linear-gradient(135deg, rgba(34,197,94,.2), rgba(16,185,129,.12));
            border: 1px solid rgba(34,197,94,.3);
        }
        .result-banner .res-label { font-size: .82rem; opacity: .8; margin-bottom: .2rem; }
        .result-banner .res-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
        .result-banner .res-tag {
            display: inline-block;
            border-radius: 999px;
            padding: .2rem .8rem;
            font-size: .78rem;
            font-weight: 600;
            margin-top: .35rem;
        }
        .tag-high { background: rgba(239,68,68,.25); color: #fca5a5; }
        .tag-low  { background: rgba(34,197,94,.25);  color: #86efac; }

        /* ── Risk badge ── */
        .risk-badge {
            display: inline-flex; align-items: center; gap: .3rem;
            background: rgba(251,191,36,.12);
            border: 1px solid rgba(251,191,36,.25);
            border-radius: 8px;
            padding: .3rem .65rem;
            font-size: .82rem;
            margin: .2rem .3rem .2rem 0;
            color: #fde68a;
        }

        /* ── Recommendation cards ── */
        .rec-card {
            background: rgba(255,255,255,.03);
            border: 1px solid rgba(255,255,255,.06);
            border-radius: 10px;
            padding: .6rem .8rem;
            margin-bottom: .45rem;
            font-size: .88rem;
            display: flex; align-items: flex-start; gap: .5rem;
        }
        .rec-card .rec-icon { font-size: 1.1rem; flex-shrink: 0; }

        /* ── Prob bar ── */
        .prob-bar-wrap {
            background: rgba(255,255,255,.06);
            border-radius: 10px;
            overflow: hidden;
            height: 28px;
            position: relative;
            margin: .5rem 0;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: .78rem; font-weight: 700;
            transition: width .5s ease;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: rgba(15,23,42,.95);
            border-right: 1px solid rgba(255,255,255,.06);
        }

        /* ── Metric override ── */
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }

        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255,255,255,.03);
            border-radius: 10px;
            padding: 3px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: .4rem 1rem;
            font-size: .85rem;
        }

        /* ── Hide default header padding ── */
        .block-container { padding-top: 1.5rem !important; }

        /* smaller slider labels */
        .stSlider label, .stSelectbox label, .stNumberInput label {
            font-size: .84rem !important;
        }

        /* ── Footer ── */
        .app-footer {
            text-align: center;
            padding: .6rem;
            font-size: .75rem;
            opacity: .55;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────
# Load model
# ──────────────────────────────────────
@st.cache_resource
def load_model():
    model_dir = "outputs/models"
    if not os.path.exists(model_dir):
        return None, None, None
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if not model_files:
        return None, None, None
    data = joblib.load(os.path.join(model_dir, model_files[0]))
    model = data.get("model")
    scaler = data.get("scaler")
    name = model_files[0].replace(".joblib", "").replace("_", " ").title()
    return model, scaler, name


def parse_enum(label: str) -> int:
    return int(label.split("(")[1].replace(")", ""))


# ──────────────────────────────────────
# Risk factors
# ──────────────────────────────────────
def get_risk_factors(age, chol, trestbps, thalch, oldpeak, cp_val, exang_val, ca):
    factors = []
    if age > 55:
        factors.append(("👤", f"Tuổi cao — {age} tuổi"))
    if chol > 240:
        factors.append(("🩸", f"Cholesterol cao — {chol} mg/dl"))
    if trestbps > 140:
        factors.append(("💉", f"Huyết áp nghỉ cao — {trestbps} mmHg"))
    if thalch < 120:
        factors.append(("💓", f"Nhịp tim tối đa thấp — {thalch} bpm"))
    if oldpeak > 2.0:
        factors.append(("📉", f"ST depression cao — {oldpeak}"))
    if cp_val == 3:
        factors.append(("🫁", "Đau ngực dạng không triệu chứng"))
    if exang_val == 1:
        factors.append(("🏃", "Đau thắt ngực khi vận động"))
    if ca > 0:
        factors.append(("🔬", f"Số mạch bất thường: {ca}"))
    return factors


# ──────────────────────────────────────
# Main
# ──────────────────────────────────────
def main():
    inject_css()

    model, scaler, model_name = load_model()
    if model is None:
        st.error("⚠️ Chưa có model. Hãy chạy `python scripts/run_pipeline.py` trước.")
        st.stop()

    # ── Hero ──
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="hero-icon">🫀</span>
            <div>
                <h1>Heart Care AI</h1>
                <p>Sàng lọc nguy cơ tim mạch · Model: <b>{model_name}</b> · Dữ liệu: UCI Heart Disease</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar: preset & info ──
    with st.sidebar:
        st.markdown("### ⚡ Chế độ nhanh")
        profile = st.radio(
            "Preset",
            ["Mặc định", "Nguy cơ thấp", "Nguy cơ cao"],
            horizontal=False,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### ℹ️ Thông tin")
        st.caption(f"**Model:** {model_name}")
        st.caption("**Seed:** 42 · **Dataset:** UCI Heart Disease (920 mẫu)")
        st.divider()
        st.caption("⚠️ Kết quả chỉ mang tính sàng lọc, không thay thế chẩn đoán y khoa.")

    # ── Defaults based on preset ──
    presets = {
        "Nguy cơ thấp": dict(age=38, sex="Nữ", cp=1, trestbps=118, chol=185, fbs=0, restecg=0, thalch=172, exang=0, oldpeak=0.4, slope=0, ca=0, thal=0),
        "Nguy cơ cao": dict(age=63, sex="Nam", cp=3, trestbps=155, chol=290, fbs=1, restecg=2, thalch=108, exang=1, oldpeak=2.8, slope=2, ca=2, thal=2),
        "Mặc định": dict(age=55, sex="Nam", cp=2, trestbps=130, chol=240, fbs=0, restecg=1, thalch=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=1),
    }
    d = presets[profile]

    # ── INPUT FORM ──
    st.markdown('<div class="section-hdr">📋 Thông tin bệnh nhân</div>', unsafe_allow_html=True)

    # Row 1: Personal info
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        age = st.slider("Tuổi", 20, 90, d["age"])
    with r1c2:
        sex = st.selectbox("Giới tính", ["Nam", "Nữ"], index=0 if d["sex"] == "Nam" else 1)
    with r1c3:
        cp_label = st.selectbox(
            "Loại đau ngực",
            ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"],
            index=d["cp"],
        )
    with r1c4:
        fbs_label = st.selectbox("Đường huyết > 120", ["Không (0)", "Có (1)"], index=d["fbs"])

    # Row 2: Cardiac metrics
    st.markdown('<div class="section-hdr">🩺 Chỉ số tim mạch</div>', unsafe_allow_html=True)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        trestbps = st.slider("Huyết áp nghỉ (mmHg)", 80, 220, d["trestbps"])
    with r2c2:
        chol = st.slider("Cholesterol (mg/dl)", 100, 650, d["chol"])
    with r2c3:
        thalch = st.slider("Nhịp tim tối đa (bpm)", 60, 230, d["thalch"])
    with r2c4:
        oldpeak = st.slider("ST Depression", 0.0, 6.5, float(d["oldpeak"]), 0.1)

    # Row 3: Test results
    st.markdown('<div class="section-hdr">🧪 Kết quả xét nghiệm</div>', unsafe_allow_html=True)
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        restecg_label = st.selectbox(
            "Điện tâm đồ nghỉ",
            ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"],
            index=d["restecg"],
        )
    with r3c2:
        exang_label = st.selectbox("Đau ngực khi vận động", ["Không (0)", "Có (1)"], index=d["exang"])
    with r3c3:
        slope_label = st.selectbox("Độ dốc ST", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"], index=d["slope"])
    with r3c4:
        ca = st.slider("Số mạch fluoroscopy", 0, 3, d["ca"])

    # Hidden-ish thal, put in an expander to save space
    with st.expander("🔬 Tùy chọn nâng cao", expanded=False):
        thal_label = st.selectbox(
            "Thalassemia",
            ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"],
            index=d["thal"],
        )

    # ── PREDICTION (auto) ──
    sex_val = 1 if sex == "Nam" else 0
    cp_val = parse_enum(cp_label)
    fbs_val = parse_enum(fbs_label)
    restecg_val = parse_enum(restecg_label)
    exang_val = parse_enum(exang_label)
    slope_val = parse_enum(slope_label)
    thal_val = parse_enum(thal_label)

    features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalch, exang_val, oldpeak, slope_val, ca, thal_val]])
    X = scaler.transform(features) if scaler is not None else features
    prob = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    prob_heart = float(prob[1])
    prob_healthy = 1 - prob_heart

    # ── RESULT SECTION ──
    st.markdown("---")

    # Top result cards
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])

    with col_res1:
        is_high = pred == 1
        cls = "result-high" if is_high else "result-low"
        tag_cls = "tag-high" if is_high else "tag-low"
        tag_txt = "⚠️ NGUY CƠ CAO" if is_high else "✅ NGUY CƠ THẤP"
        val_color = "#fca5a5" if is_high else "#86efac"
        st.markdown(
            f"""
            <div class="result-banner {cls}">
                <div class="res-label">Xác suất bệnh tim</div>
                <div class="res-value" style="color:{val_color}">{prob_heart*100:.1f}%</div>
                <span class="res-tag {tag_cls}">{tag_txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res2:
        st.markdown(
            f"""
            <div class="result-banner result-low">
                <div class="res-label">Xác suất không bệnh</div>
                <div class="res-value" style="color:#86efac">{prob_healthy*100:.1f}%</div>
                <span class="res-tag tag-low">Khỏe mạnh</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center; padding:1rem;">
                <div style="font-size:.82rem; opacity:.7; margin-bottom:.3rem;">Phân bố xác suất</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{prob_healthy*100:.1f}%; background:linear-gradient(90deg,#22c55e,#16a34a); color:#fff;">
                        {prob_healthy*100:.0f}%
                    </div>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{prob_heart*100:.1f}%; background:linear-gradient(90deg,#ef4444,#dc2626); color:#fff;">
                        {prob_heart*100:.0f}%
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:.72rem; opacity:.6; margin-top:.2rem;">
                    <span>🟢 Không bệnh</span><span>🔴 Bệnh tim</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["🧪 Yếu tố nguy cơ", "🩺 Khuyến nghị", "📊 Model Performance"])

    # ── Tab 1: Risk Factors ──
    with tab1:
        factors = get_risk_factors(age, chol, trestbps, thalch, oldpeak, cp_val, exang_val, ca)
        if factors:
            badges_html = "".join(
                f'<span class="risk-badge">{icon} {text}</span>' for icon, text in factors
            )
            st.markdown(
                f'<div class="glass-card"><b>⚠️ Phát hiện {len(factors)} tín hiệu nguy cơ</b><br><br>{badges_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ Không phát hiện yếu tố nguy cơ rõ rệt theo các ngưỡng rule-based.")

        # Quick patient summary
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Tuổi", f"{age}")
        mc2.metric("Huyết áp", f"{trestbps} mmHg")
        mc3.metric("Cholesterol", f"{chol} mg/dl")
        mc4.metric("Nhịp tim max", f"{thalch} bpm")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Recommendations ──
    with tab2:
        if pred == 1:
            st.warning("⚠️ Kết quả sàng lọc cho thấy nguy cơ cao — nên khám chuyên khoa tim mạch.")
            recs = [
                ("🏥", "Khám chuyên khoa tim mạch để chẩn đoán chính xác"),
                ("📊", "Làm điện tâm đồ và siêu âm tim"),
                ("🩸", "Kiểm tra mỡ máu, đường huyết chi tiết"),
                ("📈", "Theo dõi huyết áp và nhịp tim định kỳ"),
                ("👨‍⚕️", "Tham khảo bác sĩ về kế hoạch điều trị/can thiệp"),
            ]
        else:
            st.success("✅ Kết quả sàng lọc hiện ở mức tích cực — tiếp tục duy trì lối sống lành mạnh.")
            recs = [
                ("🥗", "Duy trì ăn uống lành mạnh, giảm mỡ bão hòa"),
                ("🏃", "Tập thể dục đều đặn ≥ 150 phút/tuần"),
                ("😴", "Ngủ đủ 7-8 tiếng mỗi đêm"),
                ("🩺", "Kiểm tra sức khỏe định kỳ 6 tháng/lần"),
                ("🧘", "Quản lý stress, thư giãn tinh thần"),
            ]

        for icon, text in recs:
            st.markdown(f'<div class="rec-card"><span class="rec-icon">{icon}</span>{text}</div>', unsafe_allow_html=True)

    # ── Tab 3: Model Performance ──
    with tab3:
        st.markdown('<div class="section-hdr">📋 Bảng so sánh các mô hình</div>', unsafe_allow_html=True)

        csv_path = "outputs/tables/classification_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            df_display = df.copy()
            df_display["F1-Score"] = df_display["F1-Score"].map(lambda x: f"{x:.4f}")
            df_display["ROC-AUC"] = df_display["ROC-AUC"].map(lambda x: f"{x:.4f}")
            df_display["PR-AUC"] = df_display["PR-AUC"].map(lambda x: f"{x:.4f}")
            df_display["FN Rate"] = df_display["FN Rate"].map(lambda x: f"{x:.2%}")
            df_display["FN Count"] = df_display["FN Count"].astype(int)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu classification. Chạy pipeline trước.")

        # Figures gallery
        st.markdown('<div class="section-hdr">📈 Biểu đồ phân tích</div>', unsafe_allow_html=True)

        fig_dir = "outputs/figures"
        figure_map = {
            "model_comparison.png": "So sánh mô hình",
            "cm_random_forest.png": "Confusion Matrix",
            "roc_pr_curves.png": "ROC & PR Curves",
            "correlation_matrix.png": "Ma trận tương quan",
            "features_vs_target.png": "Features vs Target",
            "numeric_distributions.png": "Phân phối dữ liệu",
            "target_distribution.png": "Phân phối nhãn",
            "elbow_silhouette.png": "Elbow & Silhouette",
            "cluster_profiles.png": "Cluster Profiles",
            "missing_values.png": "Missing Values",
        }

        if os.path.exists(fig_dir):
            available = [f for f in figure_map if os.path.exists(os.path.join(fig_dir, f))]
            if available:
                # Show 2 figures per row
                for i in range(0, len(available), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(available):
                            fname = available[idx]
                            with col:
                                st.caption(f"**{figure_map[fname]}**")
                                st.image(os.path.join(fig_dir, fname), use_container_width=True)
            else:
                st.info("Chưa có biểu đồ. Chạy pipeline trước.")
        else:
            st.info("Thư mục figures chưa tồn tại.")

    # ── Footer ──
    st.markdown(
        '<div class="app-footer">⚠️ Ứng dụng chỉ hỗ trợ sàng lọc rủi ro, không thay thế chẩn đoán y khoa. · Heart Care AI © 2025</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
