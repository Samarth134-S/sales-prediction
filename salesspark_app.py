"""
⚡ SalesSpark — Ad Spend → Sales Prediction Studio
Run with:  streamlit run salesspark_app.py
Needs:     sales_prediction_model.pkl in the same folder (or upload it in the sidebar)
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(
    page_title="SalesSpark ⚡ Ad-to-Sales Predictor",
    page_icon="⚡",
    layout="wide",
)

MODEL_FILE = Path(__file__).parent / "sales_prediction_model.pkl"
FEATURES = ["youtube", "facebook", "newspaper"]  # same order used in training

CHANNELS = {
    "youtube":   {"label": "YouTube",   "emoji": "▶️", "color": "#FF3B5C", "max": 400.0, "default": 150.0},
    "facebook":  {"label": "Facebook",  "emoji": "👍", "color": "#3B82F6", "max": 70.0,  "default": 30.0},
    "newspaper": {"label": "Newspaper", "emoji": "📰", "color": "#F59E0B", "max": 130.0, "default": 40.0},
}

# ───────────────────────── Styling ─────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #2b1055 0%, #1a1040 35%, #0b132b 100%);
        color: #f5f3ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3a0ca3 0%, #7209b7 60%, #f72585 140%);
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    header[data-testid="stHeader"] { background: transparent; }

    .hero {
        padding: 2.2rem 2rem; border-radius: 28px; margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #f72585, #7209b7, #3a0ca3, #4cc9f0);
        background-size: 300% 300%; animation: flow 10s ease infinite;
        box-shadow: 0 20px 50px rgba(247, 37, 133, .25);
    }
    @keyframes flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    .hero h1 { font-size: 3rem; font-weight: 800; margin: 0; color: #fff; letter-spacing: -1px; }
    .hero p  { font-size: 1.1rem; margin: .4rem 0 0; color: #ffffffdd; }

    .card {
        background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.15);
        border-radius: 22px; padding: 1.2rem 1.4rem; backdrop-filter: blur(10px);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(0,0,0,.35); }
    .card .label { font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75; }
    .card .value { font-size: 2.2rem; font-weight: 800; line-height: 1.2; }
    .card .sub   { font-size: .85rem; opacity: .7; }

    .big-result {
        text-align: center; padding: 1.6rem; border-radius: 26px; margin: .5rem 0 1rem;
        background: linear-gradient(135deg, #06d6a0, #4cc9f0);
        color: #052; box-shadow: 0 18px 40px rgba(6,214,160,.3);
    }
    .big-result .num { font-size: 4rem; font-weight: 800; color: #04283a; line-height: 1; }
    .big-result .cap { font-size: 1rem; font-weight: 600; color: #04283acc; }

    div.stButton > button {
        background: linear-gradient(90deg, #f72585, #7209b7); color: white; border: 0;
        border-radius: 14px; padding: .7rem 1.4rem; font-weight: 700;
        box-shadow: 0 8px 20px rgba(247,37,133,.35); transition: all .2s ease;
    }
    div.stButton > button:hover { transform: scale(1.04); color: white; }

    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── Helpers ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path_or_file):
    return joblib.load(path_or_file)


def predict(model, yt, fb, np_):
    X = pd.DataFrame([[yt, fb, np_]], columns=FEATURES)
    return float(model.predict(X)[0])


def card(label, value, sub="", color="#fff"):
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def plot_layout(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#fff")),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f3ff", family="Poppins"), margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SalesSpark")
    st.caption("Tune your ad mix and watch sales react in real time.")
    st.divider()

    model = None
    if MODEL_FILE.exists():
        model = load_model(MODEL_FILE)
        st.success("Model loaded ✅")
    else:
        uploaded = st.file_uploader("Upload sales_prediction_model.pkl", type=["pkl"])
        if uploaded:
            model = load_model(uploaded)
            st.success("Model loaded ✅")

    st.markdown("### 🎛️ Ad budget mix")
    inputs = {}
    for key, meta in CHANNELS.items():
        inputs[key] = st.slider(
            f"{meta['emoji']} {meta['label']}", 0.0, meta["max"], meta["default"], step=0.5,
            help=f"Budget spent on {meta['label']} (same units as your training data).",
        )
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ───────────────────────── Hero ─────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>⚡ SalesSpark</h1>
        <p>Turn ad spend into sales insight — predict, compare and optimise your marketing mix in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning("Place **sales_prediction_model.pkl** next to this file, or upload it from the sidebar to begin.")
    st.stop()

yt, fb, nw = inputs["youtube"], inputs["facebook"], inputs["newspaper"]
prediction = predict(model, yt, fb, nw)
total_budget = yt + fb + nw
coefs = dict(zip(FEATURES, model.coef_))
intercept = float(model.intercept_)

# ───────────────────────── Tabs ─────────────────────────
tab_predict, tab_whatif, tab_optimize, tab_about = st.tabs(
    ["🔮 Predict", "🧪 What-if Lab", "🎯 Budget Optimizer", "ℹ️ About"]
)

# ───── Tab 1: Predict ─────
with tab_predict:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="big-result"><div class="cap">PREDICTED SALES</div>'
            f'<div class="num">{prediction:,.2f}</div>'
            f'<div class="cap">units (as per your dataset)</div></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("Total budget", f"{total_budget:,.1f}", "all channels", "#4cc9f0"), unsafe_allow_html=True)
        roi = prediction / total_budget if total_budget else 0
        c2.markdown(card("Sales / budget", f"{roi:.3f}", "efficiency", "#06d6a0"), unsafe_allow_html=True)
        best = max(FEATURES, key=lambda k: coefs[k])
        c3.markdown(card("Top channel", CHANNELS[best]["label"], "highest impact per unit", CHANNELS[best]["color"]),
                    unsafe_allow_html=True)

        if st.button("💾 Save this scenario"):
            st.session_state.setdefault("scenarios", []).append(
                {"Scenario": f"#{len(st.session_state.get('scenarios', [])) + 1}",
                 "YouTube": yt, "Facebook": fb, "Newspaper": nw, "Predicted sales": round(prediction, 2)}
            )
            st.toast("Scenario saved! Compare it in the What-if Lab 🧪", icon="✅")

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prediction,
            number={"font": {"size": 44, "color": "#fff"}},
            gauge={
                "axis": {"range": [0, 40], "tickcolor": "#fff"},
                "bar": {"color": "#f72585"},
                "bgcolor": "rgba(255,255,255,.08)",
                "steps": [{"range": [0, 12], "color": "#7209b7"},
                          {"range": [12, 25], "color": "#3a0ca3"},
                          {"range": [25, 40], "color": "#4361ee"}],
            },
        ))
        st.plotly_chart(plot_layout(gauge, "Sales meter", 320), use_container_width=True)

    # Waterfall: how each channel builds the final number
    contrib = {k: coefs[k] * v for k, v in inputs.items()}
    wf = go.Figure(go.Waterfall(
        x=["Baseline"] + [CHANNELS[k]["label"] for k in FEATURES] + ["Predicted"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[intercept] + [contrib[k] for k in FEATURES] + [0],
        increasing={"marker": {"color": "#06d6a0"}},
        decreasing={"marker": {"color": "#ff4d6d"}},
        totals={"marker": {"color": "#f72585"}},
        connector={"line": {"color": "#ffffff55"}},
        text=[f"{intercept:.2f}"] + [f"{contrib[k]:+.2f}" for k in FEATURES] + [f"{prediction:.2f}"],
        textposition="outside",
    ))
    st.plotly_chart(plot_layout(wf, "🧩 How each channel builds your sales number", 400), use_container_width=True)

    if contrib["newspaper"] < 0:
        st.info("📰 Newspaper has a slightly **negative** effect in this model — extra newspaper budget "
                "isn't adding sales once YouTube and Facebook are accounted for.")

# ───── Tab 2: What-if Lab ─────
with tab_whatif:
    st.markdown("#### 🧪 Sensitivity — what if you spend more on one channel?")
    channel = st.selectbox("Pick a channel to vary", FEATURES, format_func=lambda k: CHANNELS[k]["label"])
    meta = CHANNELS[channel]
    xs = [meta["max"] * i / 40 for i in range(41)]
    ys = [predict(model, *[(x if f == channel else inputs[f]) for f in FEATURES]) for x in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=meta["color"], width=5),
                             fill="tozeroy", fillcolor="rgba(255,255,255,.06)", name="Predicted sales"))
    fig.add_trace(go.Scatter(x=[inputs[channel]], y=[prediction], mode="markers",
                             marker=dict(size=16, color="#fff", line=dict(color=meta["color"], width=4)),
                             name="You are here"))
    fig.update_xaxes(title=f"{meta['label']} budget", gridcolor="#ffffff22")
    fig.update_yaxes(title="Predicted sales", gridcolor="#ffffff22")
    st.plotly_chart(plot_layout(fig, f"Sales vs {meta['label']} spend (other channels fixed)", 420),
                    use_container_width=True)

    st.markdown("#### 📊 Saved scenarios")
    scen = st.session_state.get("scenarios", [])
    if scen:
        df = pd.DataFrame(scen)
        st.dataframe(df, use_container_width=True, hide_index=True)
        bar = go.Figure(go.Bar(x=df["Scenario"], y=df["Predicted sales"],
                               marker=dict(color=df["Predicted sales"], colorscale="Sunsetdark"),
                               text=df["Predicted sales"], textposition="outside"))
        st.plotly_chart(plot_layout(bar, "Scenario comparison", 340), use_container_width=True)
        if st.button("🗑️ Clear scenarios"):
            st.session_state["scenarios"] = []
            st.rerun()
    else:
        st.caption("No scenarios yet — set the sliders and press **Save this scenario** on the Predict tab.")

# ───── Tab 3: Budget Optimizer ─────
with tab_optimize:
    st.markdown("#### 🎯 Got a fixed budget? Let SalesSpark split it for you")
    budget = st.number_input("Total budget to allocate", min_value=0.0, value=200.0, step=10.0)

    # Linear model → greedy allocation: fill the highest-coefficient channel first, up to its cap.
    # Caps keep the answer inside the range the model has seen (avoids wild extrapolation).
    remaining, plan = budget, {k: 0.0 for k in FEATURES}
    for k in sorted(FEATURES, key=lambda k: coefs[k], reverse=True):
        if coefs[k] <= 0:
            continue
        spend = min(remaining, CHANNELS[k]["max"])
        plan[k], remaining = spend, remaining - spend
    opt_pred = predict(model, *[plan[k] for k in FEATURES])
    even_pred = predict(model, *[budget / 3] * 3)

    o1, o2, o3 = st.columns(3)
    o1.markdown(card("Optimised sales", f"{opt_pred:,.2f}", "smart split", "#06d6a0"), unsafe_allow_html=True)
    o2.markdown(card("Even 3-way split", f"{even_pred:,.2f}", "for comparison", "#4cc9f0"), unsafe_allow_html=True)
    o3.markdown(card("Uplift", f"{opt_pred - even_pred:+,.2f}", "vs even split", "#f72585"), unsafe_allow_html=True)

    donut = go.Figure(go.Pie(
        labels=[CHANNELS[k]["label"] for k in FEATURES], values=[plan[k] for k in FEATURES], hole=.6,
        marker=dict(colors=[CHANNELS[k]["color"] for k in FEATURES]), textinfo="label+percent",
    ))
    st.plotly_chart(plot_layout(donut, "Recommended allocation", 400), use_container_width=True)
    if remaining > 0:
        st.caption(f"💡 {remaining:,.1f} of the budget is left unallocated — the remaining channel(s) "
                   "show no positive impact in the model or exceed the range seen in training.")

# ───── Tab 4: About ─────
with tab_about:
    st.markdown(
        f"""
        ### How it works
        SalesSpark uses a **Linear Regression** model trained on historical marketing data.

        **Sales = {intercept:.3f} + ({coefs['youtube']:.4f} × YouTube) + ({coefs['facebook']:.4f} × Facebook) + ({coefs['newspaper']:.4f} × Newspaper)**

        - Every extra unit of **Facebook** budget adds about **{coefs['facebook']:.3f}** sales units.
        - Every extra unit of **YouTube** budget adds about **{coefs['youtube']:.3f}** sales units.
        - **Newspaper** adds about **{coefs['newspaper']:.3f}** — basically nothing.

        > Predictions are most reliable inside the budget ranges seen during training.
        """
    )
