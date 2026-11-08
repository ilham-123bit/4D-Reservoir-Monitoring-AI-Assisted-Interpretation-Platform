"""Shared visual language and small Streamlit presentation components."""

from copy import deepcopy

import plotly.io as pio
import streamlit as st


def format_metric_count(value: int) -> str:
    """Compact headline counts without losing the exact value shown in help text."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("metric count must be a nonnegative integer")
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def apply_theme() -> None:
    if "reservoir_time_machine" not in pio.templates:
        template = deepcopy(pio.templates["plotly_dark"])
        template.layout.font = {"family": "Arial, sans-serif", "size": 20, "color": "#ffffff"}
        template.layout.paper_bgcolor = "#101011"
        template.layout.plot_bgcolor = "#171516"
        template.layout.xaxis = {"gridcolor": "#344347", "zerolinecolor": "#597176", "title_font": {"size": 20}, "tickfont": {"size": 18}}
        template.layout.yaxis = {"gridcolor": "#344347", "zerolinecolor": "#597176", "title_font": {"size": 20}, "tickfont": {"size": 18}}
        template.layout.scene = {"xaxis": {"backgroundcolor": "#121a1c", "gridcolor": "#344347", "title_font": {"size": 20}, "tickfont": {"size": 18}},
                                "yaxis": {"backgroundcolor": "#121a1c", "gridcolor": "#344347", "title_font": {"size": 20}, "tickfont": {"size": 18}},
                                "zaxis": {"backgroundcolor": "#121a1c", "gridcolor": "#344347", "title_font": {"size": 20}, "tickfont": {"size": 18}}}
        template.layout.legend = {"font": {"size": 19}}
        pio.templates["reservoir_time_machine"] = template
    pio.templates.default = "reservoir_time_machine"
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #080d0e; color: #ffffff; }
        [data-testid="stHeader"] { background: rgba(16,16,17,.92); }
        [data-testid="stSidebar"] { background: #101719; border-right: 1px solid #31edc2; }
        [data-testid="stMetric"] { background: #101719; border: 1px solid #31edc2;
          padding: .8rem 1rem; border-radius: .8rem; min-width:0; overflow:visible; }
        [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: clamp(1rem, 1.15vw, 1.15rem) !important; font-weight:700 !important; line-height:1.2; }
        [data-testid="stMetricValue"] { color: #ffffff !important; font-size: clamp(2.625rem, 3.1vw, 3.25rem) !important; font-weight:800 !important; line-height:1.1; white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; font-variant-numeric:tabular-nums; letter-spacing:-.035em; }
        [data-testid="stMetricValue"] > div { white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; }
        .rtm-kicker { color:#31edc2; text-transform:uppercase; letter-spacing:.13em;
          font-size:clamp(1rem, 1.4vw, 1.35rem); font-weight:800; margin-bottom:.25rem; }
        .rtm-title { color:#ffffff; font-size:clamp(3.5rem, 4.3vw, 4.5rem); line-height:1.05; font-weight:850;
          margin:.1rem 0 .45rem; }
        .rtm-subtitle { color:#ffffff; font-size:clamp(1.25rem, 2vw, 1.7rem); font-weight:650; margin-bottom:1rem; }
        .rtm-section { color:#ffffff; font-size:clamp(1.75rem, 2.2vw, 2.1rem); font-weight:800;
          border-bottom:2px solid #31edc2; padding-bottom:.35rem; margin:1.1rem 0 .7rem; }
        .rtm-note { background:#101719; border-left:5px solid #31edc2; padding:1rem 1.2rem;
          border-radius:.35rem; color:#ffffff; font-size:clamp(1.2rem, 1.8vw, 1.6rem); }
        div[data-testid="stCaptionContainer"] p, .stMarkdown, label, p, span { color:#ffffff !important; }
        div[data-testid="stCaptionContainer"] p { font-size:1rem; font-weight:600; line-height:1.3; }
        html, body, [data-testid="stAppViewContainer"] { font-size:18px; }
        .stMarkdown p { font-size:1.05rem; line-height:1.4; }
        h1 { font-size:clamp(2.5rem, 4vw, 3.5rem) !important; color:#fff !important; }
        h2 { font-size:clamp(1.9rem, 2.4vw, 2.25rem) !important; color:#fff !important; }
        h3 { font-size:clamp(1.5rem, 2vw, 1.9rem) !important; color:#fff !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          font-size:clamp(1.5rem, 1.7vw, 1.75rem) !important; line-height:1.15 !important;
          overflow-wrap:normal !important; word-break:keep-all !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { white-space:normal; overflow-wrap:normal; word-break:normal; }
        [data-testid="stAlert"] p, [data-testid="stAlert"] div, [data-testid="stInfo"] p,
        [data-testid="stWarning"] p, [data-testid="stError"] p { color:#ffffff !important; }
        [data-testid="stAlert"] { background:#111a1b !important; border:1px solid #31edc2 !important; }
        [data-testid="stWarning"] { background:#332710 !important; border-color:#ffbf47 !important; }
        [data-testid="stError"] { background:#351716 !important; border-color:#ff5a52 !important; }
        [data-testid="stSuccess"] { background:#102b23 !important; border-color:#31edc2 !important; }
        input, textarea, [data-baseweb="select"] > div { background:#101719 !important; color:#ffffff !important; font-size:1.05rem !important; }
        [role="listbox"], [role="option"] { background:#101719 !important; color:#ffffff !important; font-size:1.1rem !important; }
        button { color:#ffffff !important; background:#123c36 !important; font-size:1.05rem !important; font-weight:750 !important; }
        button:hover { background:#176456 !important; }
        [data-testid="stPopoverBody"] { background:#101719 !important; color:white !important; }
        button:disabled { color:#ffffff !important; opacity:.65; }
        .stButton button, .stDownloadButton button { border:2px solid #31edc2; padding:.65rem 1rem; }
        [data-testid="stTabs"] button { font-size:1.05rem !important; color:white !important; }
        @media (max-width: 900px) { .rtm-title { font-size:clamp(2.4rem, 7vw, 3.2rem); } [data-testid="stMetricValue"] { font-size:2.625rem !important; } [data-testid="stMetric"] { padding:.7rem .75rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="rtm-kicker">Reservoir Time Machine · Challenge 3</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rtm-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rtm-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="rtm-section">{title}</div>', unsafe_allow_html=True)


def registration_warning() -> None:
    st.warning(
        "Trace-key pairing is available on the Baseline grid. Physical 4D registration is not yet verified. "
        "Interpret paired attributes as preliminary screening only."
    )
