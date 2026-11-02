"""Shared visual language and small Streamlit presentation components."""

from copy import deepcopy

import plotly.io as pio
import streamlit as st


def apply_theme() -> None:
    if "reservoir_time_machine" not in pio.templates:
        template = deepcopy(pio.templates["plotly_dark"])
        template.layout.font = {"family": "Arial, sans-serif", "size": 15, "color": "#eee7e4"}
        template.layout.paper_bgcolor = "#101011"
        template.layout.plot_bgcolor = "#171516"
        template.layout.xaxis = {"gridcolor": "#3a3535", "zerolinecolor": "#5a5050", "title_font": {"size": 15}}
        template.layout.yaxis = {"gridcolor": "#3a3535", "zerolinecolor": "#5a5050", "title_font": {"size": 15}}
        template.layout.scene = {"xaxis": {"backgroundcolor": "#171516", "gridcolor": "#484142", "title_font": {"size": 15}},
                                "yaxis": {"backgroundcolor": "#171516", "gridcolor": "#484142", "title_font": {"size": 15}},
                                "zaxis": {"backgroundcolor": "#171516", "gridcolor": "#484142", "title_font": {"size": 15}}}
        template.layout.legend = {"font": {"size": 14}}
        pio.templates["reservoir_time_machine"] = template
    pio.templates.default = "reservoir_time_machine"
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #101011; color: #f5f1ee; }
        [data-testid="stHeader"] { background: rgba(16,16,17,.92); }
        [data-testid="stSidebar"] { background: #191819; border-right: 1px solid #383233; }
        [data-testid="stMetric"] { background: #211d1e; border: 1px solid #423638;
          padding: 1rem 1.1rem; border-radius: .75rem; }
        [data-testid="stMetricLabel"] { color: #d0c8c7 !important; font-size: 1rem !important; }
        [data-testid="stMetricValue"] { color: #fff8f4 !important; font-size: 1.8rem !important; }
        .rtm-kicker { color:#ff7868; text-transform:uppercase; letter-spacing:.13em;
          font-size:.84rem; font-weight:700; margin-bottom:.25rem; }
        .rtm-title { color:#fff8f4; font-size:2.3rem; line-height:1.12; font-weight:750;
          margin:.1rem 0 .45rem; }
        .rtm-subtitle { color:#d3cbca; font-size:1.05rem; margin-bottom:1rem; }
        .rtm-section { color:#fff8f4; font-size:1.35rem; font-weight:700;
          border-bottom:1px solid #493a3c; padding-bottom:.35rem; margin:1.1rem 0 .7rem; }
        .rtm-note { background:#241e1f; border-left:4px solid #e34b3e; padding:.8rem 1rem;
          border-radius:.35rem; color:#eee4e2; font-size:1rem; }
        div[data-testid="stCaptionContainer"] p, .stMarkdown, label, p { color:#e1d9d8; }
        div[data-testid="stCaptionContainer"] p { font-size:.9rem; }
        .stButton button, .stDownloadButton button { border-color:#a83c34; }
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
