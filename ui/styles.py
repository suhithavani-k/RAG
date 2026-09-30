def apply_styles() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        :root { --ink:#182522; --muted:#66736f; --paper:#f5f4ee; --line:#dfe3dc; --green:#1e5b4b; --lime:#d8eb87; --coral:#d65b43; }
        .stApp { background:var(--paper); color:var(--ink); font-family:'Avenir Next', 'Segoe UI', sans-serif; }
        [data-testid="stHeader"] { background:rgba(245,244,238,.92); }
        [data-testid="stSidebar"] { background:#e9ece4; border-right:1px solid var(--line); }
        h1,h2,h3 { color:var(--ink); letter-spacing:0; }
        h1 { font-weight:800; }
        .eyebrow { color:var(--green); text-transform:uppercase; font:500 11px 'Cascadia Code',Consolas,monospace; letter-spacing:0; }
        .hero-title { font-size:36px; line-height:1.08; font-weight:800; margin:.3rem 0 .5rem; }
        .hero-copy { color:var(--muted); max-width:680px; margin-bottom:1.1rem; }
        .status-strip { display:flex; flex-wrap:wrap; gap:8px; margin:.8rem 0 1.4rem; }
        .status-pill { border:1px solid var(--line); border-radius:4px; padding:7px 10px; background:#fffefa; font-size:12px; color:var(--ink); }
        .status-pill strong { color:var(--green); font-weight:700; }
        .section-label { color:var(--muted); font:500 11px 'Cascadia Code',Consolas,monospace; text-transform:uppercase; }
        .metric-card { background:#fffefa; border:1px solid var(--line); border-radius:5px; padding:12px 14px; min-height:78px; }
        .metric-value { color:var(--ink); font-size:22px; font-weight:800; }
        .metric-name { color:var(--muted); font-size:11px; }
        .source-meta { color:var(--muted); font:12px 'Cascadia Code',Consolas,monospace; }
        .stButton>button[kind="primary"] { background:var(--green); border-color:var(--green); }
        .stButton>button { border-radius:4px; }
        div[data-testid="stChatMessage"] { background:#fffefa; border:1px solid var(--line); border-radius:6px; }
        div[data-testid="stExpander"] { border-color:var(--line); border-radius:5px; }
        code { font-family:'Cascadia Code',Consolas,monospace; }
        @media(max-width:700px) { .hero-title {font-size:29px;} }
        </style>
        """,
        unsafe_allow_html=True,
    )
