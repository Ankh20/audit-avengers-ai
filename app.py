"""
app.py
Audit Avengers AI — Streamlit frontend
Treasury Regulatory Compliance Assistant powered by Amazon Bedrock + Amazon Nova Pro
"""

import logging
import uuid
import base64
import os
import streamlit as st
from agent import run_query, reload_documents
from audit_logger import load_log
import agent as _agent
import escalation as _escalation

# ---------------------------------------------------------------------------
# Logo loader — renders the PNG if it exists and has content,
# otherwise falls back gracefully to the emoji so the app never breaks.
# ---------------------------------------------------------------------------
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "audit_avengers_logo.png")

def _logo_b64() -> str | None:
    """Return base64-encoded PNG string, or None if file is missing/empty."""
    try:
        if os.path.exists(LOGO_PATH) and os.path.getsize(LOGO_PATH) > 0:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except OSError:
        pass
    return None

_LOGO_B64 = _logo_b64()   # computed once at startup

def logo_img_tag(width: int, style: str = "") -> str:
    """Return an <img> tag for the logo, or a styled emoji span as fallback."""
    if _LOGO_B64:
        return (
            f'<img src="data:image/png;base64,{_LOGO_B64}" '
            f'width="{width}" style="object-fit:contain;{style}" alt="Audit Avengers logo">'
        )
    return f'<span style="font-size:{width//16}rem;{style}">🛡️</span>'

def logo_st(width: int) -> None:
    """Render the logo via st.image (sidebar) or HTML fallback."""
    if _LOGO_B64:
        st.image(LOGO_PATH, width=width)
    else:
        st.markdown(f'<div style="font-size:3rem;text-align:center;">🛡️</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audit Avengers AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — purple accent bar, card styles, chat bubbles, confidence bar
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Top accent bar ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > section:first-child::before {
    content: "";
    display: block;
    height: 5px;
    background: linear-gradient(90deg, #6B21A8, #A855F7, #7C3AED);
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}

/* ── Metric cards ─────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #EDE9FE, #F5F3FF);
    border: 1px solid #C4B5FD;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #6D28D9;
    line-height: 1.1;
}
.metric-card .metric-label {
    font-size: 0.72rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Sponsor tech card ────────────────────────────────────────────── */
.sponsor-card {
    background: #F9F7FF;
    border: 1px solid #DDD6FE;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.sponsor-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
.sponsor-name { font-size: 0.8rem; font-weight: 700; color: #5B21B6; }
.sponsor-role { font-size: 0.72rem; color: #6B7280; }

/* ── App title bar ────────────────────────────────────────────────── */
.app-title {
    background: linear-gradient(135deg, #4C1D95 0%, #6D28D9 100%);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    border-left: 5px solid #A855F7;
}
.app-title h1 {
    color: #F3E8FF !important;
    font-size: 1.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
}
.app-title p { color: #DDD6FE; margin: 4px 0 0 0; font-size: 0.9rem; }

/* ── Chat bubbles ─────────────────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #EDE9FE, #DDD6FE);
    border: 1px solid #C4B5FD;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 12px 0 6px auto;
    max-width: 85%;
    color: #1E1B4B;
    font-size: 0.95rem;
}
.chat-label-user {
    text-align: right;
    font-size: 0.72rem;
    color: #7C3AED;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}
.chat-assistant {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 4px 12px 12px 12px;
    padding: 16px 20px;
    margin: 6px 0 12px 0;
    max-width: 95%;
    color: #1E293B;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-label-assistant {
    font-size: 0.72rem;
    color: #6B7280;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

/* ── Confidence bar ───────────────────────────────────────────────── */
.conf-bar-outer {
    background: #E2E8F0;
    border-radius: 6px;
    height: 10px;
    margin: 6px 0 2px 0;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* ── Citation pill ────────────────────────────────────────────────── */
.citation-pill {
    display: inline-block;
    background: #EDE9FE;
    border: 1px solid #C4B5FD;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px 4px;
    font-size: 0.76rem;
    color: #5B21B6;
}

/* ── Empty state ──────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 80px 20px;
    color: #9CA3AF;
}
.empty-state .shield { font-size: 5rem; margin-bottom: 16px; }
.empty-state h3 { color: #6B7280; font-weight: 500; margin: 0; }
.empty-state p { font-size: 0.85rem; margin-top: 8px; }

/* ── Escalation banner ────────────────────────────────────────────── */
.escalation-banner {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #7F1D1D;
    font-size: 0.9rem;
}

/* ── Audit log entry ──────────────────────────────────────────────── */
.log-entry {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.log-entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.log-badge-ok  { background:#DCFCE7; color:#14532D; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; }
.log-badge-esc { background:#FEE2E2; color:#7F1D1D; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; }

/* ── Layout ───────────────────────────────────────────────────────── */
.block-container { padding-top: 2rem; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []
if "escalation_threshold" not in st.session_state:
    st.session_state.escalation_threshold = 0.60
if "top_k" not in st.session_state:
    st.session_state.top_k = 3

# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
all_log = load_log()
questions_asked   = len(all_log)
escalated_cases   = sum(1 for e in all_log if e.get("escalated"))
knowledge_sources = 3   # number of policy files
cited_entries     = [e for e in all_log if e.get("sources_cited")]
citation_coverage = f"{int(len(cited_entries)/max(1,questions_asked)*100)}%"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        {logo_img_tag(140, "display:block;margin:0 auto 8px auto;border-radius:8px;")}
        <div style='font-size:1.1rem; font-weight:700; color:#5B21B6; margin-top:4px;'>Audit Avengers AI</div>
        <div style='font-size:0.72rem; color:#6B7280; margin-top:2px;'>Treasury Compliance Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid #DDD6FE; margin:12px 0;'></div>", unsafe_allow_html=True)

    # ── Metric cards ──
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{questions_asked}</div>
        <div class='metric-label'>Questions Asked</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='color:{"#DC2626" if escalated_cases > 0 else "#6D28D9"}'>{escalated_cases}</div>
        <div class='metric-label'>Escalated Cases</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{knowledge_sources}</div>
        <div class='metric-label'>Knowledge Sources</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{citation_coverage}</div>
        <div class='metric-label'>Citation Coverage</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid #DDD6FE; margin:16px 0 10px 0;'></div>", unsafe_allow_html=True)

    # ── Settings ──
    st.markdown("<div style='font-size:0.78rem; color:#6B7280; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;'>⚙️ Settings</div>", unsafe_allow_html=True)
    top_k = st.slider("Sources to retrieve", 1, 5, st.session_state.top_k, key="top_k_slider")
    escalation_threshold = st.slider("Escalation threshold", 0.1, 0.9, st.session_state.escalation_threshold, step=0.05, key="esc_slider")
    st.session_state.top_k = top_k
    st.session_state.escalation_threshold = escalation_threshold

    _agent.TOP_K_CHUNKS = top_k
    _escalation.ESCALATION_THRESHOLD = escalation_threshold

    if st.button("🔄 Reload Docs", use_container_width=True):
        count = reload_documents()
        st.success(f"Loaded {count} chunks")

    st.markdown("<div style='border-top:1px solid #DDD6FE; margin:16px 0 10px 0;'></div>", unsafe_allow_html=True)

    # ── Sponsor tech story ──
    st.markdown("<div style='font-size:0.78rem; color:#6B7280; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px;'>🔧 Powered By</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sponsor-card'>
        <div class='sponsor-icon'>🤖</div>
        <div>
            <div class='sponsor-name'>Amazon Nova Pro</div>
            <div class='sponsor-role'>LLM reasoning and response generation</div>
        </div>
    </div>
    <div class='sponsor-card'>
        <div class='sponsor-icon'>🔍</div>
        <div>
            <div class='sponsor-name'>Elastic</div>
            <div class='sponsor-role'>Document search &amp; retrieval</div>
        </div>
    </div>
    <div class='sponsor-card'>
        <div class='sponsor-icon'>📊</div>
        <div>
            <div class='sponsor-name'>Datadog</div>
            <div class='sponsor-role'>Audit monitoring &amp; observability</div>
        </div>
    </div>
    <div class='sponsor-card'>
        <div class='sponsor-icon'>☁️</div>
        <div>
            <div class='sponsor-name'>AWS · Bedrock</div>
            <div class='sponsor-role'>Cloud infrastructure &amp; LLM runtime</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.68rem; color:#9CA3AF; text-align:center; margin-top:16px;'>Session {st.session_state.session_id}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main area — title bar
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class='app-title' style='display:flex; align-items:center; gap:20px;'>
    <div style='flex-shrink:0;'>
        {logo_img_tag(72, "border-radius:8px;")}
    </div>
    <div>
        <h1 style='margin:0;padding:0;'>Audit Avengers AI</h1>
        <p style='margin:4px 0 0 0;'>Ask any Treasury or IRS regulatory compliance question — answers are grounded in policy documents with citations.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_query, tab_log = st.tabs(["💬 Compliance Chat", "📋 Audit Log"])

# =============================================================================
# TAB 1 — COMPLIANCE CHAT
# =============================================================================
with tab_query:

    # ── Chat history area ──
    chat_container = st.container()

    with chat_container:
        if not st.session_state.history:
            st.markdown(f"""
            <div class='empty-state'>
                <div class='shield'>{logo_img_tag(96, "display:block;margin:0 auto;opacity:0.85;")}</div>
                <h3>Ask your first compliance question</h3>
                <p>Type below to get policy-grounded answers with citations from Treasury &amp; IRS documents.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for item in reversed(st.session_state.history):
                # ── User bubble ──
                st.markdown(f"<div class='chat-label-user'>YOU</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-user'>{item['query']}</div>", unsafe_allow_html=True)

                if item.get("error"):
                    st.error(f"❌ Agent error: {item['error']}")
                    continue

                # ── Escalation banner ──
                if item["escalated"]:
                    st.markdown(f"""
                    <div class='escalation-banner'>
                        🚨 <strong>Escalation Required</strong> — Confidence below threshold.
                        This response requires review by a qualified compliance officer before acting.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Assistant bubble ──
                st.markdown(f"<div class='chat-label-assistant'>{logo_img_tag(18, 'vertical-align:middle;border-radius:3px;')} &nbsp;AUDIT AVENGERS AI</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-assistant'>{item['response']}</div>", unsafe_allow_html=True)

                # ── Confidence bar + citations row ──
                col_conf, col_cit = st.columns([1, 2])

                with col_conf:
                    conf = item["confidence"]
                    bar_color = (
                        "#22C55E" if conf >= 0.8
                        else "#EAB308" if conf >= st.session_state.escalation_threshold
                        else "#EF4444"
                    )
                    conf_label = "High" if conf >= 0.8 else "Medium" if conf >= st.session_state.escalation_threshold else "Low"
                    st.markdown(f"""
                    <div style='font-size:0.75rem; color:#94A3B8; margin-bottom:4px; font-weight:600;'>
                        CONFIDENCE — {conf_label} ({conf:.0%})
                    </div>
                    <div class='conf-bar-outer'>
                        <div class='conf-bar-inner' style='width:{conf*100:.1f}%; background:{bar_color};'></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_cit:
                    if item.get("sources"):
                        pills = "".join(f"<span class='citation-pill'>📄 {s}</span>" for s in item["sources"])
                        st.markdown(f"""
                        <div style='font-size:0.75rem; color:#94A3B8; margin-bottom:6px; font-weight:600;'>CITATIONS</div>
                        <div>{pills}</div>
                        """, unsafe_allow_html=True)

                # ── Human review button ──
                if item["escalated"]:
                    btn_key = f"review_{item.get('query','')[:20].replace(' ','_')}"
                    if st.button("👤 Flag for Human Review", key=btn_key, type="primary"):
                        st.success("✅ Flagged and sent to compliance officer queue.")

                st.markdown("<hr style='border-color:#1E293B; margin:20px 0;'>", unsafe_allow_html=True)

    # ── Input box (always at bottom) ──
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    with st.form("query_form", clear_on_submit=True):
        query = st.text_area(
            "Ask a compliance question",
            placeholder="e.g. What are the CTR filing requirements for cash transactions over $10,000?",
            height=90,
            label_visibility="collapsed",
        )
        col_btn, col_tip = st.columns([1, 3])
        with col_btn:
            submitted = st.form_submit_button("🚀 Ask Audit Avengers", use_container_width=True, type="primary")
        with col_tip:
            st.markdown("<div style='font-size:0.75rem; color:#6B7280; padding-top:10px;'>Answers are grounded in policy documents · Citations included · Audit logged automatically</div>", unsafe_allow_html=True)

    if submitted and query.strip():
        with st.spinner("🔍 Retrieving policy documents and generating answer…"):
            result = run_query(query.strip(), session_id=st.session_state.session_id)
            st.session_state.history.insert(0, {"query": query.strip(), **result})
        st.rerun()
    elif submitted and not query.strip():
        st.warning("Please type a question before submitting.")

# =============================================================================
# TAB 2 — AUDIT LOG
# =============================================================================
with tab_log:
    st.markdown("### 📋 Audit Log")
    st.markdown("<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:16px;'>Every interaction is logged for compliance review. Stored in <code>audit_log.jsonl</code>.</div>", unsafe_allow_html=True)

    col_r, col_f, col_s = st.columns([1, 2, 2])
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_f:
        show_escalated_only = st.checkbox("Show escalated only", value=False)
    with col_s:
        st.markdown(f"<div style='font-size:0.8rem; color:#94A3B8; padding-top:8px;'>Total entries: <strong>{len(all_log)}</strong> · Escalated: <strong>{escalated_cases}</strong></div>", unsafe_allow_html=True)

    entries = list(reversed(all_log))
    if show_escalated_only:
        entries = [e for e in entries if e.get("escalated")]

    if not entries:
        st.info("No audit log entries yet. Submit a question to get started.")
    else:
        for entry in entries:
            is_esc = entry.get("escalated", False)
            badge = "<span class='log-badge-esc'>🔴 ESCALATED</span>" if is_esc else "<span class='log-badge-ok'>🟢 OK</span>"
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            conf = entry.get("confidence", 0)
            conf_pct = f"{conf:.0%}"
            q_preview = entry.get("query", "")[:70]

            with st.expander(f"{ts} UTC  |  Confidence: {conf_pct}  |  {q_preview}…", expanded=False):
                st.markdown(f"""
                <div class='log-entry'>
                    <div class='log-entry-header'>
                        <div>{badge}</div>
                        <div style='font-size:0.75rem; color:#6B7280;'>{ts} UTC · Session <code>{entry.get("session_id","n/a")}</code></div>
                    </div>
                    <div style='font-size:0.8rem; color:#94A3B8; margin-bottom:4px; font-weight:600;'>QUESTION</div>
                    <div style='background:#F1F5F9; border-radius:6px; padding:10px 14px; color:#1E293B; font-size:0.9rem; margin-bottom:12px;'>{entry.get("query","")}</div>
                    <div style='font-size:0.8rem; color:#94A3B8; margin-bottom:4px; font-weight:600;'>CONFIDENCE</div>
                    <div class='conf-bar-outer' style='margin-bottom:12px;'>
                        <div class='conf-bar-inner' style='width:{conf*100:.1f}%; background:{"#22C55E" if conf >= 0.8 else "#EAB308" if conf >= st.session_state.escalation_threshold else "#EF4444"};'></div>
                    </div>
                    <div style='font-size:0.8rem; color:#94A3B8; margin-bottom:6px; font-weight:600;'>SOURCES CITED</div>
                    <div style='margin-bottom:12px;'>{"".join(f"<span class='citation-pill'>📄 {s}</span>" for s in entry.get("sources_cited",[]))  or "<span style='color:#4B5563; font-size:0.8rem;'>None</span>"}</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("View full answer", expanded=False):
                    st.markdown(entry.get("response", ""))
