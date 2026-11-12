"""Wide, interactive, consent-gated Gemma chat for the Reservoir Time Machine."""
from __future__ import annotations

from ..ai.gemma_copilot import (
    ask_gemma, context_identity, offline_answer, validate_question,
    REGISTRATION_WARNING,
)
from ..settings import LLMSettings


def render_gemma_copilot(context):
    """Show a functional chat across source-selected investigations.

    Session state is scoped to an evidence fingerprint; replies for one finding
    cannot silently carry over when a different finding or source is selected.
    """
    import streamlit as st

    st.markdown("## Gemma · Reservoir Intelligence Copilot")
    st.caption(
        "Ask about paired seismic changes, ML evidence, well-marker hypotheses, "
        "survey repeatability, uncertainty, and next reservoir investigations."
    )
    st.warning(REGISTRATION_WARNING)
    try:
        settings = LLMSettings.from_environment()
        configuration_error = None
    except (ValueError, OSError) as exc:
        settings = LLMSettings()
        configuration_error = type(exc).__name__
    can_send = bool(settings.base_url and settings.real_data_enabled and not configuration_error)
    if configuration_error:
        st.error(f"Gemma configuration invalid ({configuration_error}); local explanation remains available.")
    if not can_send:
        st.info(
            "Evidence-first local assistant is available. Live Gemma is offline until "
            "an approved service endpoint, data-sharing policy reference and "
            "restricted-data permission are configured."
        )
    else:
        st.success("Gemma endpoint is configured; no evidence is transmitted without explicit consent.")
    consent = st.checkbox(
        "I authorize sending this bounded, restricted scientific evidence and my question "
        "to the configured Gemma service for this conversation.",
        value=False, disabled=not can_send, key="rtm_gemma_chat_consent",
    )
    st.caption(
        "Only small validated evidence fields and your question may be sent. "
        "Raw SEG-Y/LAS, seismic cubes, private file paths, and credentials are excluded. "
        "Gemma's response is model-generated interpretation, not geological proof."
    )
    identity = context_identity(context)
    if st.session_state.get("rtm_gemma_chat_context") != identity:
        st.session_state["rtm_gemma_chat_context"] = identity
        st.session_state["rtm_gemma_chat_messages"] = []
    history = st.session_state.setdefault("rtm_gemma_chat_messages", [])
    left, right = st.columns([4, 1])
    with left:
        st.caption(
            f"Active source context: {len(context['evidence'])} bounded evidence items "
            "· no verified recovery targets"
        )
    with right:
        if st.button("Clear chat", key="rtm_gemma_chat_clear"):
            history.clear()
            st.rerun()
    with st.container(height=450, border=True):
        if not history:
            with st.chat_message("assistant"):
                st.markdown(
                    "I can compare the measured 1995/2006 responses, explain "
                    "why a conditional I-100 case is or is not convincing, and "
                    "prioritize evidence-gathering. Try: **What changed at Well-6?** "
                    "or **Where should we investigate next?**"
                )
        for entry in history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if entry["role"] == "assistant":
                    st.caption(
                        f"{entry.get('mode', 'local')} · evidence: "
                        + ", ".join(entry.get("evidence_ids", []))
                    )
                    st.caption(entry.get("uncertainty", REGISTRATION_WARNING))
    question = st.chat_input(
        "Ask Gemma about the selected reservoir investigation…",
        key="rtm_gemma_copilot_question",
    )
    if question:
        try:
            question = validate_question(question)
        except ValueError as exc:
            st.warning(str(exc))
            return
        previous = list(history)
        history.append({"role": "user", "content": question})
        try:
            if can_send and consent:
                response = ask_gemma(
                    question, context, settings, consent=True, conversation=previous,
                )
            else:
                response = offline_answer(question, context)
        except Exception as exc:
            # Never print endpoint, request data, credentials, or private file paths.
            response = offline_answer(question, context)
            response["mode"] = "deterministic_local_fallback"
            response["uncertainty"] += (
                f" Live model request was unavailable ({type(exc).__name__})."
            )
        history.append({
            "role": "assistant",
            "content": response["answer"],
            "mode": response["mode"],
            "evidence_ids": response["evidence_ids"],
            "uncertainty": response["uncertainty"],
        })
        # Keep conversation bounded and ensure the latest prompt is visible.
        st.session_state["rtm_gemma_chat_messages"] = history[-16:]
        st.rerun()
