"""Render the real chat module without network, private Angsi files or Gemma."""
from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = """
import streamlit as st
from reservoir_time_machine.ai.gemma_copilot import build_copilot_context
from reservoir_time_machine.visualization.gemma_copilot import render_gemma_copilot
context = build_copilot_context(
    selected={"id":"fixture-1","kind":"qc_follow_up",
              "finding":{"array_location":[5,6,7]}}
)
render_gemma_copilot(context)
"""


class GemmaCopilotStreamlitTests(unittest.TestCase):
    def test_offline_wide_chat_renders_without_endpoint(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assertFalse(app.exception, [str(x.message) for x in app.exception])
        self.assertGreaterEqual(len(app.chat_input), 1)
        self.assertIn("Gemma", " ".join(str(x.value) for x in app.markdown))

    def test_question_gets_local_answer_without_network(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.chat_input[0].set_value("What changed in 1995 vs 2006?").run()
        self.assertFalse(app.exception, [str(x.message) for x in app.exception])
        messages = app.session_state["rtm_gemma_chat_messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["mode"], "deterministic_local")
        self.assertIn("not proven recovery targets", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
