# Gemma integration

The optional client uses an OpenAI-compatible service configured at runtime through `RTM_LLM_BASE_URL`, `RTM_LLM_MODEL`, `RTM_LLM_API_KEY`, and `RTM_LLM_TIMEOUT`. The endpoint has no source-code or template default. Configure it only in the authorized runtime environment; keep service addresses and credentials out of the repository. If no endpoint is configured, connection checks return a clear unavailable result and explanation requests fail with a configuration message. The API key default is a non-secret placeholder; no credentials are stored in source.

Test connectivity inside the authorized HPC environment after setting its local endpoint configuration:

```bash
python -c "from reservoir_time_machine.llm.gemma_client import test_connection; print(test_connection())"
```

Connectivity has not been tested in this local environment. A positive model-list response confirms endpoint/model listing only; it does not validate scientific reasoning. The client accepts bounded JSON summaries and rejects non-JSON arrays such as full seismic cubes. It is optional, invoked only by an explicit UI action, and does not participate in numerical detection. Explanations must separate observations, detections, interpretations, alternatives, missing evidence, and uncertainty.
