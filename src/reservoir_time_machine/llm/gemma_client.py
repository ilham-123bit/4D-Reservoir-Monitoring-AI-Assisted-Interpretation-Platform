"""Bounded OpenAI-compatible transport. No automatic connection or retry loop."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from typing import Any
from ..settings import LLMSettings


class ServiceBusy(RuntimeError):
    """Another request from this HPC account holds the shared-service slot."""


@contextmanager
def request_slot(settings):
    directory = Path(settings.lock_directory).expanduser() if settings.lock_directory else Path.home()/'.cache/rtm-gemma'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'request.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ServiceBusy('Another account request is in progress; no request sent') from exc
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def _client(settings: LLMSettings):
    if not settings.base_url: raise RuntimeError('No LLM endpoint configured. Set RTM_LLM_BASE_URL to enable the approved service.')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Optional OpenAI-compatible client dependency unavailable') from exc
    return OpenAI(api_key=settings.api_key or 'not-needed',base_url=settings.base_url,
                  timeout=settings.timeout_seconds,max_retries=0)


def test_connection(settings: LLMSettings | None = None) -> dict[str,Any]:
    """Explicit model-list call only; it does not validate generation or science."""
    settings = settings or LLMSettings.from_environment()
    if not settings.base_url:
        return {'connected':False,'configured_model':settings.model,'error':'No LLM service endpoint is configured.'}
    try:
        with request_slot(settings): response = _client(settings).models.list()
        models = [item.id for item in response.data]
        return {'connected':True,'configured_model':settings.model,'configured_model_listed':settings.model in models,
                'available_model_count':len(models)}
    except Exception as exc:
        return {'connected':False,'configured_model':settings.model,'error':type(exc).__name__+': configured endpoint unavailable'}


def explain_evidence(evidence: dict[str,Any], settings: LLMSettings | None = None, *,
                     max_payload_bytes: int = 24000, structured: bool = False, response_schema=None) -> str:
    """Transport compact JSON only. Classification/consent gate lives in the typed provider."""
    settings = settings or LLMSettings.from_environment()
    payload = json.dumps(evidence,ensure_ascii=False,allow_nan=False)
    if len(payload.encode()) > min(max_payload_bytes,24000):
        raise ValueError('Evidence exceeds bounded payload; arrays/volumes are not permitted')
    if not settings.base_url: raise RuntimeError('No LLM endpoint configured. Set RTM_LLM_BASE_URL to enable the approved service.')
    if evidence.get('data_classification','unknown') not in {'synthetic','public'} and not settings.real_data_enabled:
        raise PermissionError('Data-sharing policy gate blocks real or unclassified evidence')
    options = {}
    if structured:
        if response_schema is not None and settings.structured_mode=='json_schema':
            options['response_format']={'type':'json_schema','json_schema':{'name':'grounded_finding','strict':True,'schema':response_schema}}
        else: options['response_format']={'type':'json_object'}
    with request_slot(settings):
        response = _client(settings).chat.completions.create(model=settings.model,
            messages=[{'role':'system','content':(
                'You explain supplied geoscience evidence only. Return the required JSON, no prose. '
                'Use EVERY required_code exactly once and only supplied evidence IDs of the required fact type. '
                'Select at most four supplied numeric measurements and copy values exactly. '
                'Do not invent facts, fluid movement, pressure depletion, unswept oil or recovery certainty. '
                'All reservoir mechanisms are hypotheses; preserve registration, well-tie, support and dependence limits. '
                'A successful request is not scientific validation.')},
                {'role':'user','content':payload}],temperature=0,max_tokens=settings.max_tokens,**options)
    if getattr(response,'model',settings.model)!=settings.model:
        raise ValueError('Provider returned an unexpected model identifier')
    choice = response.choices[0]
    if getattr(choice,'finish_reason',None)=='length': raise ValueError('Provider response truncated at token budget')
    content = choice.message.content
    if not isinstance(content,str) or not content.strip() or len(content.encode())>24000:
        raise ValueError('Empty or oversized provider response')
    return content
