from __future__ import annotations

import json


def test_chat_json_includes_gemini_disabled_thinking_option(monkeypatch):
    import rag_document_parser.llm as llm_module
    from rag_document_parser import GeminiLlmConfig

    requests = _capture_chat_requests(monkeypatch, llm_module)

    llm_module.chat_json(
        "return ok",
        GeminiLlmConfig(
            url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            api_key="secret",
            model="gemini-2.5-flash-lite",
            thinking="disabled",
            timeout=5.0,
        ),
    )

    request, timeout = requests[0]
    assert timeout == 5.0
    body = json.loads(request.data.decode("utf-8"))
    assert body["reasoning_effort"] == "none"


def test_chat_json_omits_gemini_thinking_option_by_default(monkeypatch):
    import rag_document_parser.llm as llm_module
    from rag_document_parser import GeminiLlmConfig

    requests = _capture_chat_requests(monkeypatch, llm_module)

    llm_module.chat_json(
        "return ok",
        GeminiLlmConfig(
            url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            api_key="secret",
            model="gemini-2.5-flash-lite",
        ),
    )

    body = json.loads(requests[0][0].data.decode("utf-8"))
    assert "reasoning_effort" not in body


def test_chat_json_includes_qwen_chat_template_thinking_option(monkeypatch):
    import rag_document_parser.llm as llm_module
    from rag_document_parser import QwenLlmConfig

    requests = _capture_chat_requests(monkeypatch, llm_module)

    llm_module.chat_json(
        "return ok",
        QwenLlmConfig(
            url="http://llm.test/v1",
            api_key="secret",
            model="qwen3-vl-30b-a3b",
            thinking="disabled",
        ),
    )

    body = json.loads(requests[0][0].data.decode("utf-8"))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_json_can_use_qwen_top_level_thinking_option(monkeypatch):
    import rag_document_parser.llm as llm_module
    from rag_document_parser import QwenLlmConfig

    requests = _capture_chat_requests(monkeypatch, llm_module)

    llm_module.chat_json(
        "return ok",
        QwenLlmConfig(
            url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key="secret",
            model="qwen-plus",
            thinking="enabled",
            thinking_parameter="enable_thinking",
        ),
    )

    body = json.loads(requests[0][0].data.decode("utf-8"))
    assert body["enable_thinking"] is True


def test_chat_json_omits_provider_options_for_gemma(monkeypatch):
    import rag_document_parser.llm as llm_module
    from rag_document_parser import GemmaLlmConfig

    requests = _capture_chat_requests(monkeypatch, llm_module)

    llm_module.chat_json(
        "return ok",
        GemmaLlmConfig(
            url="http://llm.test/v1",
            api_key="secret",
            model="gemma-3-27b-it",
        ),
    )

    body = json.loads(requests[0][0].data.decode("utf-8"))
    assert "reasoning_effort" not in body
    assert "enable_thinking" not in body
    assert "chat_template_kwargs" not in body


def _capture_chat_requests(monkeypatch, llm_module):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "{\"ok\": true}"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(llm_module.request, "urlopen", fake_urlopen)
    return requests
