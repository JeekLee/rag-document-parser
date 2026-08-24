from __future__ import annotations

import json

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "extracted", "text": "", "reason": "missing text"},
        {"status": "no_text", "text": "hallucinated text", "reason": "none"},
        {"status": "uncertain", "text": "maybe", "reason": "unreadable"},
    ],
)
def test_ocr_result_enforces_text_status_contract(payload):
    from rag_document_parser import OcrResult

    with pytest.raises(ValidationError):
        OcrResult.model_validate(payload)


def test_vision_ocr_requests_and_validates_strict_json_schema(monkeypatch):
    from rag_document_parser import LlmConfig, VisionOcr
    from rag_document_parser.evidence_unit_extraction import ocr as ocr_module

    requests = []

    def fake_read_response(req, cfg):
        requests.append((req, cfg))
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "no_text",
                                    "text": "",
                                    "reason": "decorative arrow only",
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        ocr_module,
        "_read_response_with_retries",
        fake_read_response,
    )
    cfg = LlmConfig(url="http://llm.test/v1", api_key="secret", model="vision")

    result = VisionOcr(cfg)(b"\x89PNG\r\n\x1a\npng", 3)

    assert result.status == "no_text"
    assert result.text == ""
    assert result.reason == "decorative arrow only"
    req, request_cfg = requests[0]
    assert request_cfg is cfg
    assert req.full_url == "http://llm.test/v1/chat/completions"
    body = json.loads(req.data.decode("utf-8"))
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == (
        ocr_module.OcrResult.model_json_schema()
    )
    assert body["messages"][0]["role"] == "system"
    user_message = body["messages"][1]
    assert user_message["role"] == "user"
    assert len(user_message["content"]) == 1
    assert user_message["content"][0]["type"] == "image_url"
    assert user_message["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
