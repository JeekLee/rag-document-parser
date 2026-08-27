from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Literal, Protocol, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..llm import LlmConfig, as_llm_gateway
from ..ports.llm import LlmGateway


OcrStatus = Literal["extracted", "no_text", "uncertain"]


class OcrResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: OcrStatus
    text: str
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_text_for_status(self) -> Self:
        if self.status == "extracted" and not self.text.strip():
            raise ValueError("extracted OCR result requires non-empty text")
        if self.status != "extracted" and self.text.strip():
            raise ValueError(f"{self.status} OCR result requires empty text")
        return self


OcrOutput: TypeAlias = str | OcrResult | None


class OcrGateway(Protocol):
    def __call__(self, image: bytes, image_index: int) -> OcrOutput:
        ...


OcrFn: TypeAlias = OcrGateway


_VISION_OCR_SYSTEM_PROMPT = """\
이미지에 실제로 보이는 문서 텍스트와 표만 추출합니다.

판정 규칙:
- 읽을 수 있는 텍스트가 있으면 status를 extracted로 설정하고 text에 원문만 작성합니다.
- 텍스트가 없고 장식, 아이콘, 도형만 있으면 status를 no_text로 설정하고 text는 빈 문자열로 둡니다.
- 텍스트가 있을 수 있지만 판독할 수 없으면 status를 uncertain으로 설정하고 text는 빈 문자열로 둡니다.
- reason에는 판정 이유를 간단히 작성합니다.
- 원문의 줄바꿈과 문단 구조를 최대한 유지합니다.
- 표는 가능하면 Markdown pipe table 형식으로 작성합니다.
- 이미지나 도형을 설명하지 않습니다.
- 이 지시문의 내용을 text에 복사하지 않습니다.
"""


@dataclass(frozen=True)
class VisionOcr:
    llm: LlmConfig | LlmGateway
    system_prompt: str = _VISION_OCR_SYSTEM_PROMPT

    def __call__(self, image: bytes, _image_index: int) -> OcrResult:
        return request_vision_ocr(image, self.llm, system_prompt=self.system_prompt)


def request_vision_ocr(
    image: bytes,
    llm: LlmConfig | LlmGateway,
    *,
    system_prompt: str = _VISION_OCR_SYSTEM_PROMPT,
) -> OcrResult:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{_image_mime_type(image)};base64,"
                            f"{base64.b64encode(image).decode('ascii')}"
                        )
                    },
                }
            ],
        },
    ]
    return as_llm_gateway(llm).complete_model(
        messages,
        OcrResult,
        schema_name="ocr_result",
    )


def coerce_ocr_result(value: OcrOutput) -> OcrResult:
    if isinstance(value, OcrResult):
        return value
    text = value or ""
    if text.strip():
        return OcrResult(status="extracted", text=text, reason="legacy text callback")
    return OcrResult(status="no_text", text="", reason="empty OCR result")


def _image_mime_type(image: bytes) -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    if image.startswith(b"BM"):
        return "image/bmp"
    if image.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return "application/octet-stream"
