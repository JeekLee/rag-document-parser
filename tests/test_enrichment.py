from __future__ import annotations


def test_rag_chunk_enricher_batches_llm_enrichment_by_token_budget(monkeypatch):
    from rag_document_parser import Evidence, RagChunk, SourceEvidence
    import rag_document_parser.chunk.enrichment as enrichment
    from rag_document_parser.chunk.enrichment import RagChunkEnricher
    from rag_document_parser.llm import LlmConfig

    calls: list[str] = []

    def chat_fn(prompt, cfg):
        calls.append(prompt)
        ids = [
            line.removeprefix('      "id": "').removesuffix('",')
            for line in prompt.splitlines()
            if line.startswith('      "id": "') and '"chunk-id"' not in line
        ]
        return {
            "chunks": [
                {
                    "id": chunk_id,
                    "summary": f"{chunk_id} batch summary",
                    "keywords": [chunk_id, "batch"],
                    "questions": [f"{chunk_id} 질문은 무엇인가요?"],
                }
                for chunk_id in ids
            ]
        }

    chunks = [
        RagChunk(
            id=f"chunk-{index}",
            source=SourceEvidence(kind="text", text=f"source {index}"),
            evidence=Evidence(items=[]),
            summary="",
            keywords=[],
            questions=[],
            metadata={"source_unit_ids": [f"b{index}"]},
        )
        for index in range(1, 11)
    ]

    monkeypatch.setattr(enrichment, "_chunk_batch_token_cost", lambda chunk: 1, raising=False)
    enriched = RagChunkEnricher(
        llm=LlmConfig(url="http://llm.test/v1", api_key="key", model="model"),
        chat_fn=chat_fn,
        max_concurrency=1,
        batch_token_budget=8,
    ).enrich(chunks)

    assert len(calls) == 2
    assert [chunk.summary for chunk in enriched] == [
        f"chunk-{index} batch summary" for index in range(1, 11)
    ]
    assert [chunk.keywords for chunk in enriched] == [
        [f"chunk-{index}", "batch"] for index in range(1, 11)
    ]
    assert all(chunk.metadata["_enrichment"]["method"] == "llm_batch" for chunk in enriched)


def test_rag_chunk_enricher_keeps_oversized_chunk_as_single_batch(monkeypatch):
    from rag_document_parser import Evidence, RagChunk, SourceEvidence
    import rag_document_parser.chunk.enrichment as enrichment
    from rag_document_parser.chunk.enrichment import RagChunkEnricher
    from rag_document_parser.llm import LlmConfig

    calls: list[list[str]] = []

    def chat_fn(prompt, cfg):
        ids = [
            line.removeprefix('      "id": "').removesuffix('",')
            for line in prompt.splitlines()
            if line.startswith('      "id": "') and '"chunk-id"' not in line
        ]
        calls.append(ids)
        return {
            "chunks": [
                {
                    "id": chunk_id,
                    "summary": f"{chunk_id} summary",
                    "keywords": [chunk_id],
                    "questions": [f"{chunk_id} 질문은 무엇인가요?"],
                }
                for chunk_id in ids
            ]
        }

    chunks = [
        RagChunk(
            id=f"chunk-{index}",
            source=SourceEvidence(kind="text", text=f"source {index}"),
            evidence=Evidence(items=[]),
            summary="",
            keywords=[],
            questions=[],
            metadata={"source_unit_ids": [f"b{index}"]},
        )
        for index in range(1, 6)
    ]
    costs = {
        "chunk-1": 10,
        "chunk-2": 10,
        "chunk-3": 10,
        "chunk-4": 30,
        "chunk-5": 10,
    }
    monkeypatch.setattr(
        enrichment,
        "_chunk_batch_token_cost",
        lambda chunk: costs[chunk.id],
        raising=False,
    )

    enriched = RagChunkEnricher(
        llm=LlmConfig(url="http://llm.test/v1", api_key="key", model="model"),
        chat_fn=chat_fn,
        max_concurrency=1,
        batch_token_budget=25,
    ).enrich(chunks)

    assert calls == [
        ["chunk-1", "chunk-2"],
        ["chunk-3"],
        ["chunk-4"],
        ["chunk-5"],
    ]
    assert [chunk.summary for chunk in enriched] == [
        f"chunk-{index} summary" for index in range(1, 6)
    ]
