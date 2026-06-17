from __future__ import annotations


def test_public_url_for_s3_uri_uses_path_style_public_endpoint():
    from rag_document_parser.storage import public_url_for_s3_uri

    url = public_url_for_s3_uri(
        "s3://rag-assets/hwpx-validation/run 1/assets/img-0001.png",
        "http://203.0.113.10:10190",
    )

    assert (
        url
        == "http://203.0.113.10:10190/rag-assets/hwpx-validation/run%201/assets/img-0001.png"
    )


def test_public_url_for_s3_uri_preserves_endpoint_path_prefix():
    from rag_document_parser.storage import public_url_for_s3_uri

    url = public_url_for_s3_uri(
        "s3://rag-assets/doc/assets/img-0001.png",
        "https://files.example.com/minio/",
    )

    assert url == "https://files.example.com/minio/rag-assets/doc/assets/img-0001.png"
