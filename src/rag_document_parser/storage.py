from __future__ import annotations

import hashlib
import hmac
import http.client
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    prefix: str = ""
    region: str = "us-east-1"


def put_object(cfg: S3Config, key: str, data: bytes, content_type: str) -> str:
    full_key = f"{cfg.prefix.strip('/')}/{key}" if cfg.prefix else key
    now = datetime.now(timezone.utc)
    dt = now.strftime("%Y%m%dT%H%M%SZ")
    date = dt[:8]

    parsed = urllib.parse.urlparse(cfg.endpoint)
    host = parsed.netloc
    scheme = parsed.scheme
    uri = f"/{cfg.bucket}/{full_key}"
    payload_hash = _sha256_hex(data)

    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{dt}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        ["PUT", uri, "", canonical_headers, signed_headers, payload_hash]
    )

    credential_scope = f"{date}/{cfg.region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            dt,
            credential_scope,
            _sha256_hex(canonical_request.encode()),
        ]
    )
    signing_key = _hmac_sha256(
        _hmac_sha256(
            _hmac_sha256(
                _hmac_sha256(f"AWS4{cfg.secret_key}".encode(), date.encode()),
                cfg.region.encode(),
            ),
            b"s3",
        ),
        b"aws4_request",
    )
    signature = _hmac_sha256(signing_key, string_to_sign.encode()).hex()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={cfg.access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    conn_cls = (
        http.client.HTTPSConnection
        if scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(host)
    try:
        conn.request(
            "PUT",
            uri,
            body=data,
            headers={
                "Host": host,
                "Content-Type": content_type,
                "x-amz-date": dt,
                "x-amz-content-sha256": payload_hash,
                "Authorization": authorization,
                "Content-Length": str(len(data)),
            },
        )
        response = conn.getresponse()
        response.read()
    finally:
        conn.close()

    if response.status not in (200, 204):
        raise RuntimeError(
            f"S3 PUT failed: {response.status} {response.reason} (key={full_key})"
        )
    return f"s3://{cfg.bucket}/{full_key}"


def public_url_for_s3_uri(uri: str, endpoint: str) -> str:
    bucket, key = _parse_s3_uri(uri)
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"public endpoint must be an HTTP(S) URL: {endpoint}")

    path_prefix = parsed.path.rstrip("/")
    bucket_path = urllib.parse.quote(bucket, safe="")
    key_path = urllib.parse.quote(key, safe="/")
    path = f"{path_prefix}/{bucket_path}/{key_path}" if path_prefix else f"/{bucket_path}/{key_path}"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"expected s3://bucket/key URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()
