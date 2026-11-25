from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

from livekit.agents import APIStatusError

BEIJING_WS_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
SINGAPORE_WS_BASE_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/"
REGION_TO_WS_BASE_URL = {
    "beijing": BEIJING_WS_BASE_URL,
    "cn": BEIJING_WS_BASE_URL,
    "cn-beijing": BEIJING_WS_BASE_URL,
    "singapore": SINGAPORE_WS_BASE_URL,
    "sg": SINGAPORE_WS_BASE_URL,
    "intl": SINGAPORE_WS_BASE_URL,
    "ap-southeast-1": SINGAPORE_WS_BASE_URL,
}


def resolve_base_url(base_url: str | None) -> str:
    if base_url:
        return base_url

    env_base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
    if env_base_url:
        return env_base_url

    region = os.environ.get("DASHSCOPE_REGION", "").strip().lower()
    if region:
        try:
            return REGION_TO_WS_BASE_URL[region]
        except KeyError as e:
            raise ValueError(
                "Unsupported DASHSCOPE_REGION. Use one of: "
                "beijing, cn-beijing, cn, singapore, sg, intl, ap-southeast-1."
            ) from e

    return BEIJING_WS_BASE_URL


def format_handshake_error(
    error: aiohttp.WSServerHandshakeError,
    *,
    base_url: str,
    service_name: str,
) -> tuple[str, int, str | None]:
    status_code = error.status if hasattr(error, "status") else -1
    headers = error.headers if hasattr(error, "headers") else None
    request_id = headers.get("x-request-id") if headers is not None else None

    if status_code == 401:
        hint = (
            f"{service_name} handshake failed with 401 Unauthorized. "
            "Check whether DASHSCOPE_API_KEY is valid, whether its region matches the endpoint "
            f"({base_url}), whether DASHSCOPE_BASE_URL or DASHSCOPE_REGION are set correctly, "
            "and whether the API key has workspace permission or IP whitelist restrictions."
        )
        if request_id:
            hint += f" x-request-id={request_id}."
        return hint, status_code, str(headers) if headers is not None else None

    message = f"{service_name} handshake failed: {error.message}"
    if request_id:
        message += f" x-request-id={request_id}."
    return message, status_code, str(headers) if headers is not None else None


# `task-failed` error codes by prefix. Request and account problems fail the same way on every
# attempt, so they map to a 4xx status, which livekit-agents treats as non-retryable; throttling
# keeps 429 (retryable) and server faults 5xx. Unknown codes stay retryable (-1).
_ERROR_CODE_STATUS: tuple[tuple[str, int], ...] = (
    ("InvalidApiKey", 401),
    ("AccessDenied", 403),
    ("Model.AccessDenied", 403),
    ("Arrearage", 403),
    ("InvalidParameter", 400),
    ("DataInspectionFailed", 400),
    ("BadRequest", 400),
    ("ModelNotFound", 404),
    ("Throttling", 429),
    ("InternalError", 500),
    ("SystemError", 500),
    ("ServiceUnavailable", 503),
)


def status_for_error_code(code: str) -> int:
    for prefix, status in _ERROR_CODE_STATUS:
        if code.startswith(prefix):
            return status
    return -1


def task_failed_error(payload: dict[str, Any], *, service_name: str) -> APIStatusError:
    """The APIStatusError for a `task-failed` event, classified by its error code."""
    header = payload.get("header", {})
    code = str(header.get("error_code") or "unknown")
    message = header.get("error_message") or "unknown error"
    return APIStatusError(
        message=f"{service_name} failed: {code} - {message}",
        status_code=status_for_error_code(code),
        request_id=header.get("task_id") or None,
        body=json.dumps(payload, ensure_ascii=False),
    )
