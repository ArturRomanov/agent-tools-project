from __future__ import annotations

from contextvars import ContextVar

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_path_var: ContextVar[str | None] = ContextVar("request_path", default=None)
_request_method_var: ContextVar[str | None] = ContextVar("request_method", default=None)


def set_request_context(request_id: str, path: str, method: str) -> None:
    _request_id_var.set(request_id)
    _request_path_var.set(path)
    _request_method_var.set(method)


def clear_request_context() -> None:
    _request_id_var.set(None)
    _request_path_var.set(None)
    _request_method_var.set(None)


def get_request_id() -> str | None:
    return _request_id_var.get()


def get_request_path() -> str | None:
    return _request_path_var.get()


def get_request_method() -> str | None:
    return _request_method_var.get()
