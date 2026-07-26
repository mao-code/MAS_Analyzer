#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from itertools import combinations
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("stabletoolbench_virtual_server")


def _standardize(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_]", "_", str(value or ""))
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = f"get_{result}"
    return result


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_key_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return text


def _normalize_key(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_key(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_key(item) for item in value]
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return str(value)
    return str(value)


def _lookup_token(value: Any) -> str:
    return json.dumps(_normalize_key(value), ensure_ascii=False, sort_keys=True)


def _subset_lookup_tokens(value: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict) or len(value) < 2:
        return []

    items = list(value.items())
    tokens: list[str] = []
    # Try the most-specific subsets first. This handles optional/default fields
    # that are present in model calls but absent from the recorded cache key.
    for subset_size in range(len(items) - 1, 0, -1):
        for indexes in combinations(range(len(items)), subset_size):
            subset = {items[index][0]: items[index][1] for index in indexes}
            tokens.append(_lookup_token(subset))
    return tokens


@lru_cache(maxsize=512)
def _cache_entries(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    raw = _load_json_file(path)
    if not isinstance(raw, dict):
        return {}

    entries: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        entries[_lookup_token(_parse_key_text(raw_key))] = raw_value
        if isinstance(raw_key, str):
            entries[raw_key.strip()] = raw_value
    return entries


class StableToolBenchCacheServer:
    def __init__(self, *, cache_root: Path) -> None:
        self.cache_root = cache_root.expanduser().resolve()

    def lookup(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        token = _lookup_token(tool_input)
        for candidate in self._candidate_paths(category=category, tool_name=tool_name, api_name=api_name):
            if not candidate.exists():
                continue
            cache = _cache_entries(str(candidate))
            if token in cache:
                return self._coerce_response(cache[token], source=candidate)
            for subset_token in _subset_lookup_tokens(tool_input):
                if subset_token in cache:
                    return self._coerce_response(cache[subset_token], source=candidate)
            if not tool_input and len(cache) == 1:
                only_value = next(iter(cache.values()))
                return self._coerce_response(only_value, source=candidate)
        return {
            "error": (
                "StableToolBench cache miss for "
                f"category={category!r}, tool={tool_name!r}, api={api_name!r}, input={tool_input!r}"
            ),
            "response": "",
        }

    def _candidate_paths(self, *, category: str, tool_name: str, api_name: str) -> list[Path]:
        category_dir = self.cache_root / str(category or "").strip()
        standardized_tool = _standardize(tool_name)
        standardized_category = _standardize(category)
        raw_category = str(category or "").strip()
        tool_dir = category_dir / f"{standardized_tool}_for_{standardized_category}"
        tool_dir_raw_category = category_dir / f"{standardized_tool}_for_{raw_category}"
        api_filename = f"{_standardize(api_name)}.json"
        api_filename_raw = f"{str(api_name or '').strip()}.json"
        return [
            tool_dir / api_filename,
            tool_dir / api_filename_raw,
            tool_dir_raw_category / api_filename,
            tool_dir_raw_category / api_filename_raw,
            category_dir / api_filename,
            category_dir / api_filename_raw,
            self.cache_root / api_filename,
            self.cache_root / api_filename_raw,
        ]

    @staticmethod
    def _coerce_response(raw_value: Any, *, source: Path) -> dict[str, Any]:
        if isinstance(raw_value, dict) and "response" in raw_value:
            return raw_value
        return {
            "error": "",
            "response": raw_value,
            "source": str(source),
        }


class StableToolBenchHandler(BaseHTTPRequestHandler):
    server: StableToolBenchHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/healthz", "/virtual/healthz"}:
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "cache_root": str(self.server.cache_server.cache_root),
                    "api_path": self.server.api_path,
                },
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != self.server.api_path.rstrip("/"):
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            content_length = 0
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON request body"})
            return

        if not isinstance(payload, dict):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "JSON body must be an object"})
            return

        tool_input = payload.get("tool_input", {})
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except Exception:
                try:
                    tool_input = ast.literal_eval(tool_input)
                except Exception:
                    tool_input = {"raw_input": tool_input}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}

        response = self.server.cache_server.lookup(
            category=str(payload.get("category", "")),
            tool_name=str(payload.get("tool_name", "")),
            api_name=str(payload.get("api_name", "")),
            tool_input=tool_input,
        )
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def _write_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StableToolBenchHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[StableToolBenchHandler],
        *,
        cache_server: StableToolBenchCacheServer,
        api_path: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.cache_server = cache_server
        self.api_path = api_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight cache-backed StableToolBench virtual server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--path", default="/virtual")
    parser.add_argument(
        "--cache-root",
        default="benchmark/stabletoolbench/tool_response_cache",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_root = Path(args.cache_root)
    if not cache_root.exists():
        raise FileNotFoundError(f"StableToolBench cache root not found: {cache_root}")

    cache_server = StableToolBenchCacheServer(cache_root=cache_root)
    httpd = StableToolBenchHTTPServer(
        (args.host, args.port),
        StableToolBenchHandler,
        cache_server=cache_server,
        api_path=str(args.path),
    )
    LOGGER.info(
        "Serving StableToolBench cache on http://%s:%s%s using %s",
        args.host,
        args.port,
        args.path,
        cache_root.resolve(),
    )
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
