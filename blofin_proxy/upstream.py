"""aiohttp forwarding client for upstream BloFin REST, gated by the token bucket."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
from multidict import CIMultiDict

from .limiter import TokenBucket

logger = logging.getLogger(__name__)

# Describe the bytes on the wire, not the decoded body `request()` hands back.
_BODY_ENCODING_HEADERS = {"content-encoding", "content-length"}


class UpstreamClient:
    """Forwards HTTP calls to the upstream BloFin REST API through the limiter.

    Counts every upstream request so ``/health`` can prove traffic collapsed.

    Two sessions, deliberately — the two forwarding modes want opposite things
    from compression:

    * ``_verbatim`` (``auto_decompress=False``) backs :meth:`stream`, so the
      catch-all can copy upstream bytes AND headers through untouched: a gzip
      reply reaches the client still gzipped, still labelled as such.
    * ``_decoding`` (``auto_decompress=True``) backs :meth:`request`, whose
      callers all either ``json.loads`` the body or re-serve it under headers
      of their own. Sharing the verbatim session with them was the bug:
      aiohttp sends ``Accept-Encoding: gzip`` by default, BloFin (via
      Cloudflare) answers ``content-encoding: gzip``, and the raw ``\\x1f\\x8b``
      body blew up ``json.loads`` with "can't decode byte 0x8b in position 1".
    """

    def __init__(self, base_url: str, limiter: TokenBucket, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self.request_count = 0
        self._verbatim: aiohttp.ClientSession | None = None
        self._decoding: aiohttp.ClientSession | None = None
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def start(self) -> None:
        if self._verbatim is None or self._verbatim.closed:
            self._verbatim = aiohttp.ClientSession(
                timeout=self._timeout, auto_decompress=False
            )
        if self._decoding is None or self._decoding.closed:
            self._decoding = aiohttp.ClientSession(
                timeout=self._timeout, auto_decompress=True
            )

    async def close(self) -> None:
        for session in (self._verbatim, self._decoding):
            if session is not None and not session.closed:
                await session.close()

    def _url(self, path: str, query_string: str | None) -> str:
        url = self.base_url + path
        if query_string:
            url += "?" + query_string
        return url

    async def request(
        self,
        method: str,
        path: str,
        query_string: str | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, aiohttp.typedefs.LooseHeaders, bytes]:
        """Buffered request; returns ``(status, headers, body)`` with the body
        already DECOMPRESSED, ready to parse or re-serve.

        Content-Encoding and Content-Length are dropped from the returned
        headers on purpose: both describe the compressed bytes upstream sent,
        so passing them along with the decoded body would mislabel it (a
        caller re-serving that pair makes a client try to gunzip plain JSON).
        Kept as a CIMultiDict so header lookups stay case-insensitive.
        """
        async with self._open(
            method, path, query_string, body, headers, decode=True
        ) as resp:
            payload = await resp.read()
            clean = CIMultiDict(
                (k, v)
                for k, v in resp.headers.items()
                if k.lower() not in _BODY_ENCODING_HEADERS
            )
            return resp.status, clean, payload

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        query_string: str | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Streamed request; yields the live upstream response for chunk copying.

        Bytes and headers stay exactly as upstream sent them (see the class
        docstring) — the caller forwards both together, so a compressed body
        travels with its own Content-Encoding intact.
        """
        async with self._open(
            method, path, query_string, body, headers, decode=False
        ) as resp:
            yield resp

    @asynccontextmanager
    async def _open(
        self,
        method: str,
        path: str,
        query_string: str | None,
        body: bytes | None,
        headers: dict[str, str] | None,
        *,
        decode: bool,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Rate-limited, counted upstream call on the session `decode` selects."""
        await self.start()
        await self.limiter.acquire()
        self.request_count += 1
        url = self._url(path, query_string)
        logger.info("upstream #%d %s %s", self.request_count, method, url)
        session = self._decoding if decode else self._verbatim
        async with session.request(method, url, data=body, headers=headers) as resp:
            yield resp
