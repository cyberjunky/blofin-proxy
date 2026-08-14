"""TTL cache with single-flight coalescing, plus a WS-fed candle book."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

_BAR_UNITS = {
    "m": 60,
    "H": 3600,
    "D": 86400,
    "W": 604800,
    "M": 30 * 86400,  # BloFin '1M' calendar month, approximated for TTL math
}


def bar_to_seconds(bar: str) -> int:
    """Convert a BloFin bar value (``1m``, ``5m``, ``1H``, ``4H``, ``1D`` ...) to seconds."""
    if len(bar) < 2:
        raise ValueError(f"invalid bar: {bar!r}")
    unit = bar[-1]
    if unit not in _BAR_UNITS:
        raise ValueError(f"unknown bar unit: {bar!r}")
    amount = int(bar[:-1])
    if amount <= 0:
        raise ValueError(f"invalid bar amount: {bar!r}")
    return amount * _BAR_UNITS[unit]


def _num(value: float) -> str:
    """Format a float the way BloFin REST renders numeric strings."""
    return f"{value:.12g}"


class TTLCache:
    """Key -> (value, expiry) cache with single-flight ``get_or_fetch``.

    Concurrent misses for the same key await one shared fetch instead of each
    hitting upstream. Failed fetches are not cached.
    """

    def __init__(self) -> None:
        self._values: dict[str, tuple[Any, float]] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value if present and fresh, else ``None``."""
        entry = self._values.get(key)
        if entry is not None and entry[1] > time.monotonic():
            return entry[0]
        return None

    def put(self, key: str, value: Any, ttl: float) -> None:
        self._values[key] = (value, time.monotonic() + ttl)

    async def get_or_fetch(
        self, key: str, ttl: float, fetch: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Return the cached value, or coalesce concurrent misses into one fetch."""
        entry = self._values.get(key)
        if entry is not None and entry[1] > time.monotonic():
            self.hits += 1
            return entry[0]
        self.misses += 1
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._run(key, ttl, fetch))
            self._inflight[key] = task
        # Shield so one cancelled waiter does not cancel the shared fetch for
        # the other waiters.
        return await asyncio.shield(task)

    async def _run(self, key: str, ttl: float, fetch: Callable[[], Awaitable[Any]]) -> Any:
        try:
            value = await fetch()
            self._values[key] = (value, time.monotonic() + ttl)
            return value
        finally:
            self._inflight.pop(key, None)

    def stats(self) -> dict[str, int]:
        now = time.monotonic()
        fresh = sum(1 for _, expiry in self._values.values() if expiry > now)
        return {
            "entries": len(self._values),
            "fresh": fresh,
            "hits": self.hits,
            "misses": self.misses,
            "inflight": len(self._inflight),
        }


class CandleBook:
    """Per ``(instId, bar)`` candle storage fed by WS updates and REST seeds.

    Internally candles are ``[ts_ms, open, high, low, close, base_volume]``
    with ``ts_ms`` as int and the rest as floats, keyed by timestamp so a WS
    update for the in-progress candle replaces it in place.

    The volume slot is ALWAYS base volume — BloFin REST index 6, the same
    figure ccxt's ``parse_ohlcv`` reads. The REST seed previously stored index
    5 (contract volume) while WS updates stored ccxt's base volume, so a
    single column silently carried two different units depending on which
    source last touched a given candle.

    ``_contract_size`` holds the exchange's own contractValue per instrument
    (set once from loaded markets), used to re-expand a WS-updated candle into
    BloFin's 9-column REST shape. It is deliberately NOT inferred from the
    contract/base ratio of a seed row: BloFin rounds contract volume to whole
    contracts, so a quiet candle gives a badly quantised ratio (DOGE-USDT
    measured 9/9910 = 0.00091 against a true 0.001 — 9% out).
    """

    def __init__(self, max_len: int = 500) -> None:
        self._books: dict[tuple[str, str], dict[int, list[float]]] = {}
        self._contract_size: dict[str, float] = {}
        self._max_len = max_len

    def set_contract_sizes(self, sizes: dict[str, float]) -> None:
        """Record contractValue per instId, from the exchange's own market list."""
        self._contract_size.update(sizes)

    def seed(self, inst_id: str, bar: str, rows: list[list[str]]) -> None:
        """Replace the book from BloFin REST rows (newest-first, all strings).

        Rows are BloFin's 9-column shape — ``[ts, o, h, l, c, contractVol,
        baseVol, quoteVol, confirm]`` — so volume is taken from index 6, not
        index 5, to match what WS updates and ccxt both mean by "volume".
        """
        candles: dict[int, list[float]] = {}
        for row in rows:
            ts = int(row[0])
            # Tolerate a short row rather than raising: index 5 is the closest
            # thing to a volume such a row would carry.
            base = float(row[6]) if len(row) > 6 else float(row[5])
            candles[ts] = [float(row[i]) for i in range(1, 5)] + [base]
        self._books[(inst_id, bar)] = candles
        self._trim(inst_id, bar)

    def update(self, inst_id: str, bar: str, candle: list | tuple) -> None:
        """Apply one WS candle (``[ts_ms, o, h, l, c, vol]``, ccxt numeric shape).

        ccxt's BloFin ``parse_ohlcv`` fills that last slot from REST index 6,
        so what arrives here is base volume — matching what :meth:`seed` now
        stores.

        An existing timestamp is replaced (in-progress candle); a new
        timestamp is appended.
        """
        ts = int(candle[0])
        key = (inst_id, bar)
        book = self._books.setdefault(key, {})
        book[ts] = [float(candle[i]) for i in range(1, 6)]
        self._trim(inst_id, bar)

    def _trim(self, inst_id: str, bar: str) -> None:
        book = self._books[(inst_id, bar)]
        if len(book) > self._max_len:
            for ts in sorted(book)[: len(book) - self._max_len]:
                del book[ts]

    def has(self, inst_id: str, bar: str) -> bool:
        return bool(self._books.get((inst_id, bar)))

    def keys(self) -> list[tuple[str, str]]:
        return list(self._books)

    def rest_rows(self, inst_id: str, bar: str, limit: int) -> list[list[str]]:
        """Return BloFin REST-shaped rows: newest-first, 9 columns, all strings.

        The full 9 columns are not cosmetic. ccxt's BloFin ``parse_ohlcv``
        reads volume from index 6, so a 6-column row makes every candle reach
        freqtrade with ``volume=None`` — which silently disables any strategy
        gating on volume.

        Index 6 (base) is the stored figure, passed through untouched — never
        rescaled, so what ccxt reads is exactly what the exchange reported.
        Index 5 (contract = base/contractValue) and index 7 (quote =
        base*close) are reconstructions for other consumers; both are left
        blank rather than guessed when contractValue is unknown. Index 8 is
        BloFin's confirm flag — "0" (still forming) for the newest candle
        held, "1" for the rest.
        """
        book = self._books.get((inst_id, bar))
        if not book:
            return []
        size = self._contract_size.get(inst_id)
        newest = max(book)
        rows = []
        for ts in sorted(book, reverse=True)[:limit]:
            o, h, low, c, base = book[ts]
            if size:
                contract = _num(base / size)
            elif base == 0:
                contract = "0"  # no base volume means no contracts either
            else:
                contract = ""
            rows.append(
                [
                    str(ts),
                    _num(o),
                    _num(h),
                    _num(low),
                    _num(c),
                    contract,
                    _num(base),
                    _num(base * c),
                    "0" if ts == newest else "1",
                ]
            )
        return rows
