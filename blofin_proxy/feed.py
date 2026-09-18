"""WebSocket feed manager backed by a single ccxt.pro.blofin instance."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .cache import CandleBook, bar_to_seconds

logger = logging.getLogger(__name__)

_TF_UNITS = {"m": "m", "H": "h", "D": "d", "W": "w", "M": "M"}

# How often the reaper sweeps, and how long a watcher may go unrequested first.
_REAP_INTERVAL = 300.0
_BOOK_IDLE_TTL = 1800.0
_CANDLE_IDLE_FLOOR = 1800.0
_CANDLE_IDLE_CAP = 21600.0


def bar_to_ccxt_timeframe(bar: str) -> str:
    """Map a BloFin bar (``5m``, ``1H``, ``1D``) to a ccxt timeframe (``5m``, ``1h``, ``1d``)."""
    unit = bar[-1]
    if unit not in _TF_UNITS:
        raise ValueError(f"unknown bar unit: {bar!r}")
    return bar[:-1] + _TF_UNITS[unit]


def candle_idle_ttl(bar: str) -> float:
    """How long a candle watcher may go unrequested before it is reaped.

    Two bar periods, so a pair that is still in the pairlist cannot be reaped
    between refreshes — freqtrade re-requests a key at least once per bar
    close, and reaping a live key would only churn it back with a REST reseed.
    Floored so fast bars still get a usable grace window, capped so a 1D key
    cannot pin a dropped pair for days.
    """
    try:
        span = float(bar_to_seconds(bar))
    except ValueError:
        return _CANDLE_IDLE_FLOOR
    return min(max(_CANDLE_IDLE_FLOOR, span * 2), _CANDLE_IDLE_CAP)


class FeedManager:
    """Idempotent WS watcher manager.

    One shared ``ccxt.pro.blofin`` connection multiplexes ``watch_ohlcv`` per
    (symbol, timeframe) and one ``watch_tickers`` loop. Watchers reconnect with
    exponential backoff; after a gap the candle key is reseeded from REST once
    via the ``on_reseed`` callback.
    """

    def __init__(
        self,
        book: CandleBook,
        on_reseed: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self.book = book
        self.tickers: dict[str, dict[str, Any]] = {}
        # instId -> {"bids": [[px, sz], ...], "asks": [...], "ts": ms}.
        # Snapshots, not ccxt's live orderbook object: that one is mutated in
        # place by delta handling, so serving straight from it could serialise
        # a half-applied update.
        self.books: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None
        self._on_reseed = on_reseed
        self._ex: Any = None
        self._tasks: dict[tuple, asyncio.Task] = {}
        self._id_to_symbol: dict[str, str] = {}
        self._ticker_symbols: set[str] = set()
        self._contract_size: dict[str, float] = {}
        self._last_reseed: dict[tuple[str, str], float] = {}
        # Last time each per-pair watcher's route was asked for, so idle ones
        # can be reaped. See _reap.
        self._last_seen: dict[tuple, float] = {}
        self._reaper_task: asyncio.Task | None = None
        self.reaped = 0
        self._started = False
        self._stopped = False

    async def start(self) -> None:
        """Create the ccxt.pro instance and load markets once at startup."""
        import ccxt.pro as ccxtpro  # imported lazily: heavy module

        self._ex = ccxtpro.blofin({"enableRateLimit": True})
        markets = await self._ex.load_markets()
        self._id_to_symbol = {m["id"]: m["symbol"] for m in markets.values()}
        # contractValue straight from the exchange, so nothing downstream has
        # to infer it from rounded volume figures.
        self.book.set_contract_sizes(
            {
                m["id"]: float(m["contractSize"])
                for m in markets.values()
                if m.get("contractSize")
            }
        )
        self._contract_size = {
            m["id"]: float(m["contractSize"])
            for m in markets.values()
            if m.get("contractSize")
        }
        self._started = True
        self._reaper_task = asyncio.create_task(self._reap())
        logger.info("feed started, %d markets loaded", len(self._id_to_symbol))

    def contract_size(self, inst_id: str) -> float | None:
        return self._contract_size.get(inst_id)

    @property
    def started(self) -> bool:
        return self._started

    async def ensure_candles(self, inst_id: str, bar: str) -> None:
        """Idempotently spawn a ``watch_ohlcv`` loop for (instId, bar)."""
        if not self._started or self._stopped:
            return
        key = ("candles", inst_id, bar)
        # Touch before the liveness check: a running watcher still needs its
        # idle clock reset, or the reaper would eventually take a live key.
        self._last_seen[key] = time.monotonic()
        if key in self._tasks and not self._tasks[key].done():
            return
        symbol = self._id_to_symbol.get(inst_id)
        if symbol is None:
            logger.warning("no market for instId %s, skipping candle watcher", inst_id)
            return
        try:
            timeframe = bar_to_ccxt_timeframe(bar)
        except ValueError:
            logger.warning("unsupported bar %s, skipping candle watcher", bar)
            return
        self._tasks[key] = asyncio.create_task(
            self._watch_candles(inst_id, bar, symbol, timeframe)
        )
        logger.info("watching candles %s %s (%s %s)", inst_id, bar, symbol, timeframe)

    async def ensure_tickers(self, inst_id: str | None = None) -> None:
        """Idempotently run one ``watch_tickers`` loop covering every instId asked for.

        ccxt.pro's BloFin ``watch_tickers()`` rejects a bare call — the channel
        is per-symbol, so it needs an explicit list. Calling it without one
        raised NotSupported on every retry, so the watcher never delivered a
        single ticker and the route fell back to REST forever, which is the
        one thing this proxy exists to avoid.

        Symbols accumulate as routes ask for them; a genuinely new symbol
        restarts the single watcher over the wider set. A pairlist converges
        after its first pass, so this settles rather than churning.
        """
        if not self._started or self._stopped:
            return
        symbol = self._id_to_symbol.get(inst_id) if inst_id else None
        if inst_id and symbol is None:
            logger.warning("no market for instId %s, skipping ticker watcher", inst_id)
            return

        key = ("tickers",)
        task = self._tasks.get(key)
        if symbol is not None and symbol not in self._ticker_symbols:
            self._ticker_symbols.add(symbol)
            if task is not None and not task.done():
                task.cancel()  # restart over the widened symbol set
                task = None
        if not self._ticker_symbols or (task is not None and not task.done()):
            return
        self._tasks[key] = asyncio.create_task(self._watch_tickers())
        logger.info("watching tickers (%d symbols)", len(self._ticker_symbols))

    async def _watch_candles(
        self, inst_id: str, bar: str, symbol: str, timeframe: str
    ) -> None:
        backoff = 1.0
        had_gap = False
        # One reseed per bar period at most. Every WS error used to queue an
        # immediate REST reseed, so a broad outage turned N watchers into N
        # REST calls per reconnect attempt — pointing a burst of REST traffic
        # at an upstream that is already refusing, which is how a transient
        # failure escalates into a Cloudflare 1015 IP ban. Reseeding more than
        # once per bar buys nothing anyway: no new candle has closed.
        try:
            reseed_interval = max(60.0, float(bar_to_seconds(bar)))
        except ValueError:
            reseed_interval = 60.0
        while not self._stopped:
            try:
                if had_gap and self._on_reseed is not None:
                    # Reseed from REST so a WS gap cannot leave a hole, but no
                    # more often than reseed_interval. had_gap deliberately
                    # stays set when throttled, so the reseed still happens —
                    # just later, on a subsequent pass.
                    now = time.monotonic()
                    key = (inst_id, bar)
                    if now - self._last_reseed.get(key, 0.0) >= reseed_interval:
                        self._last_reseed[key] = now
                        await self._on_reseed(inst_id, bar)
                        had_gap = False
                ohlcv = await self._ex.watch_ohlcv(symbol, timeframe)
                backoff = 1.0
                if ohlcv:
                    self.book.update(inst_id, bar, ohlcv[-1])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any WS error
                self.last_error = f"candles {inst_id} {bar}: {exc}"
                logger.warning("candle watcher error: %s (retry in %.0fs)", exc, backoff)
                had_gap = True
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def ensure_books(self, inst_id: str) -> None:
        """Idempotently spawn a ``watch_order_book`` loop for one instId.

        Order books were the one hot route still going to REST on every call:
        a 1s TTL never survives to the next request for the same pair, so with
        order-book pricing enabled each bot pulled one upstream book per pair
        per cycle and walked straight into Cloudflare's rate limit. BloFin
        supports the books WS channel, so it can be fed like candles/tickers.
        """
        if not self._started or self._stopped:
            return
        key = ("books", inst_id)
        self._last_seen[key] = time.monotonic()
        if key in self._tasks and not self._tasks[key].done():
            return
        symbol = self._id_to_symbol.get(inst_id)
        if symbol is None:
            logger.warning("no market for instId %s, skipping book watcher", inst_id)
            return
        self._tasks[key] = asyncio.create_task(self._watch_books(inst_id, symbol))
        logger.info("watching books %s (%s)", inst_id, symbol)

    async def _watch_books(self, inst_id: str, symbol: str) -> None:
        backoff = 1.0
        while not self._stopped:
            try:
                ob = await self._ex.watch_order_book(symbol)
                backoff = 1.0
                # ccxt's blofin parse_order_book applies no unit conversion, so
                # these are the exchange's own price/size pairs (size in
                # contracts) and can be handed back verbatim in REST shape.
                self.books[inst_id] = {
                    "bids": [[lvl[0], lvl[1]] for lvl in ob["bids"][:50]],
                    "asks": [[lvl[0], lvl[1]] for lvl in ob["asks"][:50]],
                    "ts": ob.get("timestamp"),
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any WS error
                self.last_error = f"books {inst_id}: {exc}"
                logger.warning("book watcher error: %s (retry in %.0fs)", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _watch_tickers(self) -> None:
        backoff = 1.0
        while not self._stopped:
            try:
                symbols = sorted(self._ticker_symbols)
                if not symbols:
                    return
                tickers = await self._ex.watch_tickers(symbols)
                backoff = 1.0
                for symbol, ticker in tickers.items():
                    market = self._ex.market(symbol)
                    self.tickers[market["id"]] = ticker
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any WS error
                self.last_error = f"tickers: {exc}"
                logger.warning("ticker watcher error: %s (retry in %.0fs)", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _reap(self) -> None:
        """Cancel per-pair watchers whose route has stopped being requested.

        ensure_candles/ensure_books are idempotent on creation, but nothing
        removed a watcher when its pair left the pairlist — so under a
        rotating VolumePairList the task set grew monotonically for the life
        of the process (measured 280 distinct pairs against a 250-asset list
        after 9h, along with every candle those dead keys kept resident).

        Idle time is the signal, since the proxy has no view of the pairlist.
        The single tickers watcher is skipped: it is not per-pair, and it ends
        itself when its symbol set is empty.
        """
        while not self._stopped:
            await asyncio.sleep(_REAP_INTERVAL)
            now = time.monotonic()
            for key, task in list(self._tasks.items()):
                if key[0] == "candles":
                    ttl = candle_idle_ttl(key[2])
                elif key[0] == "books":
                    ttl = _BOOK_IDLE_TTL
                else:
                    continue
                if now - self._last_seen.get(key, now) < ttl:
                    continue
                task.cancel()
                self._tasks.pop(key, None)
                self._last_seen.pop(key, None)
                self.reaped += 1
                if key[0] == "candles":
                    self.book.drop(key[1], key[2])
                    self._last_reseed.pop((key[1], key[2]), None)
                else:
                    self.books.pop(key[1], None)
                logger.info(
                    "reaped idle watcher %s", "/".join(str(part) for part in key)
                )

    def status(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "watchers": {
                "/".join(str(part) for part in key): not task.done()
                for key, task in self._tasks.items()
            },
            "tickers_cached": len(self.tickers),
            "reaped": self.reaped,
            "last_error": self.last_error,
        }

    async def close(self) -> None:
        self._stopped = True
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        if self._ex is not None:
            await self._ex.close()
        self._started = False
