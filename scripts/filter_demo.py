#!/usr/bin/env python3
"""Generate demo charts: raw CNN output vs rolling-median filter on E2E data.

Produces one chart per contiguous date block (gaps > 1 day start a new block)
so the time axis is always meaningful.
"""
import csv
import collections
import datetime
import pathlib
import statistics
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = pathlib.Path(__file__).parent.parent


class PoolValueFilter:
    WINDOW = 5
    MAX_GAP_SEC = 600
    PH_MIN, PH_MAX = 5.0, 10.0
    RX_MIN, RX_MAX = 0, 999
    PH_SPIKE = 0.20
    RX_SPIKE = 35

    def __init__(self):
        self._accepted = collections.deque(maxlen=self.WINDOW)
        self._history  = []
        self._lock = threading.Lock()

    def update(self, ph, redux, ts=None):
        ts = ts or datetime.datetime.now()
        if not (self.PH_MIN <= ph <= self.PH_MAX) or \
           not (self.RX_MIN <= redux <= self.RX_MAX):
            self._rec(ts, ph, redux, False); return False
        with self._lock:
            ok = self._check(ph, redux, ts)
            ph_f, rx_f = self._last()
            if ok:
                self._accepted.append((ph, redux, ts))
                ph_f, rx_f = ph, redux
        self._rec(ts, ph, redux, ok, ph_f, rx_f)
        return ok

    def _check(self, ph, redux, ts):
        if not self._accepted: return True
        if (ts - self._accepted[-1][2]).total_seconds() > self.MAX_GAP_SEC:
            self._accepted.clear(); return True
        med_ph = statistics.median(a[0] for a in self._accepted)
        med_rx = statistics.median(a[1] for a in self._accepted)
        return abs(ph - med_ph) <= self.PH_SPIKE and \
               abs(redux - med_rx) <= self.RX_SPIKE

    def _last(self):
        return (self._accepted[-1][0], self._accepted[-1][1]) \
               if self._accepted else (None, None)

    def _rec(self, ts, ph, rx, ok, ph_f=None, rx_f=None):
        self._history.append(dict(ts=ts, ph_raw=ph, rx_raw=rx,
                                  ph_f=ph_f, rx_f=rx_f, accepted=ok))


def split_into_blocks(data: list[dict], max_gap_hours: float = 12.0) -> list[list[dict]]:
    """Split data into contiguous blocks (large time gaps start a new block)."""
    if not data:
        return []
    blocks = [[data[0]]]
    for entry in data[1:]:
        gap = (entry["ts"] - blocks[-1][-1]["ts"]).total_seconds() / 3600
        if gap > max_gap_hours:
            blocks.append([])
        blocks[-1].append(entry)
    return blocks


def make_block_chart(block: list[dict], title: str, out_path: pathlib.Path) -> None:
    ts_all   = [d["ts"]     for d in block]
    ph_raw   = [d["ph_raw"] for d in block]
    rx_raw   = [d["rx_raw"] for d in block]
    ph_filt  = [d["ph_f"]   for d in block]
    rx_filt  = [d["rx_f"]   for d in block]
    accepted = [d["accepted"] for d in block]

    ts_acc = [t for t, a in zip(ts_all, accepted) if a]
    ph_acc = [v for v, a in zip(ph_raw, accepted) if a]
    rx_acc = [v for v, a in zip(rx_raw, accepted) if a]
    ts_rej = [t for t, a in zip(ts_all, accepted) if not a]
    ph_rej = [v for v, a in zip(ph_raw, accepted) if not a]
    rx_rej = [v for v, a in zip(rx_raw, accepted) if not a]
    ts_f   = [t for t, v in zip(ts_all, ph_filt) if v is not None]
    ph_f   = [v for v in ph_filt if v is not None]
    rx_f   = [v for v in rx_filt if v is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.grid(color="#333355", linewidth=0.5)

    ax1.scatter(ts_acc, ph_acc, s=6, color="#6688cc", alpha=0.6,
                label=f"CNN raw (accepted  n={len(ts_acc)})", zorder=2)
    ax1.scatter(ts_rej, ph_rej, s=35, color="#ff5555", marker="x",
                linewidths=1.5, label=f"CNN raw (rejected  n={len(ts_rej)})", zorder=3)
    if ts_f:
        ax1.plot(ts_f, ph_f, color="#88ddff", linewidth=1.5,
                 label="filtered output", zorder=4)
    ax1.set_ylabel("pH", color="white")
    ax1.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white",
               loc="upper left", framealpha=0.8)
    # Y-axis scaled to accepted data + small margin (so spikes don't squash the range)
    if ph_acc:
        margin = 0.15
        ax1.set_ylim(min(ph_acc) - margin, max(ph_acc) + margin)

    ax2.scatter(ts_acc, rx_acc, s=6, color="#66bb88", alpha=0.6,
                label=f"CNN raw (accepted  n={len(ts_acc)})", zorder=2)
    ax2.scatter(ts_rej, rx_rej, s=35, color="#ff5555", marker="x",
                linewidths=1.5, label=f"CNN raw (rejected  n={len(ts_rej)})", zorder=3)
    if ts_f:
        ax2.plot(ts_f, rx_f, color="#aaffcc", linewidth=1.5,
                 label="filtered output", zorder=4)
    ax2.set_ylabel("Redux (mV)", color="white")
    ax2.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white",
               loc="upper left", framealpha=0.8)
    if rx_acc:
        margin = 10
        ax2.set_ylim(min(rx_acc) - margin, max(rx_acc) + margin)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=25)
    fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)"
          f"  n={len(block)}  rej={len(ts_rej)}")


def run_csv(csv_path: pathlib.Path, model_label: str, out_prefix: str) -> None:
    rows = list(csv.DictReader(open(csv_path)))

    # Parse timestamps and feed CNN predictions through filter
    filt = PoolValueFilter()
    for r in rows:
        try:
            ts = datetime.datetime.strptime(r["file"], "%Y%m%d_%H%M%S")
        except ValueError:
            ts = datetime.datetime.now()
        filt.update(float(r["ph_pred"]), int(float(r["redux_pred"])), ts)

    blocks = split_into_blocks(filt._history, max_gap_hours=12)
    print(f"\n{model_label}  ({len(filt._history)} readings, {len(blocks)} block(s)):")

    for i, block in enumerate(blocks):
        date_str = block[0]["ts"].strftime("%Y-%m-%d")
        title = f"{model_label}\nBlock {i+1}: {date_str}"
        out = ROOT / "results" / f"{out_prefix}_block{i+1}_{date_str}.png"
        make_block_chart(block, title, out)


if __name__ == "__main__":
    run_csv(ROOT / "results" / "e2e_baseline.csv",
            "BASELINE model — CNN raw vs rolling-median filter", "chart_baseline")
    run_csv(ROOT / "results" / "e2e_clean.csv",
            "NEW model (clean training) — CNN raw vs rolling-median filter", "chart_clean")



class PoolValueFilter:
    WINDOW = 5
    MAX_GAP_SEC = 600
    PH_MIN, PH_MAX = 5.0, 10.0
    RX_MIN, RX_MAX = 0, 999
    PH_SPIKE = 0.20
    RX_SPIKE = 35

    def __init__(self):
        self._accepted = collections.deque(maxlen=self.WINDOW)
        self._history  = []
        self._lock = threading.Lock()

    def update(self, ph, redux, ts=None):
        ts = ts or datetime.datetime.now()
        if not (self.PH_MIN <= ph <= self.PH_MAX) or \
           not (self.RX_MIN <= redux <= self.RX_MAX):
            self._rec(ts, ph, redux, False); return False
        with self._lock:
            ok = self._check(ph, redux, ts)
            ph_f, rx_f = self._last()
            if ok:
                self._accepted.append((ph, redux, ts))
                ph_f, rx_f = ph, redux
        self._rec(ts, ph, redux, ok, ph_f, rx_f)
        return ok

    def _check(self, ph, redux, ts):
        if not self._accepted: return True
        if (ts - self._accepted[-1][2]).total_seconds() > self.MAX_GAP_SEC:
            self._accepted.clear(); return True
        med_ph = statistics.median(a[0] for a in self._accepted)
        med_rx = statistics.median(a[1] for a in self._accepted)
        return abs(ph - med_ph) <= self.PH_SPIKE and \
               abs(redux - med_rx) <= self.RX_SPIKE

    def _last(self):
        return (self._accepted[-1][0], self._accepted[-1][1]) \
               if self._accepted else (None, None)

    def _rec(self, ts, ph, rx, ok, ph_f=None, rx_f=None):
        self._history.append(dict(ts=ts, ph_raw=ph, rx_raw=rx,
                                  ph_f=ph_f, rx_f=rx_f, accepted=ok))


def run_chart(csv_path: pathlib.Path, title: str, out_path: pathlib.Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))

    # Feed CNN predictions through the filter
    filt = PoolValueFilter()
    for r in rows:
        try:
            ts = datetime.datetime.strptime(r["file"], "%Y%m%d_%H%M%S")
        except ValueError:
            ts = datetime.datetime.now()
        filt.update(float(r["ph_pred"]), int(float(r["redux_pred"])), ts)

    data = filt._history

    ts_all   = [d["ts"]    for d in data]
    ph_raw   = [d["ph_raw"]  for d in data]
    rx_raw   = [d["rx_raw"]  for d in data]
    ph_filt  = [d["ph_f"]    for d in data]
    rx_filt  = [d["rx_f"]    for d in data]
    accepted = [d["accepted"] for d in data]

    ts_acc = [t for t, a in zip(ts_all, accepted) if a]
    ph_acc = [v for v, a in zip(ph_raw, accepted) if a]
    rx_acc = [v for v, a in zip(rx_raw, accepted) if a]
    ts_rej = [t for t, a in zip(ts_all, accepted) if not a]
    ph_rej = [v for v, a in zip(ph_raw, accepted) if not a]
    rx_rej = [v for v, a in zip(rx_raw, accepted) if not a]
    ts_f   = [t for t, v in zip(ts_all, ph_filt) if v is not None]
    ph_f   = [v for v in ph_filt if v is not None]
    rx_f   = [v for v in rx_filt if v is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.grid(color="#333355", linewidth=0.5)

    ax1.scatter(ts_acc, ph_acc, s=6, color="#6688cc", alpha=0.6,
                label="CNN raw (accepted)", zorder=2)
    ax1.scatter(ts_rej, ph_rej, s=35, color="#ff5555", marker="x",
                linewidths=1.5, label=f"CNN raw (rejected  n={len(ts_rej)})", zorder=3)
    if ts_f:
        ax1.plot(ts_f, ph_f, color="#88ddff", linewidth=1.5,
                 label="filtered output", zorder=4)
    ax1.set_ylabel("pH", color="white")
    ax1.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white",
               loc="upper left", framealpha=0.8)

    ax2.scatter(ts_acc, rx_acc, s=6, color="#66bb88", alpha=0.6,
                label="CNN raw (accepted)", zorder=2)
    ax2.scatter(ts_rej, rx_rej, s=35, color="#ff5555", marker="x",
                linewidths=1.5, label=f"CNN raw (rejected  n={len(ts_rej)})", zorder=3)
    if ts_f:
        ax2.plot(ts_f, rx_f, color="#aaffcc", linewidth=1.5,
                 label="filtered output", zorder=4)
    ax2.set_ylabel("Redux (mV)", color="white")
    ax2.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white",
               loc="upper left", framealpha=0.8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=25)
    fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"  Total: {len(data)}  accepted: {len(ts_acc)}  rejected: {len(ts_rej)}")


if __name__ == "__main__":
    for csv_name, label in [
        ("e2e_baseline.csv", "BASELINE model — CNN raw vs rolling-median filter"),
        ("e2e_clean.csv",    "NEW model (clean training) — CNN raw vs rolling-median filter"),
    ]:
        csv_path = ROOT / "results" / csv_name
        if csv_path.exists():
            out = ROOT / "results" / csv_path.stem.replace("e2e_", "chart_") \
                  .replace(".csv", ".png") 
            run_chart(csv_path, label, ROOT / "results" / f"chart_{csv_path.stem.split('_',1)[1]}.png")
