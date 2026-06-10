"""Host-memory / process tuning shared by the training & eval entry points.

CPU-fraction resolution for the channel precompute, glibc malloc-arena capping,
RSS milestone logging, and the DataLoader IPC sharing strategy. None of this is
model-specific; it just keeps the main process (and its fork workers) from being
OOM-killed on the constrained WSL box this trains on.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform

from .constants import CPU_FRACTION


def resolve_cpu_fraction(args: argparse.Namespace | None) -> float:
    """CPU fraction for image precompute: --cpu-fraction if given, else CPU_FRACTION."""
    frac = getattr(args, "cpu_fraction", None)
    if frac is None:
        return CPU_FRACTION
    if frac <= 0:
        raise ValueError(f"--cpu-fraction must be > 0, got {frac}")
    return frac


_MALLOC_TUNED = False


def cap_malloc_arenas(max_arenas: int = 2) -> None:
    """Limit glibc's per-process malloc arenas to curb host-RAM (RSS) growth.

    glibc defaults to up to ``8 * ncpu`` allocator arenas; with DataLoader
    ``num_workers > 0`` every forked worker inherits and independently grows its
    own arena set, fragmenting RSS — a large, silent host-RAM cost on Linux/WSL.
    Capping arenas (and trimming eagerly) is a near-free reduction. ``mallopt`` is
    applied to the live parent allocator so workers forked later inherit it; the
    env vars cover any *child* processes spawned afterwards (the channel-cache
    pool). Idempotent. No-op when ``max_arenas <= 0`` or off glibc (musl/non-Linux)."""
    global _MALLOC_TUNED
    if _MALLOC_TUNED or max_arenas <= 0:
        return
    _MALLOC_TUNED = True
    os.environ.setdefault("MALLOC_ARENA_MAX", str(max_arenas))
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        M_TRIM_THRESHOLD, M_ARENA_MAX = -1, -8
        libc.mallopt(M_ARENA_MAX, int(max_arenas))
        libc.mallopt(M_TRIM_THRESHOLD, 0)
        print(f"[mem] capped glibc malloc arenas to {max_arenas} (MALLOC_ARENA_MAX)", flush=True)
    except (OSError, AttributeError):
        return  # not glibc, or mallopt unavailable


def resolve_malloc_arena_max(args: argparse.Namespace) -> int:
    """CLI --malloc-arena-max wins; default 2. 0 leaves the glibc default (no cap)."""
    val = getattr(args, "malloc_arena_max", None)
    return 2 if val is None else int(val)


def host_rss_mb() -> tuple[float, float]:
    """Return ``(current_rss_mb, peak_rss_mb)`` for this process by reading
    ``/proc/self/status`` (VmRSS / VmHWM). No psutil dependency. VmHWM is the
    monotonic high-water mark, so it captures transient spikes (torch.compile /
    Inductor, the embedding-table ``np.stack``) even after they are freed.
    Returns ``(nan, nan)`` off Linux or if the file is unreadable."""
    try:
        rss = hwm = float("nan")
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0  # kB -> MB
                elif line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / 1024.0
        return rss, hwm
    except (OSError, ValueError, IndexError):
        return float("nan"), float("nan")


def log_rss(tag: str) -> None:
    """Print a host-RAM milestone in the existing ``[mem]`` style: current RSS and
    the process peak (VmHWM). Used to localize where the main-process footprint is
    spent during startup (loaders, embedding-table build, model->device, compile)."""
    rss, hwm = host_rss_mb()
    if rss != rss:  # NaN: not Linux / unreadable
        return
    print(f"[mem] {tag}: RSS={rss:,.0f} MB (peak {hwm:,.0f} MB)", flush=True)


_MP_SHARING_TUNED = False


def configure_dataloader_sharing(num_workers: int) -> None:
    """Avoid the '/dev/shm out of shared memory' Bus error that kills DataLoader workers
    on shm-constrained boxes (notably WSL, where /dev/shm is a small tmpfs). PyTorch's
    default 'file_descriptor' strategy backs every worker->main tensor with a /dev/shm
    segment; with multi-view image batches in flight (8 views/sample for prior-aware) that
    pool fills and a worker takes SIGBUS mid-epoch (seen as "DataLoader worker ... killed by
    signal: Bus error"). The 'file_system' strategy backs shared tensors with files in the
    system temp dir (disk-backed on WSL) instead, sidestepping the /dev/shm size limit.

    Idempotent; no-op when num_workers==0 (no worker IPC). Set
    CAMCHEX_MP_SHARING_STRATEGY=file_descriptor to restore the PyTorch default."""
    global _MP_SHARING_TUNED
    if _MP_SHARING_TUNED or num_workers <= 0:
        return
    _MP_SHARING_TUNED = True
    strategy = os.environ.get("CAMCHEX_MP_SHARING_STRATEGY", "file_system")
    try:
        import torch.multiprocessing as mp
        if strategy not in mp.get_all_sharing_strategies():
            return
        if mp.get_sharing_strategy() != strategy:
            mp.set_sharing_strategy(strategy)
            print(
                f"[mem] DataLoader IPC sharing strategy -> {strategy} (avoids the /dev/shm "
                f"Bus error under num_workers>0; set CAMCHEX_MP_SHARING_STRATEGY to override)",
                flush=True,
            )
    except (RuntimeError, ImportError):
        return  # leave the default in place if the platform rejects the strategy
