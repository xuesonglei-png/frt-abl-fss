"""Efficiency instrumentation for FRT-ABL runs.

This module is inert on import: it never imports torch, builds a model, reads
data or starts a process.  `EfficiencyRecorder` is created by `train.run` only
when the caller explicitly enables the efficiency protocol.

Measurement contract (aligned with `experiment_protocol.md` §6-7):

* clock origin is the start of `train.run`, so setup, model/KB construction,
  warm-up, proposal/search, graph, update, validation and checkpoint writes are
  all inside the reported wall clock;
* validation accuracy is observed once per round, therefore "time to target"
  from the curve is an upper bound at validation granularity; the summarizer
  also reports a linear interpolation as a descriptive estimate;
* inference cost is reported both as abduction candidate checks
  (`sum(search.checked)`, the paper's "推理检查次数") and as neural forward
  calls / image presentations;
* memory is peak process-tree RSS sampled by a background thread (the
  abduction search workers are child processes) plus the parent process peak
  working set and, when CUDA is used, torch's peak allocated/reserved bytes.

All values are diagnostics.  A single run is not a statistical claim, and no
formal test split is read here.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import socket
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
TZ = timezone(timedelta(hours=8))

# Pre-declared reference points for the two tasks.  They are below the v10
# five-seed mean validation accuracy (MNIST 0.9812, HWF 0.7844) so that a
# completed run can actually cross them; they are fixed before any efficiency
# run and must not be changed after looking at timing results.
DEFAULT_TARGETS = {
    "mnist-add2": (0.95, 0.97, 0.98),
    "hwf": (0.70, 0.74, 0.78),
}
DEFAULT_BUDGETS_SECONDS = (300.0, 900.0, 1800.0, 2700.0)


def now_iso():
    return datetime.now(TZ).isoformat(timespec="seconds")


def parse_seconds(value):
    """Parse a positive duration such as 90, '90s', '5m', '1.5h' or '1h30m'."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower().replace(" ", "")
        if not text:
            raise ValueError("Empty duration")
        units = {"s": 1.0, "sec": 1.0, "secs": 1.0, "m": 60.0, "min": 60.0,
                 "mins": 60.0, "h": 3600.0, "hr": 3600.0, "hrs": 3600.0}
        total, number = 0.0, ""
        for char in text:
            if char.isdigit() or char == ".":
                number += char
                continue
            if not number:
                raise ValueError(f"Invalid duration: {value!r}")
            if char not in units:
                raise ValueError(f"Unknown duration unit {char!r} in {value!r}")
            total += float(number) * units[char]
            number = ""
        if number:
            total += float(number)
        seconds = total
    if not (seconds > 0):
        raise ValueError("Duration must be positive")
    return seconds


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Hardware identity
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def _windows_ram_mb():
        try:
            status = _MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024 ** 2)
        except Exception:
            pass
        return None
else:
    def _windows_ram_mb():
        return None


def gpu_information(torch_module=None):
    """Return GPU identity/peak fields; never import torch at module import."""
    try:
        if torch_module is None:
            import torch as torch_module
        if not torch_module.cuda.is_available():
            return {"available": False, "name": None, "total_memory_mb": None,
                    "cuda_version": getattr(torch_module.version, "cuda", None)}
        props = torch_module.cuda.get_device_properties(0)
        return {"available": True, "name": props.name,
                "total_memory_mb": props.total_memory / (1024 ** 2),
                "cuda_version": getattr(torch_module.version, "cuda", None),
                "capability": f"{props.major}.{props.minor}"}
    except Exception as error:  # pragma: no cover - environment dependent
        return {"available": None, "name": None, "total_memory_mb": None,
                "error": repr(error)}


def hardware_fingerprint(cfg=None, torch_module=None):
    """A stable-ish description of the measurement machine and runtime."""
    cfg = cfg or {}
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_logical": os.cpu_count(),
        "ram_total_mb": _windows_ram_mb(),
        "python": platform.python_version(),
        "device": cfg.get("device"),
        "threads": cfg.get("threads"),
        "search_workers": cfg.get("search_workers"),
        "gpu": gpu_information(torch_module),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = None
    try:
        if torch_module is None:
            import torch as torch_module
        info["torch"] = torch_module.__version__
        info["torch_num_threads"] = torch_module.get_num_threads()
        info["deterministic_algorithms"] = bool(torch_module.are_deterministic_algorithms_enabled())
    except Exception:
        info["torch"] = None
    identity = {key: info[key] for key in
                ("platform", "machine", "processor", "cpu_logical", "ram_total_mb",
                 "python", "torch", "device", "threads", "search_workers")}
    identity["gpu"] = (info.get("gpu") or {}).get("name")
    identity["cuda"] = (info.get("gpu") or {}).get("cuda_version")
    info["fingerprint"] = _sha256_json(identity)[:16]
    return info


# ---------------------------------------------------------------------------
# Process memory (Windows process tree, dependency-free)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
    _KERNEL32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _KERNEL32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _KERNEL32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _KERNEL32.Process32FirstW.restype = wintypes.BOOL
    _KERNEL32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _KERNEL32.Process32NextW.restype = wintypes.BOOL
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _PSAPI.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    _PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL

    _TH32CS_SNAPPROCESS = 0x00000002
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def _process_memory(pid):
        """Return (working_set_mb, peak_working_set_mb) or (None, None)."""
        handle = _KERNEL32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            handle = _KERNEL32.OpenProcess(0x0410, False, int(pid))  # QUERY_INFORMATION|VM_READ
        if not handle:
            return None, None
        try:
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            if not _PSAPI.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), ctypes.sizeof(counters)):
                return None, None
            return (counters.WorkingSetSize / (1024 ** 2),
                    counters.PeakWorkingSetSize / (1024 ** 2))
        finally:
            _KERNEL32.CloseHandle(handle)

    def _process_table():
        snapshot = _KERNEL32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if not snapshot or snapshot == _INVALID_HANDLE_VALUE:
            return {}
        table = {}
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            ok = _KERNEL32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                table[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                ok = _KERNEL32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            _KERNEL32.CloseHandle(snapshot)
        return table

    def process_tree_memory(parent_pid=None):
        """Sample RSS of the parent and all descendants.

        Summing RSS double-counts shared pages; the value is a conservative
        peak-cost indicator, not a precise unique-memory measurement.
        """
        parent_pid = int(parent_pid if parent_pid is not None else os.getpid())
        table = _process_table()
        children = {}
        for pid, ppid in table.items():
            children.setdefault(ppid, []).append(pid)
        stack, seen = [parent_pid], set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            stack.extend(children.get(pid, ()))
        total, peak_parent, peak_sum, observed = 0.0, None, 0.0, 0
        for pid in seen:
            working, peak = _process_memory(pid)
            if working is None:
                continue
            total += working
            if peak is not None:
                peak_sum += peak
            observed += 1
            if pid == parent_pid:
                peak_parent = peak
        return {"process_tree_rss_mb": total,
                "process_tree_peak_working_set_mb": peak_sum or None,
                "parent_peak_working_set_mb": peak_parent,
                "processes_observed": observed}
else:
    def process_tree_memory(parent_pid=None):
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
            peak_mb = peak / divisor
            return {"process_tree_rss_mb": peak_mb,
                    "process_tree_peak_working_set_mb": peak_mb,
                    "parent_peak_working_set_mb": peak_mb,
                    "processes_observed": 1}
        except Exception:
            return {"process_tree_rss_mb": None,
                    "process_tree_peak_working_set_mb": None,
                    "parent_peak_working_set_mb": None,
                    "processes_observed": 0}


# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Peak RSS/GPU sampler; the thread is daemonic and stopped explicitly."""

    def __init__(self, interval_seconds=0.25, torch_module=None):
        self.interval_seconds = max(float(interval_seconds), 0.05)
        self.torch = torch_module
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._reset()
        self.gpu_reset_error = None

    def _reset(self):
        self.tree_peak_mb = 0.0
        self.tree_peak_working_set_mb = 0.0
        self.parent_peak_mb = 0.0
        self.samples = 0
        self.processes_observed = 0
        self.last = {"process_tree_rss_mb": None,
                     "process_tree_peak_working_set_mb": None,
                     "parent_peak_working_set_mb": None}

    def start(self):
        self._reset()
        if self.torch is not None and getattr(self.torch.cuda, "is_available", lambda: False)():
            try:
                self.torch.cuda.synchronize()
                self.torch.cuda.reset_peak_memory_stats()
            except Exception as error:  # pragma: no cover - environment dependent
                self.gpu_reset_error = repr(error)
        self._sample()
        self._thread = threading.Thread(target=self._run, name="efficiency-rss",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception:
                pass

    def _sample(self):
        snapshot = process_tree_memory()
        with self._lock:
            tree = snapshot.get("process_tree_rss_mb")
            tree_peak = snapshot.get("process_tree_peak_working_set_mb")
            parent = snapshot.get("parent_peak_working_set_mb")
            if tree is not None:
                self.tree_peak_mb = max(self.tree_peak_mb, float(tree))
            if tree_peak is not None:
                self.tree_peak_working_set_mb = max(self.tree_peak_working_set_mb,
                                                    float(tree_peak))
            if parent is not None:
                self.parent_peak_mb = max(self.parent_peak_mb, float(parent))
            self.processes_observed = max(self.processes_observed,
                                          int(snapshot.get("processes_observed") or 0))
            self.last = snapshot
            self.samples += 1

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 4 * self.interval_seconds))
        try:
            self._sample()
        except Exception:
            pass
        return self.result()

    def result(self):
        gpu = {"available": None, "peak_allocated_mb": None,
               "peak_reserved_mb": None}
        if self.torch is not None and getattr(self.torch.cuda, "is_available", lambda: False)():
            try:
                self.torch.cuda.synchronize()
                gpu = {
                    "available": True,
                    "peak_allocated_mb": self.torch.cuda.max_memory_allocated(0) / (1024 ** 2),
                    "peak_reserved_mb": self.torch.cuda.max_memory_reserved(0) / (1024 ** 2),
                }
            except Exception as error:  # pragma: no cover - environment dependent
                gpu = {"available": True, "peak_allocated_mb": None,
                       "peak_reserved_mb": None, "error": repr(error)}
        with self._lock:
            return {
                "process_tree_rss_peak_mb": self.tree_peak_mb or None,
                "process_tree_peak_working_set_mb": self.tree_peak_working_set_mb or None,
                "parent_peak_working_set_mb": self.parent_peak_mb or None,
                "samples": self.samples,
                "processes_observed": self.processes_observed,
                "sample_interval_seconds": self.interval_seconds,
                "gpu": gpu,
                "method": ("sampled parent+descendants RSS (shared pages counted per process)"
                           if sys.platform == "win32" else "parent ru_maxrss"),
            }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class PhaseClock:
    def __init__(self):
        self.totals = {}

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + max(0.0, float(seconds))


class EfficiencyRecorder:
    """Per-run efficiency record, flushed atomically after every round."""

    def __init__(self, out_dir, cfg, run_id, targets=None,
                 time_budget_seconds=None, sample_interval_seconds=0.25,
                 started_perf=None, started_wall=None, torch_module=None):
        self.out_dir = Path(out_dir)
        self.cfg = dict(cfg)
        self.run_id = run_id
        self.targets = sorted({float(x) for x in (targets if targets is not None
                                                  else DEFAULT_TARGETS.get(cfg["task"], ()))})
        if any(not 0 < target <= 1 for target in self.targets):
            raise ValueError("Accuracy targets must lie in (0, 1]")
        if time_budget_seconds is not None and float(time_budget_seconds) <= 0:
            raise ValueError("Time budget must be positive")
        self.time_budget_seconds = (None if time_budget_seconds is None
                                    else float(time_budget_seconds))
        self.started_perf = started_perf if started_perf is not None else time.perf_counter()
        self.started_wall = started_wall if started_wall is not None else time.time()
        self.phases = PhaseClock()
        self._round_phases = {}
        self.torch = torch_module
        self.monitor = MemoryMonitor(sample_interval_seconds, torch_module)
        self.counters = {
            "optimizer_steps": 0,
            "candidate_checks": 0,
            "search_calls": 0,
            "proposal_images": 0,
            "proposal_forwards": 0,
            "training_images": 0,
            "training_forwards": 0,
            "validation_images": 0,
            "validation_forwards": 0,
        }
        self.rounds = []
        self.target_crossings = {}
        self.status = "RUNNING"
        self.error = None
        self.budget_hit_seconds = None
        self.flush_errors = 0
        self.started_at = now_iso()
        self.finished_at = None
        self.hardware = hardware_fingerprint(cfg, torch_module)
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._started:
            return
        self.monitor.start()
        self._started = True
        self.flush()

    def finish(self, status, error=None, extra=None):
        self.status = status
        self.error = error
        self.finished_at = now_iso()
        if self._started:
            self.monitor.stop()
        if extra:
            for key, value in extra.items():
                self.counters[key] = value
        self.flush()
        return self.summary()

    # -- timing ------------------------------------------------------------
    def elapsed(self):
        return time.perf_counter() - self.started_perf

    def begin_round(self, round_index=None):
        self._round_phases = {}
        self._round_index = round_index

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_phase(name, time.perf_counter() - started)

    def add_phase(self, name, seconds):
        seconds = max(0.0, float(seconds))
        self.phases.add(name, seconds)
        self._round_phases[name] = self._round_phases.get(name, 0.0) + seconds

    # -- counters ----------------------------------------------------------
    def peak_now(self):
        """Cheap live peak snapshot; does not synchronize CUDA."""
        with self.monitor._lock:
            return {
                "process_tree_rss_peak_mb": self.monitor.tree_peak_mb or None,
                "process_tree_peak_working_set_mb":
                    self.monitor.tree_peak_working_set_mb or None,
                "parent_peak_working_set_mb": self.monitor.parent_peak_mb or None,
            }

    def set_counters(self, **values):
        for key, value in values.items():
            if value is not None:
                self.counters[key] = value

    def add_counters(self, **values):
        for key, value in values.items():
            self.counters[key] = self.counters.get(key, 0) + value

    def derived_counters(self):
        counters = dict(self.counters)
        counters["inference_forwards"] = (
            counters.get("proposal_forwards", 0)
            + counters.get("training_forwards", 0)
            + counters.get("validation_forwards", 0))
        counters["inference_images"] = (
            counters.get("proposal_images", 0)
            + counters.get("training_images", 0)
            + counters.get("validation_images", 0))
        return counters

    # -- budget ------------------------------------------------------------
    def budget_exceeded(self):
        if self.time_budget_seconds is None:
            return False
        if self.elapsed() >= self.time_budget_seconds:
            if self.budget_hit_seconds is None:
                self.budget_hit_seconds = self.elapsed()
            return True
        return False

    def budget_remaining(self):
        if self.time_budget_seconds is None:
            return None
        return max(0.0, self.time_budget_seconds - self.elapsed())

    def budget_overshoot(self):
        if self.time_budget_seconds is None or self.budget_hit_seconds is None:
            return None
        return max(0.0, self.budget_hit_seconds - self.time_budget_seconds)

    # -- observation -------------------------------------------------------
    def record_round(self, point):
        """Append one validation point and update target crossings."""
        point = dict(point)
        point.setdefault("elapsed_seconds", self.elapsed())
        point["round_phase_seconds"] = {key: round(value, 6)
                                        for key, value in sorted(self._round_phases.items())}
        point["counters"] = self.derived_counters()
        accuracy = point.get("accuracy")
        if accuracy is not None:
            for target in self.targets:
                key = f"{target:.6f}"
                if key not in self.target_crossings and float(accuracy) >= target:
                    counters = point["counters"]
                    self.target_crossings[key] = {
                        "target": target,
                        "elapsed_seconds": point["elapsed_seconds"],
                        "round": point.get("round"),
                        "accuracy": accuracy,
                        "optimizer_steps": counters.get("optimizer_steps"),
                        "candidate_checks": counters.get("candidate_checks"),
                        "search_calls": counters.get("search_calls"),
                        "inference_forwards": counters.get("inference_forwards"),
                        "definition": "first validation at or above target",
                    }
        self.rounds.append(point)
        self.flush()
        return point

    # -- output ------------------------------------------------------------
    def snapshot(self):
        counters = self.derived_counters()
        peak = self.monitor.result() if self._started else {}
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "error": self.error,
            "run_id": self.run_id,
            "method_version": self.cfg.get("method_version"),
            "task": self.cfg.get("task"),
            "variant": self.cfg.get("variant"),
            "seed": self.cfg.get("seed"),
            "track": self.cfg.get("track"),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "clock": {
                "elapsed_seconds": self.elapsed(),
                "started_wall_unix": self.started_wall,
                "time_budget_seconds": self.time_budget_seconds,
                "budget_exceeded": self.budget_hit_seconds is not None,
                "budget_hit_seconds": self.budget_hit_seconds,
                "budget_overshoot_seconds": self.budget_overshoot(),
                "budget_remaining_seconds": self.budget_remaining(),
            },
            "targets": self.targets,
            "target_crossings": self.target_crossings,
            "counters": counters,
            "phase_seconds": {key: round(value, 6)
                              for key, value in sorted(self.phases.totals.items())},
            "peak": peak,
            "hardware": self.hardware,
            "config_sha256": _sha256_json(self.cfg),
            "rounds": self.rounds,
            "notes": [
                "Time-to-target from rounds is an upper bound at validation granularity.",
                "Process-tree RSS sums resident pages of parent and search workers; "
                "shared pages are counted per process.",
                "No formal test split is read by this recorder.",
            ],
        }

    def summary(self):
        snapshot = self.snapshot()
        peak = snapshot.get("peak") or {}
        gpu = peak.get("gpu") or {}
        return {
            "status": snapshot["status"],
            "elapsed_seconds": snapshot["clock"]["elapsed_seconds"],
            "time_budget_seconds": self.time_budget_seconds,
            "budget_exceeded": snapshot["clock"]["budget_exceeded"],
            "budget_overshoot_seconds": snapshot["clock"]["budget_overshoot_seconds"],
            "targets": self.targets,
            "target_crossings": self.target_crossings,
            "counters": snapshot["counters"],
            "phase_seconds": snapshot["phase_seconds"],
            "peak_process_tree_rss_mb": peak.get("process_tree_rss_peak_mb"),
            "peak_process_tree_working_set_mb": peak.get("process_tree_peak_working_set_mb"),
            "parent_peak_working_set_mb": peak.get("parent_peak_working_set_mb"),
            "peak_gpu_allocated_mb": gpu.get("peak_allocated_mb"),
            "peak_gpu_reserved_mb": gpu.get("peak_reserved_mb"),
            "hardware_fingerprint": (snapshot.get("hardware") or {}).get("fingerprint"),
            "efficiency_file": "efficiency.json",
        }

    def _flush_curve_csv(self):
        path = self.out_dir / "efficiency_curve.csv"
        fields = ["round", "elapsed_seconds", "accuracy", "correct", "total",
                  "optimizer_steps", "candidate_checks", "search_calls",
                  "inference_forwards", "partial_round"]
        lines = [",".join(fields)]
        for point in self.rounds:
            counters = point.get("counters") or {}
            values = [
                point.get("round"), f"{point.get('elapsed_seconds', 0):.6f}",
                point.get("accuracy"), point.get("correct"), point.get("total"),
                counters.get("optimizer_steps"), counters.get("candidate_checks"),
                counters.get("search_calls"), counters.get("inference_forwards"),
                point.get("partial_round", False),
            ]
            lines.append(",".join("" if value is None else str(value) for value in values))
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def flush(self):
        try:
            payload = self.snapshot()
            temporary = self.out_dir / "efficiency.json.tmp"
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, self.out_dir / "efficiency.json")
            self._flush_curve_csv()
        except Exception:
            # Instrumentation must not abort a valid training run.
            self.flush_errors += 1


def target_curve_summary(recorder_snapshot):
    """Small helper used by tests and ad-hoc inspection."""
    return [
        {"round": point.get("round"), "elapsed_seconds": point.get("elapsed_seconds"),
         "accuracy": point.get("accuracy")}
        for point in recorder_snapshot.get("rounds", [])
    ]
