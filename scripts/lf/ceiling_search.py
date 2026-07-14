#!/usr/bin/env python3
"""Seq-ceiling search driver for scripts/lf/profile_lora_lf_test_both.sh.

For each config row (model+backend+recompute+liger fixed), finds the largest
sequence length that trains without GPU-HBM OOM (G_OOM) or host-RAM OOM
(C_OOM), tuning the -ohbm<N> knob (keep every Nth outer Unsloth-GC checkpoint
root in HBM) as the host<->HBM trade.

Algorithm per config:
  1. Probe the prior (seq0, ohbm0). No global min/max is assumed.
  2. Gallop: from the prior step +seq_step, doubling the step while feasible
     (downward if the prior is infeasible) until the first flip -> a tight
     [lo, hi) bracket. This replaces bisection-from-absolute-bounds.
  3. Bisect the bracket on the seq_resolution grid.
  4. Inner feasibility at a given seq: endpoint-first ladder search over ohbm.
     The ladder is ordered by HBM share, NOT by N:
       share(0) = 0 (all roots to CPU), share(N) = 1/N  =>  0 < 8 < 6 < 4 < 3 < 2 < 1.
     C_OOM -> jump to the max-share end (if that still C_OOMs, seq infeasible);
     G_OOM -> jump to share 0 (if that still G_OOMs, seq infeasible);
     otherwise shrink the window.
  5. Confirm the winner with a longer run (late host peaks, allocator creep).
     On confirm failure, tighten the ohbm window / step seq down and repeat.

Every live probe is appended to a JSONL ledger (resume-safe). Monotone
pruning avoids rerunning dominated points:
  G_OOM at (s, share) => G_OOM for all s' >= s, share' >= share
  C_OOM at (s, share) => C_OOM for all s' >= s, share' <= share

Outcome classification: OK / G_OOM / C_OOM / UNKNOWN. UNKNOWN (timeouts,
non-OOM crashes like the fla illegal-memory-access) is retried once and then
ABORTS the config instead of guessing -- a non-OOM failure must not silently
shrink the reported ceiling.

Config list: JSONL, one object per line ('#' lines ignored):
  {"name": "q3-32b_asymds",
   "template": "q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false",
   "seq0": 50000, "ohbm0": 0}
Optional per-config keys (defaults in Config below): ohbm_ladder, seq_step,
seq_resolution, seq_min, seq_max, probe_steps, confirm_steps, warmup_steps,
probe_timeout_s, confirm_timeout_s, max_probes, max_confirm_attempts, env{}.

Usage:
  python scripts/lf/ceiling_search.py <configs>.jsonl   # normally generated + passed by ceiling_search_{both,source}.sh
  python scripts/lf/ceiling_search.py cfgs.jsonl --dry-run
  # validate the classifier on a point you already know C-OOMs:
  python scripts/lf/ceiling_search.py cfgs.jsonl --single q3-32b_asymds 55000 0
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # third_party/AsymGEMM
WRAPPER = ROOT / "scripts" / "lf" / "profile_lora_lf_test_both.sh"

OK, G_OOM, C_OOM, UNKNOWN = "OK", "G_OOM", "C_OOM", "UNKNOWN"

# --- outcome signatures (editable; matched signal is recorded in the ledger) ---
STRICT_G_OOM = [
    r"CUDA out of memory\. Tried to allocate",
    r"torch\.(cuda\.)?OutOfMemoryError",
    r"CUBLAS_STATUS_ALLOC_FAILED",
    r"CUDNN_STATUS_ALLOC_FAILED",
]
STRICT_C_OOM_ALLOC = [  # allocation-failure/host-watchdog evidence: valid always
    r"DefaultCPUAllocator: can't allocate memory",
    r"std::bad_alloc",
    r"\bMemoryError\b",
    r"Cannot allocate memory",
    # pinned-host allocation failures surface as CUDA errors but are host-side
    r"cudaHostAlloc",
    r"failed to pin",
    # run_lf's host-mem watchdog (HOST_MEM_WATCHDOG=true default) is the asym
    # family's primary C-OOM reporter -- kernel OOM-kill of the launcher leaves
    # NO torchrun signature on stdout for these configs (verified on real logs)
    r"\[host-mem-watchdog\]",
    r"soft host OOM",
    r"HOST_OOM_EVIDENCE=true",
]
STRICT_C_OOM_KILL = [  # SIGKILL evidence: IGNORED on timeout, because our own
    # timeout escalation SIGKILLs the tree and fabricates exactly these lines
    r"Out of memory: Killed process",
    r"oom-kill",
    r"[Kk]illed by signal[: ]*\s*(9|Killed|SIGKILL)",
    r"exit_?code\s*[:=]\s*-9\b",
    r"Signal 9 \(SIGKILL\)",
    # run_lf reporting its trainer died by SIGKILL (asym family prints no
    # torchrun summary; the wrapper flattens everything to exit 1)
    r"Training command failed with status (9|137)\b",
]
GENERIC_G_OOM = [
    r"CUDA error: out of memory",
    r"CUDA out of memory",
    # a failed cudaHostAlloc returns exactly this code, so it must rank BELOW
    # the host-side signatures; alone it still reads as device OOM
    r"cudaErrorMemoryAllocation",
    # cuDNN workspace alloc at full HBM reports INTERNAL_ERROR; ranked last
    r"CUDNN_STATUS_INTERNAL_ERROR",
]


def share(n: int) -> float:
    return 0.0 if n == 0 else 1.0 / n


def round_grid(x: float, grid: int) -> int:
    return int(round(x / grid)) * grid


def host_cpu_mem_avail_gib() -> int:
    """Available memory on CPU NUMA nodes only. On GB200, global MemAvailable
    also counts free HBM (GPU NUMA nodes) that host allocations cannot use, so
    sum MemFree + max(FilePages - Shmem, 0) over nodes with a non-empty cpulist
    (mirrors run_lf's host-mem watchdog). Falls back to global MemAvailable."""
    total_kb, found = 0, False
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        try:  # per-node: one unreadable node must not discard the others
            if not (node / "cpulist").read_text().strip():
                continue  # GPU/HBM node: its free memory is not host RAM
            vals = {}
            for line in (node / "meminfo").read_text().splitlines():
                parts = line.split()  # "Node 0 MemFree: 123 kB"
                if len(parts) >= 4:
                    vals[parts[2].rstrip(":")] = int(parts[3])
            if "MemFree" in vals:
                found = True
                total_kb += vals["MemFree"] + max(vals.get("FilePages", 0) - vals.get("Shmem", 0), 0)
        except (OSError, ValueError):
            continue  # skipping a CPU node undercounts, which fails closed
    if not found:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // (1024 * 1024)
        return 0
    return total_kb // (1024 * 1024)


def marked_pids(needle: bytes) -> list:
    """PIDs whose environment contains `needle` (see CEILING_SEARCH_MARK)."""
    out = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            if needle in (pid_dir / "environ").read_bytes():
                out.append(int(pid_dir.name))
        except OSError:
            continue
    return out


@dataclass
class Config:
    name: str
    template: str
    seq0: int
    ohbm0: int = 0
    ohbm_ladder: list = field(default_factory=lambda: [0, 16, 8, 7, 6, 5, 4, 3, 2, 1])
    seq_step: int = 4000
    seq_resolution: int = 1000
    seq_min: int = 4000
    seq_max: int = 300000
    probe_steps: int = 3
    confirm_steps: int = 8
    warmup_steps: int = 1
    probe_timeout_s: int = 5400
    confirm_timeout_s: int = 10800
    max_probes: int = 40
    max_confirm_attempts: int = 10  # max CONSECUTIVE failed confirm-length runs (a confirm OK resets the streak)
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        if "{seq}" not in self.template:
            raise ValueError(f"[{self.name}] template must contain {{seq}}")
        try:  # fail at load time, not on the config's first probe two days in
            self.template.format(seq=self.seq0, ohbm=self.ohbm0)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"[{self.name}] template does not format cleanly: {e!r}")
        if self.probe_steps < 1:
            raise ValueError(f"[{self.name}] probe_steps must be >= 1 (0 steps would fake-OK probes)")
        if self.confirm_steps < self.probe_steps:
            raise ValueError(f"[{self.name}] confirm_steps ({self.confirm_steps}) must be >= "
                             f"probe_steps ({self.probe_steps}); cross-grade ledger inference assumes it")
        if "{ohbm}" not in self.template:
            # no ohbm knob in this recompute mode (norecomp/recomp): seq-only search
            self.ohbm_ladder = [self.ohbm0]
        if self.ohbm0 not in self.ohbm_ladder:
            self.ohbm_ladder.append(self.ohbm0)
        if any(int(n) != n or n < 0 for n in self.ohbm_ladder):
            raise ValueError(f"[{self.name}] ohbm_ladder must be nonnegative ints")
        # ladder sorted by HBM share ascending: 0, ..., 8, 4, 3, 2, 1
        self.ladder = sorted(set(int(n) for n in self.ohbm_ladder), key=share)
        grid = self.seq_resolution
        if grid <= 0:
            raise ValueError(f"[{self.name}] seq_resolution must be > 0")
        # align everything to the grid so gallop/bisect arithmetic stays exact
        self.seq_step = max(round_grid(self.seq_step, grid), grid)
        self.seq_min = max(((self.seq_min + grid - 1) // grid) * grid, grid)
        self.seq_max = (self.seq_max // grid) * grid
        if self.seq_min > self.seq_max:
            raise ValueError(f"[{self.name}] seq_min > seq_max after grid alignment")
        self.seq0 = min(max(round_grid(self.seq0, grid), self.seq_min), self.seq_max)
        # outcome-defining fingerprint: any change here makes old ledger entries
        # for this name describe a DIFFERENT experiment (seq0/ladder/timeouts
        # only steer the search, so they are deliberately excluded)
        self.fp = hashlib.sha1(json.dumps([
            self.template, self.probe_steps, self.confirm_steps,
            self.warmup_steps, sorted((str(k), str(v)) for k, v in self.env.items()),
        ]).encode()).hexdigest()[:12]

    def row(self, seq: int, ohbm: int) -> str:
        return self.template.format(seq=seq, ohbm=ohbm)


class SearchAbort(Exception):
    pass


class Ledger:
    """Append-only JSONL of probe outcomes; memo + monotone inference."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        self.memo: dict[tuple, dict] = {}
        if path.exists():
            for ln, line in enumerate(path.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
                    # tolerate a torn write from a killed driver
                    print(f"warning: {path}:{ln}: skipping corrupt ledger line ({err})",
                          file=sys.stderr)

    def _index(self, e: dict):
        # touch (and type-coerce) every needed key BEFORE appending, so the
        # loader's catch fully excludes a malformed (hand-edited) entry
        e["seq"], e["ohbm"] = int(e["seq"]), int(e["ohbm"])
        key = (e["name"], e["seq"], e["ohbm"], e["kind"])
        outcome = e["outcome"]
        self.entries.append(e)
        # UNKNOWN never enters the memo: a cached UNKNOWN would otherwise be
        # returned as a real outcome on resume and corrupt the search.
        if outcome in (OK, G_OOM, C_OOM):
            self.memo[key] = e

    def record(self, e: dict):
        self._index(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(e) + "\n")

    def lookup(self, name: str, seq: int, ohbm: int, kind: str):
        e = self.memo.get((name, seq, ohbm, kind))
        if e:
            return e["outcome"]
        if kind == "probe":
            c = self.memo.get((name, seq, ohbm, "confirm"))
            if c and c["outcome"] == OK:  # confirm OK implies probe OK
                return OK
        # monotone inference from recorded failures. Probe failures imply
        # confirm failures too; confirm failures only imply confirm-grade.
        s = share(ohbm)
        for e in self.entries:
            if e["name"] != name or e["outcome"] not in (G_OOM, C_OOM):
                continue
            # skip superseded entries: a --single reseed overwrites the memo,
            # and dominance must follow the LATEST outcome for each point
            if self.memo.get((e["name"], e["seq"], e["ohbm"], e["kind"])) is not e:
                continue
            if e["kind"] == "confirm" and kind != "confirm":
                continue
            es = share(e["ohbm"])
            if e["outcome"] == G_OOM and e["seq"] <= seq and es <= s:
                return G_OOM
            if e["outcome"] == C_OOM and e["seq"] <= seq and es >= s:
                return C_OOM
        return None


class PeakSampler(threading.Thread):
    """Samples per-GPU HBM (nvidia-smi, all pool GPUs) and the probe tree's
    summed RSS (/proc of marker-carrying pids) while a probe runs. ~1 Hz, so
    peaks are lower bounds, but they cover OOM-killed probes whose fatal
    spike never reaches step_samples.csv."""

    def __init__(self, gpus: list, marker: bytes, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.gpu_peak_mib = {g: 0.0 for g in gpus}
        self.rss_peak_kib = 0.0
        self.marker = marker
        self.interval_s = interval_s
        self._halt = threading.Event()  # not _stop: shadows Thread._stop(), breaks join()

    def stop(self):
        self._halt.set()
        self.join(timeout=15)

    def run(self):
        while not self._halt.wait(self.interval_s):
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                for line in r.stdout.strip().splitlines():
                    idx, used = [t.strip() for t in line.split(",")]
                    if idx in self.gpu_peak_mib:
                        self.gpu_peak_mib[idx] = max(self.gpu_peak_mib[idx], float(used))
            except Exception:
                pass  # sampling is advisory; never disturb the probe
            rss = 0.0
            for pid in marked_pids(self.marker):
                if pid == os.getpid():
                    continue
                try:
                    with open(f"/proc/{pid}/status") as f:
                        for ln in f:
                            if ln.startswith("VmRSS:"):
                                rss += float(ln.split()[1])  # kB
                                break
                except OSError:
                    continue
            self.rss_peak_kib = max(self.rss_peak_kib, rss)

    def peaks(self) -> dict:
        return {"peak_hbm_gib": {g: round(v / 1024, 1) for g, v in self.gpu_peak_mib.items()},
                "peak_rss_gib": round(self.rss_peak_kib / (1024 ** 2), 1)}


class Driver:
    def __init__(self, cfg: Config, ledger: Ledger, args):
        self.cfg = cfg
        self.ledger = ledger
        self.args = args
        self.live_probes = 0
        self.live_confirms = 0
        # Every probe process tree carries this env marker so leftovers can be
        # found and killed via /proc even after the wrapper itself is gone
        # (the wrapper starts training under `setsid --fork`, i.e. a separate
        # session that killpg on the wrapper's group does NOT reach).
        self.mark = f"csearch-{os.getpid()}"
        # (SIGTERM grace, SIGKILL grace): TERM grace must cover the wrapper's
        # trap doing its own INT->TERM->KILL child cleanup
        self.kill_grace = (120, 60)
        self.probe_dir = Path(args.state_dir) / "probes" / cfg.name
        self.probe_dir.mkdir(parents=True, exist_ok=True)
        # Artifacts root follows the row's pinned profiler + host tag (both
        # injected into env{} by ceiling_search_{source,both}.sh), so per-host
        # searches never mix artifact roots. Fallbacks: wrapper default (both)
        # and the local hostname.
        _prof = str(cfg.env.get("PROFILERS") or "both")
        _host = str(cfg.env.get("HOST_TAG") or os.uname().nodename.split(".")[0])
        self.artifacts_root = ROOT / "profiling_results" / f"profiling_{_prof}_ceiling_{_host}"

    def say(self, msg: str):
        print(f"[{self.cfg.name}] {msg}", flush=True)

    # ---------------- probe execution ----------------

    def build_env(self, seq: int, ohbm: int, steps: int) -> dict:
        env = os.environ.copy()
        # Probe-critical wrapper knobs: pinned so ambient shell exports can't leak
        # in and silently change probe behavior. Override per config via the row's
        # env{} field instead. PROFILERS is deliberately NOT pinned here -- it
        # inherits the wrapper's own default.
        env.update({
            "SFT_ROOT": str(ROOT.parents[1]),
            "PLOT": "true",
            "RUN_POST": "false",
            "OVERWRITE": "true",
            # dedicated root isolates probe artifacts from curated sweeps
            # (normal wrapper naming inside; OVERWRITE=true can only clobber
            # other probes). RUN_NAME empty so an export can't relabel.
            "OUTPUT_ROOT": str(self.artifacts_root),
            "RUN_NAME": "",
        })
        env.update({k: str(v) for k, v in self.cfg.env.items()})
        env.update({
            "RUNS": self.cfg.row(seq, ohbm),
            "MAX_STEPS": str(steps),
            "WARMUP_STEPS": str(self.cfg.warmup_steps),
            "CONTINUE_ON_ERROR": "false",   # probe outcome must reach our exit code
            "DRY_RUN": "false",             # exported DRY_RUN=true would fake-OK every probe
            "COLLECT_EXISTING": "false",    # ditto: would skip training entirely
            # an ambient export would silently override -ohbm0 rows (the row
            # suffix only wins when N > 0), shifting the search's ohbm axis
            "UNSLOTH_GC_OUTER_HBM_EVERY_N": "0",
            "CEILING_SEARCH_MARK": self.mark,
        })
        return env

    def sweep_leftovers(self):
        """SIGKILL any process still carrying this driver's probe marker."""
        killed = []
        # environ entries are NUL-terminated: the trailing \0 makes the match
        # exact, so mark "csearch-1234" cannot hit a "csearch-12345" process
        for pid in marked_pids(f"CEILING_SEARCH_MARK={self.mark}".encode() + b"\x00"):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        if killed:
            self.say(f"killed leftover probe processes: {killed}")
            time.sleep(5)

    def _kill_probe(self, p: subprocess.Popen):
        """Stop a probe: SIGTERM the wrapper's group first and give its TERM
        trap time to signal the setsid'd training session (INT -> TERM -> KILL
        with INTERRUPT_GRACE_SECONDS pauses), then escalate; finish with a
        marker sweep for anything the wrapper failed to reap."""
        for sig, grace in zip((signal.SIGTERM, signal.SIGKILL), self.kill_grace):
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        self.sweep_leftovers()

    def pool_gpus(self) -> list:
        return [g for g in re.split(r"[,\s]+", str(self.cfg.env.get("GPU_POOL", os.environ.get("GPU_POOL", "0")))) if g]

    def preflight(self):
        """Wait until GPUs and host RAM are back to baseline; abort if stuck."""
        self.sweep_leftovers()
        deadline = time.time() + self.args.preflight_timeout_s
        gpus = self.pool_gpus()
        while True:
            problems = []
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=60)
                if r.returncode != 0 or not r.stdout.strip():
                    problems.append(f"nvidia-smi rc={r.returncode}: {r.stderr.strip()[:200]}")
                for line in r.stdout.strip().splitlines():
                    idx, used = [t.strip() for t in line.split(",")]
                    if idx in gpus and int(used) > self.args.gpu_max_used_mib:
                        problems.append(f"gpu{idx} used {used} MiB")
            except Exception as e:  # nvidia-smi flake: report, don't crash
                problems.append(f"nvidia-smi failed: {e}")
            avail_gib = host_cpu_mem_avail_gib()
            if avail_gib < self.args.min_ram_avail_gib:
                problems.append(f"CPU-node MemAvail {avail_gib} GiB < {self.args.min_ram_avail_gib}")
            disk_free_gib = shutil.disk_usage(ROOT).free // (1024 ** 3)
            if disk_free_gib < self.args.min_disk_free_gib:
                # a full disk mid-probe fails in confusing non-OOM ways
                problems.append(f"disk free {disk_free_gib} GiB < {self.args.min_disk_free_gib} on {ROOT}")
            if not problems:
                return
            if time.time() > deadline:
                raise SearchAbort(f"preflight never recovered: {'; '.join(problems)}")
            self.say(f"preflight waiting: {'; '.join(problems)}")
            time.sleep(20)

    def classify(self, rc, text: str, timed_out: bool):
        # Patterns are scanned BEFORE the timeout check: a run that OOMs on one
        # rank and then hangs (collective peers stuck) must classify as the OOM,
        # not as UNKNOWN. Exception: on timeout, SIGKILL-shaped evidence is
        # skipped -- our own kill escalation writes those lines. The wrapper
        # exits 1 on any failed run (CONTINUE_ON_ERROR=false), so classification
        # leans on the log, with SIGKILL-family exit codes as the
        # signature-less host-OOM fallback.
        groups = [(G_OOM, STRICT_G_OOM), (C_OOM, STRICT_C_OOM_ALLOC)]
        if not timed_out:
            groups.append((C_OOM, STRICT_C_OOM_KILL))
        groups.append((G_OOM, GENERIC_G_OOM))
        hit = None
        for outcome, pats in groups:
            for pat in pats:
                if re.search(pat, text):
                    hit = (outcome, pat)
                    break
            if hit:
                break
        if rc == 0 and not timed_out:
            # trust the exit code; benign log noise must not abort the search
            return OK, (f"warning: exit 0 but log matched {hit[1]}" if hit else "")
        if hit:
            return hit
        if timed_out:
            return UNKNOWN, "timeout"
        if rc in (137, -9):
            return C_OOM, f"exit {rc} (SIGKILL, no signature)"
        return UNKNOWN, f"exit {rc} (no OOM signature)"

    def run_once(self, seq: int, ohbm: int, kind: str) -> tuple[str, dict]:
        steps = self.cfg.confirm_steps if kind == "confirm" else self.cfg.probe_steps
        timeout = self.cfg.confirm_timeout_s if kind == "confirm" else self.cfg.probe_timeout_s
        self.run_seq = getattr(self, "run_seq", 0) + 1  # unique even at settle_s=0
        log_path = self.probe_dir / f"{kind}_s{seq}_ohbm{ohbm}_{int(time.time())}_{self.run_seq:03d}.log"
        self.say(f"{kind} seq={seq} ohbm={ohbm} (steps={steps}, timeout={timeout}s) -> {log_path.name}")
        self.preflight()
        t0 = time.time()
        timed_out = False
        sampler = PeakSampler(self.pool_gpus(),
                              f"CEILING_SEARCH_MARK={self.mark}".encode() + b"\x00")
        sampler.start()
        try:
            with log_path.open("wb") as f:
                p = subprocess.Popen(["bash", str(WRAPPER)], cwd=ROOT,
                                     env=self.build_env(seq, ohbm, steps),
                                     stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                try:
                    rc = p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill_probe(p)
                    rc = None
                except (KeyboardInterrupt, SystemExit):  # Ctrl-C or SIGTERM handler
                    self._kill_probe(p)
                    raise
        finally:
            sampler.stop()
        dur = int(time.time() - t0)
        with log_path.open("rb") as lf:  # tail only: hang logs can be huge
            lf.seek(max(0, log_path.stat().st_size - 4_000_000))
            text = lf.read().decode("utf-8", errors="replace")
        outcome, sig = self.classify(rc, text, timed_out)
        entry = {"ts": int(time.time()), "name": self.cfg.name, "fp": self.cfg.fp,
                 "seq": seq, "ohbm": ohbm, "kind": kind, "outcome": outcome,
                 "signal": sig, "rc": rc, "secs": dur, "log": str(log_path),
                 **sampler.peaks()}
        self.say(f"  -> {outcome} {('[' + sig + ']') if sig else ''} in {dur}s")
        if outcome != OK:
            time.sleep(self.args.settle_s)
        return outcome, entry

    def outcome_for(self, seq: int, ohbm: int, kind: str = "probe") -> str:
        cached = self.ledger.lookup(self.cfg.name, seq, ohbm, kind)
        if cached is not None:
            if (self.cfg.name, seq, ohbm, kind) not in self.ledger.memo:
                self.say(f"{kind} seq={seq} ohbm={ohbm} -> {cached} (inferred)")
            else:
                self.say(f"{kind} seq={seq} ohbm={ohbm} -> {cached} (ledger)")
            return cached
        if self.live_probes >= self.cfg.max_probes:
            raise SearchAbort(f"max_probes={self.cfg.max_probes} exhausted")
        self.live_probes += 1
        if kind == "confirm":
            self.live_confirms += 1
        outcome, entry = self.run_once(seq, ohbm, kind)
        if outcome == UNKNOWN:
            entry["note"] = "first attempt, retrying"
            self.ledger.record(entry)
            self.say("UNKNOWN outcome, retrying once")
            self.live_probes += 1  # the retry is a live run too
            if kind == "confirm":
                self.live_confirms += 1
            outcome, entry = self.run_once(seq, ohbm, kind)
            if outcome == UNKNOWN:
                self.ledger.record(entry)
                raise SearchAbort(f"UNKNOWN twice at seq={seq} ohbm={ohbm} ({entry['signal']}); "
                                  f"inspect {entry['log']}")
        self.ledger.record(entry)
        if kind == "confirm" and outcome == OK:
            self.live_confirms = 0  # streak budget: a confirm OK resets it
        return outcome

    # ---------------- inner ohbm feasibility ----------------

    def feasible(self, seq: int, hint: int, bounds=None, kind: str = "probe"):
        """Return a working ohbm at this seq, or None. bounds=[lo_excl, hi_excl]
        (ladder indices) is tightened in place if provided."""
        lad = self.cfg.ladder
        b = bounds if bounds is not None else [-1, len(lad)]
        i = lad.index(hint) if hint in lad else 0
        first = True
        while b[1] - b[0] > 1:
            if first:
                i = min(max(i, b[0] + 1), b[1] - 1)
                first = False
            out = self.outcome_for(seq, lad[i], kind)
            if out == OK:
                return lad[i]
            if out == C_OOM:
                b[0] = max(b[0], i)   # need MORE share than this
                i = b[1] - 1          # endpoint-first: max share still in window
            else:  # G_OOM
                b[1] = min(b[1], i)   # need LESS share than this
                i = b[0] + 1          # endpoint-first: min share still in window
        return None

    # ---------------- outer search ----------------

    def search(self) -> dict:
        cfg = self.cfg
        grid = cfg.seq_resolution
        t0 = time.time()
        hint = cfg.ohbm0
        lo_seq = lo_w = hi_seq = None
        best_conf = None  # highest confirm-OK (seq, ohbm); consulted on abort too
        capped = False    # gallop hit seq_max without ever failing
        try:
            # fail closed if this name's ledger entries describe a DIFFERENT
            # experiment (template/steps/env changed under the same auto-name):
            # replaying them would return a stale ceiling with zero live runs
            stale_fp = {e["fp"] for e in self.ledger.entries
                        if e["name"] == cfg.name and e.get("fp") not in (None, cfg.fp)}
            if stale_fp:
                raise SearchAbort(
                    f"ledger holds entries for '{cfg.name}' with different config "
                    f"fingerprint(s) {sorted(stale_fp)} (template/steps/env changed); "
                    f"use a fresh --state-dir or delete those ledger lines")
            w = self.feasible(cfg.seq0, hint)
            if w is not None:
                lo_seq, lo_w, hint = cfg.seq0, w, w
                step = cfg.seq_step  # gallop up, doubling
                while lo_seq < cfg.seq_max:
                    nxt = min(round_grid(lo_seq + step, grid), cfg.seq_max)
                    if nxt <= lo_seq:
                        nxt = lo_seq + grid
                    w = self.feasible(nxt, hint)
                    if w is None:
                        hi_seq = nxt
                        break
                    lo_seq, lo_w, hint = nxt, w, w
                    step *= 2
                capped = hi_seq is None
            else:
                hi_seq, s, step = cfg.seq0, cfg.seq0, cfg.seq_step  # gallop down
                while True:
                    s = max(round_grid(s - step, grid), cfg.seq_min)
                    w = self.feasible(s, hint)
                    if w is not None:
                        lo_seq, lo_w, hint = s, w, w
                        break
                    hi_seq = s
                    if s <= cfg.seq_min:
                        return self._result(None, None, False, t0, "infeasible even at seq_min")
                    step *= 2
            # bisect the bracket on the grid
            while hi_seq is not None and hi_seq - lo_seq > grid:
                mid = round_grid((lo_seq + hi_seq) / 2, grid)
                mid = min(max(mid, lo_seq + grid), hi_seq - grid)
                w = self.feasible(mid, hint)
                if w is not None:
                    lo_seq, lo_w, hint = mid, w, w
                else:
                    hi_seq = mid
            # confirm phase: the probe-grade ceiling can hide late peaks, so
            # re-search at confirm grade -- retry other ohbm at the same seq,
            # gallop down on failure, then bisect back up to the boundary.
            conf_bounds: dict[int, list] = {}

            budget_cut = False  # confirm budget expired with untried ohbm rungs

            def confirm_at(s: int):
                nonlocal budget_cut
                while True:
                    if self.live_confirms >= cfg.max_confirm_attempts:
                        budget_cut = True  # window not exhausted, just out of budget
                        return False, None
                    b = conf_bounds.setdefault(s, [-1, len(cfg.ladder)])
                    w = self.feasible(s, hint, bounds=b)
                    if w is None:
                        return False, None
                    out = self.outcome_for(s, w, kind="confirm")
                    if out == OK:
                        return True, w
                    i = cfg.ladder.index(w)
                    if out == C_OOM:
                        b[0] = max(b[0], i)
                    else:
                        b[1] = min(b[1], i)

            cap_note = " (hit seq_max; true ceiling may be higher)" if capped else ""
            s, step, last_fail = lo_seq, grid, None
            while s >= cfg.seq_min and self.live_confirms < cfg.max_confirm_attempts:
                ok2, w = confirm_at(s)
                if not ok2:
                    if budget_cut:
                        break  # undetermined, not evidence of infeasibility at s
                    last_fail = s
                    if s <= cfg.seq_min:
                        break  # nowhere lower to go; don't spin at seq_min
                    s = max(round_grid(s - step, grid), cfg.seq_min)
                    step *= 2
                    continue
                best_conf = (s, w)
                if last_fail is None or last_fail - s <= grid:
                    note = "confirm budget hit; ceiling may be conservative" if budget_cut else ""
                    return self._result(s, w, True, t0, (note + cap_note).strip())
                lo_c, hi_c = s, last_fail  # bisect back up at confirm grade
                while hi_c - lo_c > grid and self.live_confirms < cfg.max_confirm_attempts:
                    mid = round_grid((lo_c + hi_c) / 2, grid)
                    mid = min(max(mid, lo_c + grid), hi_c - grid)
                    ok3, w3 = confirm_at(mid)
                    if ok3:
                        lo_c, best_conf = mid, (mid, w3)
                    elif budget_cut:
                        break  # undetermined mid, don't tighten the bracket on it
                    else:
                        hi_c = mid
                note = "" if (hi_c - lo_c <= grid and not budget_cut) else \
                    "confirm budget hit; ceiling may be conservative"
                return self._result(best_conf[0], best_conf[1], True, t0, (note + cap_note).strip())
            # best_conf is always None here: both confirm-success branches
            # return inside the loop
            return self._result(lo_seq, lo_w, False, t0, "no confirm-grade success; ceiling is probe-grade only")
        except SearchAbort as e:
            if best_conf:
                return self._result(best_conf[0], best_conf[1], True, t0,
                                    f"confirmed before abort ({e}); ceiling may be conservative")
            return self._result(lo_seq, lo_w, False, t0, f"ABORTED: {e}")

    def confirm_metrics(self, seq: int, ohbm: int) -> dict:
        """Steady-state + peak metrics and an artifact pointer from the confirm
        run's step_samples.csv (source leaf preferred, nsys fallback). Best
        effort: extraction trouble leaves fields null, never fails the search.
        Peaks are rank0's (multi-GPU rows understate the peer)."""
        out = {"artifact_dir": None, "confirm_steady_step_s": None,
               "confirm_peak_hbm_gib": None, "confirm_peak_cpu_rss_gib": None}
        try:
            fields = [p.strip() for p in self.cfg.row(seq, ohbm).split(";")]
            backend, recompute = [t.strip() for t in fields[1].split("|")[:2]]
            _, batch, ga = [t.strip() for t in fields[2].split("|")[:3]]
            # the w/s step label pins this to the confirm run; probes differ in _s<steps>
            run_glob = (f"*/*__b{batch}_s{seq}_ga{ga}"
                        f"_w{self.cfg.warmup_steps}_s{self.cfg.confirm_steps}_*")
            leaf = None
            for prof in ("source", "nsys"):
                cands = sorted(
                    self.artifacts_root.glob(
                        f"{run_glob}/{backend}*__{prof}__{recompute}__*/b{batch}_s{seq}_ga{ga}"),
                    key=lambda p: p.stat().st_mtime)
                if cands:
                    leaf = cands[-1]
                    break
            if leaf is None:
                return out
            out["artifact_dir"] = str(leaf.relative_to(ROOT))
            with (leaf / "step_samples.csv").open() as f:
                rows = list(csv.DictReader(f))
            gib = 1024 ** 3
            hbm = [float(r["peak_reserved_hbm_bytes"]) for r in rows
                   if r.get("peak_reserved_hbm_bytes")]
            rss = [float(r["process_rss_peak_bytes"]) for r in rows
                   if r.get("process_rss_peak_bytes")]
            if hbm:
                out["confirm_peak_hbm_gib"] = round(max(hbm) / gib, 1)
            if rss:
                out["confirm_peak_cpu_rss_gib"] = round(max(rss) / gib, 1)
            meas = sorted((r for r in rows
                           if r.get("is_warmup", "").strip() in ("False", "false", "0")),
                          key=lambda r: int(float(r["measured_step"])))
            if len(meas) >= 3:  # steady state: drop first and last measured step
                mid = [float(r["step_milliseconds"]) / 1000.0 for r in meas[1:-1]]
                out["confirm_steady_step_s"] = round(sum(mid) / len(mid), 1)
        except Exception as e:
            self.say(f"confirm-metrics extraction failed (fields left null): {e}")
        return out

    def _result(self, seq, ohbm, confirmed, t0, note) -> dict:
        r = {"name": self.cfg.name, "ceiling_seq": seq, "ohbm": ohbm,
             "confirmed": confirmed, "note": note,
             "live_probes": self.live_probes, "wall_s": int(time.time() - t0),
             "row": self.cfg.row(seq, ohbm) if seq is not None else None}
        r.update(self.confirm_metrics(seq, ohbm) if confirmed and seq is not None else
                 {"artifact_dir": None, "confirm_steady_step_s": None,
                  "confirm_peak_hbm_gib": None, "confirm_peak_cpu_rss_gib": None})
        status = "CONFIRMED" if confirmed else ("PARTIAL" if seq else "FAILED")
        self.say(f"result: {status} ceiling_seq={seq} ohbm={ohbm} "
                 f"({self.live_probes} live probes, {r['wall_s']}s) {note}")
        return r


def load_configs(path: Path) -> list[Config]:
    cfgs = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cfgs.append(Config(**json.loads(line)))
        except Exception as e:
            sys.exit(f"error: {path}:{ln}: {e}")
    names = [c.name for c in cfgs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"error: duplicate config names: {', '.join(dupes)}\n"
                 f"       (rows differing only in batch/liger auto-derive the same name; "
                 f"add a \"name\":\"...\" override in the extra-json field)")
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", type=Path, help="JSONL config list")
    ap.add_argument("--state-dir", required=True,
                    help="ledger/probe state dir, e.g. scripts/lf/ceiling_search_state_<profiler>_<host>")
    ap.add_argument("--only", nargs="+", help="run only these config names")
    ap.add_argument("--dry-run", action="store_true", help="print plan + prior probe rows, no runs")
    ap.add_argument("--single", nargs=3, metavar=("NAME", "SEQ", "OHBM"),
                    help="run exactly one probe (validates classification, seeds the ledger)")
    ap.add_argument("--single-confirm", action="store_true", help="--single at confirm length")
    ap.add_argument("--gpu-max-used-mib", type=int, default=6000)
    ap.add_argument("--min-ram-avail-gib", type=int, default=64)
    ap.add_argument("--min-disk-free-gib", type=int, default=50)
    ap.add_argument("--preflight-timeout-s", type=int, default=900)
    ap.add_argument("--settle-s", type=int, default=20, help="pause after a failed probe")
    args = ap.parse_args()

    # a `kill <driver>` must stop the running probe too, not orphan it
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    if not WRAPPER.exists():
        sys.exit(f"error: wrapper not found: {WRAPPER}")
    cfgs = load_configs(args.configs)
    if not cfgs:
        sys.exit(f"error: no configs in {args.configs}")
    if args.only:
        cfgs = [c for c in cfgs if c.name in set(args.only)]
        if not cfgs:
            sys.exit("error: --only matched nothing")

    stale = marked_pids(b"CEILING_SEARCH_MARK=")
    if stale:
        print(f"warning: processes from a previous ceiling search are still alive: {stale}\n"
              f"         preflight will block until they exit; kill them if they are stale.",
              file=sys.stderr)

    state = Path(args.state_dir)
    if not args.dry_run:
        # one driver per state dir: two concurrent searches would fight over
        # the ledger, the GPUs, and each other's preflight
        state.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(state / "driver.lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"error: another ceiling search already holds {state / 'driver.lock'}; "
                     f"run one search at a time (or use a different --state-dir).")
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()}\n".encode())  # fd stays open for process lifetime
    ledger = Ledger(state / "ledger.jsonl")

    if args.dry_run:
        for c in cfgs:
            print(f"[{c.name}] prior=(seq={c.seq0}, ohbm={c.ohbm0}) ladder={c.ladder} "
                  f"step={c.seq_step} res={c.seq_resolution} probe_steps={c.probe_steps} "
                  f"confirm_steps={c.confirm_steps}")
            print(f"  RUNS={c.row(c.seq0, c.ohbm0)}")
        return

    if args.single:
        name = args.single[0]
        try:
            seq, ohbm = int(args.single[1]), int(args.single[2])
        except ValueError:
            sys.exit(f"error: --single SEQ and OHBM must be integers, got {args.single[1:]}")
        cfg = next((c for c in cfgs if c.name == name), None) or sys.exit(
            f"error: no config '{name}' (available: {', '.join(c.name for c in cfgs)})")
        stale_fp = {e["fp"] for e in ledger.entries
                    if e["name"] == name and e.get("fp") not in (None, cfg.fp)}
        if stale_fp:
            sys.exit(f"error: ledger holds entries for '{name}' with different config "
                     f"fingerprint(s) {sorted(stale_fp)}; use a fresh --state-dir")
        d = Driver(cfg, ledger, args)
        kind = "confirm" if args.single_confirm else "probe"
        outcome, entry = d.run_once(seq, ohbm, kind)
        ledger.record(entry)
        print(json.dumps(entry, indent=2))
        return

    results_path = state / "results.jsonl"
    results = []
    for cfg in cfgs:
        res = Driver(cfg, ledger, args).search()
        results.append(res)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write(json.dumps(res) + "\n")

    print("\n=== ceiling search summary ===")
    for r in results:
        tag = "OK " if r["confirmed"] else "!! "
        print(f"{tag}{r['name']}: seq={r['ceiling_seq']} ohbm={r['ohbm']} "
              f"probes={r['live_probes']} {r['note']}")
        if r["row"]:
            print(f"    {r['row']}")
    sys.exit(0 if all(r["confirmed"] for r in results) else 1)


if __name__ == "__main__":
    main()
