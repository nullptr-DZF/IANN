"""Durable input, result, and process utilities for serial experiments."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "GLOVE100": (100, 10000, 1183514),
    "DEEP10M": (96, 10000, 9990000),
    "MSONG420": (420, 1000, 983185),
    "LASTFM65": (65, 10000, 292385),
    "IMAGENET": (768, 10000, 1281167),
    "GLINK": (512, 10000, 17091649),
}
KINDS = {k: (["missed", "unvisited"] if k in ("IMAGENET", "GLINK")
             else ["top100", "deep", "deep_unvisited"]) for k in DATASETS}
FIELDS = ["method", "query_id", "expected_id", "ground_truth_rank", "ef_best",
          "k_best", "score", "execution_time_ms", "raw_execution_time_ms",
          "status", "stop_reason", "lower_bound", "visited", "expanded",
          "searches", "optimal"]
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS")}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path, full=False):
    p = Path(path).resolve()
    s = p.stat()
    d = dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
    if full:
        d["sha256"] = sha(p)
    else:
        with p.open("rb") as f:
            d["header_sha256"] = hashlib.sha256(f.read(4096)).hexdigest()
    return d


def assert_identity(info):
    actual = identity(info["path"], "sha256" in info)
    if any(actual[k] != v for k, v in info.items()):
        raise RuntimeError("Input changed: " + info["path"])


def point_key(row):
    return int(row["query_id"]), int(row.get("expected_id", row.get("omega_id")))


def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        def complete_lines():
            for line in f:
                if not line.endswith("\n"):
                    break
                yield line
        return [r for r in csv.DictReader(complete_lines())
                if None not in r and all(v is not None for v in r.values())]


def errors(path):
    p = Path(path)
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            try:
                r = json.loads(line)
                key = point_key(r)
            except (ValueError, KeyError, TypeError):
                continue
            if any(r.get(k) not in (None, "") for k in
                   ("execution_time_ms", "raw_execution_time_ms")):
                raise ValueError("Exception records cannot contain timing values")
            if key in out and out[key] != r:
                raise ValueError("Conflicting exception records")
            out[key] = r
    return out


def write_rows(path, data, fields=FIELDS):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def write_errors(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in data:
            f.write(json.dumps(r, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def vector_shape(path):
    p = Path(path)
    with p.open("rb") as f:
        header = f.read(4)
    if len(header) != 4:
        raise ValueError("Empty vector file: " + str(p))
    d = struct.unpack("<i", header)[0]
    if d <= 0 or p.stat().st_size % (4 * (d + 1)):
        raise ValueError("Invalid vector dimensions/length: " + str(p))
    return d, p.stat().st_size // (4 * (d + 1))


def expected_path(dataset, kind, m=16, unreachable=False):
    suffix = kind + ("_unreachable" if unreachable else "")
    return ROOT / "datasets/processed" / dataset / (
        "expected_vectors_%s_M%d_efC500_efS500_k100_%s.txt" % (dataset, m, suffix))


def expected_entries(path, dimension=None, query_count=None):
    with Path(path).open(encoding="utf-8") as f:
        if not f.readline() or f.readline().split() != ["500", "100"]:
            raise ValueError("Expected-file header must specify 500 100: " + str(path))
        qid = None
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Query":
                qid = int(parts[1])
                if qid < 0 or (query_count is not None and qid >= query_count):
                    raise ValueError("Expected query ID outside paper workload")
            else:
                if qid is None or len(parts) < 3:
                    raise ValueError("Malformed expected point")
                eid, rank = int(parts[0]), int(parts[1])
                if eid < 0 or rank <= 0 or (dimension is not None and len(parts) != dimension + 2):
                    raise ValueError("Invalid expected point ID, rank, or dimension")
                yield (qid, eid), rank, line.rstrip("\r\n")


def keys_from_expected(path, output, dimension, query_count, base_count):
    seen = set()
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "expected_id", "ground_truth_rank"])
        for key, rank, _ in expected_entries(path, dimension, query_count):
            if key in seen or key[1] >= base_count:
                raise ValueError("Duplicate or out-of-range expected point")
            seen.add(key)
            writer.writerow([key[0], key[1], rank])
    return len(seen)


def merge_expected(paths, output, query_count):
    grouped = {}
    for path in paths:
        for key, rank, line in expected_entries(path, query_count=query_count):
            if key in grouped and grouped[key][0] != rank:
                raise ValueError("Inconsistent rank for overlapping expected point")
            grouped[key] = rank, line
    by_query = {}
    for key in sorted(grouped):
        by_query.setdefault(key[0], []).append(key)
    with Path(output).open("w", encoding="utf-8", newline="\n") as f:
        f.write("ef_search -- k\n500 100\n")
        for q in range(query_count):
            f.write("Query %d\n" % q)
            for key in by_query.get(q, []):
                f.write(grouped[key][1] + "\n")


def filtered_expected(source, output, wanted):
    wanted = set(wanted)
    with Path(output).open("w", encoding="utf-8", newline="\n") as f:
        f.write("ef_search -- k\n500 100\n")
        last = None
        found = set()
        for key, _, line in expected_entries(source):
            if key not in wanted:
                continue
            if key[0] != last:
                f.write("Query %d\n" % key[0])
                last = key[0]
            f.write(line + "\n")
            found.add(key)
    if found != wanted:
        raise ValueError("Expected selection is not present in input")


def process_id(pid):
    try:
        text = Path("/proc/%d/stat" % pid).read_text()
        tail = text[text.rfind(")") + 2:].split()
        return dict(pid=pid, start_ticks=tail[19], state=tail[0])
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    current = process_id(record["pid"]) if record else None
    return bool(current and current["state"] != "Z" and
                current["start_ticks"] == record.get("start_ticks"))


def cpu_selection(explicit=None):
    available = sorted(os.sched_getaffinity(0))
    if explicit is not None:
        if explicit not in available:
            raise ValueError("CPU is outside allowed affinity")
        return explicit
    def snapshot():
        result = {}
        for line in Path("/proc/stat").read_text().splitlines():
            a = line.split()
            if a and a[0].startswith("cpu") and a[0][3:].isdigit():
                v = list(map(int, a[1:9]))
                result[int(a[0][3:])] = (sum(v), v[3] + v[4])
        return result
    first = snapshot()
    time.sleep(0.2)
    second = snapshot()
    def score(cpu):
        folder = Path("/sys/devices/system/cpu/cpu%d" % cpu)
        capacity = folder / "cpu_capacity"
        freq = folder / "cpufreq/cpuinfo_max_freq"
        cap = int(capacity.read_text()) if capacity.exists() else 0
        hz = int(freq.read_text()) if freq.exists() else 0
        a, b = first.get(cpu, (0, 0)), second.get(cpu, (1, 1))
        busy = 1 - (b[1] - a[1]) / max(1, b[0] - a[0])
        return (-cap, -hz, busy, cpu)
    return min(available, key=score)


def stats(values):
    if not values:
        return dict(mean_ms=None, median_ms=None, p95_ms=None, max_ms=None)
    v = sorted(values)
    ix = .95 * (len(v) - 1)
    lo = int(ix)
    p95 = v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (ix - lo)
    return dict(mean_ms=statistics.mean(v), median_ms=statistics.median(v),
                p95_ms=p95, max_ms=max(v))
