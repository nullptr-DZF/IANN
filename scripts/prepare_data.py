#!/usr/bin/env python3
"""Prepare the complete paper query, index, and expected-point inputs serially."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
from convert_data import ROOT, DATASETS, atomic_json, convert_dataset, identity


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def executable(build, name):
    path = build/(name + (".exe" if os.name == "nt" else ""))
    if not path.is_file():
        raise FileNotFoundError("Build the C++ targets first: " + str(path))
    return path.resolve()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def expected_prefix(key, m):
    return "expected_vectors_%s_M%d_efC500_efS500_k100" % (key, m)


def expected_statistics(directory, key, m, nquery, dimension):
    kinds = ["missed", "unvisited"] if key in ("IMAGENET", "GLINK") else [
        "top100", "deep", "deep_unvisited"]
    files, totals = {}, [0]*nquery
    for kind in kinds:
        per_kind = [0]*nquery
        for unreachable in (False, True):
            suffix = kind + ("_unreachable" if unreachable else "")
            path = directory/(expected_prefix(key, m)+"_"+suffix+".txt")
            count, query_id, headers = 0, -1, 0
            with path.open(encoding="utf-8") as stream:
                if stream.readline().strip() != "ef_search -- k" or stream.readline().split() != ["500", "100"]:
                    raise ValueError("Invalid expected-point header: " + str(path))
                seen = set()
                for line in stream:
                    fields = line.split()
                    if not fields:
                        continue
                    if fields[0] == "Query":
                        if len(fields) != 2 or int(fields[1]) != headers:
                            raise ValueError("Expected query blocks must be contiguous: " + str(path))
                        query_id = headers
                        headers += 1
                        seen.clear()
                        continue
                    if query_id < 0 or query_id >= nquery or len(fields) != dimension+2:
                        raise ValueError("Invalid expected-point row: " + str(path))
                    identifier, rank = int(fields[0]), int(fields[1])
                    if identifier < 0 or rank <= 0 or identifier in seen:
                        raise ValueError("Invalid or duplicate expected point: " + str(path))
                    seen.add(identifier)
                    per_kind[query_id] += 1
                    totals[query_id] += 1
                    count += 1
            if headers != nquery:
                raise ValueError("Expected-point query count mismatch: " + str(path))
            files[path.name] = dict(pairs=count, sha256=digest(path), **identity(path))
        cap = 5 if key in ("IMAGENET", "GLINK") else (2 if kind == "top100" else 8)
        if any(n > cap for n in per_kind):
            raise ValueError("Per-query expected-point cap exceeded: " + kind)
    if key in ("IMAGENET", "GLINK") and any(n > 5 for n in totals):
        raise ValueError("The combined application expected-point cap must be five")
    return files


def prepare_dataset(key, root, build, force=False, base_chunk=10000, query_chunk=64):
    converted = convert_dataset(key, root, force, base_chunk, query_chunk)
    directory = root/"datasets"/"processed"/key
    index_directory = root/"IndexSave"
    index_directory.mkdir(parents=True, exist_ok=True)
    builder = executable(build, "expected_generate")
    application = key in ("IMAGENET", "GLINK")
    generator = executable(build, "task_expected_generate") if application else builder
    conversion = dict(base=identity(directory/"base.fvecs"),
                      query=identity(directory/"query.fvecs"),
                      conversion_sha256=digest(directory/"conversion.json"))
    for m in (16, 8):
        index_path = index_directory/(key + ("_M8" if m == 8 else "") + ".index")
        marker = directory/("preparation_M%d.json" % m)
        previous = read_json(marker) or {}
        index_inputs = dict(conversion=conversion, m=m, ef_construction=500, seed=100,
                            executable_sha256=digest(builder))
        index_valid = (not force and previous.get("index_inputs") == index_inputs
                       and index_path.is_file()
                       and previous.get("index") == identity(index_path))
        environment = dict(os.environ, DATASET_NAME=key,
                           DATASET_DIR=str(root/"datasets"/"processed")+"/",
                           EXPECTED_DIR=str(directory), METRIC="l2", M=str(m),
                           EF_CONSTRUCTION="500", EF_SEARCH="500", K="100",
                           RANDOM_SEED="100",
                           TASK_CAP="5", TASK_BASE_LABELS=str(directory/"base.labels.u32"),
                           TASK_QUERY_LABELS=str(directory/"query.labels.u32"))
        if not index_valid:
            temporary = index_path.with_name(index_path.name+"."+uuid.uuid4().hex+".partial")
            try:
                subprocess.run([str(builder)], env=dict(environment, INDEX_ONLY="1",
                               INDEX_PATH=str(temporary)), cwd=root, check=True)
                temporary.replace(index_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            # Commit a completed index independently of the longer expected-point pass.
            previous = dict(index_inputs=index_inputs, index=identity(index_path))
            atomic_json(marker, previous)
        environment.update(INDEX_PATH=str(index_path), INDEX_ONLY="0")
        inputs = dict(index=identity(index_path), conversion=conversion,
                      generator_sha256=digest(generator), ef_search=500, k=100,
                      seed=12345, application_cap=5)
        valid = not force and previous.get("expected_inputs") == inputs
        if valid:
            for name, info in previous.get("expected_files", {}).items():
                path = directory/name
                if not path.is_file() or info.get("sha256") != digest(path):
                    valid = False
                    break
            valid = valid and bool(previous.get("expected_files"))
        if valid:
            print("Prepared expected points retained: %s M%d" % (key, m), flush=True)
            continue
        if application:
            subprocess.run([str(generator)], env=environment, cwd=root, check=True)
        else:
            for mode in ("result", "unvisited"):
                subprocess.run([str(generator)], env=dict(environment, EXPECTED_MODE=mode),
                               cwd=root, check=True)
        files = expected_statistics(directory, key, m, converted["n_queries"], converted["dim"])
        atomic_json(marker, dict(index_inputs=index_inputs, index=identity(index_path),
                    expected_inputs=inputs, expected_files=files,
                    n_queries=converted["n_queries"], n_base=converted["n_base"],
                    dimension=converted["dim"]))
        print("Prepared %s M%d: %d pairs" %
              (key, m, sum(row["pairs"] for row in files.values())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--build-dir", type=Path, default=ROOT/"build")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--base-chunk", type=int, default=10000)
    parser.add_argument("--query-chunk", type=int, default=64)
    args = parser.parse_args()
    if min(args.base_chunk, args.query_chunk) <= 0:
        parser.error("Chunk sizes must be positive")
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        cpu = min(allowed) if args.cpu is None else args.cpu
        if cpu not in allowed:
            parser.error("The selected CPU is outside the process affinity mask")
        os.sched_setaffinity(0, {cpu})
    elif args.cpu is not None:
        parser.error("CPU selection requires sched_setaffinity support")
    root, build = args.root.resolve(), args.build_dir.resolve()
    for key in DATASETS if args.all else args.datasets:
        prepare_dataset(key, root, build, args.force, args.base_chunk, args.query_chunk)


if __name__ == "__main__":
    main()
