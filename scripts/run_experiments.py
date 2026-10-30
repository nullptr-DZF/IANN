"""Run the full paper workload in one detached serial queue."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
try:
    import fcntl
except ImportError:
    fcntl = None
from runtime import (ROOT, DATASETS, KINDS, FIELDS, THREAD_ENV, utc, load, save,
                     sha, identity, assert_identity, point_key, rows, errors,
                     write_rows, write_errors, vector_shape, expected_path,
                     expected_entries, keys_from_expected, merge_expected,
                     filtered_expected, process_id, alive, cpu_selection, stats)

FAMILIES = ["all", "overall", "ablations", "paths", "repair",
            "weights", "rank-gain", "patience"]
METHOD_NAMES = {"iann": "IANN", "inc-only": "IncOnly", "lb-only": "LBOnly",
                "inc-rerun": "IncRerun", "beams": "BeamS", "bisp": "BiSP",
                "unisp": "UniSP", "repair-refine": "RepairIANN",
                "repair-eval": "RepairEvaluation", "ground": "Ground"}
QUEUE_LOCK_FD = None


def binary(name):
    p = ROOT / "build" / name
    if not p.is_file():
        raise FileNotFoundError("Build the project first; missing " + str(p))
    return p


def build_identity():
    paths = [ROOT / "CMakeLists.txt"]
    for folder in ["hnswlib", "whynot/hpp", "whynot/cpp", "scripts"]:
        paths.extend(p for p in (ROOT / folder).glob("*")
                     if p.suffix in [".h", ".cpp", ".py"] and p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def make_manifest(out, args):
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    family = args.experiment
    include_repair = family in ["all", "repair"]
    cases = []
    for dataset, (dimension, query_count, base_count) in DATASETS.items():
        folder = ROOT / "datasets/processed" / dataset
        qpath, bpath = folder / "query.fvecs", folder / "base.fvecs"
        if vector_shape(qpath) != (dimension, query_count):
            raise ValueError("Query file does not match the paper workload: " + dataset)
        if vector_shape(bpath) != (dimension, base_count):
            raise ValueError("Base file does not match the paper workload: " + dataset)
        for kind in KINDS[dataset]:
            ep = expected_path(dataset, kind)
            key_file = inputs / ("%s_%s.keys.csv" % (dataset, kind))
            n = keys_from_expected(ep, key_file, dimension, query_count, base_count)
            cases.append(dict(id=dataset + "_" + kind, dataset=dataset, kind=kind,
                              m=16, queries=str(qpath), expected=str(ep),
                              index=str(ROOT / "IndexSave" / (dataset + ".index")),
                              keys=str(key_file), keys_sha256=sha(key_file), pairs=n,
                              identities=[identity(qpath, True), identity(ep, True),
                                          identity(ROOT / "IndexSave" / (dataset + ".index"))]))
        if include_repair:
            for m in [16, 8]:
                sources = [expected_path(dataset, k, m, True) for k in KINDS[dataset]]
                merged = inputs / ("%s_M%d_unreach.txt" % (dataset, m))
                merge_expected(sources, merged, query_count)
                key_file = inputs / ("%s_M%d_unreach.keys.csv" % (dataset, m))
                n = keys_from_expected(merged, key_file, dimension, query_count, base_count)
                index = ROOT / "IndexSave" / (
                    dataset + ("" if m == 16 else "_M8") + ".index")
                cases.append(dict(id="%s_M%d_unreach" % (dataset, m), dataset=dataset,
                                  kind="unreach", m=m, queries=str(qpath),
                                  expected=str(merged), index=str(index),
                                  keys=str(key_file), keys_sha256=sha(key_file),
                                  expected_sha256=sha(merged), pairs=n,
                                  identities=[identity(qpath, True), identity(index)] +
                                  [identity(s, True) for s in sources]))
    jobs = []
    seen = set()
    def add(case, method, ratio=99, gain=5, patience=5):
        config = "default" if (ratio, gain, patience) == (99, 5, 5) else (
            "r%d_i%d_l%d" % (ratio, gain, patience))
        jid = "%s__%s__%s" % (method, case["id"], config)
        if jid in seen:
            return
        seen.add(jid)
        jobs.append(dict(id=jid, case=case["id"], method=method, ratio=ratio,
                         alpha=ratio / float(ratio + 1), beta=1.0 / (ratio + 1),
                         i=gain, ell=patience, expected_pairs=case["pairs"]))
    main_cases = [c for c in cases if c["kind"] != "unreach"]
    if family != "ground":
        if family in ["all", "overall", "ablations", "weights", "rank-gain", "patience"]:
            for c in main_cases:
                add(c, "iann")
        if family in ["all", "ablations"]:
            for method in ["inc-only", "lb-only", "inc-rerun", "beams"]:
                for c in main_cases:
                    add(c, method)
        if family in ["all", "paths"]:
            for method in ["bisp", "unisp"]:
                for c in main_cases:
                    add(c, method)
        if include_repair:
            for c in cases:
                if c["kind"] == "unreach":
                    add(c, "repair-refine")
                    add(c, "repair-eval")
        for name, values in [("weights", [20, 40, 60, 80]),
                             ("rank-gain", [1, 3, 7, 9]),
                             ("patience", [1, 2, 3, 10, 15])]:
            if family not in ["all", name]:
                continue
            for value in values:
                for c in main_cases:
                    add(c, "iann", ratio=value if name == "weights" else 99,
                        gain=value if name == "rank-gain" else 5,
                        patience=value if name == "patience" else 5)
    else:
        for c in main_cases:
            add(c, "ground")
    executables = ["experiment"]
    if family in ["all", "paths"]:
        executables.append("pure_path_search")
    if include_repair:
        executables += ["xr_end_to_end", "index_reconstruction"]
    result = dict(schema=1, created=utc(), experiment=family, run_id=args.run_id,
                  cpu=cpu_selection(args.cpu), point_budget_ms=600000,
                  source_run=str(args.source_run.resolve()) if args.source_run else None,
                  sources=build_identity(),
                  binaries={n: identity(binary(n), True) for n in executables},
                  cases=cases, jobs=jobs, single_thread=True, workers=1)
    save(out / "manifest.json", result)
    return result


def normalized_result(row, method=None):
    r = dict(row)
    point_key(r)
    if r.get("method") not in METHOD_NAMES or (method and r["method"] != method):
        raise ValueError("Unexpected result method")
    if r.get("status") not in ("ok", "timeout", "no_feasible"):
        raise ValueError("Invalid terminal result status")
    elapsed = float(r["execution_time_ms"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid recorded elapsed time")
    if r.get("raw_execution_time_ms") not in (None, ""):
        raw = float(r["raw_execution_time_ms"])
        if not math.isfinite(raw) or raw < 0:
            raise ValueError("Invalid raw elapsed time")
    if r["status"] == "timeout":
        r["execution_time_ms"] = "600000"
        r["optimal"] = "0"
    if r["status"] == "ok" and r["method"] in (
            "iann", "inc-only", "lb-only", "inc-rerun", "beams", "ground"):
        ef, k = int(r.get("ef_best") or 0), int(r.get("k_best") or 0)
        if not 0 < k <= ef:
            raise ValueError("Successful result has infeasible parameters")
    if r["method"] == "ground" and r.get("optimal") == "1" and r["status"] != "ok":
        raise ValueError("A non-successful Ground result cannot be certified optimal")
    return r


def terminal_data(folder):
    result = {}
    for r in rows(folder / "results.csv"):
        r = normalized_result(r)
        key = point_key(r)
        if key in result:
            raise ValueError("Duplicate result point")
        result[key] = r
    exc = errors(folder / "exceptions.jsonl")
    if set(result) & set(exc):
        raise ValueError("A point cannot have both a timing and an exception")
    return result, exc


def collect_attempt(folder, attempt, job):
    result, exc = terminal_data(folder)
    incoming = rows(attempt / "results.csv")
    if job["method"] in ["bisp", "unisp"]:
        incoming = []
        for p in attempt.glob("pure_path_search/paths_*.csv"):
            for r in rows(p):
                incoming.append(dict(method=job["method"], query_id=r["query_id"],
                                     expected_id=r["expected_id"],
                                     execution_time_ms=r["time_ms"],
                                     raw_execution_time_ms=r["time_ms"],
                                     status=r.get("run_status", "ok"),
                                     visited=r.get("dist_comps", "")))
    elif job["method"] == "repair-refine":
        incoming = []
        for p in attempt.glob("xr_end_to_end/*.csv"):
            for r in rows(p):
                incoming.append(dict(method=job["method"], query_id=r["query_id"],
                    expected_id=r["omega_id"], ground_truth_rank=r["ground_truth_rank"],
                    execution_time_ms=r["total_time_ms"],
                    raw_execution_time_ms=r["total_time_ms"],
                    ef_best=r["pr_ef"], k_best=r["pr_rank"],
                    status=r.get("run_status") or (
                        "ok" if r["repair_success"] == "1" and r.get("pr_found") == "1"
                        else "no_feasible")))
    attempt_seen = set()
    for r in incoming:
        r = normalized_result(r, job["method"])
        key = point_key(r)
        if key in attempt_seen or key in exc:
            raise ValueError("Duplicate terminal point across attempts: " + str(key))
        attempt_seen.add(key)
        if key in result:
            if any(str(result[key].get(k) if result[key].get(k) is not None else "") !=
                   str(v if v is not None else "")
                   for k, v in r.items() if k in FIELDS):
                raise ValueError("Conflicting durable result: " + str(key))
            continue
        result[key] = r
    for key, r in errors(attempt / "errors.jsonl").items():
        if key in result:
            raise ValueError("Conflicting timing and exception outcome")
        if key in exc and exc[key] != r:
            raise ValueError("Conflicting durable exception")
        exc[key] = r
    for line in (attempt / "progress.tsv").read_text(encoding="utf-8").splitlines() if (
            attempt / "progress.tsv").exists() else []:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "TIMEOUT":
            key = (int(fields[1]), int(fields[2]))
            if key not in result and key not in exc:
                result[key] = dict(method=job["method"], query_id=key[0],
                                  expected_id=key[1], status="timeout",
                                  execution_time_ms="600000", stop_reason="timeout")
    write_rows(folder / "results.csv", [result[k] for k in sorted(result)])
    write_errors(folder / "exceptions.jsonl", [exc[k] for k in sorted(exc)])
    return result, exc


def expected_keys(case):
    result = {}
    for r in rows(case["keys"]):
        key = point_key(r)
        if key in result or min(key) < 0 or int(r["ground_truth_rank"]) <= 0:
            raise ValueError("Invalid or duplicate expected key")
        result[key] = r
    if len(result) != case["pairs"]:
        raise ValueError("Expected-key coverage differs from its manifest")
    return result


def validate_outcomes(case, job, result, exc):
    expected = expected_keys(case)
    if not (set(result) | set(exc)) <= set(expected):
        raise ValueError("Unexpected result IDs in " + job["id"])
    for key, row in result.items():
        if row["method"] != job["method"]:
            raise ValueError("Result method differs from its job")
        if row.get("ground_truth_rank") not in (None, "") and (
                int(row["ground_truth_rank"]) != int(expected[key]["ground_truth_rank"])):
            raise ValueError("Result rank differs from its expected point")
    for row in exc.values():
        if row.get("method") not in (None, "", job["method"]):
            raise ValueError("Exception method differs from its job")
    return expected


def default_job_id(case):
    return "iann__%s__default" % case["id"]


def reuse_default(out, manifest, job, case):
    if (not manifest.get("source_run") or job["method"] != "iann" or
            (job["ratio"], job["i"], job["ell"]) != (99, 5, 5)):
        return False
    prior = Path(manifest["source_run"])
    src = load(prior / "manifest.json", {})
    original = next((c for c in src.get("cases", []) if c["id"] == case["id"]), None)
    if (not original or src.get("sources") != manifest["sources"] or
            original["identities"] != case["identities"]):
        raise ValueError("Source default run has different inputs or source version")
    current_binary = manifest.get("binaries", {}).get("experiment", {}).get("sha256")
    if (not current_binary or src.get("binaries", {}).get("experiment", {}).get("sha256") !=
            current_binary):
        raise ValueError("Source default run has a different executable")
    source_job = next((j for j in src.get("jobs", []) if j["id"] == job["id"]), None)
    if not source_job or any(source_job.get(k) != job.get(k) for k in
                             ("method", "ratio", "alpha", "beta", "i", "ell", "expected_pairs")):
        raise ValueError("Source default job has different parameters")
    source = prior / "jobs" / default_job_id(case)
    expected = expected_keys(case)
    r, e = terminal_data(source)
    validate_outcomes(case, job, r, e)
    if set(r) | set(e) != set(expected):
        raise ValueError("Source default job is incomplete: " + str(source))
    dest = out / "jobs" / job["id"]
    dest.mkdir(parents=True, exist_ok=True)
    write_rows(dest / "results.csv", r.values())
    write_errors(dest / "exceptions.jsonl", e.values())
    save(dest / "state.json", dict(state="completed", reused_from=str(source), ended=utc(),
                                  results_sha256=sha(dest / "results.csv"),
                                  exceptions_sha256=sha(dest / "exceptions.jsonl")))
    return True


def beams_keys(out, case, expected, folder):
    saved, exc = terminal_data(out / "jobs" / default_job_id(case))
    validate_outcomes(case, dict(id=default_job_id(case), method="iann"), saved, exc)
    if set(saved) | set(exc) != set(expected):
        raise ValueError("BeamS requires a complete default IANN job")
    valid = {}
    skipped = []
    for key, r in expected.items():
        v = saved.get(key)
        if (v and v["status"] == "ok" and
                0 < int(v.get("k_best") or 0) <= int(v.get("ef_best") or 0)):
            valid[key] = dict(r, ef_best=v["ef_best"], k_best=v["k_best"],
                              iann_time_ms=v["execution_time_ms"])
        else:
            skipped.append(dict(query_id=key[0], expected_id=key[1],
                                error="No feasible saved IANN parameters",
                                status="unavailable_parameters"))
    existing = errors(folder / "exceptions.jsonl")
    for r in skipped:
        existing[point_key(r)] = r
    write_errors(folder / "exceptions.jsonl", existing.values())
    return valid


def child_environment(case, job, attempt, expected):
    remove = ("ALGO4_", "PR_SUITE_", "MAX_NO_IMPROVEMENT_", "GROUND_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(remove)}
    env.update(THREAD_ENV)
    env.update(DATASET_NAME=case["dataset"],
               DATASET_DIR=str(ROOT / "datasets/processed") + "/",
               INDEX_PATH=case["index"], EXPECTED_FILE=str(expected),
               EXPECTED_FILEPATH=str(expected), OUTPUT_DIR=str(attempt),
               M=str(case["m"]), K="100", EF_SEARCH="500", EF_CONSTRUCTION="500",
               RANDOM_SEED="100", METRIC="l2", MAX_QUERIES=(
                   "-1" if job["method"] in ("bisp", "unisp", "repair-eval") else "0"),
               ALPHA=str(job["alpha"]), BETA=str(job["beta"]), VERIFY_RESULTS="0",
               ALGO4_JUMP_I=str(job["i"]), ALGO4_FREEZE_ROUNDS=str(job["ell"]),
               EXPERIMENT_PROGRESS=str(attempt / "progress.tsv"),
               EXPERIMENT_ERRORS=str(attempt / "errors.jsonl"),
               EXPERIMENT_BUDGET_MS="600000", MAX_POINT_TIME_MS="600000",
               EFS_Q="500", EFS_OMEGA="500", EF_EVAL="500", STRATEGY="wvd-pareto")
    for name in ["ALGO4_FREEZE_EF", "MAX_EF", "QUERY_START", "QUERY_END"]:
        env.pop(name, None)
    return env


def run_child(command, env, attempt, cpu, update):
    def prepare_child():
        os.sched_setaffinity(0, {cpu})
        record = process_id(os.getpid())
        save(attempt / "process.json", record)
        save(ROOT / "ExperResults/active_process.json", record)
    with (attempt / "stdout.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 preexec_fn=prepare_child,
                                 pass_fds=(() if QUEUE_LOCK_FD is None else (QUEUE_LOCK_FD,)))
        save(attempt / "process.json", process_id(child.pid))
        update(child=process_id(child.pid), attempt=str(attempt))
        while child.poll() is None:
            time.sleep(1)
            update(child=process_id(child.pid), attempt=str(attempt))
        return child.returncode


def wait_for_existing_children(update):
    records = [ROOT / "ExperResults/active_process.json"]
    records.extend((ROOT / "ExperResults").glob("*/jobs/**/process.json"))
    seen = set()
    for path in records:
        who = load(path)
        if not who or who.get("pid") == os.getpid():
            continue
        key = (who.get("pid"), who.get("start_ticks"))
        if key in seen:
            continue
        seen.add(key)
        while alive(who):
            update(child=who, phase="waiting_for_existing_child", attempt=str(path.parent))
            time.sleep(1)


def run_job(out, manifest, job, case, update):
    folder = out / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = load(folder / "state.json", {})
    for name, field in [("results.csv", "results_sha256"),
                        ("exceptions.jsonl", "exceptions_sha256")]:
        if receipt.get("state") == "completed" and field in receipt:
            if sha(folder / name) != receipt[field]:
                raise ValueError("Completed job output changed: " + job["id"])
    if reuse_default(out, manifest, job, case):
        return
    for p in (folder / "attempts").glob("*/process.json") if (folder / "attempts").exists() else []:
        who = load(p)
        if alive(who):
            update(child=who, attempt=str(p.parent), phase="waiting_for_existing_child")
            while alive(who):
                time.sleep(1)
                update(child=who, attempt=str(p.parent), phase="waiting_for_existing_child")
        if not (p.parent / "collected.json").exists() and job["method"] != "repair-eval":
            collect_attempt(folder, p.parent, job)
            save(p.parent / "collected.json", dict(collected=utc()))
    if job["method"] == "repair-eval":
        run_repair_evaluation(out, manifest, job, case, update)
        return
    expected = expected_keys(case)
    if job["method"] == "beams":
        expected = beams_keys(out, case, expected, folder)
    attempts = folder / "attempts"
    attempts.mkdir(exist_ok=True)
    failures = 0
    while True:
        result, exc = terminal_data(folder)
        validate_outcomes(case, job, result, exc)
        missing = {k: v for k, v in expected.items() if k not in result and k not in exc}
        if not missing:
            if not (folder / "results.csv").exists():
                write_rows(folder / "results.csv", [])
            if not (folder / "exceptions.jsonl").exists():
                write_errors(folder / "exceptions.jsonl", [])
            save(folder / "state.json", dict(state="completed", ended=utc(),
                                           valid=len(result), exceptions=len(exc),
                                           results_sha256=sha(folder / "results.csv"),
                                           exceptions_sha256=sha(folder / "exceptions.jsonl")))
            return
        attempt = attempts / ("%04d" % (len(list(attempts.iterdir())) + 1))
        attempt.mkdir()
        fields = ["query_id", "expected_id", "ground_truth_rank"]
        if job["method"] == "beams":
            fields += ["ef_best", "k_best", "iann_time_ms"]
        write_rows(attempt / "keys.csv", missing.values(), fields)
        secondary = job["method"] in ["bisp", "unisp", "repair-refine"]
        selected = attempt / "expected.txt"
        if secondary:
            filtered_expected(case["expected"], selected, missing)
        env = child_environment(case, job, attempt, selected if secondary else case["expected"])
        if secondary:
            program = "pure_path_search" if job["method"] in ["bisp", "unisp"] else "xr_end_to_end"
            env["PURE_MODE"] = "bi_serial" if job["method"] == "bisp" else "single"
            command = [str(binary(program))]
        else:
            command = [str(binary("experiment")), "--method", job["method"],
                       "--index", case["index"], "--queries", case["queries"],
                       "--keys", str(attempt / "keys.csv"), "--output", str(attempt / "results.csv"),
                       "--errors", str(attempt / "errors.jsonl"),
                       "--progress", str(attempt / "progress.tsv"),
                       "--alpha", str(job["alpha"]), "--beta", str(job["beta"]),
                       "--i", str(job["i"]), "--ell", str(job["ell"]), "--budget-ms", "600000"]
        before = len(result) + len(exc)
        rc = run_child(command, env, attempt, manifest["cpu"], update)
        result, exc = collect_attempt(folder, attempt, job)
        save(attempt / "collected.json", dict(collected=utc(), exit_code=rc))
        progressed = len(result) + len(exc) > before
        if rc not in (0, 124) or not progressed:
            failures += 1
        else:
            failures = 0
        if failures >= 3:
            raise RuntimeError("Repeated process failure for %s; see %s" % (job["id"], attempt))


def run_repair_evaluation(out, manifest, job, case, update):
    folder = out / "jobs" / job["id"]
    if load(folder / "state.json", {}).get("state") == "completed":
        validate_repair_evaluation(Path(load(folder / "state.json")["summary"]).parents[1], case)
        return
    previous = sorted(p for p in folder.glob("evaluation*") if p.is_dir())
    for attempt in previous:
        who = load(attempt / "process.json")
        while alive(who):
            update(child=who, attempt=str(attempt), phase="waiting_for_existing_child")
            time.sleep(1)
        try:
            summary = validate_repair_evaluation(attempt, case)
        except (ValueError, KeyError, OSError):
            continue
        save(folder / "state.json", dict(state="completed", ended=utc(),
                                       summary=str(summary), expected_pairs=case["pairs"]))
        return
    failures = 0
    while True:
        attempt = folder / ("evaluation%04d" % (len(previous) + 1))
        attempt.mkdir()
        previous.append(attempt)
        checkpoint = folder / "repair_state.tsv"
        before = sha(checkpoint) if checkpoint.exists() else None
        env = child_environment(case, job, attempt, case["expected"])
        env["EXPERIMENT_REPAIR_STATE"] = str(checkpoint)
        env["REPAIRED_INDEX_PATH"] = str(folder / "repaired.index")
        rc = run_child([str(binary("index_reconstruction"))], env, attempt, manifest["cpu"], update)
        if rc == 124:
            after = sha(checkpoint) if checkpoint.exists() else None
            failures = failures + 1 if before == after else 0
            if failures >= 3:
                raise RuntimeError("Cumulative repair repeatedly timed out without recording a new outcome")
            continue
        if rc:
            raise RuntimeError("Cumulative repair/evaluation interrupted; resume this run after checking its log")
        summary = validate_repair_evaluation(attempt, case)
        save(folder / "state.json", dict(state="completed", ended=utc(),
                                       summary=str(summary), expected_pairs=case["pairs"]))
        return


def validate_repair_evaluation(attempt, case):
    summaries = list(attempt.glob("index_reconstruction/summary_*.csv"))
    records = rows(summaries[0]) if len(summaries) == 1 else []
    if len(records) != 2 or {r["strategy"] for r in records} != {"baseline", "repaired"}:
        raise ValueError("Missing complete before/after repair evaluation")
    count = DATASETS[case["dataset"]][1]
    for r in records:
        if int(r["n_queries"]) != count or int(r["k"]) != 100 or int(r["ef_eval"]) != 500:
            raise ValueError("Repair evaluation did not cover the paper query workload")
        if r["strategy"] == "repaired" and int(r["reconstruct_total"]) != case["pairs"]:
            raise ValueError("Cumulative repair omitted expected points")
        evaluation = list((attempt / "index_reconstruction/eval").glob(
            "eval_*_" + r["strategy"] + ".csv"))
        evaluation_rows = rows(evaluation[0]) if len(evaluation) == 1 else []
        qids = [int(v["query_id"]) for v in evaluation_rows]
        if len(qids) != count or set(qids) != set(range(count)):
            raise ValueError("Incomplete before/after query records")
    reconstruction = list((attempt / "index_reconstruction").glob("reconstruction_*_wvd-pareto.csv"))
    reconstructed = rows(reconstruction[0]) if len(reconstruction) == 1 else []
    keys = [point_key(r) for r in reconstructed]
    terminal_errors = set(errors(attempt / "errors.jsonl"))
    checkpoint = attempt.parent / "repair_state.tsv"
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0] == "EXCEPTION":
                terminal_errors.add((int(fields[1]), int(fields[2])))
    if len(keys) != len(set(keys)) or set(keys) & terminal_errors or (
            set(keys) | terminal_errors) != set(expected_keys(case)):
        raise ValueError("Cumulative repair coverage is incomplete")
    completed = list((attempt / "index_reconstruction").glob("completed_*.json"))
    receipt = load(completed[0], {}) if len(completed) == 1 else {}
    repaired = next(r for r in records if r["strategy"] == "repaired")
    if (receipt.get("queries") != count or receipt.get("repair_pairs") != case["pairs"] or
            receipt.get("repair_success") != int(repaired["reconstruct_success"])):
        raise ValueError("Missing committed cumulative-evaluation completion receipt")
    return summaries[0]


def summarize(out):
    manifest = load(out / "manifest.json")
    if not manifest:
        return {"state": "not_prepared"}
    summaries = []
    for job in manifest["jobs"]:
        folder = out / "jobs" / job["id"]
        state = load(folder / "state.json", {})
        base = dict(job_id=job["id"], case=job["case"], method=METHOD_NAMES[job["method"]],
                    ratio=job["ratio"], i=job["i"], ell=job["ell"],
                    expected_pairs=job["expected_pairs"], state=state.get("state", "pending"),
                    reused_from=state.get("reused_from"))
        if job["method"] == "repair-eval":
            base.update(evaluation_summary=state.get("summary"))
        else:
            r, e = terminal_data(folder)
            case = next(c for c in manifest["cases"] if c["id"] == job["case"])
            validate_outcomes(case, job, r, e)
            timed = list(r.values())
            times = [float(v["execution_time_ms"]) for v in timed]
            successful = [v for v in timed if v["status"] == "ok"]
            widths = [int(v["ef_best"]) for v in successful if v.get("ef_best")]
            ks = [int(v["k_best"]) for v in successful if v.get("k_best")]
            base.update(valid=len(timed), successful=len(successful),
                        timeouts=sum(v["status"] == "timeout" for v in timed),
                        no_feasible=sum(v["status"] == "no_feasible" for v in timed),
                        exceptions=len(e),
                        missing=job["expected_pairs"] - len(r) - len(e),
                        mean_ef=sum(widths) / len(widths) if widths else None,
                        mean_k=sum(ks) / len(ks) if ks else None, **stats(times))
        summaries.append(base)
    save(out / "summary.json", summaries)
    fields = list(dict.fromkeys(k for r in summaries for k in r))
    if summaries:
        write_rows(out / "summary.csv", summaries, fields)
    totals = dict(jobs_total=len(summaries),
                  jobs_complete=sum(r["state"] == "completed" for r in summaries),
                  timeouts=sum(r.get("timeouts", 0) for r in summaries),
                  exceptions=sum(r.get("exceptions", 0) for r in summaries),
                  missing=sum(r.get("missing", 0) for r in summaries))
    return totals


def worker(args, out):
    global QUEUE_LOCK_FD
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    lock_path = ROOT / "ExperResults/queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    who = process_id(os.getpid())
    state = dict(state="queued", worker=who, updated=utc(), current=None)
    def update(**changes):
        state.update(changes)
        state["updated"] = utc()
        save(out / "status.json", state)
    update()
    with lock_path.open("a+") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        QUEUE_LOCK_FD = queue_lock.fileno()
        try:
            wait_for_existing_children(update)
            manifest = load(out / "manifest.json")
            if manifest is None:
                manifest = make_manifest(out, args)
            elif manifest["sources"] != build_identity():
                raise ValueError("Source changed since this run; use a new run ID")
            for info in manifest["binaries"].values():
                assert_identity(info)
            for case in manifest["cases"]:
                for info in case["identities"]:
                    assert_identity(info)
                if case.get("keys_sha256") and sha(case["keys"]) != case["keys_sha256"]:
                    raise ValueError("Saved expected-point keys changed")
                if case.get("expected_sha256") and sha(case["expected"]) != case["expected_sha256"]:
                    raise ValueError("Saved merged expected points changed")
            case_map = {c["id"]: c for c in manifest["cases"]}
            update(state="running", cpu=manifest["cpu"], jobs_total=len(manifest["jobs"]))
            for job in manifest["jobs"]:
                update(current=job, child=None, phase="starting")
                run_job(out, manifest, job, case_map[job["case"]], update)
                update(summary=summarize(out), phase="completed")
            totals = summarize(out)
            if totals["missing"] or totals["jobs_complete"] != totals["jobs_total"]:
                raise RuntimeError("Finished queue has missing point outcomes")
            update(state="completed_with_exceptions" if totals["exceptions"] else "completed",
                   current=None, child=None, ended=utc(), summary=totals)
        except BaseException as exc:
            update(state="failed", error=str(exc), traceback=traceback.format_exc(), ended=utc())
            raise
        finally:
            QUEUE_LOCK_FD = None


def main(ground=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["launch", "resume", "status", "summarize", "_worker"])
    p.add_argument("--run-id", required=True)
    if not ground:
        p.add_argument("--experiment", choices=FAMILIES, default="all")
    p.add_argument("--source-run", type=Path)
    p.add_argument("--cpu", type=int)
    args = p.parse_args()
    if ground:
        args.experiment = "ground"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        p.error("Run ID may contain only letters, digits, underscores, and hyphens")
    out = ROOT / "ExperResults" / args.run_id
    if args.action == "status":
        state = load(out / "status.json", {"state": "not_started"})
        state["worker_alive"] = alive(state.get("worker"))
        print(json.dumps(state, indent=2))
        return
    if args.action == "summarize":
        print(json.dumps(summarize(out), indent=2))
        return
    if fcntl is None or not hasattr(os, "sched_setaffinity"):
        p.error("Detached experiment execution requires Linux")
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "_worker":
        worker(args, out)
        return
    with (out / "launch.lock").open("a+") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        state = load(out / "status.json", {})
        previous = load(out / "launch.json", {})
        record = state.get("worker") if alive(state.get("worker")) else previous.get("worker")
        if alive(record):
            print(json.dumps(dict(state="already_running", worker=record), indent=2))
            return
        if state.get("state") in ["completed", "completed_with_exceptions"]:
            print(json.dumps(dict(state="already_completed", directory=str(out)), indent=2))
            return
        script = ROOT / "scripts" / ("generate_ground.py" if ground else "run_experiments.py")
        command = [sys.executable, "-u", str(script), "_worker", "--run-id", args.run_id]
        if not ground:
            command += ["--experiment", args.experiment]
        if args.source_run:
            command += ["--source-run", str(args.source_run.resolve())]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        with (out / "supervisor.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     close_fds=True, env=dict(os.environ, **THREAD_ENV))
        record = dict(worker=process_id(child.pid), launched=utc(), directory=str(out))
        save(out / "launch.json", record)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
