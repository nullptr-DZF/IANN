#!/usr/bin/env python3
"""Convert the six paper datasets and compute exact top-1000 neighbors."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "GLOVE100": dict(source="glove-100-angular.hdf5", dim=100, queries=10000,
                     vectors=1183514, normalize=True),
    "DEEP10M": dict(source="deep-image-96-angular.hdf5", dim=96, queries=10000,
                    vectors=9990000, normalize=True),
    "MSONG420": dict(source="msong-420.hdf5", dim=420, queries=1000,
                     vectors=983185, normalize=False),
    "LASTFM65": dict(source="lastfm-64-dot.hdf5", dim=65, queries=10000,
                     vectors=292385, normalize=True),
    "IMAGENET": dict(source="imagenet-dinov2/dinov2-train.bin",
                     query="imagenet-dinov2/dinov2-validation.bin",
                     labels="imagenet-dinov2/dinov2-train-label.bin",
                     query_labels="imagenet-dinov2/dinov2-validation-label.bin",
                     dim=768, queries=10000, vectors=1281167, normalize=False),
    "GLINK": dict(source="glink-vit/vit-train.bin", query="glink-vit/vit-validation.bin",
                  labels="glink-vit/vit-train-label.bin",
                  query_labels="glink-vit/vit-validation-label.bin",
                  dim=512, queries=10000, vectors=17091649, normalize=False),
}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def normalize_rows(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1
    rows /= norms
    return rows


def write_xvecs(stream, rows, kind="f"):
    n, d = rows.shape
    dtype = "<f4" if kind == "f" else "<i4"
    output = np.empty((n, d + 1), dtype=dtype)
    output[:, :1].view("<i4")[:] = d
    output[:, 1:] = rows
    output.tofile(stream)


class HDF5Source:
    def __init__(self, dataset):
        if dataset.ndim != 2:
            raise ValueError("An HDF5 vector array must have two dimensions")
        self.dataset = dataset
        self.n, self.d = dataset.shape

    def read(self, start, end):
        return np.array(self.dataset[start:end], dtype=np.float32, copy=True)


class FvecsSource:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as stream:
            header = stream.read(4)
        if len(header) != 4:
            raise ValueError("Empty fvecs input: " + str(path))
        self.d = int(np.frombuffer(header, dtype="<i4")[0])
        if self.d <= 0 or path.stat().st_size % (4 * (self.d + 1)):
            raise ValueError("Invalid fvecs length: " + str(path))
        self.n = path.stat().st_size // (4 * (self.d + 1))

    def read(self, start, end):
        records = np.fromfile(self.path, dtype="<f4",
                              count=(end-start)*(self.d+1), offset=start*(self.d+1)*4)
        records = records.reshape(end-start, self.d+1)
        if not np.all(records[:, :1].view("<i4") == self.d):
            raise ValueError("Mixed fvecs dimensions: " + str(self.path))
        return records[:, 1:].copy()


class LabelledSource:
    """Iceberg records store a uint32 label followed by float32 coordinates."""
    def __init__(self, path, labels_path, dim):
        self.path, self.labels_path, self.d = path, labels_path, dim
        self.dtype = np.dtype([("label", "<u4"), ("vector", "<f4", (dim,))])
        if path.stat().st_size % self.dtype.itemsize:
            raise ValueError("Invalid labelled-record file length: " + str(path))
        self.n = path.stat().st_size // self.dtype.itemsize
        self._cached_range = None
        self._cached_records = None
        if labels_path.stat().st_size != self.n * 4:
            raise ValueError("Standalone label count differs: " + str(labels_path))

    def records(self, start, end):
        if self._cached_range == (start, end):
            return self._cached_records
        records = np.fromfile(self.path, dtype=self.dtype, count=end-start,
                              offset=start*self.dtype.itemsize)
        labels = np.fromfile(self.labels_path, dtype="<u4", count=end-start,
                             offset=start*4)
        if len(records) != end-start or not np.array_equal(records["label"], labels):
            raise ValueError("Embedded and standalone labels differ: " + str(self.path))
        self._cached_range, self._cached_records = (start, end), records
        return records

    def read(self, start, end):
        return self.records(start, end)["vector"].copy()

    def labels(self, start, end):
        return self.records(start, end)["label"].copy()


def select_hdf5_array(handle, candidates):
    for key in candidates:
        if key in handle:
            return handle[key]
    raise ValueError("Missing vector array; expected one of " + ", ".join(candidates))


def open_sources(key, root):
    cfg = DATASETS[key]
    raw = root / "datasets"
    handles = []
    paths = []
    if "query" in cfg:
        paths = [raw/cfg[name] for name in ["source", "query", "labels", "query_labels"]]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError("Download all feature and label files for " + key)
        sources = (LabelledSource(paths[0], paths[2], cfg["dim"]),
                   LabelledSource(paths[1], paths[3], cfg["dim"]))
    else:
        source = raw/cfg["source"]
        if key == "MSONG420" and not source.is_file():
            for base, query in [(raw/"msong"/"base.fvecs", raw/"msong"/"query.fvecs"),
                                (raw/"msong_base.fvecs", raw/"msong_query.fvecs")]:
                if base.is_file() and query.is_file():
                    return FvecsSource(base), FvecsSource(query), [], [base, query]
            alternative = raw/"msong-420-euclidean.hdf5"
            if alternative.is_file():
                source = alternative
        if not source.is_file():
            raise FileNotFoundError("Required source file: " + str(source))
        import h5py
        handle = h5py.File(source, "r")
        handles.append(handle)
        sources = (HDF5Source(select_hdf5_array(handle, ["train", "base", "dataset", "db"])),
                   HDF5Source(select_hdf5_array(handle, ["test", "query", "queries"])))
        paths = [source]
    return sources[0], sources[1], handles, paths


def convert_dataset(key, root=ROOT, force=False, base_chunk=10000, query_chunk=64):
    root = Path(root).resolve()
    cfg = DATASETS[key]
    output = root/"datasets"/"processed"/key
    output.mkdir(parents=True, exist_ok=True)
    base, query, handles, source_paths = open_sources(key, root)
    try:
        if base.d != cfg["dim"] or query.d != cfg["dim"]:
            raise ValueError("%s requires %d feature coordinates, got base=%d query=%d" %
                             (key, cfg["dim"], base.d, query.d))
        wanted = cfg["queries"]
        if wanted <= 0:
            raise ValueError("Query count must be positive")
        nbase = base.n
        nquery = min(query.n, wanted)
        if nquery < wanted and key != "MSONG420":
            raise ValueError("%s requires %d queries, source has %d" % (key, wanted, query.n))
        sources = [identity(p) for p in source_paths]
        settings = dict(schema=3, dataset=key, sources=sources, dim=cfg["dim"],
                        queries=wanted, seed=100,
                        gt_k=min(1000, nbase-max(0, wanted-nquery)),
                        distance_accumulation="float64", tie_order="distance_then_id",
                        normalized=cfg["normalize"], base_chunk=base_chunk,
                        query_chunk=query_chunk)
        marker = output/"conversion.json"
        if marker.exists() and not force:
            old = json.loads(marker.read_text(encoding="utf-8"))
            names = ["base.fvecs", "query.fvecs", "query.groundtruth.k1000.ivecs"]
            if "query" in cfg:
                names += ["base.labels.u32", "query.labels.u32"]
            if old.get("settings") == settings and all(
                    (output/name).is_file() and (output/name).stat().st_size ==
                    old.get("output_bytes", {}).get(name) for name in names):
                print("Prepared vectors retained:", key, flush=True)
                return old
        queries = query.read(0, nquery)
        moved = np.empty(0, dtype=np.int64)
        keep = None
        if nquery < wanted:
            rng = np.random.default_rng(100)
            moved = np.sort(rng.choice(nbase, size=wanted-nquery, replace=False))
            heldout = np.stack([base.read(int(i), int(i)+1)[0] for i in moved])
            queries = np.concatenate([queries, heldout])
            keep = np.ones(nbase, dtype=bool)
            keep[moved] = False
        if not np.isfinite(queries).all():
            raise ValueError("Query vectors contain nonfinite coordinates")
        if cfg["normalize"]:
            normalize_rows(queries)
        count = int(keep.sum()) if keep is not None else nbase
        if count < 1:
            raise ValueError("The base contains no vectors")
        if count != cfg["vectors"]:
            raise ValueError("%s paper base has %d vectors; supplied split has %d. "
                             "Use the published benchmark split, not an unrelated raw-song split."
                             % (key, cfg["vectors"], count))
        depth = min(1000, count)
        settings["gt_k"] = depth
        best_d = np.full((wanted, depth), np.inf, dtype=np.float64)
        best_i = np.full((wanted, depth), -1, dtype=np.int32)
        exact_queries = queries.astype(np.float64)
        qpath = output/"query.fvecs.partial"
        with qpath.open("wb") as stream:
            write_xvecs(stream, queries)
        if "query" in cfg:
            query.labels(0, wanted).astype("<u4").tofile(output/"query.labels.u32.partial")
        cursor = 0
        t0 = time.monotonic()
        label_stream = (output/"base.labels.u32.partial").open("wb") if "query" in cfg else None
        try:
            with (output/"base.fvecs.partial").open("wb") as stream:
                for start in range(0, nbase, base_chunk):
                    end = min(nbase, start+base_chunk)
                    values = base.read(start, end)
                    if keep is not None:
                        values = values[keep[start:end]]
                    if not len(values):
                        continue
                    if not np.isfinite(values).all():
                        raise ValueError("Base vectors contain nonfinite coordinates")
                    if cfg["normalize"]:
                        normalize_rows(values)
                    write_xvecs(stream, values)
                    if label_stream is not None:
                        base.labels(start, end).astype("<u4").tofile(label_stream)
                    ids = np.arange(cursor, cursor+len(values), dtype=np.int32)
                    # Double precision prevents float32 dot-product cancellation ties.
                    exact_values = values.astype(np.float64)
                    norms = np.einsum("ij,ij->i", exact_values, exact_values)
                    for qs in range(0, wanted, query_chunk):
                        qe = min(wanted, qs+query_chunk)
                        distances = norms[None, :] - 2*(exact_queries[qs:qe] @ exact_values.T)
                        candidates_d = np.concatenate([best_d[qs:qe], distances], axis=1)
                        candidates_i = np.concatenate(
                            [best_i[qs:qe], np.broadcast_to(ids, distances.shape)], axis=1)
                        for row in range(qe-qs):
                            chosen = np.argpartition(candidates_d[row], depth-1)[:depth]
                            boundary = np.max(candidates_d[row, chosen])
                            strict = np.flatnonzero(candidates_d[row] < boundary)
                            tied = np.flatnonzero(candidates_d[row] == boundary)
                            tied = tied[np.argsort(candidates_i[row, tied], kind="stable")]
                            selected = np.concatenate([strict, tied[:depth-len(strict)]])
                            order = np.lexsort((candidates_i[row, selected], candidates_d[row, selected]))
                            selected = selected[order]
                            best_d[qs+row] = candidates_d[row, selected]
                            best_i[qs+row] = candidates_i[row, selected]
                    cursor += len(values)
                    print("%s: converted %d/%d base vectors (%.1fs)" %
                          (key, end, nbase, time.monotonic()-t0), flush=True)
        finally:
            if label_stream is not None:
                label_stream.close()
        with (output/"query.groundtruth.k1000.ivecs.partial").open("wb") as stream:
            write_xvecs(stream, best_i, "i")
        names = ["base.fvecs", "query.fvecs", "query.groundtruth.k1000.ivecs"]
        if "query" in cfg:
            names += ["base.labels.u32", "query.labels.u32"]
        for name in names:
            (output/(name+".partial")).replace(output/name)
        record = dict(settings=settings, n_base=count, n_queries=wanted, dim=cfg["dim"],
                      moved_base_ids=moved.tolist(),
                      output_bytes={name:(output/name).stat().st_size for name in names})
        atomic_json(marker, record)
        return record
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--base-chunk", type=int, default=10000)
    parser.add_argument("--query-chunk", type=int, default=64)
    args = parser.parse_args()
    if min(args.base_chunk, args.query_chunk) <= 0:
        parser.error("Chunk sizes must be positive")
    convert_dataset(args.dataset, args.root, args.force, args.base_chunk, args.query_chunk)


if __name__ == "__main__":
    main()
