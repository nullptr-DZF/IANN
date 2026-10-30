# IANN: Experimental Code

[English](README.md) | [简体中文](README.zh-CN.md)

This repository provides the C++ implementation and scripts for the IANN experiments.

Due to repository size limits, the code archive does not include datasets, pre-extracted vectors, labels, indexes, expected-point files, precomputed Ground results, experimental outputs, binaries, or plotting code. Download the data and generate the required inputs using the steps below.

## 1. Components

| Component | Function |
|---|---|
| IANN | Provenance computation, reachability repair when required, and parameter refinement |
| BiSP / UniSP | Bidirectional / forward-only provenance-path computation |
| IncOnly | Incremental exploration without lower-bound initialization |
| LBOnly | Lower-bound initialization followed by fresh beam searches |
| IncRerun | Fresh beam searches at consecutive widths from the original setting |
| BeamS | One native ANN query using previously refined parameters |
| Ground | Generation of optimal parameter references |

## 2. Build

Requirements: Linux x86-64, GCC with C++11 and OpenMP support, CMake 3.13 or later, GNU Make, Python 3, NumPy, h5py, and curl.

Run all commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy h5py

cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

All experiments execute serially, with one worker and one computation thread. Index construction, preprocessing, and Ground generation also use a single thread. The runner binds each computation process to one allowed CPU before loading the index. SIMD distance computation remains enabled on supported CPUs.

## 3. Download and place the data

### Benchmark datasets

The [ANN-Benchmarks dataset list](https://github.com/erikbern/ann-benchmarks#data-sets) provides the Glove, Deep, and Last HDF5 files:

```bash
mkdir -p datasets

curl -fL --retry 3 https://ann-benchmarks.com/glove-100-angular.hdf5 \
  -o datasets/glove-100-angular.hdf5
curl -fL --retry 3 https://ann-benchmarks.com/deep-image-96-angular.hdf5 \
  -o datasets/deep-image-96-angular.hdf5
curl -fL --retry 3 https://ann-benchmarks.com/lastfm-64-dot.hdf5 \
  -o datasets/lastfm-64-dot.hdf5
```

MSong originates from [MSongsDB / Million Song Dataset](https://github.com/tbertinmahieux/MSongsDB/tree/master). The 420-dimensional benchmark matrix is available through the [PDX authors' dataset collection](https://github.com/cwida/PDX/blob/main/BENCHMARKING.md#manually-downloading-data):

```bash
curl -fL --retry 3 \
  'https://drive.usercontent.google.com/download?id=10NBa0OZqLc46P2tMEXMDRTbVoM10uj0c&export=download&confirm=t' \
  -o datasets/msong-420.hdf5
```

The input contains `train` with shape `(983185,420)` and `test` with shape `(1000,420)`. Alternatively, download the [processed HDF5 file](https://drive.google.com/file/d/10NBa0OZqLc46P2tMEXMDRTbVoM10uj0c/view) through the browser and place it at the same path.

### Application datasets

Download the Image and Glink features and labels from [PIIR/Iceberg-dataset](https://huggingface.co/datasets/PIIR/Iceberg-dataset):

```bash
HF=https://huggingface.co/datasets/PIIR/Iceberg-dataset/resolve/main
mkdir -p datasets/imagenet-dinov2 datasets/glink-vit

for file in dinov2-train.bin dinov2-validation.bin \
            dinov2-train-label.bin dinov2-validation-label.bin; do
  curl -fL --retry 3 "$HF/imagenet-1k/$file" \
    -o "datasets/imagenet-dinov2/$file"
done

for file in vit-train.bin vit-validation.bin \
            vit-train-label.bin vit-validation-label.bin; do
  curl -fL --retry 3 "$HF/glink/$file" \
    -o "datasets/glink-vit/$file"
done
```

Each application record contains a 32-bit label followed by 768 Image or 512 Glink float32 coordinates. Preprocessing separates the label from the feature vector. The standalone label files provide the class/identity information for expected-point generation.

## 4. Prepare the full paper workload

```bash
python3 scripts/prepare_data.py --all
```

This command converts the data, computes exact top-1000 neighbors, builds the HNSW indexes, and generates expected points for the following workload.

| Dataset | Internal key | Base vectors | Dimensions | Queries |
|---|---|---:|---:|---:|
| Glove | `GLOVE100` | 1,183,514 | 100 | 10,000 |
| Deep | `DEEP10M` | 9,990,000 | 96 | 10,000 |
| MSong | `MSONG420` | 983,185 | 420 | 1,000 |
| Last | `LASTFM65` | 292,385 | 65 | 10,000 |
| Image | `IMAGENET` | 1,281,167 | 768 | 10,000 |
| Glink | `GLINK` | 17,091,649 | 512 | 10,000 |

Glove, Deep, and Last vectors are L2-normalized for cosine-equivalent ranking. MSong, Image, and Glink use their Euclidean coordinates. If the MSong query set is short, the converter holds out the required vectors from the base using a fixed seed and records the resulting IDs.

### Expected-point generation

| Datasets | Set | Rule per query |
|---|---|---|
| Glove, Deep, MSong, Last | `top` | Select up to two missing exact top-100 neighbors |
| Glove, Deep, MSong, Last | `deep` | Partition exact ranks (100,1000] into eight equal-width buckets; select one missing point from each nonempty bucket |
| Glove, Deep, MSong, Last | `du` | Apply the same buckets, restricting candidates to unvisited points |
| Image, Glink | `vis`, `unvis` | First select up to five missing same-class/same-identity points in total, then split them by visit status |
| All datasets | `unreach` | Collect selected expected points unreachable from the fixed layer-0 entry |

For an application query, the five-point cap applies to `vis` and `unvis` combined, before the reachability split. All generated expected points are processed by the experiments. Sampling decisions are saved and reused across methods and parameter settings.

The number of generated pairs depends on available candidates and reachability. It is computed from the generated files, not set to a previously measured total.

### Generated files

```text
datasets/processed/<KEY>/
  base.fvecs
  query.fvecs
  query.groundtruth.k1000.ivecs
  expected_vectors_<KEY>_M16_efC500_efS500_k100_<kind>.txt
  expected_vectors_<KEY>_M16_efC500_efS500_k100_<kind>_unreachable.txt

IndexSave/<KEY>.index
IndexSave/<KEY>_M8.index
```

File suffixes are `top100`, `deep`, and `deep_unvisited` for the benchmark sets, and `missed` and `unvisited` for the application sets. The corresponding `M8` expected-point files are generated separately for index-repair experiments.

## 5. Default parameters

| Parameter | Value |
|---|---:|
| HNSW `M` | 16 |
| `efConstruction` | 500 |
| Index seed | 100 |
| Original `efSearch` | 500 |
| Original `k` | 100 |
| `alpha`, `beta` | 0.99, 0.01 |
| `i`, `ell` | 5, 5 |
| Time budget per query/expected-point pair | 600,000 ms |

The layer-0 entry obtained by upper-layer descent stays fixed during refinement. The stride and stopping interval are:

```text
s   = max(1, floor(i * alpha / beta))
tau = ell * s
```

At the defaults, `s=495` and `tau=2475`. IncOnly, LBOnly, and IncRerun use the same refinement stopping rules.

## 6. Run the experiments

### Complete experiment queue

```bash
python3 scripts/run_experiments.py launch \
  --experiment all --run-id paper
```

The detached supervisor runs the experiment families below serially. SSH disconnection does not stop the queue. Default results are reused when input files, parameters, and executable versions match.

### Individual experiment families

The overall experiment runs IANN on all 16 reachable expectation sets:

```bash
python3 scripts/run_experiments.py launch \
  --experiment overall --run-id overall
```

After the overall run completes, the following commands can reuse its default results:

```bash
python3 scripts/run_experiments.py launch \
  --experiment ablations --run-id ablations \
  --source-run ExperResults/overall

python3 scripts/run_experiments.py launch \
  --experiment paths --run-id paths

python3 scripts/run_experiments.py launch \
  --experiment repair --run-id repair

python3 scripts/run_experiments.py launch \
  --experiment weights --run-id weights \
  --source-run ExperResults/overall

python3 scripts/run_experiments.py launch \
  --experiment rank-gain --run-id rank_gain \
  --source-run ExperResults/overall

python3 scripts/run_experiments.py launch \
  --experiment patience --run-id patience \
  --source-run ExperResults/overall
```

| Family | Full execution scope |
|---|---|
| `overall` | IANN on every generated reachable pair in the 16 expectation sets |
| `ablations` | IncOnly, LBOnly, IncRerun, and BeamS on the same full workload; reuse default IANN results |
| `paths` | BiSP and UniSP on the same full expectation sets |
| `repair` | At both `M=16` and `M=8`, run per-point repair/refinement and a separate cumulative-repair evaluation on all standard queries |
| `weights` | `alpha/beta = 20,40,60,80,99`, with `alpha=r/(r+1)` and `beta=1/(r+1)` |
| `rank-gain` | `i = 1,3,5,7,9`, with other parameters fixed |
| `patience` | `ell = 1,2,3,5,10,15`, with other parameters fixed |

The sensitivity families use the same saved expected-point selections. Their default settings (`r=99`, `i=5`, `ell=5`) reuse one default measurement, giving 14 unique parameter configurations per expectation set. IANN and ablation runtimes include upper-layer descent, BiSP provenance computation, and parameter refinement. These experiments use reachable expected points and do not perform repair.

BeamS performs one native ANN query with each saved feasible `(ef_best, k_best)` pair. Its timer includes upper-layer descent and result construction, excluding parameter refinement, index loading, parameter assignment, and CSV I/O. Pairs without feasible parameters are reported separately.

The per-point repair experiment restores the original index after each point. Its reported runtime sums repair and subsequent provenance/refinement time, excluding the initial query, index snapshots, and restoration. The separate cumulative experiment applies all repairs, then measures the full standard query workload with `k=100` and `efSearch=500` before and after repair. Its query runtimes exclude repair time.

### Check progress, resume interrupted work, and generate result tables

`--run-id paper` identifies this experiment run, whose records are stored under `ExperResults/paper/`. For an individual experiment family, substitute its actual run ID.

**Check progress (`status`):** Read the saved state and display the current experiment family, dataset, parameter configuration, completed/total jobs, and running/completed/failed state. This command performs one status query and does not start an experiment.

```bash
python3 scripts/run_experiments.py status --run-id paper
```

**Resume interrupted work (`resume`):** After an unexpected process exit or machine restart, read the original run manifest and durable results, then continue unfinished work. Recorded successful, timeout, and exception outcomes are preserved; input samples are not regenerated. An already running worker is not launched again.

```bash
python3 scripts/run_experiments.py resume --run-id paper
```

Resume continues from saved results, not from the internal state of a C++ search. Only interrupted points without a complete terminal-outcome record are recomputed; points with recorded exceptions are not rerun. An SSH disconnection alone does not require `resume` if the background process is still running.

**Generate result tables (`summarize`):** Read existing per-point results and compute runtime/refined-parameter statistics and successful, timeout, exception, and missing counts, grouped by dataset, expectation set, method, and parameter configuration. Write `ExperResults/paper/summary.csv` and `summary.json`. This command does not rerun searches or compare against Ground. Summaries of an unfinished run explicitly report the remaining missing outcomes.

```bash
python3 scripts/run_experiments.py summarize --run-id paper
```

## 7. Generate Ground parameters

```bash
python3 scripts/generate_ground.py launch --run-id ground
python3 scripts/generate_ground.py status --run-id ground
python3 scripts/generate_ground.py resume --run-id ground
python3 scripts/generate_ground.py summarize --run-id ground
```

This standalone tool generates parameter references for the full reachable expected-point workload using the current exact-search implementation. It saves the parameter pair, objective value, completion status, and runtime. A timed-out search is marked explicitly and is not labeled as an optimal result. Ground generation is a separate queue from `--experiment all`.

## 8. Results

```text
ExperResults/<run-id>/
  manifest.json
  status.json
  supervisor.log
  inputs/
  jobs/<job-id>/results.csv
  jobs/<job-id>/exceptions.jsonl
  summary.csv
  summary.json
```

The manifest records query coverage, selected expected-point IDs, parameters, random seeds, and source/build identities.

Point-based statistics use one `(query, expected-point)` pair as the unit; cumulative-repair evaluation uses one standard query. Runtime excludes index loading and file I/O. A timeout counts as a valid 600,000 ms observation; an exception has no timing value and is excluded from the mean. Summaries report valid, successful, timeout, exception, and missing counts separately. For LBOnly and IncRerun, the timeout check occurs after a complete cold search returns; raw elapsed time is saved separately.
