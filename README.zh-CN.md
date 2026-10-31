# IANN：实验代码

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供 IANN 实验的 C++ 实现及运行脚本。

## 1. 组件

| 组件 | 功能 |
|---|---|
| IANN | 溯源计算、必要时恢复可达性，以及参数精炼 |
| BiSP / UniSP | 双向／仅正向的溯源路径计算 |
| IncOnly | 不使用下界初始化的增量探索 |
| LBOnly | 从下界开始，在每个宽度上重新执行 beam search |
| IncRerun | 从原始设置开始，在连续宽度上重新执行 beam search |
| BeamS | 使用已精炼参数执行一次原生 ANN 查询 |
| Ground | 生成最优参数参考结果 |

## 2. 编译

环境要求：Linux x86-64、支持 C++11 和 OpenMP 的 GCC、CMake 3.13 或更高版本、GNU Make、Python 3、NumPy、h5py 及 curl。

所有命令均在仓库根目录运行：

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

全部实验采用单工作进程、单计算线程串行执行。索引构建、预处理和 Ground 生成也使用单线程。运行器在加载索引前，将每个计算进程绑定到一个允许使用的 CPU；支持的 CPU 仍启用 SIMD 距离计算。

## 3. 下载与放置数据

### 基准数据集

[ANN-Benchmarks 数据集列表](https://github.com/erikbern/ann-benchmarks#data-sets)提供 Glove、Deep 和 Last 的 HDF5 文件：

```bash
mkdir -p datasets

curl -fL --retry 3 https://ann-benchmarks.com/glove-100-angular.hdf5 \
  -o datasets/glove-100-angular.hdf5
curl -fL --retry 3 https://ann-benchmarks.com/deep-image-96-angular.hdf5 \
  -o datasets/deep-image-96-angular.hdf5
curl -fL --retry 3 https://ann-benchmarks.com/lastfm-64-dot.hdf5 \
  -o datasets/lastfm-64-dot.hdf5
```

MSong 来源为 [MSongsDB / Million Song Dataset](https://github.com/tbertinmahieux/MSongsDB/tree/master)。420维基准矩阵可通过 [PDX 作者的数据集目录](https://github.com/cwida/PDX/blob/main/BENCHMARKING.md#manually-downloading-data)获取：

```bash
curl -fL --retry 3 \
  'https://drive.usercontent.google.com/download?id=10NBa0OZqLc46P2tMEXMDRTbVoM10uj0c&export=download&confirm=t' \
  -o datasets/msong-420.hdf5
```

该输入包含形状为 `(983185,420)` 的 `train` 和形状为 `(1000,420)` 的 `test`。也可通过浏览器下载[处理后的 HDF5 文件](https://drive.google.com/file/d/10NBa0OZqLc46P2tMEXMDRTbVoM10uj0c/view)，放到上述路径。

### 应用数据集

从 [PIIR/Iceberg-dataset](https://huggingface.co/datasets/PIIR/Iceberg-dataset) 下载 Image 和 Glink 的特征及标签：

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

每条应用数据记录由一个32位标签和 float32 特征坐标组成，Image 为768维，Glink 为512维。预处理时将标签与特征向量分离。独立标签文件提供生成期望点所需的类别／身份信息。

## 4. 准备论文全量工作负载

```bash
python3 scripts/prepare_data.py --all
```

该命令转换数据、计算精确 top-1000 近邻、构建 HNSW 索引，并为以下工作负载生成期望点。

| 数据集 | 内部标识 | 基础向量数 | 维度 | 查询数 |
|---|---|---:|---:|---:|
| Glove | `GLOVE100` | 1,183,514 | 100 | 10,000 |
| Deep | `DEEP10M` | 9,990,000 | 96 | 10,000 |
| MSong | `MSONG420` | 983,185 | 420 | 1,000 |
| Last | `LASTFM65` | 292,385 | 65 | 10,000 |
| Image | `IMAGENET` | 1,281,167 | 768 | 10,000 |
| Glink | `GLINK` | 17,091,649 | 512 | 10,000 |

Glove、Deep 和 Last 向量进行 L2 归一化，以获得与余弦距离等价的排名。MSong、Image 和 Glink 使用其欧氏坐标。如果 MSong 查询集数量不足，转换器使用固定种子从基础数据中留出所需向量，并记录生成的 ID。

### 期望点生成

| 数据集 | 期望集 | 每查询的选择规则 |
|---|---|---|
| Glove、Deep、MSong、Last | `top` | 最多选择两个缺失的精确 top-100 近邻 |
| Glove、Deep、MSong、Last | `deep` | 将精确排名 (100,1000] 划分为八个等宽桶，每个非空桶选择一个缺失点 |
| Glove、Deep、MSong、Last | `du` | 使用相同分桶，仅从未访问点中选择 |
| Image、Glink | `vis`、`unvis` | 先总共选择最多五个缺失的同类别／同身份点，再按访问状态拆分 |
| 全部数据集 | `unreach` | 收集从固定底层入口不可达的已选期望点 |

对于应用查询，五点上限作用于 `vis` 与 `unvis` 的合计数量，并在可达性拆分前执行。实验处理全部生成的期望点。采样选择会保存，并在不同方法和参数设置间复用。

生成的点对数取决于可用候选及可达性，依据生成文件统计，不预设为此前实验测得的总数。

### 生成文件

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

基准期望集的文件后缀为 `top100`、`deep` 和 `deep_unvisited`；应用期望集的后缀为 `missed` 和 `unvisited`。索引修复实验所需的对应 `M8` 期望点文件单独生成。

## 5. 默认参数

| 参数 | 数值 |
|---|---:|
| HNSW `M` | 16 |
| `efConstruction` | 500 |
| 索引种子 | 100 |
| 原始 `efSearch` | 500 |
| 原始 `k` | 100 |
| `alpha`、`beta` | 0.99、0.01 |
| `i`、`ell` | 5、5 |
| 每个查询／期望点对的时间预算 | 600,000 ms |

通过上层下降得到的底层入口在精炼过程中保持固定。步长与停止区间为：

```text
s   = max(1, floor(i * alpha / beta))
tau = ell * s
```

默认设置下，`s=495`、`tau=2475`。IncOnly、LBOnly 和 IncRerun 使用相同的精炼停止规则。

## 6. 运行实验

### 完整实验队列

```bash
python3 scripts/run_experiments.py launch \
  --experiment all --run-id paper
```

脱离终端会话的调度器会串行运行下列实验类别，SSH 断开不会停止队列。当输入文件、参数和可执行程序版本一致时，复用默认参数结果。

### 单独运行各实验类别

整体性能实验对全部16个可达期望集运行 IANN：

```bash
python3 scripts/run_experiments.py launch \
  --experiment overall --run-id overall
```

整体性能实验完成后，以下命令可复用其默认参数结果：

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

| 类别 | 全量执行范围 |
|---|---|
| `overall` | 对16个期望集中全部生成的可达点对运行 IANN |
| `ablations` | 在相同全量工作负载上运行 IncOnly、LBOnly、IncRerun 和 BeamS，复用默认 IANN 结果 |
| `paths` | 在相同的全量期望集上运行 BiSP 和 UniSP |
| `repair` | 在 `M=16` 和 `M=8` 下，分别执行逐点修复与精炼实验，以及累计修复后的全量标准查询评估 |
| `weights` | `alpha/beta = 20,40,60,80,99`，其中 `alpha=r/(r+1)`、`beta=1/(r+1)` |
| `rank-gain` | `i = 1,3,5,7,9`，其余参数固定 |
| `patience` | `ell = 1,2,3,5,10,15`，其余参数固定 |

敏感性实验使用相同的已保存期望点选择。其默认设置（`r=99`、`i=5`、`ell=5`）复用同一次默认测量，因此每个期望集共有14个唯一参数配置。IANN 与消融方法的耗时包括上层下降、BiSP 路径计算及参数精炼。这些实验使用可达期望点，不执行修复。

BeamS 对每个已保存且可行的 `(ef_best, k_best)` 参数对执行一次原生 ANN 查询。计时包括上层下降和结果构造，不包括参数精炼、索引加载、参数赋值及 CSV 读写。没有可行参数的点对单独报告。

逐点修复实验在处理每个点后恢复原始索引。其耗时为修复与后续路径计算、参数精炼之和，不包括初始查询、索引快照与恢复操作。独立的累计修复实验应用全部修复，再对修复前后索引运行全部标准查询，均使用 `k=100` 和 `efSearch=500`；该查询耗时不包含修复时间。

### 查看进度、中断续跑与生成结果表

`--run-id paper` 指定本次实验，相关记录保存在 `ExperResults/paper/`。单独运行某个实验类别时，替换为该类别实际使用的运行 ID。

**查看进度（`status`）：** 读取已保存状态，显示当前实验类别、数据集、参数配置、已完成／总任务数，以及运行、完成或失败状态。该命令只查询一次，不启动实验。

```bash
python3 scripts/run_experiments.py status --run-id paper
```

**中断续跑（`resume`）：** 实验进程意外退出或机器重启后，读取原运行清单和已落盘结果，继续处理尚未完成的任务。已记录的成功、超时和异常结果保留，输入样本不重新生成。若任务仍在运行，不会重复启动。

```bash
python3 scripts/run_experiments.py resume --run-id paper
```

续跑从已保存结果处继续，不恢复 C++ 搜索内部的执行状态；只重新计算尚无完整终态记录的中断点，已有异常记录的点不重跑。仅 SSH 断开而后台进程仍在运行时，无需执行 `resume`。

**生成结果表（`summarize`）：** 读取已有逐点结果，按数据集、期望集、方法和参数配置计算运行时间与精炼参数的统计量，以及成功、超时、异常和缺失数量，写入 `ExperResults/paper/summary.csv` 和 `summary.json`。该命令不重新运行搜索，也不进行 Ground 比较。实验未结束时，汇总表会明确显示尚未完成的数量。

```bash
python3 scripts/run_experiments.py summarize --run-id paper
```

## 7. 生成 Ground 参数

```bash
python3 scripts/generate_ground.py launch --run-id ground
python3 scripts/generate_ground.py status --run-id ground
python3 scripts/generate_ground.py resume --run-id ground
python3 scripts/generate_ground.py summarize --run-id ground
```

该独立工具使用当前精确搜索实现，为全量可达期望点工作负载生成参数参考结果，并保存参数对、目标值、完成状态与耗时。超时搜索会明确标记，不视为最优结果。Ground 生成队列独立于 `--experiment all`。

## 8. 结果

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

运行清单记录查询覆盖范围、所选期望点 ID、参数、随机种子及源码／构建身份。

逐点统计以一个 `(query, expected-point)` 点对为单位；累计修复评估以一条标准查询为单位。运行时间不包含索引加载和文件读写。超时按有效的600,000 ms观测值计入；异常没有计时值，并从均值中排除。汇总分别报告有效数、成功数、超时数、异常数与缺失数。对于 LBOnly 和 IncRerun，超时检查在一次完整冷搜索返回后进行；原始实际耗时另行保存。
