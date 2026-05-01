# LoCoMo Pipeline

当前目录按职责拆分：

- `run_experiments.py`
  - 顶层唯一高层入口
  - 只负责编排 `build -> compose -> answer -> evaluate`
- `method/hyper_simulation/`
  - 放 `hyper_simulation` 专属实现
  - 包括 hypergraph build 和 compose
- `method/rag/retrieval.py`
  - 独立负责 rag retrieval
  - 从原始对话生成通用 `entries` 数据集
- `utils/`
  - 放通用基准实现和共享工具
  - 包括 `context` compose、answer、evaluate、prompt、metrics、qa utils、路径工具

## 常用流程

下面默认在 `tests/tasks/memory/locomo` 目录下执行命令。

先约定几个相对路径：

```bash
LOCOMO_ROOT=.
PROJECT_ROOT=../../../..
LOCOMO_DATA=./data
CONTEXT_DATASET=./data/context/locomo_context.json
LANGMEM_DATASET=./data/langmem/locomo10_rag.json
INSTANCES_CONTEXT="$PROJECT_ROOT/data/hypergraphs/locomo/context"
RAG_SOURCE=./data/rag/locomo10_rag.json
RAG_DATASET=./data/rag/5_128/locomo10_rag.json
RAG_INSTANCES="$PROJECT_ROOT/data/hypergraphs/locomo/rag/5_128"
```

优先使用 `script/` 目录下按方法拆分的脚本：

```bash
bash ./script/context.sh
bash ./script/hyper_simulation.sh
bash ./script/rag.sh
bash ./script/langmem.sh
```

- `script/context.sh`
  - 默认直接跑 `context --stage all`
  - 注释里保留 `compose / answer / evaluate` 的完整示例
- `script/hyper_simulation.sh`
  - 默认直接跑 `hyper_simulation --stage all`
  - 注释里保留 `build -> all` 和分阶段示例
- `script/rag.sh`
  - 默认直接跑 `retrieval`
  - 注释里保留 `rag -> context` 和 `rag -> hyper_simulation` 的完整示例
- `script/langmem.sh`
  - 默认直接跑 `langmem --stage all`
  - 真正使用 `langmem + langgraph` 做双 speaker memory 检索
  - 注释里保留 `compose / answer / evaluate` 的完整示例

如果想手动执行原始命令，参考下面各节。

### context

```bash
pixi run -e simulation python ./run_experiments.py \
  --method context \
  --stage all \
  --dataset-path "$CONTEXT_DATASET" \
  --output-dir "$LOCOMO_DATA" \
  --answer-batch-size 5 \
  --judge-max-workers 4 \
  --llm-judge-repeat 5
```

### hyper\_simulation / context

- 这条 baseline 实际上是 `context + hyper_simulation`。
- 如果 hypergraph 已经建好，`all` 只需要 `instances-root`：

```bash
pixi run -e simulation python ./run_experiments.py \
  --method hyper_simulation \
  --stage all \
  --instances-root "$INSTANCES_CONTEXT" \
  --output-dir "$LOCOMO_DATA/hyper_simulation/context" \
  --answer-batch-size 5 \
  --judge-max-workers 4 \
  --llm-judge-repeat 5
```

- 如果还没建图，先单独 `build`：

```bash
pixi run -e hypergraph python ./run_experiments.py \
  --method hyper_simulation \
  --stage build \
  --dataset-path "$CONTEXT_DATASET" \
  --instances-root "$INSTANCES_CONTEXT"
```

### rag

```bash
pixi run -e simulation python ./method/rag/retrieval.py \
  --rag-source-path "$RAG_SOURCE" \
  --output-dir "$LOCOMO_DATA" \
  --chunk-size 128 \
  --top-k 5
```

默认会生成：

```text
$RAG_DATASET
```

retrieval 之后，这个文件就是通用 `entries` 数据集，后面可以继续选 `context` 或 `hyper_simulation`：

```bash
pixi run -e simulation python ./run_experiments.py \
  --method context \
  --stage all \
  --dataset-path "$RAG_DATASET" \
  --output-dir "$LOCOMO_DATA"
```

```bash
pixi run -e hypergraph python ./run_experiments.py \
  --method hyper_simulation \
  --stage build \
  --dataset-path "$RAG_DATASET" \
  --instances-root "$RAG_INSTANCES"
```

```bash
pixi run -e simulation python ./run_experiments.py \
  --method hyper_simulation \
  --stage all \
  --instances-root "$RAG_INSTANCES" \
  --output-dir "$LOCOMO_DATA/hyper_simulation/rag/5_128"
```

### langmem

```bash
pixi run -e simulation python ./run_experiments.py \
  --method langmem \
  --stage all \
  --dataset-path "$LANGMEM_DATASET" \
  --output-dir "$LOCOMO_DATA" \
  --model-name qwen3.5:9b \
  --answer-batch-size 5 \
  --judge-max-workers 4 \
  --llm-judge-repeat 5
```

默认会先做 `langmem compose`，把每个 query 对应的双 speaker memory 检索结果写进 prepared 文件，再走统一的 `answer / evaluate`。

prepared 里除了兼容通用流水线的 `d` 字段，还会额外保留 `speaker_memory`：

- `speaker_1.name` / `speaker_2.name`
- `speaker_1.memory` / `speaker_2.memory`
- `speaker_1.search_time` / `speaker_2.search_time`

这样后面如果你想接 `hyper_simulation` 或单独做 contradiction 分析，会比只存一段拼接文本更方便。

## 阶段说明

- `build` 只对 `hyper_simulation` 可用。
- `compose` 只负责准备 prompt。
- `langmem` 的 `compose` 会真用 `langmem` 维护双 speaker memory store，再把两侧检索结果保存到中间产物。
- `answer` 和 `evaluate` 使用统一通用流程，方法差异只在前面的 build/compose。
- 日常使用优先直接跑 `script/context.sh`、`script/hyper_simulation.sh`、`script/rag.sh`、`script/langmem.sh`。
- `all` 会按中间产物自动续跑：
  - 已有完整 prepared 时跳过 compose
  - 已有完整 answers 时跳过 answer
  - 已有 final 且 answers 完整时跳过 evaluate
- `all` 不会默认重建 hypergraph；需要时显式执行 `--stage build`。

## 分阶段示例

```bash
pixi run -e hypergraph python ./run_experiments.py --method hyper_simulation --stage build --dataset-path "$CONTEXT_DATASET" --instances-root "$INSTANCES_CONTEXT"
pixi run -e simulation python ./run_experiments.py --method hyper_simulation --stage compose --instances-root "$INSTANCES_CONTEXT" --output-dir "$LOCOMO_DATA/hyper_simulation/context"
pixi run -e simulation python ./run_experiments.py --method hyper_simulation --stage answer --prepared-path "$LOCOMO_DATA/hyper_simulation/context/prepared.json"
pixi run -e simulation python ./run_experiments.py --method hyper_simulation --stage evaluate --answers-path "$LOCOMO_DATA/hyper_simulation/context/answers.json"
pixi run -e simulation python ./run_experiments.py --method langmem --stage compose --dataset-path "$LANGMEM_DATASET" --output-dir "$LOCOMO_DATA/langmem" --model-name qwen3.5:9b
pixi run -e simulation python ./run_experiments.py --method langmem --stage answer --prepared-path "$LOCOMO_DATA/langmem/prepared.json" --model-name qwen3.5:9b
pixi run -e simulation python ./run_experiments.py --method langmem --stage evaluate --answers-path "$LOCOMO_DATA/langmem/answers.json" --judge-max-workers 4 --llm-judge-repeat 5
```

## 推荐参数

- `answer-batch-size=5`
  - 通常比逐条调用更稳
- `judge-max-workers=4`
  - 先保守并发，避免本地模型服务抢资源
- `llm-judge-repeat=5`
  - 每条样本 judge 5 次，summary 里会记录 `LLM-as-judge_mean` 和 `LLM-as-judge_std`
- `chunk-size`
  - 控制 rag retrieval 的字符切块粒度
- `top-k`
  - 控制 rag 每个问题保留的 chunk 数量
- `langmem`
  - `compose` 固定使用 `qwen3-embedding:0.6b` 作为 memory embedding 模型
