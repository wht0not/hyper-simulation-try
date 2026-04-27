# Hyper Simulation

A powerful python project for hypergraph simulation, focusing on knowledge reasoning and question-answering over diverse datasets like HotpotQA, MuSiQue, ConTRoL, and ECON.

## 🚀 Quick Start

This project uses [Pixi](https://pixi.sh/) for environment and task management.

### 1. Build Hypergraphs
```shell
# Build for HotpotQA
pixi run -e hypergraph build --rebuild --data_path /path/to/dataset --task hotpotqa

# Build for MuSiQue
pixi run -e hypergraph build --rebuild --data_path /path/to/dataset --task musique
```

### 2. Run Simulations
```shell
# Run Hyper Simulation
pixi run -e simulation hyper_simulation --data_path /path/to/dataset --task <task_name>
```

### 3. Baselines & Debugging
- **Run Baseline (e.g., Contradoc)**: 
  ```shell
  pixi run -e simulation rag_no_retrival --data_path /path/to/dataset --output_path data/baseline/contradoc --method contradoc --task musique
  ```
- **SpaCy Debugging**:
  ```shell
  pixi run -e hypergraph display --steps 1
  ```

## 📊 Scalability & Performance
The project includes a comprehensive scalability analysis across multiple datasets:
- **ECON**: Most efficient (≤15.5s total for 100 instances).
- **HotpotQA & MuSiQue**: Execution times scale linearly with parameter `b` and are heavily influenced by `sigma`.
- **Delta Parameter**: Minimal impact on overall execution time (~1-2% variance).

*For detailed analysis, refer to [Scalability.md](Scalability.md).*
*To run time benchmarks, execute `sh scripts/run/time_benchmark.sh`.*

## ⚙️ Offline Mode (Local Models)
For offline environments, set the following environment variables before running:
```shell
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

# Example run with local models
pixi run -e simulation remote --task docnli --dataset-path data/nli/docnli_50.jsonl --source-root data/debug/docnli/sample50 --max-workers 8
```
