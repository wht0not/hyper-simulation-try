# pixi run -e hypergraph python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/build_locomo.py --dataset /home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_1K.json --use-gpu-batch --output-dir /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-1K
# pixi run -e hypergraph python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/build_locomo.py --dataset /home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_4K.json --use-gpu-batch --output-dir /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-4K
# pixi run -e hypergraph python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/build_locomo.py --dataset /home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_8K.json --use-gpu-batch --output-dir /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-8K
# pixi run -e hypergraph python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/build_locomo.py --dataset /home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_16K.json --use-gpu-batch --output-dir /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-16K
# pixi run -e hypergraph python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/build_locomo.py --dataset /home/vincent/hyper-simulation-try/data/bench/locomo-main/locomo-data/locomo_32K.json --use-gpu-batch --output-dir /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-32K

pixi run -e simulation python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/locomo.py --instances-root /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-1K --output-dir /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/hypersim
pixi run -e simulation python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/locomo.py --instances-root /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-4K --output-dir /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/hypersim
pixi run -e simulation python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/locomo.py --instances-root /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-8K --output-dir /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/hypersim
pixi run -e simulation python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/locomo.py --instances-root /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-16K --output-dir /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/hypersim
pixi run -e simulation python /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/locomo.py --instances-root /home/vincent/hyper-simulation-try/data/hypergraphs/locomo-32K --output-dir /home/vincent/hyper-simulation-try/tests/tasks/memory/locomo/results/hypersim
# 1. reranking / filter
# 2. 对于conversation的文本转超图可能会有问题
# 3. 把“5”和前面的混在一起，从数据侧去看能不能解决这个问题
# 4. 在识别出最贴切的基础上完成冲突识别都需要做
# # 先花时间去统一数据集格式再去跑
# 5. RAG可以cover很多记忆的任务，