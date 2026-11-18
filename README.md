# VLM-Reasoning

视觉推理实验的服务器启动入口。

| 实验 | 入口 |
|---|---|
| MultihopSpatial 官方评测复现 | [实验说明](experiments/multihopspatial/README.md) |

## 在服务器上启动

服务器目录为 `/home/ubuntu/vLM-reasoning/`。首次在该目录保存拉取脚本，然后启动 5 题试跑。

~~~bash
mkdir -p /home/ubuntu/vLM-reasoning
cd /home/ubuntu/vLM-reasoning
curl -fsSLo pull-and-run.sh https://raw.githubusercontent.com/21cimu/VLM-Reasoning/main/pull-and-run.sh
bash pull-and-run.sh multihopspatial smoke
~~~

试跑验收后，在同一目录运行全量。

~~~bash
bash pull-and-run.sh multihopspatial full
~~~

每次启动都会拉取 `main`。服务器文件分布如下。

| 目录 | 内容 |
|---|---|
| `repo/` | GitHub 实验代码 |
| `runs/multihopspatial/` | Python 环境、固定版作者源码、权重和测试数据 |
| `runs/multihopspatial/results/` | 每次试跑和全量评测的独立结果目录 |

服务器需要能访问 GitHub、Hugging Face 及 PyTorch/PyPI 包源。默认使用服务器的 `python3`，可用 `PYTHON_BIN` 指定另一套 Python，例如 `PYTHON_BIN=python3.10 bash pull-and-run.sh multihopspatial smoke`。
