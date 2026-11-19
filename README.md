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

试跑覆盖实际启用的 GPU，默认最多用 4 张。验收后，在同一目录运行全量。如有 4 张可见卡，每张处理 1,125 题，合并后算一组官方指标。

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

## 租卡平台无卡准备

在平台会保留的数据盘里选好实验目录。在该目录运行以下命令，准备 Python 环境、固定版本的权重和 4,500 题全部测试图片，并执行校验。

~~~bash
export VLM_ROOT_DIR="$PWD"
curl -fsSLo pull-and-run.sh https://raw.githubusercontent.com/21cimu/VLM-Reasoning/main/pull-and-run.sh
bash pull-and-run.sh multihopspatial prepare
~~~

挂卡后进入同一个目录，再运行：

~~~bash
export VLM_ROOT_DIR="$PWD"
bash pull-and-run.sh multihopspatial smoke
~~~

试跑通过后运行 `bash pull-and-run.sh multihopspatial full`。脚本自动按可见卡数分片。一张卡时，由它完成全部题目。实验资产和结果都保存在 `runs/multihopspatial/`。
