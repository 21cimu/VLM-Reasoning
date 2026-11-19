# MultihopSpatial 官方评测复现

使用官方 4B Instruct 权重和作者的 Transformers greedy 入口。`smoke` 跑原始顺序的前 5 题。`full` 跑原始 4,500 题，按清单下载 3,562 张不同的测试图像。两种模式共用同一份权重、图片和 Python 环境。正式实验为同一模型、同一测试集的一组评测。

## 固定版本

| 对象 | 版本 |
|---|---|
| [作者源码](https://github.com/youngwanLEE/multihopspatial/tree/b1b6a7be261e3cde814e117eabe330d5ef47c1f0) | `b1b6a7be261e3cde814e117eabe330d5ef47c1f0` |
| [官方 4B 权重](https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct/tree/c7148d4d58d4708032b5386ade8bb1b8ac6219a5) | `c7148d4d58d4708032b5386ade8bb1b8ac6219a5` |
| [官方测试集](https://huggingface.co/datasets/etri-vilab/MultihopSpatial/tree/6c962f5227acc133d2196cbf89f09441c7ccb703) | `6c962f5227acc133d2196cbf89f09441c7ccb703` |
| [论文](https://arxiv.org/html/2603.18892v2) | `2603.18892v2` |

默认使用最多 4 张可见 GPU，每张卡独立加载一份 4B 模型，按题分片。4 卡时，试跑将前 5 题分为 1、1、1、2 题。全量时每张卡推理 1,125 题，按原顺序合并再统一评分。可用 `CUDA_VISIBLE_DEVICES` 指定可见卡，用 `VLM_GPU_COUNT` 指定进程数。无卡时可运行 `prepare`，完成全量资产下载和校验后挂卡，沿用同一实验目录。运行环境使用 Linux、Git、Bash、`flock`、支持 venv 的 Python（优先 3.10–3.12），以及可运行 PyTorch CUDA 12.8 构建的 NVIDIA 驱动。依赖固定在 `requirements-eval.txt`。模型分片合计约 8.88 GB，脚本校验两片权重和测试 JSON 的 SHA-256。

运行时直接调用作者的 `eval/benchmark_qwen.py`，保留最长生成长度 4,096 和最多 3 次解析尝试，并禁用 FlashAttention 2。作者评分排除无法解析选项的样本，保留原始分母和数据用于对齐。若下载中断，可重跑相同命令，脚本会复用已下载的文件。

## 结果

每次运行生成合并的 `predictions.json`、总分 `metrics.json`、`console.log` 和 `versions.tsv`。各卡的输入、预测和推理日志在 `shard-*/`。任一进程报错时脚本保留分片日志并返回失败状态。试跑时检查应有 5 条记录、运行错误为 0，并查看解析出的选项和框。全量核对输入题数、日志中的实际评分分母和三项指标。脚本记录原始输出与模型预测，结果文件在服务器本地。

| 参照 | MCQ Accuracy | Acc@50IoU | Avg. IoU |
|---|---:|---:|---:|
| 作者仓库 Transformers greedy 复跑 | 63.47% | 54.27% | 72.38% |
| 论文表 2 | 62.9% | 53.8% | 72.6% |

Avg. IoU 在作者日志中按 0 到 1 输出。正式对齐优先使用[作者仓库的 Transformers 复跑结果](https://github.com/youngwanLEE/multihopspatial/blob/b1b6a7be261e3cde814e117eabe330d5ef47c1f0/README.md#results)。
