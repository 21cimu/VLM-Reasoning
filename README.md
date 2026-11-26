# InstaOrder-Occ 遮挡失败归因实验 E2–E4

代码沿用第一阶段打包目录里的 580 道原题、原图、选项、两模型推理入口和答案解析器。从本仓库根目录运行以下命令，用 `--study` 指定已有第一阶段打包目录。

## 快速开始

克隆本仓库后，在 Bash 中设置脚本和第一阶段打包目录。

~~~bash
CLI=experiments/instaorder_occ/eval_occ_e234.py
STUDY=/path/to/stage1-package
~~~

开展真实评测需要已有的数据和旧评测代码。第一阶段目录至少包含：

~~~text
03_datasets/instaorder_occ.jsonl
03_datasets/images/val2017/
02_dataset_construction/raw_annotations/instances_val2017.json
04_evaluation_and_analysis/eval_blue2_pilot.py
04_evaluation_and_analysis/analysis/analyze_blue2_pilot.py
~~~

仓库提供实验脚本、说明、合成样本测试与输入审计汇总。真实样本、旧逐题预测、E4 审核表和模型权重由各自的原始目录提供。

## 实验输入

| 条件 | 给模型的单张图片和文本 | 对照 |
| --- | --- | --- |
| E0 | 原图和原题 | 同批运行的基础表现 |
| E1 | 原图上画红蓝真值框，标识题干里的两个对象 | 沿用此前红蓝框的颜色与线宽 |
| E2 | 原图上叠加两个 COCO 可见实例掩码的并集，统一用半透明灰 | E0、E1 |
| E3 | 原图上在两实例接缝邻域叠加半透明圆形提示 | 同形、同色、同透明度、等像素面积的随机标记和无关区域标记 |
| E4 | 原图加人工复核的可见视觉细节 | 同一提示模板下的人工错误描述和视觉中性描述 |

每次推理只输入一张完整图片。E0 使用与新条件一致的答题指令，旧实验的 `A_full` 用于固定历史分组。E1 用颜色标识题干对象顺序。E2 两个实例用同一种颜色和透明度，交叠像素只涂一次。

旧题号最后两个 ID 按真实前后顺序写入。程序先排序得到无序实例对，再按已有 A/B 框核对身份。E3 接缝候选是两个可见掩码各做 15 像素方形膨胀后的交集，交换掩码顺序结果相同。E3 的标记位置只用这份几何信息。它使用了可见实例标注，可以测量提供局部视觉提示的作用，无法单独识别模型内部的处理阶段。

随机圆避开接缝和另外两个圆，可以覆盖目标的其他部分。无关圆还避开两个实例及其 7 像素邻域。位置由固定种子和无序案例键决定。无法完整放下三枚圆时整组 E3 记为不可用。等面积指原始分辨率的二值标记区域，模型缩放后可能出现轻微栅格差异。实例掩码支持 COCO polygon、压缩 RLE 和非压缩 RLE。[COCO 掩码格式与接口](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)

## 检查实际输入

已有服务器 `treevgr` 环境可直接使用。本地只做预览和测试时，在独立 Python 环境安装 Pillow、NumPy 和 pycocotools 即可，依赖版本可参考旧包清单。

~~~bash
python -m pip install Pillow numpy pycocotools==2.0.11
python "$CLI" --study "$STUDY" audit --limit 6 --preview 6 --out outputs/occ_e234/audit6
python -B -m unittest discover -s experiments/instaorder_occ -v
~~~

合成样本测试可在没有第一阶段数据和模型的环境中运行。去掉 `--limit 6` 并换用新输出目录可检查全部 580 题。`summary.json` 记录覆盖率，`cases.jsonl` 记录逐题可用条件、跳过原因以及 E3 圆的位置与面积，`previews/` 保存指定数量的 E0–E3 预览。[580 道原题的输入审计汇总](experiments/instaorder_occ/audit_580.json)记录了 E2 通过 580 题、E3 通过 579 题的检查结果。`--limit` 按旧题单顺序截断，适合调试。

## E4 人工复核

~~~bash
python "$CLI" --study "$STUDY" annotate --out outputs/occ_e234/blind --seed 42
~~~

把盲审目录交给标注者和复核者。`cases.jsonl` 提供随机排序的 `case_id`、去掉元数据的 `image` 和原题 `question`。同一源图只存一次。原题号、选项字母、答案、框、掩码、模型输出和历史正误都留在评测侧。

每题在导出的 `reviews.jsonl` 中填写一行，保留 `case_id` 和 `image_sha256`。下列文字仅用于说明字段，实际内容由标注者看原图填写。

~~~json
{"case_id":"从模板保留","image_sha256":"从模板保留","status":"pending","author":"","reviewer":"","verified":"A thin vertical edge ends where a horizontal stripe begins.","wrong":"A thick diagonal edge ends where a horizontal stripe begins.","neutral":"A small patch in the image has a muted yellow color.","checks":{"image_only":false,"no_direct_answer":false,"verified_visible":false,"wrong_contains_false_observation":false,"neutral_relation_irrelevant":false},"reason":""}
~~~

- `verified` 写独立从图上观察到的线段终止、边缘连续性、纹理变化等细节。
- `wrong` 写句式和长度相近、经复核与原图冲突的观察。可改线条方向、连续性或纹理等特征。避免直接反转题目结论。
- `neutral` 写长度相近、确实可见且和遮挡方向无关的细节。每段一行并使用相同语言。最长段的非空白字符数不得超过最短段的 1.5 倍。
- `approved` 要求记录两位不同人员的标识，且所有复核项为 `true`。`excluded` 要填写原因，例如边缘太模糊。`pending` 和 `excluded` 计入覆盖率，三组 E4 均跳过。
- 程序拦截常见直接答案措辞。人工复核仍需识别隐含的方向断言、和选项相近的改写以及文字控制质量。建议在查看模型表现前冻结审核文件。

可用 `audit --reviews 已复核文件 --out 新目录` 检查批准覆盖率。关联键及源图 SHA-256 由程序核验，表格行顺序可变。

## 服务器推理

使用旧包的 CUDA、BF16、FlashAttention-2 和 Qwen 依赖环境。以下在 Bash 中执行。把 `MODEL_ROOT` 指向含 `TreeVGR-7B/` 和 `Qwen2.5-VL-7B-Instruct/` 的权重父目录。

~~~bash
MODEL_ROOT=/path/to/models

# 先用两个真实样本验证服务器环境与模型输出。
CUDA_VISIBLE_DEVICES=0 python "$CLI" --study "$STUDY" run --model treevgr --model-root "$MODEL_ROOT" \
  --limit 2 --out outputs/occ_e234/treevgr.smoke.jsonl

# 正式的视觉条件默认运行 E0、E1、E2 和 E3 三组。
for model in treevgr qwen; do
  CUDA_VISIBLE_DEVICES=0 python "$CLI" --study "$STUDY" run --model "$model" --model-root "$MODEL_ROOT" \
    --out "outputs/occ_e234/$model.visual.jsonl"
done

# 审核完毕后运行 E4，再与同模型的视觉预测合并评分。
for model in treevgr qwen; do
  CUDA_VISIBLE_DEVICES=0 python "$CLI" --study "$STUDY" run --model "$model" --model-root "$MODEL_ROOT" \
    --arms E4 --reviews outputs/occ_e234/blind/reviews.jsonl \
    --out "outputs/occ_e234/$model.text.jsonl"
done
~~~

`--arms E3` 或 `--arms E4` 各展开为三个条件。运行记录按 `(id, condition)` 追加，原命令可续跑。模型、代码、数据、环境版本或审核文件变更时用新输出文件。若强制退出后残留 `.lock`，确认原进程已结束即可删除该锁并重试。并行作业使用不同输出路径。审计、盲审导出、评分各用新的输出路径。

## 配对评分与历史 400 题

~~~bash
python "$CLI" --study "$STUDY" score --model treevgr \
  --predictions outputs/occ_e234/treevgr.visual.jsonl outputs/occ_e234/treevgr.text.jsonl \
  --treevgr-e0 /path/to/preds_instaorder_occ_treevgr.jsonl \
  --out outputs/occ_e234/treevgr.paired.json
~~~

Qwen 换成 `--model qwen` 和对应预测文件。只有视觉预测时只传视觉文件。旧包里的 400／238／162 是汇总计数。重建同一批题还需要旧 `eval_blue2_pilot.py` 所生成的正式 TreeVGR 逐题预测。程序用其中的 `A_full` 行和旧分析脚本的双框一对一 IoU ≥ 0.5 规则，核对 580／400／238／162。两个模型可传入同一份历史 TreeVGR 文件，以便使用同一批双框命中样本。旧实验如果按 GPU 生成了多份 JSONL，可在同一个 `--treevgr-e0` 后列出全部分片文件。省略 `--treevgr-e0` 可以先算全体与原数据分层。

- 每个实验家族在其所需条件均可评分的同一批题上计算配对结果。报告同时列出可用、缺失、跳过和无有效答案的数量。解析不到选项时按错误计，沿用旧评分口径。
- 纠错率为错改对数除以该对比的基准错题数。改错率为对改错数除以该对比的基准对题数。分母为零时记为 `null`。历史双框正确且答错分组里的 `available_by_arm.correct` 直接给出历史错题在当前条件下答对的数量。该分组的配对纠错率仍以当前对比的基准正误为准。
- 报告给出四格表、准确率的百分点变化、精确 McNemar 检验和按同一 COCO 图片成组交换条件标签的双侧检验。有重复图片时优先看按图聚类的结果。报告里的 p 值未经多重比较校正。按历史 TreeVGR 正误筛出的分组用于诊断，解读时同时看选择规则和覆盖率。

### 与旧数据对接的字段

- 原题单 `03_datasets/instaorder_occ.jsonl` 每行需要 `id`、`image_id`、`image_path`、`img_size`、`question`、`options`、`valid_opts`、`gt`、`gt_boxes`、`stratum`。原图路径相对于题单。`gt_boxes` 依次对应 A、B 选项第一对象的像素 `xyxy` 框，COCO 框是像素 `xywh`。
- 推理 JSONL 记录原题号、案例键、图片 ID、条件、模型、正确选项、源图指纹和 `status`。有效记录保留原文、提示词、解析选项、预测框、处理后尺寸、耗时和标记元数据。跳过记录保留 `reason`。同名 `.meta.json` 记录运行配置与版本。
- 历史分组使用的旧记录需要 `condition`、`dataset`、`id`、`gt`、`gt_boxes`、`img_size`、`pred_letter`、`correct`、`pred_boxes`、`proc_wh`。
