# PAC-Bayes PyTorch port 与两个 ablation 的查验

查验日期：2026-10-07。对象：当前工作目录中的 `PAC_Bayes_modeifeid/`，包括尚未提交的修改和现有结果。此次只新增本报告，没有修改训练代码、checkpoint 或实验结果，也没有重跑完整训练。

Follow-up implementation update: All ablation entry points now run three conditions concurrently: fresh random initialization, initialization after one SGD mini-batch update by default, and frozen weights after 20 SGD epochs. A fresh initialization seed is shared within each suite, independently of baseline checkpoints. The runner floors only the initial posterior standard deviation. The subsequent bound correction repairs the dynamic mean KL gradient, positive prior grid, confidence parameters, Monte Carlo correction, inverse KL endpoints, and complete final-state saving. See [BOUND_FIXES.md](BOUND_FIXES.md) for current behavior and validation. Python comments and docstrings are in English. The findings below describe the historical review snapshot and historical result files; they do not describe the corrected code.

## 结论

**基础 PyTorch 迁移与作者公开代码在已检查的 FC/binary-MNIST 路径上基本对应；固定 SGD 权重的 ablation 已实现并有结果证据；随机初始化的 ablation 已实现跳过 SGD，但额外改动了初始化与 prior，目前没有完成结果。整个项目尚不能认定为数学正确的论文复现。**

必须区分“对应作者的公开实现”和“符合论文公式及 Table 1 实验协议”。作者公开实现本身有若干与论文不一致的地方，你的版本继承了这些问题；近期加入的 `abs(j)` 处理还引入了新的理论问题。

参考来源：

- [作者仓库](https://github.com/gkdziugaite/pacbayes-opt)，本次获取 master commit：`d828b8f162f2f3430c492c37262a09f4717e519f`。下载该 revision 后逐文件比较，本地 `pacbayes-opt/` 的 18 个上游文件全部一致（忽略 CRLF/LF 差异）。
- [Dziugaite & Roy 2017，论文 v2](https://arxiv.org/html/1703.11008)，重点为式 (4)–(6) 和 4.2–4.4 节。
- [作者 network.py](https://github.com/gkdziugaite/pacbayes-opt/blob/d828b8f162f2f3430c492c37262a09f4717e519f/snn/core/network.py)。
- [作者 run_pacb.py](https://github.com/gkdziugaite/pacbayes-opt/blob/d828b8f162f2f3430c492c37262a09f4717e519f/snn/experiments/run_pacb.py)。

## 两个 ablation 的准确含义与现状

| 情况 | SGD 阶段 | PAC-Bayes 后验均值 | PAC-Bayes 方差与先验方差 | 查验结果 |
|---|---|---|---|---|
| 基线 | 20 epochs | 从 SGD 解开始，继续优化 | 优化 | 当前有 `results/custom/T600/seed11` 结果，数学问题见下文 |
| 随机初始化 / no_sgd | 0 epochs | 从随机初始化开始，继续优化 | 优化 | checkpoint 中均值等于 prior，确实跳过 SGD；只有 checkpoint 与 `sgd_metrics.json`，没有完整 PAC-Bayes 结果 |
| 固定 SGD 权重 / no_trainw | 20 epochs | 固定为 SGD 解 | 继续优化 | 已完成一次结果；23 组保存的均值全部与 SGD checkpoint 完全相同 |

这里的“固定 weight”是固定后验均值及所有 bias；每次前向计算仍会按后验方差随机扰动，方差和 prior variance 仍参与优化。这符合“保留 SGD 学到的 weight，只做方差优化”的 ablation。

代码证据：`snn/core/network.py:230` 根据 `trainWeights` 设置均值的 `requires_grad`；`:308` 在 `False` 时将均值排除出优化器。解析器的 `--no-trainw` 返回 `False`。

实际文件证据：`results/ablation/no_trainw/seed11/checkpoint.pickle` 与 `results/custom/T600/seed11/checkpoint.pickle` 的 SGD 权重、初始 prior 均逐数组完全相同；固定权重结果中 23 组 `PACB_weights` 相对 SGD 权重的最大变化全部为 `0.0`。

### 随机初始化额外改变了 prior

`experiments/run_ablation_only.py:13–35` 的 `patch_zero_params` 同时改写 checkpoint 中的网络参数和 prior。当前 no_sgd checkpoint 相比基线的初始 prior，唯一差异是最后一层原为零的 bias，被改成约 `0.0006802697`。

这是为避免 `PACB_init` 中 `log(2*abs(w))` 在零 bias 上产生 `-inf`。数值问题确实存在，但直接改均值和 prior，使实验不再仅改变是否经过 SGD，也不再保留论文的零 bias 初始化。

建议保留原始均值与 prior，只对**初始化的后验标准差/方差**设统一正数下限，所有实验采用同一规则并记录下限。不要原地改写初始 checkpoint。当前指标文件是在 patch 前写入的，也没有描述 patch 后的模型。

## 必须修正的问题

### 1. 高优先级：训练 KL 没有使用正在更新的后验均值

位置：`snn/core/network.py:234,246`。

`norm_params` 从 NumPy 的初始 `network_weights` 计算一次，之后作为常量参与训练。即使 `self.params` 已更新，KL 中的距离仍是 `||w_sgd-w0||²`，而论文式 (5) 要求的是当前 `||w-w0||²`。后验均值无法得到 KL 距离项的梯度。

该问题也存在于作者公开的 `PACB_objective`：它使用 `network_weights` 而非 `model_with_noise` 返回的可训练 `param_var_list`。因此迁移与原代码对应，不能据此证明符合论文公式。

小模型实测：手动把当前均值各参数增加 `0.1` 后，训练计算的 KL 仍为 `25.1292686`，正确 Gaussian KL 为 `55.5477977`；对均值求 KL 梯度全部得到 `None`。

现有 T600 结果也显示这一问题：最后一行 history 的训练 KL 约 `384.078`，原保存模型最终评估的 KL 约 `97642.726`。这不是正常的离散网格舍入差异。

修正：将 prior 转为固定 tensor，在每次 objective 调用中用当前 `self.params` 计算 `sum((param-prior)**2)`。固定均值实验仍可使用这一写法，值会自然保持固定。更正后需重新训练基线和随机初始化实验。

### 2. 高优先级：`abs(j)` 消除 NaN，但破坏论文的 prior 网格与概率预算

位置：`snn/core/network.py:429–462`，尤其 `:460`。

论文要求 `lambda = c*exp(-j/b)`，其中 `j >= 1`，`c=0.1`，`b=100`。如果 `j<0`，对应的 variance 已超过 `c`，不属于论文使用的 prior 家族。仅在惩罚项里取 `max(1,abs(j))`，并没有把 variance 变成有效 prior；正负 j 还会重复占用相同的 union-bound 概率预算。

现有 T600 summary：`log_prior_std=-0.7155263`，因此 `lambda≈0.2390571>0.1`，`j≈-87.1532`。目前报告的 `0.8364835` 不能按论文解释为有效的 PAC-Bayes 保证。

小模型实测也证实：给定 `lambda≈0.3678794`、`j≈-130.2585`，当前评估仍返回有限 bound `0.7915205`。

修正：训练时约束 prior 参数在有效域，评估只比较合法的正整数网格点。例如限制 `rho=log_prior_std <= (log(c)-1/b)/2`，确保 `j>=1`。如果希望扩展到更大 variance，需要重新定义 prior 家族及对应概率预算，不能直接沿用论文公式。

### 3. 高优先级：最终评估缺少 Monte Carlo 误差修正

位置：`experiments/run_ablation_only.py:58`、`snn/core/network.py:478–492`。

当前 `snn_samples=1`，直接把一个后验采样网络的训练准确率代入 PAC-Bayes 的逆 KL。论文式 (6) 有两层：先用 `KL^{-1}(empirical_error, log(2/delta')/N)` 控制 Monte Carlo 误差，再计算最终 PAC-Bayes bound。4.4 节采用 `N=150000`、`delta'=0.01`，总置信度 `1-0.025-0.01=0.965`。

作者公开入口也使用 `N=1` 且未应用这个修正。你目前对应的是公开入口的简化评估，不能声称与论文 Table 1 的认证指标一致。代码打印 `deltaPAC+0.01`，但没有实际使用这个 `0.01` 做修正。

以固定权重结果为例，单次经验训练错误为 `0.0383637`；当 `N=1, delta'=0.01` 时，仅第一层修正就把错误上界推至约 `0.9965822`，再代入复杂度项会得到数值上接近 1 的最终保证。

修正：实现两层逆 KL；明确分开经验 SNN error 和 SNN error 上界，记录 N、delta、delta'。增大 N 可改善估计，但不能代替修正步骤。150000 次评估成本高，此次没有启动。

### 4. 高优先级：重算脚本混用了不同时刻的均值与方差

位置：`experiments/recompute_summary.py:59–61`；`snn/core/network.py:264,281–293,389–401,497`。

训练每 25000 次 iteration 保存一次后验。550000 次更新的最后一次定期保存是 iteration `525000`，最终更新是 `549999`。最终评估又追加了均值和 prior std，但没有追加最终 `log_post_all`。

实际两个模型均为 `PACB_weights` 23 组、`log_prior_std` 23 组、`log_post_all` 22 组。重算脚本取三个列表各自的 `[-1]`，得到最终均值、较早的后验标准差和最终 prior std，形成混合时刻的 Q。

T600 当前 summary 的 KL `97727.12475`、B `0.94268938` 能由这个混合快照复算匹配；它与原模型 pickle 的最终 KL `97642.72562` 不同。修正先验网格后仍不能把该 summary 视为原训练的最终结果。

修正：单独保存一个包含最终 mean、log posterior std、log prior std 和参数配置的完整 final-state checkpoint；重算时加载同一状态。旧文件不能恢复未保存的最终完整 posterior，需重训或明确评估某个完整的较早快照。

### 5. 中优先级：官方入口仍运行旧的三种 ablation

位置：`experiments/run_experiments.py:63–90`；README 和 SUMMARY 的 ablation 命令。

`python experiments/run_experiments.py --suite ablation` 运行 5000 PAC-Bayes epochs、no_trainw 和 200 evaluation samples，没有 no_sgd。只有 `python experiments/run_ablation_only.py` 实现你现在指定的两个情况。

建议统一 runner 与文档，保留基线加两个目标 ablation。checkpoint 复用目前只检查文件是否存在，没有核验训练配置；未来换 seed、epochs 或实验条件时，需要检查 checkpoint 元数据。此次实际固定权重 checkpoint 与基线相同，实际随机 checkpoint 均值等于 prior，因此没有观察到复用了错误训练权重的情况。

## 论文实验协议与当前 runner 的其他差异

以下主要涉及“严格复现 Table 1”的声明；不要将它们误归为两个 ablation 的定义问题：

| 项目 | 论文 | 当前实现/runner |
|---|---|---|
| PAC-Bayes 梯度数据 | 全部训练集；正文明确没有尝试 mini-batch | 每次 100 个样本；与作者公开实现一致 |
| true-label PAC-Bayes 更新数 | 150000 次 lr=0.001，再 50000 次 lr=0.0001 | 1000 epochs × 550 = 550000 次；在 epoch>250 后降 lr |
| 后验初始 covariance | true labels: `abs(w)`；random labels: `abs(w)/10` | 初始 log std=`log(2*abs(w))`，即 covariance=`4*w²`；与作者代码一致 |
| R600 SGD | 120 epochs | runner 对所有网络使用 20 epochs |
| R600 PAC-Bayes | lr=0.0001，500000 次更新 | 与 true-label run 使用相同配置 |
| random-label 协议 | 独立、均匀随机生成二元标签 | 打乱已有标签，保持原标签计数 |
| 优化的 delta | 0.025 | objective 硬编码 0.05；最终评估使用 0.025；作者代码亦如此 |

二元标签正负方向与作者公开代码一致，和论文文字相反；全局翻转标签本身不构成关键问题。

目前结果目录只看到 custom/T600、no_trainw 的完整结果和 no_sgd 初始化，不能据此声明已完成 7 个架构的 Table 1 复现。

## 其他结果可靠性问题

- `experiments/plot_results.py:24–25` 从 final_summary 读取 SGD error，但该 summary 没有这些字段，于是把缺失值写成 `0.0000`。实际 T600 SGD train/test error 是约 `0.0017455/0.0169000`，保存在 `sgd_metrics.json`。应合并该文件或显示缺失，不能默认为零。
- 表格 `Prior Std` 实际写入 log prior std；应更名或先取指数。
- `snn/core/extra_fn.py:39–58` 的 Newton inverse-KL 在经验错误恰好为零时返回 NaN，已用 `train_accur=1.0, B=0.1` 验证。应处理 q=0/1 边界，并使用有区间保护的数值求根。
- `recompute_summary.py` 固定 `random_labels=False`、seed=11，且可能在多个模型文件中任取第一个；不适用于一般结果目录。完整保存配置可以同时解决这个问题。
- `requirements.txt` 没有列出绘图脚本依赖的 pandas/matplotlib。

## 本次验证范围与数值证据

1. 上游 18 文件比较：一致。
2. 全部修改版 `.py` 文件语法编译：通过。
3. 从作者原文件提取 FC forward 和 logistic loss，使用本地 TensorFlow 2.14 的 `compat.v1` 与 PyTorch 2.5.1 做相同输入/初值对照：前向最大差异 `0.0`；loss 差异 `0.0`；5 步 SGD 最大参数差异 `4.66e-10`。
4. TensorFlow compat.v1 RMSProp 与 `TFRMSprop` 在相同梯度下做 3 步更新：最大差异 `0.0`。这支持基本更新规则迁移正确，不代表跨框架随机数或完整训练轨迹完全一致。
5. 小型 FC 做 3 epochs PAC-Bayes：no_trainw 的均值变化 `0.0`，后验 std 和 prior std 都改变；trainw=True 的均值最大变化约 `0.000569716`。
6. 0 SGD epochs：初始权重、最终权重、返回的 prior 逐数组相同。
7. 现有完整模型与 checkpoint 的逐数组检查、history/final_summary 检查、Gaussian KL 标量复算：完成。

没有执行作者完整的旧 TF1 环境或完整 MNIST 训练。此次验证支持基础迁移和冻结机制；上列数学、评估与保存问题仍需修正后再验证实验结论。
