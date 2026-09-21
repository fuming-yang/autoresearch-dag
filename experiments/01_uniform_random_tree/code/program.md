# Controlled Random Tree AutoResearch (Tree-30)

## 搜索规则与职责

本实验是 naive Random Tree Search，目标是降低 validation bits per byte (`val_bpb`)。
root 没有 parent；每个 candidate 恰好只有一个 parent，操作名保留为 `expand`。
parent 已由 orchestrator 从所有 eligible finished research nodes 中均匀随机选定。
所有成功 finished、具有有效结果的节点（包括 root）永久保留在 parent pool 中，
不论 `val_bpb` 好坏、是否已有子节点。failed、running 和违反 controlled invariants
的 candidate 不进入 parent pool。不使用分数加权、best-first、UCB、pruning、beam search
或 credit assignment。Agent 不选择 parent，不执行融合，也不读取其他研究分支拼接方案。

Agent 的任务只有：
1. 读取当前已检出的 parent 的 `train.py` 和本说明。
2. 提出一个有意义、连贯的 research hypothesis。
3. 修改 `train.py` 实现该假设。
4. 写入未跟踪的 `decision.json`，内容格式为：

```json
{"status": "accept", "hypothesis": "简洁说明研究假设及预期影响"}
```

只允许修改 `train.py`；`decision.json` 是额外的决策产物。不要修改本说明、
`prepare.py`、orchestrator、sampler、依赖配置或数据，不要自行提交 Git，
不要自行启动完整训练。orchestrator 负责单亲 Git commit、worktree、训练、
结果解析、校验和 `dag.json` 更新。

## 固定实验条件（controlled invariants）

- 使用相同 RTX 5070 兼容实现：`SDPA_QUERY_CHUNK_SIZE = 128`，
  `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`，以及现有硬件回退和执行方式。
- `AUTORESEARCH_SEED = 42`；保留读取该环境变量（默认 42）以及 PyTorch/CUDA seed 设置。
- `TRAIN_TIME_BUDGET = 300.0` 秒。
- 终止条件为 `while total_training_time < TRAIN_TIME_BUDGET`。
- 使用 accumulated synchronized training time：每步前后 CUDA synchronize，
  由 `t1 - t0` 得到该步时间；按原协议仅在 `step > 10` 时累加到
  `total_training_time`。前 11 步不计入该预算，不得改变此口径。
- 使用 time-based progress：
  `progress = min(total_training_time / TRAIN_TIME_BUDGET, 1.0)`。
  学习率 warmup / warmdown 和使用 progress 的 schedule 均沿用此时间进度。
  保留原有 Muon momentum schedule 的实现；它仍是允许研究的 optimizer 超参数。
- 300 秒是累计训练时间预算，不是进程总墙钟时间；最后一步允许自然越过预算，
  初始化、不计入预算的预热和最终 evaluation 使总运行时间更长。
- 数据、tokenizer、evaluation 及其调用
  `evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)` 保持不变。
- 正式 candidate-generation model 显式固定为 `gpt-5.6-luna`：每次 Codex
  调用均带 `--model gpt-5.6-luna`，不依赖默认模型，不自动切换模型。
  禁用 `--agent-command` 自定义覆盖入口；调用失败沿用原失败处理，不回退到其他模型。
  外层 interactive session 的模型不影响此配置。
- 除上述模型固定外，保留原 Codex agent setup、本地 subprocess backend、`--workers 1`。
  watchdog 默认关闭，只用于异常挂起保护，不是实验预算。
- 保持原结果字段和解析规则。`num_steps` 与 `target_steps` 均报告实际完成步数，
  解析器继续检查两者相等，不以此决定训练终止。
- 保持原 candidate 计数规则：违反 controlled invariants 的尝试删除且退还配额；
  其他失败尝试仍占用 launch 配额，不得擅自把配额改为只统计成功训练。

不得修改以上受控条件或绕过 validator。允许的研究变量包括 learning rate、
weight decay、warmup / warmdown 比例、optimizer hyperparameters、architecture、
attention pattern 和 RoPE 等；研究修改仍必须满足同一训练预算和校验规则。

## 状态与执行

`dag.json` 是兼容旧工具的文件名，当前内容只表示单亲树。
保留节点 status、parents、result、descendants、best_child_result、commit、hypothesis
等记录字段；统计字段不参与 parent sampling。`rejected_merges` 仅为空的兼容字段。
`results/` 记录本次实验日志，`worktrees/` 在当前项目内提供隔离工作目录。
不要读取归档中的旧 candidate 历史，也不要以其他实验的最优 candidate 起步。

本次准备仅实现与审查。未经后续明确启动指令，不启动正式 Tree-30 candidates。
