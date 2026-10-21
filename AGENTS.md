# swefactory_self 协作规则（SWE 数据工厂）

本仓库是 SWE 数据合成工厂（AweAgent 脚手架 + 独立判分），产出用于本地后训练的 SFT/RL 样本。本文件只放稳定约束；当前进度、通过率、token 数与预算一律从台账和索引库现读，不写进这里。

## 合成数据必须登记到 SQL（强制）

任何合成数据产物——池 A verified、池 B correction、池 C 行为样本，以及未来的新池、重导出、重采样——**在导出后必须登记进 SQLite 索引，禁止只留 JSONL 或目录**。

- **权威源**：JSONL 台账（`results/pool_a_formal/admission_ledger.jsonl` 等）为准；`results/pool_a_formal/index.sqlite3` 是唯一查询入口。
- **不一致时**：以台账为准重建索引，不得手改数据库充数；也不得反过来按库改台账。
- **登记时机**：每次 `formal_export.py` 或新池导出之后立即执行

  ```bash
  python results/build_index_db.py          # 重建
  python results/build_index_db.py --check  # 校验，非零退出即视为未登记
  ```

- **提交前必跑** `--check`：索引与台账不一致（行数、准入数、token 合计、门分布、轨迹缺失、重复 ID）时禁止提交涉及数据数字的报告。
- **最低字段**：`instance_id` 唯一、`family`、`batch`、`admitted`、`rejected`（门列表）、F2P/P2P 计数、`node_evidence`、`seq_tokens`、`length_gate`、步数与教师成本、轨迹路径 + sha256。
- **新池接入**：池 B/C 若不复用 `samples` 表，必须新增同库表并满足"可查询 + 可校验"；池 B 在配对导出实现前**计 0**，禁止把独立重试拼成纠正对。
- **报告纪律**：报告里的每个数字都必须能从库里复现；查不到的数字不许写进报告或对外结论。

## 数据质量边界

- 隐藏判分、F2P 测试、gold patch 只允许独立判分入口读取；不进入教师提示、agent view 或训练数据。
- 不伪造工具观察、测试结果与执行收据；池 C 的诚实失败单独分桶，不套池 A 的"全门通过 + 非空补丁"标准。
- 训练与评测隔离：SWE-bench Verified 正式集不回流训练；任务家族与近重复统一划分 split，每家族在池 A 中至多一题。

## 工具面与提示词

- 工具面与 DSH 评测端对齐：`bash` / `read` / `edit`、原生 `tool_calls`、自然停止（无 finish 工具）。任何工具面改动都要开新 run，不得与旧轨迹混训。
- `extra.dsh_contract_review` 默认关闭；提示词改动必须先做同条件对照，未观察到收益前不得设为默认。

## 验证与提交

- 提交前至少：`python results/build_index_db.py --check`、相关 unittest 通过、`git diff --check` 干净。
- 只提交任务相关文件；**不提交**密钥、轨迹目录、派生索引库（`results/` 已忽略）、临时日志。需要入库的脚本用 `git add -f` 显式加入。
- 工程文档写入 `docs/工程问题/YYYY-MM-DD-任务或问题简述.md`，包含 Mermaid 流程图与修改说明。
