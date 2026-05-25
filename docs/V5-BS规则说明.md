# V5 B/S 规则说明（草案）

> **状态**：骨架已接入前后端，具体买卖规则待完善。  
> **原则**：V5 与 V2/V3/V4 **完全独立**，代码位于 `backend/v5/`，不复用旧版 weighted/mandatory 打标链。

**V 左几何识别（因果、三层架构、验证记录）** → 见 [V5.md](./V5.md)

## 目录结构

| 文件 | 职责 |
|------|------|
| `backend/v5/rules.py` | 规则常量、`get_conditions()` 文档化说明 |
| `backend/v5/engine.py` | `analyze_signals_v5()` 主引擎 |
| `backend/engine_v2.py` | 仅编排：`analyze_signals_dual` 并行调用 V5 |
| `frontend/index.html` | V5 Tab、池收益列、规则面板 |

## API

- 运行分析：`GET /api/v2/analyze` → 响应含 `v5` 块（与 `v2`/`v3`/`v4` 并列）
- 读取缓存：`GET /api/v2/signals` → 含 `signals_v5` / `portfolio_v5` 映射的 `v5` 块
- 运行日志：`analysis_run_log` 文档含 `v5` 收益快照

## 待完善（规则）

以下章节在规则定稿后补充：

1. 买入必要条件  
2. 买入充分条件  
3. 卖出必要条件  
4. 卖出充分条件  
5. 与 V2 权重公式的关系（预计 V5 **不使用** V2 迭代权重，若需独立优化再增模块）

## 版本

- `V5_VERSION` 见 `backend/v5/rules.py`
