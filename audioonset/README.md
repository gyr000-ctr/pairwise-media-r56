# audioonset

## 题面（可直接粘贴）

起始点检测（谱通量 + 峰值挑选）。约束：谱通量由半波整流差分给出且与解析式 ≤1e-9；峰值挑选用局部极大+最小间隔（间隔可配）；检测点与真值偏差 ≤10 ms；静音返回空；100 万样本不超过 800 ms；非法阈值与非法间隔可区分报错。 工作目录为仓库内的 audioonset/。 验收：cd audioonset && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd audioonset && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
