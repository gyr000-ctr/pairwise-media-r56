# audiopitch

## 题面（可直接粘贴）

自相关基频估计。约束：基频与真值误差 ≤1%（80-1000 Hz）；倍频歧义必须消解（选最低有效周期）；静音与纯噪声返回 0 而非 NaN；窗长与重叠可配；100 万样本不超过 700 ms；非法采样率与非法范围可区分报错。 工作目录为仓库内的 audiopitch/。 验收：cd audiopitch && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd audiopitch && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
