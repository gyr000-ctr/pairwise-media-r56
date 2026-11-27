# audiotimestretch

## 题面（可直接粘贴）

时间伸缩（相位声码器）。约束：输出长度 = ceil(输入长度 / rate)（精确，≤1e-12）；分析/合成窗必须互补（COLA，窗和误差 ≤1e-12）；相位累积按帧中心修正，重建正弦 SNR ≥20 dB；rate=1 时恒等（≤1e-6）；100 万样本不超过 900 ms；非法速率与非法帧长可区分报错。 工作目录为仓库内的 audiotimestretch/。 验收：cd audiotimestretch && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd audiotimestretch && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
