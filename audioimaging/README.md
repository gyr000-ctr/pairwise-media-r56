# audioimaging

## 题面（可直接粘贴）

立体声成像（相关矩阵与相位安全）。约束：相关度必须由 2x2 相关矩阵的特征值分解给出，与解析式 ≤1e-9；宽度参数作用在 S 上且为线性（≤1e-9）；宽 0 时左右严格相等（≤1e-12）；单声道下混损失 ≤0.3 dB（相位安全，可测）；M/S 互逆 ≤1e-12；100 万样本不超过 500 ms；非法宽度可区分报错。 工作目录为仓库内的 audioimaging/。 验收：cd audioimaging && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd audioimaging && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
