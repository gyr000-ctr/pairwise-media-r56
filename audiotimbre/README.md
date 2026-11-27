# audiotimbre

## 题面（可直接粘贴）

音色特征（MFCC 式）。约束：梅尔滤波器组按解析式生成且三角窗和为 1（≤1e-12）；DCT-II 用正交归一形式，与朴素 DCT 对照 ≤1e-9；倒谱提升（lifter）线性作用；系数对增益不变（除 c0）；100 万样本不超过 800 ms；非法滤波器数可区分报错。 工作目录为仓库内的 audiotimbre/。 验收：cd audiotimbre && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd audiotimbre && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
