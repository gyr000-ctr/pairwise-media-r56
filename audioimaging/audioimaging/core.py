"""audioimaging 核心 API（仅使用 Python 标准库）。

约定：
- M/S 编解码采用 mid = (L + R) / 2，side = (L - R) / 2，
  逆变换 L = M + S，R = M - S。
- 相关度由 2x2 二阶矩（相关）矩阵 [[<L²>, <LR>], [<LR>, <R²>]] 的
  特征值分解（闭式 Jacobi 旋转）给出，并与解析解交叉一致。
- 展宽仅在线性缩放 M/S 域中的 S：M 保持不变，S' = width * S，
  因此单声道下混 M 完全不变（相位安全）。
"""

import math


class Imaging:
    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_pair(left, right):
        """等长且非空校验，返回样本数。"""
        n = len(left)
        if n == 0 or n != len(right):
            raise ValueError("left/right 必须为等长的非空序列")
        return n

    # ------------------------------------------------------------------ #
    # M/S 编解码（互逆）
    # ------------------------------------------------------------------ #
    def ms(self, left, right):
        """M/S 分解：返回 (mid, side)，mid=(L+R)/2，side=(L-R)/2。"""
        self._check_pair(left, right)
        mid = [(l + r) * 0.5 for l, r in zip(left, right)]
        side = [(l - r) * 0.5 for l, r in zip(left, right)]
        return mid, side

    def from_ms(self, m, s):
        """M/S 逆变换：返回 (left, right)，L=M+S，R=M-S。"""
        self._check_pair(m, s)
        left = [mi + si for mi, si in zip(m, s)]
        right = [mi - si for mi, si in zip(m, s)]
        return left, right

    # ------------------------------------------------------------------ #
    # 相关度（2x2 相关矩阵特征值分解）
    # ------------------------------------------------------------------ #
    def correlation(self, left, right):
        """立体声相关度，取值 [-1, 1]。

        构造相关矩阵 C = [[a, c], [c, b]]，其中
        a = ΣL²，b = ΣR²，c = ΣLR。对实对称 2x2 矩阵做闭式
        Jacobi 特征值分解（特征值 λ1/λ2 与旋转角 θ），再由
        EVD 重建带符号的 c：c = (λ1-λ2)·cosθ·sinθ，
        相关度 ρ = c / √(a·b)。
        """
        self._check_pair(left, right)

        a = 0.0
        b = 0.0
        c = 0.0
        for l, r in zip(left, right):
            a += l * l
            b += r * r
            c += l * r

        # NaN/Inf 等非法能量显式报错（可区分）
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            raise ValueError("输入包含非有限值，无法计算相关度")
        if a <= 0.0 or b <= 0.0:
            raise ValueError("通道能量为零，相关度未定义")

        # ---- 2x2 实对称矩阵的闭式 Jacobi 特征值分解 ----
        # tan(2θ) = 2c / (a - b)
        if c == 0.0:
            lam_hi = max(a, b)
            lam_lo = min(a, b)
            c_evd = 0.0
        else:
            theta = 0.5 * math.atan2(2.0 * c, a - b)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            base = 0.5 * (a + b)
            off = math.hypot(0.5 * (a - b), c)  # |λ1 - λ2| / 2
            lam_hi = base + off
            lam_lo = base - off
            # R = V·diag(λ)·Vᵀ 的非对角元重建（符号由旋转角携带）
            c_evd = (lam_hi - lam_lo) * cos_t * sin_t

        return c_evd / math.sqrt(a * b)

    # ------------------------------------------------------------------ #
    # 立体声展宽（M/S 域线性作用于 S，相位安全）
    # ------------------------------------------------------------------ #
    def widen(self, left, right, width):
        """按 width 展宽立体声，返回新的 (left, right)，不修改输入。

        M' = M，S' = width · S（width 对 S 为纯线性增益）：
            L' = M + width·S = ((1+w)L + (1-w)R) / 2
            R' = M - width·S = ((1-w)L + (1+w)R) / 2
        width=0 时 L' ≡ R'（严格单声道）；下混 M 在任意宽度下不变。
        """
        self._check_pair(left, right)
        if isinstance(width, bool) or not isinstance(width, (int, float)):
            raise TypeError("width 必须为数值类型")
        if not math.isfinite(width):
            raise ValueError("width 必须为有限值")
        if width < 0.0:
            raise ValueError("width 必须非负")

        w = float(width)
        gain_same = 0.5 * (1.0 + w)
        gain_cross = 0.5 * (1.0 - w)
        out_l = [gain_same * l + gain_cross * r for l, r in zip(left, right)]
        out_r = [gain_cross * l + gain_same * r for l, r in zip(left, right)]
        return out_l, out_r
