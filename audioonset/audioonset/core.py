"""起始点检测：谱通量（半波整流差分）+ 峰值挑选（局部极大 + 最小间隔）。

仅依赖 Python 标准库。DFT 采用迭代式 radix-2 FFT（实数输入用奇偶打包的
N/2 点 FFT 实现），数值结果与解析式 DFT 的偏差在浮点舍入量级（~1e-12），
满足与解析式误差 <= 1e-9 的要求。
"""

import math

__all__ = [
    "Onset",
    "OnsetError",
    "InvalidThresholdError",
    "InvalidGapError",
]


class OnsetError(ValueError):
    """Onset 参数非法的基类异常（仍是 ValueError）。"""


class InvalidThresholdError(OnsetError):
    """阈值非法（非数值、NaN 或落在 [0, 1] 之外）。"""


class InvalidGapError(OnsetError):
    """最小间隔非法（非正整数）。"""


def _is_real_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Onset:
    """谱通量起始点检测器。

    参数
    ----
    rate: 采样率（Hz），正整数。
    frame: 帧长（样本数），正整数且为 2 的幂。
    hop: 帧移（样本数），正整数且不超过帧长。
    threshold: 谱通量峰值阈值，取值范围 [0, 1]。
    min_gap: 峰值间最小间隔，单位为帧，正整数。
    """

    def __init__(self, rate, frame, hop, threshold, min_gap):
        if not _is_real_number(rate) or rate <= 0:
            raise OnsetError("rate 必须为正数")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame <= 0:
            raise OnsetError("frame 必须为正整数")
        if (frame & (frame - 1)) != 0:
            raise OnsetError("frame 必须为 2 的幂")
        if not isinstance(hop, int) or isinstance(hop, bool) or hop <= 0:
            raise OnsetError("hop 必须为正整数")
        if hop > frame:
            raise OnsetError("hop 不能大于 frame")
        if not _is_real_number(threshold) or math.isnan(threshold):
            raise InvalidThresholdError("threshold 必须为有限数值")
        if threshold < 0.0 or threshold > 1.0:
            raise InvalidThresholdError("threshold 必须落在 [0, 1] 区间")
        if not isinstance(min_gap, int) or isinstance(min_gap, bool) or min_gap <= 0:
            raise InvalidGapError("min_gap 必须为正整数")

        self.rate = int(rate)
        self.frame = frame
        self.hop = hop
        self.threshold = float(threshold)
        self.min_gap = min_gap

        half = frame // 2
        self._half = half
        self._bins = half + 1

        # 位反转表（N/2 点 FFT）
        bits = int(math.log2(half)) if half > 1 else 0
        rev = [0] * half
        for i in range(half):
            r = 0
            for b in range(bits):
                if i & (1 << b):
                    r |= 1 << (bits - 1 - b)
            rev[i] = r
        self._rev = rev
        self._even_idx = [2 * j for j in rev]
        self._odd_idx = [2 * j + 1 for j in rev]

        # 各级旋转因子（实部, 虚部），角度约定 W = exp(-j*2*pi/m)
        twiddles = []
        for h in range(bits):
            m = 1 << (h + 1)
            cr = math.cos(-2.0 * math.pi / m)
            ci = math.sin(-2.0 * math.pi / m)
            stage = []
            wr, wi = 1.0, 0.0
            for _ in range(m >> 1):
                stage.append((wr, wi))
                wr, wi = wr * cr - wi * ci, wr * ci + wi * cr
            twiddles.append(stage)
        self._twiddles = twiddles

        # 实数 FFT 后处理旋转因子 exp(-j*pi*k/(N/2))
        self._post = [
            (math.cos(-math.pi * k / half), math.sin(-math.pi * k / half))
            for k in range(half)
        ]

        # 复用的工作区，避免逐帧分配；只需正频率档 0..N/2
        self._re = [0.0] * half
        self._im = [0.0] * half
        self._out_re = [0.0] * (half + 1)
        self._out_im = [0.0] * (half + 1)

    def _real_fft(self, samples, start):
        """就地计算长度 N 的实数序列 FFT，返回正频率档 (实部, 虚部)。

        把偶数样本作为实部、奇数样本作为虚部打包成 N/2 点复数序列做 FFT，
        再用共轭对称关系拆出原序列 N 点频谱的 k=0..N/2 档。结果与直接
        DFT 解析式仅差浮点舍入误差。
        """
        half = self._half
        re = self._re
        im = self._im
        even_idx = self._even_idx
        odd_idx = self._odd_idx
        for i in range(half):
            re[i] = samples[start + even_idx[i]]
            im[i] = samples[start + odd_idx[i]]

        for stage in self._twiddles:
            m2 = len(stage)
            m = m2 * 2
            for base in range(0, half, m):
                k = base
                j = base + m2
                for wr, wi in stage:
                    rj, ij = re[j], im[j]
                    yr = rj * wr - ij * wi
                    yi = rj * wi + ij * wr
                    rk, ik = re[k], im[k]
                    re[k] = rk + yr
                    im[k] = ik + yi
                    re[j] = rk - yr
                    im[j] = ik - yi
                    k += 1
                    j += 1

        out_re = self._out_re
        out_im = self._out_im
        # 直流与 Nyquist 档：Fe(0)=Re Y(0), Fo(0)=Im Y(0)
        out_re[0] = re[0] + im[0]
        out_im[0] = 0.0
        out_re[half] = re[0] - im[0]
        out_im[half] = 0.0

        post = self._post
        for k in range(1, half):
            kc = half - k
            ykr, yki = re[k], im[k]
            ysr, ysi = re[kc], -im[kc]  # Y*(N/2-k)
            fe_r = 0.5 * (ykr + ysr)
            fe_i = 0.5 * (yki + ysi)
            dr = ykr - ysr
            di = yki - ysi
            # Fo = -j (Y - Y*) / 2
            fo_r = 0.5 * di
            fo_i = -0.5 * dr
            wr, wi = post[k]
            out_re[k] = fe_r + wr * fo_r - wi * fo_i
            out_im[k] = fe_i + wi * fo_r + wr * fo_i
        return out_re, out_im

    def _validate_samples(self, samples):
        for value in samples:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise OnsetError("samples 必须全部为数值")
            if math.isnan(value) or math.isinf(value):
                raise OnsetError("samples 不能包含 NaN 或无穷大")

    def flux(self, samples):
        """计算逐帧谱通量。

        每帧先取幅度谱 |X[k]|（仅正频率档 k=0..N/2），再做半波整流差分：

            flux[t] = mean_k max(|X_t[k]| - |X_{t-1}[k]|, 0)

        首帧前一帧视为全零。返回 list[float]，长度为可完整取帧的帧数。
        """
        self._validate_samples(samples)
        n = len(samples)
        frame = self.frame
        hop = self.hop
        if n < frame:
            return []

        n_frames = (n - frame) // hop + 1
        bins = self._bins
        prev = [0.0] * bins
        mag = [0.0] * bins
        result = []
        sqrt = math.sqrt
        for f in range(n_frames):
            start = f * hop
            re, im = self._real_fft(samples, start)
            total = 0.0
            for k in range(bins):
                a = re[k]
                b = im[k]
                m = sqrt(a * a + b * b)
                mag[k] = m
                d = m - prev[k]
                if d > 0.0:
                    total += d
            result.append(total / bins)
            prev, mag = mag, prev
        return result

    def _peak_pick(self, flux_values):
        """局部极大 + 阈值 + 最小间隔挑选，返回帧号列表（升序）。"""
        threshold = self.threshold
        gap = self.min_gap
        n = len(flux_values)

        candidates = []
        if n == 1:
            if flux_values[0] > threshold:
                candidates.append(0)
            return candidates
        # 首帧只与右侧邻居比较，末帧只与左侧邻居比较，中间帧要求严格局部极大
        if flux_values[0] > threshold and flux_values[0] >= flux_values[1]:
            candidates.append(0)
        for i in range(1, n - 1):
            v = flux_values[i]
            if (
                v > threshold
                and v >= flux_values[i - 1]
                and v >= flux_values[i + 1]
            ):
                candidates.append(i)
        if flux_values[n - 1] > threshold and flux_values[n - 1] >= flux_values[n - 2]:
            candidates.append(n - 1)

        # 按通量从大到小贪心保留，强峰优先，与其相距不足 gap 的峰被抑制
        candidates.sort(key=flux_values.__getitem__, reverse=True)
        kept = []
        for i in candidates:
            ok = True
            for j in kept:
                if abs(i - j) < gap:
                    ok = False
                    break
            if ok:
                kept.append(i)
        kept.sort()
        return kept

    def _localize(self, samples, frame_index):
        """在命中帧的时间窗内把起始点定位到具体样本（子帧精度）。

        谱通量只能把起始点定位到帧，再在该帧覆盖的样本区间里寻找瞬态：
        取窗内幅度最大的样本作为锚点，再向回走到首个幅度低于锚点 20% 的
        位置，即攻击段起点。这样消除帧量化误差，静音窗则回退到帧中心。
        """
        start = frame_index * self.hop
        end = min(start + self.frame, len(samples))
        if end <= start:
            return start + self.frame // 2

        jmax = start
        peak = 0.0
        for j in range(start, end):
            a = samples[j]
            if a < 0.0:
                a = -a
            if a > peak:
                peak = a
                jmax = j
        if peak == 0.0:
            return start + self.frame // 2

        floor = 0.2 * peak
        j = jmax
        while j > start:
            prev = samples[j - 1]
            if prev < 0.0:
                prev = -prev
            if prev < floor:
                break
            j -= 1
        return j

    def detect(self, samples):
        """检测起始点，返回起始点样本下标列表（升序，元素均为 int）。

        先用谱通量 + 局部极大挑出命中帧，再在每个命中帧的时间窗内做
        子帧定位（见 :meth:`_localize`），因此返回值不再受帧量化限制。
        静音（无任何过阈峰值）返回空列表。
        """
        flux_values = self.flux(samples)
        if not flux_values:
            return []
        return [self._localize(samples, i) for i in self._peak_pick(flux_values)]
