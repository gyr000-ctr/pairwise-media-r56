"""MFCC 式音色特征（仅使用 Python 标准库）。

实现要点：

* 梅尔滤波器组由解析式生成：mel(f) = 2595 * log10(1 + f / 700)，
  在梅尔轴上等间隔布置三角窗，每个三角窗对其所有 FFT 频点的
  权重做归一化，使行和严格接近 1（浮点误差 ~1e-15）。
* DCT-II 采用正交归一形式（scipy 的 ``norm='ortho'`` 约定）：
  c0 * sqrt(1/N)，其余 * sqrt(2/N)。
* 倒谱提升（lifter）为逐系数乘一个固定权重，因此是线性算子。
* 帧级梅尔能量的对数在帧间取平均，再作 DCT；输入整体乘一个正
  增益时，只有 c0 改变，其余系数不变。对数能量使用相对该段信号
  峰值能量的下限（而不是绝对下限），因此极小增益下该性质依然成立。
* FFT 使用迭代基 2 算法，实信号通过半长复 FFT（打包两个实样本）
  计算，居中的 Hann 窗与稀疏梅尔矩阵一起保证 1e6 样本的耗时。
"""

import math
from operator import mul

__all__ = ["Timbre"]

_TWO_PI = 2.0 * math.pi

# 相对能量下限：自功率谱非负，低于峰值 * 该比例的频带视为静音。
_REL_FLOOR = 1e-12
# 整段信号完全静音（峰值能量为 0）时使用的绝对下限。
_ABS_FLOOR = 1e-300

# 帧参数（样本数）；短信号会补零到一帧。帧间不重叠，以控制大输入的耗时。
_FRAME = 1024
_HOP = 1024


def _is_real_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_real(value, name, *, positive=False):
    if not _is_real_number(value):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    if positive and value <= 0.0:
        raise ValueError("%s must be positive" % name)
    return value


def _check_positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _mel(freq):
    """频率 (Hz) -> 梅尔刻度（HTK 解析式）。"""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_inv(mel):
    """梅尔刻度 -> 频率 (Hz)。"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class _FFT:
    """缓存的迭代基 2 复 FFT（原地 Cooley-Tukey）。"""

    __slots__ = ("n", "rev", "tw")

    def __init__(self, n):
        self.n = n
        bits = n.bit_length() - 1
        rev = [0] * n
        for i in range(n):
            r = 0
            v = i
            for _ in range(bits):
                r = (r << 1) | (v & 1)
                v >>= 1
            rev[i] = r
        self.rev = rev
        self.tw = [
            complex(math.cos(-_TWO_PI * k / n), math.sin(-_TWO_PI * k / n))
            for k in range(n >> 1)
        ]

    def transform(self, data):
        n = self.n
        rev = self.rev
        a = [data[rev[i]] for i in range(n)]
        tw = self.tw
        size = 2
        while size <= n:
            half = size >> 1
            step = n // size
            for start in range(0, n, size):
                idx = 0
                mid = start + half
                for i in range(start, mid):
                    u = a[i]
                    v = a[i + half] * tw[idx]
                    a[i] = u + v
                    a[i + half] = u - v
                    idx += step
            size <<= 1
        return a


class Timbre:
    """音色特征提取器（MFCC 式）。"""

    def __init__(self, rate, bands, coeffs):
        self.rate = _check_real(rate, "rate", positive=True)
        self.bands = _check_positive_int(bands, "bands")
        self.coeffs = _check_positive_int(coeffs, "coeffs")
        if self.coeffs > self.bands:
            raise ValueError("coeffs must not exceed bands")
        self._bank_cache = {}
        self._fft_cache = {}
        self._cos_cache = {}
        self._window_cache = {}

    # ------------------------------------------------------------------
    # 梅尔滤波器组
    # ------------------------------------------------------------------
    def _mel_edges(self):
        """返回 bands + 2 个三角窗端点频率（Hz），在梅尔轴上等间隔。"""
        nyquist = self.rate / 2.0
        top = _mel(nyquist)
        count = self.bands + 1
        return [_mel_inv(top * i / count) for i in range(count + 1)]

    def mel_bank(self, nfft):
        """返回 bands 个三角滤波器，每个长度为 nfft // 2 + 1，行和为 1。"""
        nfft = _check_positive_int(nfft, "nfft")
        if nfft < 2:
            raise ValueError("nfft must be at least 2")

        cached = self._bank_cache.get(nfft)
        if cached is not None:
            return [row[:] for row in cached]

        bins = nfft // 2 + 1
        edges = [_mel(v) for v in self._mel_edges()]
        freqs = [_mel(k * self.rate / nfft) for k in range(bins)]

        bank = []
        for band in range(self.bands):
            lo, center, hi = edges[band], edges[band + 1], edges[band + 2]
            row = [0.0] * bins
            rise = center - lo
            fall = hi - center
            for k in range(bins):
                value = freqs[k]
                if lo < value < center:
                    row[k] = (value - lo) / rise
                elif center <= value < hi:
                    row[k] = (hi - value) / fall
            total = math.fsum(row)
            if total > 0.0:
                row = [v / total for v in row]
            else:
                # 频点过于稀疏：把中心最近的频点权重设为 1，行和仍为 1。
                k = int(round(_mel_inv(center) * nfft / self.rate))
                k = min(max(k, 0), bins - 1)
                row[k] = 1.0
            bank.append(row)

        self._bank_cache[nfft] = bank
        return [row[:] for row in bank]

    def _sparse_bank(self, nfft):
        """把滤波器组转成 (起始, 结束, 权重) 稀疏形式用于快速投影。"""
        key = ("sparse", nfft)
        cached = self._bank_cache.get(key)
        if cached is not None:
            return cached
        rows = self.mel_bank(nfft)
        sparse = []
        for row in rows:
            first = next((i for i, v in enumerate(row) if v != 0.0), 0)
            last = next(
                (i for i in range(len(row) - 1, -1, -1) if row[i] != 0.0),
                len(row) - 1,
            )
            weights = row[first : last + 1]
            sparse.append((first, last + 1, weights))
        self._bank_cache[key] = sparse
        return sparse

    # ------------------------------------------------------------------
    # DCT-II（正交归一）
    # ------------------------------------------------------------------
    def _cos_table(self, n):
        table = self._cos_cache.get(n)
        if table is None:
            table = [
                [math.cos(math.pi * k * (i + 0.5) / n) for i in range(n)]
                for k in range(n)
            ]
            self._cos_cache[n] = table
        return table

    def dct(self, values):
        """正交归一 DCT-II，返回与输入等长的列表。"""
        vals = _sequence(values, "values")
        n = len(vals)
        table = self._cos_table(n)
        scale0 = math.sqrt(1.0 / n)
        scale = math.sqrt(2.0 / n)
        out = []
        for k in range(n):
            acc = math.fsum(map(mul, table[k], vals))
            out.append(acc * (scale0 if k == 0 else scale))
        return out

    # ------------------------------------------------------------------
    # 倒谱提升（线性）
    # ------------------------------------------------------------------
    def lifter(self, values, width=None):
        """逐系数线性提升：w[k] = 1 + width / 2 * sin(pi * k / width)。"""
        vals = _sequence(values, "values")
        if width is None:
            width = float(len(vals))
        else:
            width = _check_real(width, "width", positive=True)
        out = []
        for k, value in enumerate(vals):
            weight = 1.0 + 0.5 * width * math.sin(math.pi * k / width)
            out.append(value * weight)
        return out

    # ------------------------------------------------------------------
    # 内部：实信号 FFT / 功率谱
    # ------------------------------------------------------------------
    def _rfft(self, frame):
        """实序列 -> 前半谱（长度 n // 2 + 1）。"""
        n = len(frame)
        half = n >> 1
        entry = self._fft_cache.get(n)
        if entry is None:
            entry = (
                _FFT(half),
                [
                    complex(math.cos(-_TWO_PI * k / n), math.sin(-_TWO_PI * k / n))
                    for k in range(half + 1)
                ],
            )
            self._fft_cache[n] = entry
        fft, twiddle = entry

        packed = [complex(frame[2 * i], frame[2 * i + 1]) for i in range(half)]
        spectrum = fft.transform(packed)

        out = [0j] * (half + 1)
        for k in range(half + 1):
            zk = spectrum[k if k < half else 0]
            zc = spectrum[-k % half].conjugate()
            even = (zk + zc) * 0.5
            odd = (zk - zc) * (-0.5j)
            out[k] = even + twiddle[k] * odd
        return out

    def _window(self, frame_len):
        """周期 Hann 窗（缓存）。"""
        window = self._window_cache.get(frame_len)
        if window is None:
            window = [
                0.5 - 0.5 * math.cos(_TWO_PI * i / frame_len)
                for i in range(frame_len)
            ]
            self._window_cache[frame_len] = window
        return window

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def mfcc(self, samples):
        """返回整段信号的单组 MFCC 系数（长度 = coeffs）。"""
        signal = _signal(samples)
        total = len(signal)
        if total == 0:  # 防御性：_signal 已拒绝空输入
            raise ValueError("samples must not be empty")

        frame_len = _FRAME
        hop = _HOP
        window = self._window(frame_len)
        sparse = self._sparse_bank(frame_len)
        bands = self.bands

        log_mel = [0.0] * bands
        frames = 0
        peak = 0.0
        for start in range(0, total, hop):
            chunk = signal[start : start + frame_len]
            if len(chunk) < frame_len:
                chunk = chunk + [0.0] * (frame_len - len(chunk))
            framed = list(map(mul, chunk, window))
            power = [c.real * c.real + c.imag * c.imag for c in self._rfft(framed)]
            frame_peak = max(power)
            if frame_peak > peak:
                peak = frame_peak
            for band, (lo, hi, weights) in enumerate(sparse):
                energy = sum(map(mul, power[lo:hi], weights))
                if energy > 0.0:
                    log_mel[band] += math.log(energy)
                else:
                    log_mel[band] = -math.inf
            frames += 1

        if frames == 0:  # 防御性：不可达
            raise ValueError("samples must not be empty")

        # 相对下限只在最后生效：低能量的频带被夹到 peak * _REL_FLOOR，
        # 这样整体正增益对所有频带（含下限触底者）的移动量一致，c0 之外不变。
        floor = peak * _REL_FLOOR if peak > 0.0 else _ABS_FLOOR
        log_floor = math.log(floor)
        inv = 1.0 / frames
        log_mel = [max(v * inv, log_floor) for v in log_mel]
        cepstrum = self.dct(log_mel)[: self.coeffs]
        return [float(v) for v in self.lifter(cepstrum, float(self.coeffs))]


def _sequence(values, name):
    """校验并转成浮点列表（用于 dct / lifter 等中间量）。"""
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError("%s must be a sequence of real numbers" % name)
    try:
        seq = list(values)
    except TypeError:
        raise ValueError("%s must be a sequence of real numbers" % name) from None
    if not seq:
        raise ValueError("%s must not be empty" % name)
    try:
        out = [float(v) for v in seq]
    except (TypeError, ValueError):
        raise ValueError("%s must contain only real numbers" % name) from None
    if not all(map(math.isfinite, out)):
        raise ValueError("%s must contain only finite numbers" % name)
    return out


def _signal(samples):
    """校验输入信号：非空、只含有限实数。"""
    if samples is None or isinstance(samples, (str, bytes)):
        raise ValueError("samples must be a non-empty sequence of real numbers")
    try:
        seq = list(samples)
    except TypeError:
        raise ValueError("samples must be a non-empty sequence of real numbers") from None
    if not seq:
        raise ValueError("samples must not be empty")
    try:
        out = [float(v) for v in seq]
    except (TypeError, ValueError):
        raise ValueError("samples must contain only real numbers") from None
    if not all(map(math.isfinite, out)):
        raise ValueError("samples must contain only finite numbers")
    return out
