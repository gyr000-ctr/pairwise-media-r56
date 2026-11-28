"""audiopitch 核心 API（自相关基频估计，仅标准库）。"""

import math


class PitchError(ValueError):
    """所有参数/信号错误的基类（同时是 ValueError）。"""


class InvalidRateError(PitchError):
    """非法采样率。"""


class InvalidRangeError(PitchError):
    """非法频率范围 / 窗长 / 重叠率。"""


class InvalidSignalError(PitchError):
    """非法输入信号（空、NaN、Inf、非数值）。"""


def _solve3(a, b):
    """3x3 线性方程组的克莱姆求解（插值用，矩阵恒良态）。"""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(a)
    return [det([[b[r] if c == col else a[r][c] for c in range(3)]
                 for r in range(3)]) / d for col in range(3)]


class Pitch:
    """基于短时自相关的基频（pitch）估计器。

    每帧三阶段，兼顾纯 Python 速度与 1% 精度：

    1. 扫描：在完整合法 lag 区间上对信号抽稀求归一化自相关（短积分），
       得到全部候选周期峰。全区间覆盖使倍频/次谐波都在视野内；抽稀让
       固定数量的样本即可跨越很长的物理跨度，高采样率下依然低成本。
    2. 细化：对最短达标峰、全局最强峰及半周期候选，在原始分辨率邻域、
       长积分窗口（>=2 个周期）下重算自相关——长积分频率选择性好，能
       可靠区分真正的基频周期与高次谐波/噪声巧合峰。
    3. 选周期：在细化后的高质量相关值上，取同时满足绝对门限与相对
       全局峰门限的最短严格局部极大，天然消解倍频歧义（选最低有效
       周期），对弱/缺失基频同样成立；再做五点加权最小二乘二次插值
       得到亚采样周期。

    多帧结果按周期做共识聚类（成员帧数优先、帧数相同取更短周期），
    抑制强噪声下偶发的离群帧。静音、纯直流与纯噪声帧由去均值后的
    相关峰/能量比判为清音；没有浊音帧时 detect 返回 0.0（从不返回
    NaN）。
    """

    _SCAN_K = 200          # 扫描阶段每条自相关使用的抽稀样本数上限
    _REFINE_K = 900        # 细化阶段积分长度上限（保证低基频也能覆盖 >=2 周期）
    _REFINE_MIN_K = 128    # 细化积分长度下限（强噪声下保证峰位稳定）
    _REFINE_PERIODS = 3    # 细化积分覆盖的候选周期数（3 个已足够选择性）
    _VOICED = 0.40         # 浊音判定：相关峰 / 零滞后能量
    _OCTAVE_ABS = 0.45     # 倍频消解：基频峰的绝对相关门限
    _OCTAVE_REL = 0.85     # 倍频消解：相对全局峰的相关门限
    _SILENCE = 1e-9        # 静音峰值幅度门限

    def __init__(self, rate, lo, hi, window=4096, overlap=0.5):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise InvalidRateError("sample rate must be a real number")
        if not math.isfinite(rate) or rate <= 0:
            raise InvalidRateError("sample rate must be positive and finite")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               for v in (lo, hi)) or not (math.isfinite(lo) and math.isfinite(hi)):
            raise InvalidRangeError("frequency bounds must be finite real numbers")
        if lo <= 0.0 or hi <= lo:
            raise InvalidRangeError("require 0 < lo < hi")
        if hi > rate / 2.0:
            raise InvalidRangeError("hi must not exceed Nyquist frequency (rate/2)")
        if isinstance(window, bool) or not isinstance(window, int) or window < 32:
            raise InvalidRangeError("window must be an int >= 32")
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) \
                or not math.isfinite(overlap) or not (0.0 <= overlap < 1.0):
            raise InvalidRangeError("overlap must satisfy 0 <= overlap < 1")

        self.rate = float(rate)
        self.lo = float(lo)
        self.hi = float(hi)
        self.window = int(window)
        self.overlap = float(overlap)
        self._last_conf = 0.0

    def confidence(self):
        """最近一次 detect 的浊音置信度，范围 [0.0, 1.0]。"""
        return float(self._last_conf)

    # ------------------------------------------------------------------
    @staticmethod
    def _interp5(r, c):
        """在局部极大 c 上做五点加权最小二乘二次插值，返回亚采样 lag。

        权重取相关值的平方（下限 0.05），使峰顶附近的点主导拟合、远离
        峰顶的点（易受谐波/噪声畸变）影响变小，从而在含谐波的非对称
        相关峰上仍给出无偏的亚采样周期。
        """
        ts = (c - 2, c - 1, c, c + 1, c + 2)
        w = []
        for t in ts:
            wv = r[t]
            if wv < 0.05:
                wv = 0.05
            w.append(wv * wv)
        s0 = sum(w)
        s1 = sum(w[i] * ts[i] for i in range(5))
        s2 = sum(w[i] * ts[i] * ts[i] for i in range(5))
        s3 = sum(w[i] * ts[i] ** 3 for i in range(5))
        s4 = sum(w[i] * ts[i] ** 4 for i in range(5))
        t0 = sum(w[i] * r[ts[i]] for i in range(5))
        t1 = sum(w[i] * ts[i] * r[ts[i]] for i in range(5))
        t2 = sum(w[i] * ts[i] * ts[i] * r[ts[i]] for i in range(5))
        a, b, _ = _solve3(
            [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]],
            [t2, t1, t0])
        if a >= 0.0:
            return float(c)
        vertex = -b / (2.0 * a)
        if c - 1.5 < vertex < c + 1.5:
            return vertex
        return float(c)

    def detect(self, samples):
        """估计信号基频；静音/纯噪声返回 0.0。"""
        try:
            n = len(samples)
        except TypeError:
            raise InvalidSignalError("samples must be a sequence of real numbers")
        if n == 0:
            raise InvalidSignalError("samples must not be empty")

        x = [0.0] * n
        isfinite = math.isfinite
        for i in range(n):
            v = samples[i]
            if type(v) is bool or not isinstance(v, (int, float)) \
                    or not isfinite(v):
                raise InvalidSignalError(
                    "samples must all be finite real numbers (no NaN/Inf)")
            x[i] = float(v)

        hop = max(1, int(round(self.window * (1.0 - self.overlap))))
        w = self.window if n >= self.window else n
        starts = range(0, n - w + 1, hop) if n >= self.window else (0,)

        min_lag = max(2, int(math.floor(self.rate / self.hi)))
        max_lag = min(w // 2, int(math.ceil(self.rate / self.lo)))
        if max_lag < min_lag:
            self._last_conf = 0.0
            return 0.0

        # 扫描对“信号求和”抽稀：用固定数量的抽稀样本跨越很长的物理跨度，
        # 从而在高采样率/低基频下仍保证两个比较窗口充分重叠。
        # - decim: 信号求和抽稀因子（沿 i 方向）
        # - lag_step: lag 网格步长；仅当最短合法 lag 足够大（抽稀不会
        #   漏掉短周期）时才对 lag 轴抽稀，否则 lag 保持全分辨率
        span_lags = max_lag - min_lag + 1
        if span_lags <= 120:
            decim = 1
            lag_step = 1
        elif span_lags <= 260:
            decim = 2
            # lag 抽稀仅在最短周期足够长（半步误差远小于周期）时启用
            lag_step = 2 if min_lag >= 16 else 1
        else:
            decim = 4
            if min_lag >= 32:
                lag_step = 4
            elif min_lag >= 16:
                lag_step = 2
            else:
                lag_step = 1

        scan_k_target = self._SCAN_K
        refine_k = self._REFINE_K
        voiced_thr = self._VOICED
        silence = self._SILENCE

        periods = []
        confs = []

        for start in starts:
            # 抽稀样本数：物理跨度 ks*decim 必须覆盖最长候选周期
            # （max_lag）才有意义，故下限取 max_lag/decim；同时设上限
            # 避免低采样率下做无用功。
            need = (max_lag + decim - 1) // decim
            scan_k = min(scan_k_target, max(need, 96))
            ks = min(scan_k, (w - max_lag - 1) // decim)
            if ks < 8:
                continue

            # 静音/直流门限 + 零滞后能量（在抽稀序列上计算，与相关同口径）。
            # 先求均值再用去均值能量，使纯直流（常数信号，方差为 0）与
            # 静音一并被判为清音；当 lag 网格与信号抽稀对齐
            # （lag_step==decim）时一次性构造抽稀视图 xd，让扫描内层
            # 循环使用连续下标、去掉乘法
            aligned = lag_step == decim and decim > 1
            if aligned:
                # 视图需覆盖到最大 padding lag（max_lag//decim + 2）+ ks 求和长度。
                # 帧从 start 起、长度为 w，故可用抽稀样本数为 w//decim；
                # 若不足则先缩短 ks 再定视图长度，保证不越界。
                avail = w // decim
                need_view = ks + max_lag // decim + 3
                if need_view > avail:
                    ks = max(8, avail - max_lag // decim - 3)
                    need_view = ks + max_lag // decim + 3
                view_len = min(need_view, avail)
                xd = [0.0] * view_len
                m = 0
                src = start
                while m < view_len:
                    xd[m] = x[src]
                    src += decim
                    m += 1
                mean = sum(xd[:ks]) / ks
                e0s = 0.0
                peak_amp = 0.0
                for m in range(ks):
                    dv = xd[m] - mean
                    xd[m] = dv
                    e0s += dv * dv
                    av = dv if dv >= 0.0 else -dv
                    if av > peak_amp:
                        peak_amp = av
                # 视图后半段（lag 偏移用到）同样去均值
                for m in range(ks, view_len):
                    xd[m] -= mean
            else:
                xd = None
                mean = 0.0
                j = 0
                while j < ks:
                    mean += x[start + j * decim]
                    j += 1
                mean /= ks
                e0s = 0.0
                peak_amp = 0.0
                j = 0
                while j < ks:
                    dv = x[start + j * decim] - mean
                    e0s += dv * dv
                    av = dv if dv >= 0.0 else -dv
                    if av > peak_amp:
                        peak_amp = av
                    j += 1
            if peak_amp < silence or e0s <= 0.0:
                continue

            # ---------------- 扫描：信号抽稀的全区间自相关 ----------------
            scan = {}
            sg = -1.0
            sgl = min_lag
            if aligned:
                # 连续下标快速路径：lag 以抽稀单位计；两端各外扩 2 个
                # 网格点作为 padding，供边界峰的双侧局部极大检验
                td = max(1, min_lag // decim - 2)
                td_max = max_lag // decim + 2
                while td <= td_max:
                    acc = 0.0
                    j = 0
                    while j < ks:
                        acc += xd[j] * xd[j + td]
                        j += 1
                    tau = td * decim
                    val = acc / e0s
                    scan[tau] = val
                    if min_lag <= tau <= max_lag and val > sg:
                        sg = val
                        sgl = tau
                    td += 1
            else:
                # 通用路径：lag 按 lag_step 递增，两端各外扩 2 个网格点
                # 作为 padding；两侧均减去抽稀序列均值，去除直流分量
                tau = max(lag_step, min_lag - 2 * lag_step)
                tau_end = max_lag + 2 * lag_step
                while tau <= tau_end:
                    acc = 0.0
                    j = 0
                    while j < ks:
                        idx = start + j * decim
                        acc += (x[idx] - mean) * (x[idx + tau] - mean)
                        j += 1
                    val = acc / e0s
                    scan[tau] = val
                    if min_lag <= tau <= max_lag and val > sg:
                        sg = val
                        sgl = tau
                    tau += lag_step
            if sg < voiced_thr:
                continue

            # 抽稀网格上的严格双侧局部极大（两侧都严格更小）。
            # 严格性可排除单调包络上的伪峰（如低基频时短 lag 处的高相关），
            # 而真正的基频周期与其谐波周期都是真正的局部极大。
            # 扫描时额外计算合法区间外的 padding lag（见下），因此落在
            # min_lag/max_lag 边界上的真实峰也能通过双侧检验被认定。
            grid = sorted(scan)
            speaks = []
            for idx, t in enumerate(grid):
                if t < min_lag or t > max_lag:
                    continue
                if (t - lag_step) not in scan or (t + lag_step) not in scan:
                    continue
                if scan[t] > scan[t - lag_step] \
                        and scan[t] > scan[t + lag_step]:
                    speaks.append(t)
            if sgl not in speaks and min_lag <= sgl <= max_lag:
                speaks.append(sgl)

            # 倍频消解只需细化“最短达标峰”及其半周期（次谐波安全网）；
            # 其余高次谐波峰不影响“最低有效周期”的判定，无需细化。
            # 候选门限与细化阶段一致：绝对强度 + 相对全局峰比例，避免
            # 强噪声下真正的基频峰（略低于全局噪声峰）在扫描阶段被漏掉。
            gate = sg * self._OCTAVE_REL
            qualifying = [t for t in sorted(speaks)
                          if scan[t] >= self._OCTAVE_ABS and scan[t] >= gate]
            if not qualifying:
                qualifying = [sgl]
            shortest = qualifying[0]
            cand_set = {shortest}
            # 全局最强扫描峰也纳入细化：强噪声下最短达标峰可能是短 lag
            # 处的随机巧合，而真正的基频周期恰是全局最强峰；在全分辨率、
            # 长积分的细化值上再用倍频规则统一裁决，可剔除这类伪短周期。
            if min_lag <= sgl <= max_lag:
                cand_set.add(sgl)
            half = shortest // 2
            if half >= min_lag:
                cand_set.add(half)

            # ---------------- 细化：全分辨率、长积分 ----------------
            # 合并相邻候选的邻域（候选须按升序处理）；半宽 = 扫描网格
            # 半步 + 1 个逐点余量 + 两端各 2 个 lag 供五点加权插值
            half_w = 2 + (lag_step // 2) + 1
            intervals = []
            for c in sorted(cand_set):
                lo = max(2, c - half_w)
                hi = min(w // 2 - 2, c + half_w)
                if intervals and lo <= intervals[-1][1] + 1:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], hi))
                else:
                    intervals.append((lo, hi))

            # 细化积分长度：
            #  - 至少覆盖 2 个候选周期，否则峰顶过平、整数峰位会偏移；
            #  - 且不少于 _REFINE_MIN_K 个样本（约 6 个最短周期），否则
            #    强噪声下一两个周期内的随机会量就会把局部峰推偏一个 lag；
            #  - 目标取 3 个周期，并钳制在 refine_k 以内控制成本。
            target_k = self._REFINE_PERIODS * shortest
            min_k = max(2 * shortest, self._REFINE_MIN_K)
            if target_k < min_k:
                target_k = min_k
            if target_k > refine_k:
                target_k = refine_k
            kr = min(target_k, w - max_lag - 1)
            if kr < 16:
                kr = ks * decim
                if kr < 16:
                    continue
            # 零滞后能量与自相关均去均值，去除直流分量
            rmean = 0.0
            i = start
            endr = start + kr
            while i < endr:
                rmean += x[i]
                i += 1
            rmean /= kr
            e0r = 0.0
            i = start
            while i < endr:
                dv = x[i] - rmean
                e0r += dv * dv
                i += 1
            if e0r <= 0.0:
                continue

            rr = {}
            for lo, hi in intervals:
                tau = lo
                while tau <= hi:
                    acc = 0.0
                    i = start
                    end = start + kr
                    while i < end:
                        acc += (x[i] - rmean) * (x[i + tau] - rmean)
                        i += 1
                    rr[tau] = acc / e0r
                    tau += 1

            # 细化值上的严格局部极大（邻点严格更小），避免单调包络伪峰；
            # 周期峰只在合法 [min_lag, max_lag] 内认定，padding 仅供插值。
            # 细化区间边界是人为切分的，落在边界上的峰改用单侧邻点判定，
            # 以免恰好位于区间边缘的真实峰被漏判。
            rpeaks = []
            for t in rr:
                if not (min_lag <= t <= max_lag):
                    continue
                lv = rr.get(t - 1)
                rv = rr.get(t + 1)
                if lv is not None and rr[t] <= lv:
                    continue
                if rv is not None and rr[t] <= rv:
                    continue
                rpeaks.append(t)
            valid_vals = [rr[t] for t in rr if min_lag <= t <= max_lag]
            g = max(valid_vals)
            gl = max((t for t in rr if min_lag <= t <= max_lag),
                     key=lambda t: rr[t])
            if gl not in rpeaks:
                rpeaks.append(gl)
            if g < voiced_thr:
                continue

            # 倍频消解：在“足够强”的局部极大中取最短者 = 最低有效周期。
            # 双重门限：绝对值门限（浊音峰 r>=~0.45，纯噪声峰不超过 ~0.2）
            # 与相对门限（达到全局峰的 85%，容忍谐波导致的基频峰略弱），
            # 从而在含谐波/噪声时仍选到真正的基频周期而非其倍频。
            rgate = g * self._OCTAVE_REL
            final_lag = gl
            for t in sorted(rpeaks):
                if rr[t] >= self._OCTAVE_ABS and rr[t] >= rgate:
                    final_lag = t
                    break
            f_val = rr[final_lag]

            # 五点插值（缺的邻点补算；区间 padding 已保证边界峰也有邻域）
            need = (final_lag - 2, final_lag - 1, final_lag + 1,
                    final_lag + 2)
            if all(2 <= t <= w // 2 - 2 for t in need):
                for t in need:
                    if t not in rr:
                        acc = 0.0
                        i = start
                        end = start + kr
                        while i < end:
                            acc += (x[i] - rmean) * (x[i + t] - rmean)
                            i += 1
                        rr[t] = acc / e0r
                period = self._interp5(rr, final_lag)
            else:
                period = float(final_lag)

            if period < min_lag:
                period = float(min_lag)
            elif period > max_lag:
                period = float(max_lag)

            conf = g if g <= 1.0 else 1.0
            periods.append(period)
            confs.append(conf)

        if not periods:
            self._last_conf = 0.0
            return 0.0

        # 跨帧聚合：按周期聚类。单帧在强噪声下可能偶然落到倍/次谐波上，
        # 但真实周期会在多数帧重复出现，而噪声峰的位置帧间不重复。因此
        # 先按“成员帧数”多寡选簇（多数共识），帧数相同时再用总置信度
        # 决胜；簇内代表周期按置信度加权平均。
        order = sorted(range(len(periods)), key=lambda idx: periods[idx])
        clusters = []          # 每个簇: [代表周期, 权重和, 成员下标]
        for idx in order:
            t = periods[idx]
            w = confs[idx]
            placed = False
            for cl in clusters:
                # 相对距离 < 8% 视为同一周期（容差大于 1% 精度要求）
                if abs(t - cl[0]) / cl[0] < 0.08:
                    # 增量更新加权代表值
                    new_w = cl[1] + w
                    cl[0] = (cl[0] * cl[1] + t * w) / new_w
                    cl[1] = new_w
                    cl[2].append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([t, w, [idx]])

        # 主键：成员帧数（多数共识）；帧数相同时（典型为强噪声下 2 帧
        # 各执一词）取更短周期——倍频/次谐波失误几乎都表现为周期被高估，
        # 真实基频是其中更短的强峰。
        best = max(clusters, key=lambda cl: (len(cl[2]), -cl[0]))
        med_period = best[0]
        member_w = sum(confs[idx] for idx in best[2])
        f0 = self.rate / med_period
        if f0 < self.lo:
            f0 = self.lo
        elif f0 > self.hi:
            f0 = self.hi

        # 置信度：获胜簇权重占比 × 簇内平均相关强度
        share = member_w / max(sum(confs), 1e-12)
        conf = share * (sum(confs[idx] for idx in best[2]) / len(best[2]))
        self._last_conf = conf if conf <= 1.0 else 1.0
        return float(f0)
