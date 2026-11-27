"""audioonset 核心 API（初始未实现，仅标准库）。"""


class Onset:
    def __init__(self, rate, frame, hop, threshold, min_gap):
        """构造"""
        raise NotImplementedError

    def flux(self, samples):
        """谱通量"""
        raise NotImplementedError

    def detect(self, samples):
        """检测"""
        raise NotImplementedError
