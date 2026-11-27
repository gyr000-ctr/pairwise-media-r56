"""audiopitch 核心 API（初始未实现，仅标准库）。"""


class Pitch:
    def __init__(self, rate, lo, hi):
        """构造"""
        raise NotImplementedError

    def detect(self, samples):
        """估计"""
        raise NotImplementedError

    def confidence(self):
        """置信度"""
        raise NotImplementedError
