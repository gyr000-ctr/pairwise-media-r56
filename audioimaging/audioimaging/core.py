"""audioimaging 核心 API（初始未实现，仅标准库）。"""


class Imaging:
    def ms(self, left, right):
        """M/S 分解"""
        raise NotImplementedError

    def from_ms(self, m, s):
        """逆变换"""
        raise NotImplementedError

    def correlation(self, left, right):
        """相关度"""
        raise NotImplementedError

    def widen(self, left, right, width):
        """展宽"""
        raise NotImplementedError
