"""audiotimestretch 核心 API（初始未实现，仅标准库）。"""


class TimeStretch:
    def __init__(self, frame, hop, rate):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError

    def cola_error(self):
        """窗和误差"""
        raise NotImplementedError
