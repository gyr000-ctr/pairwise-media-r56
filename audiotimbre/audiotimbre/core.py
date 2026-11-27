"""audiotimbre 核心 API（初始未实现，仅标准库）。"""


class Timbre:
    def __init__(self, rate, bands, coeffs):
        """构造"""
        raise NotImplementedError

    def mel_bank(self, nfft):
        """滤波器组"""
        raise NotImplementedError

    def mfcc(self, samples):
        """系数"""
        raise NotImplementedError
