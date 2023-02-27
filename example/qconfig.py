import torch

class CustomQConfigs:
    @staticmethod
    def get_default_qconfig():
        return torch.quantization.QConfig(activation=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                reduce_range=False),
            weight=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric))
