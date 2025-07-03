# models/registry.py

from models.unet import UNet
from models.segformer import SegFormer

MODEL_REGISTRY = {
    "unet": UNet,
    "segformer":SegFormer,
    # 之后添加： "deeplabv3": DeepLabV3(), "segformer": SegFormer()
}

def get_model(name, in_channels=1, out_channels=1):
    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    elif name == "segformer":
        return SegFormer(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model: {name}")