# models/registry.py

from models.unet import UNet
from models.segformer import SegFormer
from models.deeplabv3 import DeepLabV3
from models.unetpp import UNetPP  # ✅ 添加这一行
from models.hrnetv2 import HRNetV2

MODEL_REGISTRY = {
    "unet": UNet,
    "segformer": SegFormer,
    "deeplabv3": DeepLabV3,
    "unetpp": UNetPP,  # ✅ 添加这一行
    "hrnetv2": HRNetV2  # ✅ 添加这一行
}

def get_model(name, in_channels=1, out_channels=1):
    if name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    elif name == "segformer":
        return SegFormer(in_channels=in_channels, out_channels=out_channels)
    elif name == "deeplabv3":
        return DeepLabV3(in_channels=in_channels, out_channels=out_channels)
    elif name == "unetpp":
        return UNetPP(in_channels=in_channels, out_channels=out_channels)
    elif name=="hrnetv2":
        return HRNetV2(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model: {name}")
