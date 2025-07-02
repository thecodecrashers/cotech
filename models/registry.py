# models/registry.py

from models.unet import UNet

MODEL_REGISTRY = {
    "unet": UNet,
    # 之后添加： "deeplabv3": DeepLabV3(), "segformer": SegFormer()
}

def get_model(name, **kwargs):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{name}' 未注册！可用模型：{list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
