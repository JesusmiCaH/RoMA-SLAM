import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def load_vggt(pth = "facebook/VGGT-1B", device = "cuda"):
    model = VGGT.from_pretrained(pth).to(device)
    return model