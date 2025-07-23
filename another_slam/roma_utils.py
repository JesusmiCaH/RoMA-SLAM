import os
import torch
import numpy as np
from romatch import roma_outdoor, roma_indoor
from PIL import Image

def load_roma(device = "cuda"):
    roma_model = roma_indoor(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False


def get_warps(model, viewpointA, viewpointB):
    imA = viewpointA.img.cpu().numpy().transpose(1, 2, 0)
    imA = (imA * 255).astype(np.uint8)
    imA = Image.fromarray(imA)
    imB = viewpointB.img.cpu().numpy().transpose(1, 2, 0)
    imB = (imB * 255).astype(np.uint8)
    imB = Image.fromarray(imB)

    with torch.no_grad():
        warp, certainty_warp = model.match(
            imA, imB
        )
    return warp, certainty_warp

def get_matches(viewpointA, viewpointB, num_matches=15000):
    warp, certainty_warp = get_warps(
        viewpointA, viewpointB
    )
    warp = warp.reshape(-1, 4)  # H*W x 4
    certainty_warp = certainty_warp.reshape(-1).clone()  # H*W
    certainty_warp[certainty_warp < 0.6] = 0
    good_samples = torch.multinomial(certainty_warp, num_matches, replacement=False)
    kpts_A, kpts_B = warp[good_samples].split(2, dim=1)
    return kpts_A, kpts_B