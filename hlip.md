# Repository Analysis

## Summary

```
Repository: tinycrops/hlip
Files analyzed: 11

Estimated tokens: 8.8k
```

## Important Files

```
Directory structure:
└── tinycrops-hlip/
    ├── README.md
    ├── LICENSE
    ├── data/
    │   ├── ct_rate/
    │   │   ├── files/
    │   │   └── metafiles/
    │   ├── pub_brain_5/
    │   │   ├── brats23/
    │   │   ├── nyu_mets/
    │   │   ├── open_bhb/
    │   │   ├── stroke/
    │   │   └── ucsf_mets/
    │   └── rad_chestct/
    │       ├── files/
    │       ├── metafiles/
    │       └── notebooks/
    ├── docs/
    │   ├── BraTS-GLI-00459-000/
    │   │   ├── BraTS-GLI-00459-000-t1c.pt
    │   │   └── BraTS-GLI-00459-000-t1n.pt
    │   └── tst32751/
    └── src/
        ├── hlip/
        │   ├── patch_embed.py
        │   ├── pos_embed.py
        │   ├── visual_encoder.py
        │   └── model_configs/
        │       ├── vit_base_multiscan_h2_token1176.json
        │       ├── vit_base_multiscan_h2_token588.json
        │       ├── vit_base_multiscan_h3_token1176.json
        │       └── vit_base_singlescan_h2_token2744.json
        └── hlip_test/

```

## Content

```
================================================
File: README.md
================================================
# HLIP
> Official PyTorch implementation of the following paper:\
> Towards Scalable Language-Image Pre-training for 3D Medical Imaging\
> University of Michigan\
> [![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.21862-b31b1b.svg)](https://arxiv.org/abs/2505.21862)&nbsp;


## Overview
<p align="center"><img src="https://github.com/Zch0414/hlip/blob/master/docs/github.png" width=96% height=96% class="center"></p>

We propose **H**ierarchical attention for **L**anguage-**I**mage **P**re-training (**HLIP**), inspired by the natural hierarchy of radiology data: slice, scan, and study. With this lightweight attention mechanism, HLIP can be trained directly on uncurated clinical datasets, enabling scalable language-image pre-training in 3D medical imaging. For real-world clinical use, HLIP can be applied to studies containing either a single scan (e.g., chest CT) or multiple scans (e.g., brain MRI).

## Updates
- **(Todo)** Release training code.
- **(Todo)** Release evaluation code.
- **(2025-06)** Release data process pipeline.
- **(2025-05)** Release HLIP models trained on chest CT and brain MRI, feel free to try our demos.

## Getting Started

### Install 
[open-clip](https://github.com/mlfoundations/open_clip/tree/main)
```bash
python3 -m venv env
source env/bin/activate
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone git@github.com:mlfoundations/open_clip.git
cd open_clip
make install
make install-training
```

### Pre-trained Weight
| Modality | Attention | Patch Size | Model |
| -------- | -------- | -------- | -------- |
| Chest CT | <code>slice</code> <code>scan</code> | <code>8, 24, 24</code> | [ViT-Base](https://drive.google.com/file/d/1muu7L9H3KaL3nq3fNtN8kKF1eDK3R5Z4/view?usp=drive_link) |
| Brain MRI | <code>scan</code> <code>study</code> | <code>16, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/1uUdcE0TYx3K2YU7FQMfwb2FsFQjQcGil/view?usp=drive_link) |
| Brain MRI | <code>scan</code> <code>study</code> | <code>8, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/12BwJvd6IEZynXM8jkled0ND7t11iuySj/view?usp=drive_link) |
| Brain MRI | <code>slice</code> <code>scan</code> <code>study</code> | <code>8, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/1FgOS3W6LhnhH4gJlbASPopUEXChcjeqy/view?usp=drive_link) |

### Demo
Chest CT
```bash
python inference_rad_chestct.py \
  --model vit_base_singlescan_h2_token1176 \
  --resume /path/to/vit_base_singlescan_h2_token1176.pt \
  --data /docs/tst32751/tst32751.pt \
```
Brain MRI
```bash
python inference_pub_brain_5.py \
  --model vit_base_multiscan_h2_token1176 \
  --resume /path/to/vit_base_multiscan_h2_token1176.pt \
  --patch-size 8 16 16 \
  --num-slices 72 \
  --data /docs/BraTS-GLI-00459-000/ \
```
Visualizing the activation with <code>--interpret</code>.




================================================
File: LICENSE
================================================
MIT License

Copyright (c) 2025 Zach

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.













================================================
File: docs/BraTS-GLI-00459-000/BraTS-GLI-00459-000-t1c.pt
================================================
[Non-text file]


================================================
File: docs/BraTS-GLI-00459-000/BraTS-GLI-00459-000-t1n.pt
================================================
[Non-text file]



================================================
File: src/hlip/patch_embed.py
================================================
import torch.nn as nn
    

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224, 224), patch_size=(16, 16, 16), in_chans=1, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert len(list(img_size)) == 3, 'Specify the input size at every dimension'
        assert len(list(patch_size)) == 3, 'Specify the patch size at every dimension'

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=kwargs["bias"])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W)
        x = self.proj(x)
        _, _, D, H, W = x.shape
        
        # BN * C' * D' * H' * W' -> B * N * D' * H' * W' * C'
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, D, H, W, -1) 
        x = self.norm(x)
        return x


================================================
File: src/hlip/pos_embed.py
================================================
import numpy as np
from timm.layers import resample_abs_pos_embed as resample_2d_posemb

import torch
import torch.nn.functional as F


def resample_1d_posemb(posemb, num_samples, is_train=True):
    """resample study position embedding.
    """
    posemb = posemb.float()
    _max = posemb.shape[1]

    # interpolate
    if _max < num_samples:
        assert not is_train
        posemb = F.interpolate(posemb.permute(0, 2, 1), size=num_samples, mode='linear').permute(0, 2, 1)
        return posemb
    
    # sample
    if num_samples <= _max:
        if is_train:
            perm = torch.randperm(_max)[:num_samples]
        else:
            perm = torch.arange(num_samples)
        posemb = posemb[:, perm, :]
        return posemb


def resample_3d_posemb(posemb, new_size, old_size):
    # new_size and old_size should be provided with the same shape: [d, h, w]
    # d: through-place dimension, h and w: in-place dimension.
    posemb = posemb.float()
    if new_size == old_size:
        return posemb

    # interpolate
    if old_size[0] != new_size[0] or old_size[1] != new_size[1] or old_size[2] != new_size[2]:
        posemb = F.interpolate(posemb.permute(0, 4, 1, 2, 3), size=(new_size[0], new_size[1], new_size[2]), mode='trilinear').permute(0, 2, 3, 4, 1)

    return posemb


def study_pos_embed(max_num_scans, grid_size, embed_dim, pretrained_posemb=None): 
    """
    pretrained_posemb should be a 2D position embedding without prefix_posemb
    Return:
        spatial_posemb: A tensor of shape [1, d, h, w, embed_dim]
        sequential_posemb: A tensor of shape [1, max_num_scans, embed_dim]
    """
    if pretrained_posemb is not None: 
        # build mri position embedding from pretrained_posemb:
        # study_posemb (1d) + slice_posemb (1d) + pretrained_posemb (2d)
        pretrained_posemb = resample_2d_posemb(
            pretrained_posemb,
            new_size=(grid_size[1], grid_size[2]),
            num_prefix_tokens=0 # enforce 0
        )
        pretrained_posemb = pretrained_posemb.reshape(1, grid_size[1], grid_size[2], embed_dim)
        slice_posemb = get_1d_sincos_pos_embed(
            embed_dim=embed_dim,
            sequence_len=grid_size[0],
            cls_token=False
        ) # [n, embed_dim]
        slice_posemb = torch.from_numpy(slice_posemb).float()
        spatial_posemb = slice_posemb[None, :, None, None, :] + pretrained_posemb[:, None, :, :, :] 
        if max_num_scans > 1:
            sequential_posemb = get_1d_sincos_pos_embed(
                embed_dim=embed_dim,
                sequence_len=max_num_scans,
                cls_token=False
            ) # [n, embed_dim]
            sequential_posemb = torch.from_numpy(sequential_posemb).float()[None, ...]
        else:
            sequential_posemb = None
    else: 
        # build mri position embedding from scratch: study_posemb (1d) + sequential_posemb (3d)
        spatial_posemb = get_3d_sincos_pos_embed(
            embed_dim=embed_dim, 
            grid_sizes=grid_size,
            cls_token=False,
            flatten=False,
        ) # [d, h, w, embed_dim]
        spatial_posemb = torch.from_numpy(spatial_posemb).float()[None, ...]
        if max_num_scans > 1:
            sequential_posemb = get_1d_sincos_pos_embed(
                embed_dim=embed_dim,
                sequence_len=max_num_scans,
                cls_token=False,
            ) # [n, embed_dim]
            sequential_posemb = torch.from_numpy(sequential_posemb).float()[None, ...]
        else:
            sequential_posemb = None
            
    return spatial_posemb, sequential_posemb


def get_3d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False, flatten=True):
    """
    grid_sizes: sequence of the grid depth, height, and width
    return:
    pos_embed: [dot(grid_sizes), embed_dim] or [1+dot(grid_sizes), embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_sizes[0], dtype=np.float32)
    grid_h = np.arange(grid_sizes[1], dtype=np.float32)
    grid_w = np.arange(grid_sizes[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_sizes[0], grid_sizes[1], grid_sizes[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if flatten:
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = pos_embed.reshape([grid_sizes[0], grid_sizes[1], grid_sizes[2], embed_dim])
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False, flatten=True):
    """
    grid_sizes: sequence of the grid height and width.
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_sizes[0], dtype=np.float32)
    grid_w = np.arange(grid_sizes[1], dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_sizes[0], grid_sizes[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if flatten:
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = pos_embed.reshape([grid_sizes[0], grid_sizes[1], embed_dim])
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, sequence_len, cls_token=False):
    """
    sequence_len: int of the sequence length
    return:
    pos_embed: [sequence_len, embed_dim] or [1+sequence_len, embed_dim] (w/ or w/o cls_token)
    """
    sequence = np.arange(sequence_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, sequence)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use one third of dimensions to encode grid_d
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D', D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D', D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D', D/3)

    emb = np.concatenate([emb_w, emb_h, emb_d], axis=1) # (H*W*D', D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


================================================
File: src/hlip/visual_encoder.py
================================================
import os
import sys
sys.path.append(os.path.abspath('.'))

from functools import partial

import torch
import torch.nn as nn

from timm.models import register_model, build_model_with_cfg, checkpoint
from timm.models.vision_transformer import VisionTransformer

from .patch_embed import PatchEmbed3D
from .pos_embed import study_pos_embed, resample_1d_posemb, resample_3d_posemb


class HLIPVisualEncoder(VisionTransformer):
    def __init__(self, **kwargs):
        max_num_scans = kwargs.pop('max_num_scans')
        self.slice_attn_indexes = kwargs.pop('slice_attn_indexes', ())
        self.scan_attn_indexes = kwargs.pop('scan_attn_indexes', ())
        self.study_attn_indexes = kwargs.pop('study_attn_indexes', ())
        super().__init__(**kwargs)

        # reset pos_embed
        spatial_posemb, sequential_posemb = study_pos_embed(
            max_num_scans=max_num_scans,
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
            pretrained_posemb=None,
        )
        self.spatial_posemb = nn.Parameter(spatial_posemb)
        self.spatial_posemb.requires_grad = False
        if sequential_posemb is not None:
            self.sequential_posemb = nn.Parameter(sequential_posemb)
            self.sequential_posemb.requires_grad = False
        else:
            self.sequential_posemb = None

    def _pos_embed(self, x):
        # x: [bs, n, d, h, w, c]
        bs, n, d, h, w, _ = x.shape
        spatial_posemb = resample_3d_posemb(self.spatial_posemb, (d, h, w), self.patch_embed.grid_size)
        if self.sequential_posemb is not None:
            sequential_posemb = resample_1d_posemb(self.sequential_posemb, n, is_train = bs!=1)
            pos_embed = sequential_posemb[:, :, None, None, None, :] + spatial_posemb[:, None, :, :, :, :]
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)
        else:
            pos_embed = spatial_posemb[:, None, :, :, :, :].expand(bs, n, -1, -1, -1, -1)

        # start status for vit blocks
        if 0 in self.slice_attn_indexes:
            pos_embed = pos_embed.flatten(3, 4).flatten(0, 2) # [bs * n * d, h * w, c]
            x = x.flatten(3, 4).flatten(0, 2) # [bs * n * d, h * w, c]
        elif 0 in self.scan_attn_indexes:
            pos_embed = pos_embed.flatten(2, 4).flatten(0, 1) # [bs * n, d * h * w, c]
            x = x.flatten(2, 4).flatten(0, 1) # [bs * n, d * h * w, c]
        elif 0 in self.study_attn_indexes:
            pos_embed = pos_embed.flatten(1, 4) # [bs , n * d * h * w, c]
            x = x.flatten(1, 4) # [bs , n * d * h * w, c]

        x = self.pos_drop(x + pos_embed)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x, n, d
     
    def _slice2scan(self, x, num_slices):
        """
        Slice unpartition into the original scan.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_slices (int): number of slices in one scan.

        Returns:
            x: [B * num_scans, num_prefix_tokens + num_slices * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape

        prefix_tokens = prefix_tokens.view(BND//num_slices, num_slices, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND//num_slices, num_slices, L, C).view(BND//num_slices, num_slices * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x
    
    def _slice2study(self, x, num_scans, num_slices):
        """
        Slices unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.
            num_slices (int): number of slices in on scan.

        Returns:
            x: [B, num_prefix_tokens + num_scans * num_slices * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape
        
        prefix_tokens = prefix_tokens.view(BND//(num_scans*num_slices), num_scans * num_slices, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND//(num_scans*num_slices), num_scans * num_slices, L, C).view(BND//(num_scans*num_slices), num_scans * num_slices * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x

    def _scan2study(self, x, num_scans):
        """
        Scans unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.

        Returns:
            x: [B, num_prefix_tokens + num_scans * L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BN, L, C = src.shape
        
        prefix_tokens = prefix_tokens.view(BN//num_scans, num_scans, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BN//num_scans, num_scans, L, C).view(BN//num_scans, num_scans * L, C)

        x = torch.cat([prefix_tokens, src], dim=1)
        return x
    
    def _study2slice(self, x, num_scans, num_slices):
        """
        Study partition into non-overlapping slices.
        Args:
            x (tensor): input tokens with [B, num_prefix_tokens + num_scans * num_slices * L, C].
            num_scans (int): number of scans in one study.
            num_slices (int): number of slices in one scan.

        Returns:
            x: [B * num_scans * num_slices, num_prefix_tokens + L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        B, NDL, C = src.shape

        prefix_tokens = prefix_tokens.view(B, 1, 1, self.num_prefix_tokens, C).expand(-1, num_scans, num_slices, -1, -1).contiguous()
        src = src.view(B, num_scans, num_slices, NDL//(num_scans*num_slices), C)
        
        x = torch.cat([prefix_tokens, src], dim=3)
        x = x.view(-1, self.num_prefix_tokens+NDL//(num_scans*num_slices), C)
        return x
    
    def _study2scan(self, x, num_scans):
        """
        Study partition into non-overlapping scans.
        Args:
            x (tensor): input tokens with [B, num_prefix_tokens + num_scans * L, C].
            num_scans (int): number of scans in one study.

        Returns:
            x: [B * num_scans, num_prefix_tokens + L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        B, NL, C = src.shape

        prefix_tokens = prefix_tokens.view(B, 1, self.num_prefix_tokens, C).expand(-1, num_scans, -1, -1).contiguous()
        src = src.view(B, num_scans, NL//num_scans, C)
        
        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens+NL//num_scans, C)
        return x

    def _scan2slice(self, x, num_slices):
        """
        Scan partition into non-overlapping slices.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + num_slices * L, C].
            num_slices (int): number of slices in one scan.

        Returns:
            x: [B * num_scans * num_slices, num_prefix_tokens + L, C].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BN, DL, C = src.shape

        prefix_tokens = prefix_tokens.view(BN, 1, self.num_prefix_tokens, C).expand(-1, num_slices, -1, -1).contiguous()
        src = src.view(BN, num_slices, DL//(num_slices), C)
        
        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens+DL//num_slices, C)
        return x

        
    def forward_features(self, x):
        x = self.patch_embed(x) # [b, n, d, h, w, c]
        x, num_scans, num_slices = self._pos_embed(x) # starts from: [b * n * d, h * w, c] if have slice attn else [b * n, d * h * w, c]
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if idx-1 in self.slice_attn_indexes and idx in self.study_attn_indexes:
                x = self._slice2study(x, num_scans, num_slices)
            elif idx-1 in self.slice_attn_indexes and idx in self.scan_attn_indexes:
                x = self._slice2scan(x, num_slices)
            elif idx-1 in self.scan_attn_indexes and idx in self.study_attn_indexes:
                x = self._scan2study(x, num_scans)
            elif idx-1 in self.study_attn_indexes and idx in self.slice_attn_indexes:
                x = self._study2slice(x, num_scans, num_slices)
            elif idx-1 in self.study_attn_indexes and idx in self.scan_attn_indexes:
                x = self._study2scan(x, num_scans)
            elif idx-1 in self.scan_attn_indexes and idx in self.slice_attn_indexes:
                x = self._scan2slice(x, num_slices)
            
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        if len(self.blocks) - 1 in self.scan_attn_indexes: 
            x = self._scan2study(x, num_scans)
        elif len(self.blocks) - 1 in self.slice_attn_indexes:
            x = self._slice2study(x, num_slices, num_scans)
            
        return self.norm(x)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def custom_checkpoint_filter_fn(state_dict, model, patch_size=(16, 16, 16)):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)

    # determine whether the cls_token has corresponding pos_embed
    embed_len = state_dict['pos_embed'].shape[1]
    if torch.sqrt(torch.tensor(embed_len)) != torch.sqrt(torch.tensor(embed_len)).floor():
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]

    for k, v in state_dict.items():
        if 'patch_embed' in k:
            if model.patch_embed.__class__ == PatchEmbed3D:
                if 'weight' in k:
                    if (v.shape[2], v.shape[3]) != (patch_size[1], patch_size[2]):
                        v = torch.nn.functional.interpolate(v, size=(patch_size[1], patch_size[2]), mode='bicubic')
                    v = v.sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1).div(patch_size[0])
            else:
                continue
        if 'pos_embed' in k:
            spatial_posemb, _ = study_pos_embed(
                max_num_scans = 1,
                grid_size = model.patch_embed.grid_size,
                embed_dim = model.embed_dim,
                pretrained_posemb = v
            )
            out_dict['spatial_posemb'] = spatial_posemb
            continue
        out_dict[k] = v
    return out_dict


def custom_create_vision_transformer(variant, **kwargs):
    kwargs.pop('pretrained_cfg_overlay')
    return build_model_with_cfg(
        model_cls=HLIPVisualEncoder,
        variant=variant,
        pretrained_cfg_overlay=dict(first_conv=None),
        pretrained_strict=False,
        pretrained_filter_fn=partial(custom_checkpoint_filter_fn, patch_size=kwargs['patch_size']),
        **kwargs,
    )


@register_model
def vit_base_singlescan_h2_token2744(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=1, slice_attn_indexes=(0, 1, 3, 4, 6, 7, 9, 10), study_attn_indexes=(2, 5, 8, 11),
        img_size=(112, 336, 336), patch_size=(8, 24, 24),
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D, 
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_multiscan_h2_token588(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=40, scan_attn_indexes=(0, 1, 3, 4, 6, 7, 9, 10), study_attn_indexes=(2, 5, 8, 11),
        img_size=(48, 224, 224), patch_size=(16, 16, 16),
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D, 
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_multiscan_h2_token1176(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=40, scan_attn_indexes=(0, 1, 3, 4, 6, 7, 9, 10), study_attn_indexes=(2, 5, 8, 11),
        img_size=(48, 224, 224), patch_size=(8, 16, 16),
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D, 
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_multiscan_h3_token1176(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=40, slice_attn_indexes=(0, 3, 6, 9), scan_attn_indexes=(1, 4, 7, 10), study_attn_indexes=(2, 5, 8, 11),
        img_size=(48, 224, 224), patch_size=(8, 16, 16),
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D, 
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


================================================
File: src/hlip/model_configs/vit_base_multiscan_h2_token1176.json
================================================
{
    "embed_dim": 512,
    "vision_cfg": {
        "timm_model_name": "vit_base_multiscan_h2_token1176",
        "timm_model_pretrained": true,
        "timm_drop_path": 0.2,
        "timm_pool": "token",
        "timm_proj": "linear"
    },
    "text_cfg": {
        "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_pooler_type": "cls_last_hidden_state_pooler",
        "hf_proj_type": "linear",
        "context_length": 256
    }
}


================================================
File: src/hlip/model_configs/vit_base_multiscan_h2_token588.json
================================================
{
    "embed_dim": 512,
    "vision_cfg": {
        "timm_model_name": "vit_base_multiscan_h2_token588",
        "timm_model_pretrained": true,
        "timm_drop_path": 0.2,
        "timm_pool": "token",
        "timm_proj": "linear"
    },
    "text_cfg": {
        "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_pooler_type": "cls_last_hidden_state_pooler",
        "hf_proj_type": "linear",
        "context_length": 256
    }
}


================================================
File: src/hlip/model_configs/vit_base_multiscan_h3_token1176.json
================================================
{
    "embed_dim": 512,
    "vision_cfg": {
        "timm_model_name": "vit_base_multiscan_h3_token1176",
        "timm_model_pretrained": true,
        "timm_drop_path": 0.2,
        "timm_pool": "token",
        "timm_proj": "linear"
    },
    "text_cfg": {
        "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_pooler_type": "cls_last_hidden_state_pooler",
        "hf_proj_type": "linear",
        "context_length": 256
    }
}


================================================
File: src/hlip/model_configs/vit_base_singlescan_h2_token2744.json
================================================
{
    "embed_dim": 512,
    "vision_cfg": {
        "timm_model_name": "vit_base_singlescan_h2_token2744",
        "timm_model_pretrained": true,
        "timm_drop_path": 0.2,
        "timm_pool": "token",
        "timm_proj": "linear"
    },
    "text_cfg": {
        "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hf_tokenizer_name": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "hf_pooler_type": "cls_last_hidden_state_pooler",
        "hf_proj_type": "linear",
        "context_length": 512
    }
}



```

