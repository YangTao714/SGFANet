import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


def patch_divide(x: torch.Tensor, step: int, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h == patch_size and w == patch_size:
        step = patch_size

    patches = []
    for top in range(0, h + step - patch_size, step):
        down = top + patch_size
        if down > h:
            top, down = h - patch_size, h

        for left in range(0, w + step - patch_size, step):
            right = left + patch_size
            if right > w:
                left, right = w - patch_size, w
            patches.append(x[:, :, top:down, left:right])

    patches = torch.stack(patches, dim=0)                  # [n, b, c, ps, ps]
    patches = patches.permute(1, 0, 2, 3, 4).contiguous() # [b, n, c, ps, ps]
    return patches


def patch_reverse(
    patches: torch.Tensor,
    ref: torch.Tensor,
    step: int,
    patch_size: int
) -> torch.Tensor:
    b, c, h, w = ref.shape
    output = torch.zeros_like(ref)

    idx = 0
    for top in range(0, h + step - patch_size, step):
        down = top + patch_size
        if down > h:
            top, down = h - patch_size, h

        for left in range(0, w + step - patch_size, step):
            right = left + patch_size
            if right > w:
                left, right = w - patch_size, w

            output[:, :, top:down, left:right] += patches[:, idx]
            idx += 1

    for top in range(step, h + step - patch_size, step):
        down = top + patch_size - step
        if top + patch_size > h:
            top = h - patch_size
        output[:, :, top:down, :] /= 2

    for left in range(step, w + step - patch_size, step):
        right = left + patch_size - step
        if left + patch_size > w:
            left = w - patch_size
        output[:, :, :, left:right] /= 2

    return output


class DWConv(nn.Module):
    def __init__(self, hidden_features: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_features = hidden_features
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=hidden_features,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        x = x.transpose(1, 2).view(
            x.shape[0], self.hidden_features, x_size[0], x_size[1]
        ).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        kernel_size: int = 5,
        act_layer=nn.GELU,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_dim: int):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


class HF(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class HAFA(nn.Module):
    def __init__(self, dim: int, qk_dim: int, mlp_dim: int, heads: int = 1):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, qk_dim))
        self.hf = HF(dim)
        self.ffn = PreNorm(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        step = patch_size - 2
        patches = patch_divide(x, step, patch_size)
        b, n, c, ph, pw = patches.shape

        residual = patches

        hf_in = rearrange(patches, "b n c h w -> (b n) c h w")
        hf_out = self.hf(hf_in)
        hf_out = rearrange(hf_out, "(b n) c h w -> b n c h w", b=b, n=n)

        tokens = rearrange(patches, "b n c h w -> (b n) (h w) c")
        tokens = self.attn(tokens)
        tokens = rearrange(tokens, "(b n) (h w) c -> b n c h w", b=b, n=n, h=ph, w=pw)

        patches = residual + tokens + hf_out
        x = patch_reverse(patches, x, step, patch_size)

        _, _, h, w = x.shape
        x_seq = rearrange(x, "b c h w -> b (h w) c")
        x_seq = x_seq + self.ffn(x_seq, x_size=(h, w))
        x = rearrange(x_seq, "b (h w) c -> b c h w", h=h)
        return x


class SGFAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        qk_dim: int,
        *,
        top_n: int = 5,
        win_ks: Tuple[int, int] = (3, 3),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_gqa_group_share: bool = False,
        recompute_attn_cmp: bool = False,
        max_sp: int = 4048,
    ):
        super().__init__()
        assert qk_dim % heads == 0, "qk_dim must be divisible by heads"
        assert dim % heads == 0, "dim must be divisible by heads"

        self.dim = dim
        self.heads = heads
        self.qk_dim = qk_dim
        self.dk = qk_dim // heads
        self.dv = dim // heads
        self.scale = self.dk ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 3),
            nn.Sigmoid(),
        )

        self.top_n = int(top_n)
        self.use_gqa_group_share = bool(use_gqa_group_share)
        self.recompute_attn_cmp = bool(recompute_attn_cmp)
        self.max_sp = int(max_sp)
        self.win_ks = win_ks

    def _attn_q_k_v(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        return out, attn

    @torch.no_grad()
    def _clip_num_sp(self, labels: torch.Tensor) -> torch.Tensor:
        b, _, _ = labels.shape
        new_labels = labels.clone()
        for bi in range(b):
            uniq = torch.unique(new_labels[bi])
            if uniq.numel() > self.max_sp:
                factor = int(math.ceil(uniq.numel() / float(self.max_sp)))
                new_labels[bi] = new_labels[bi] // factor
        return new_labels

    def _sp_aggregate_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        h: int,
        w: int,
        sp_labels: torch.Tensor,
    ):
        bsz, heads, n, dk = k.shape
        dv = self.dv
        device = k.device

        sp = self._clip_num_sp(sp_labels.to(device)).view(bsz, -1)

        k_2d = k.transpose(2, 3).reshape(bsz * heads, dk, h, w)
        v_2d = v.transpose(2, 3).reshape(bsz * heads, dv, h, w)

        k_means_per_b, v_means_per_b, num_segments = [], [], []
        for bi in range(bsz):
            ids = torch.unique(sp[bi], sorted=True)
            ns = ids.numel()
            num_segments.append(int(ns))

            remap = torch.zeros(int(ids.max().item()) + 1, dtype=torch.long, device=device)
            remap[ids] = torch.arange(ns, device=device)
            pix2seg = remap[sp[bi]]

            k_b = k_2d[bi * heads:(bi + 1) * heads].reshape(heads, dk, n)
            v_b = v_2d[bi * heads:(bi + 1) * heads].reshape(heads, dv, n)

            one = torch.ones(n, device=device, dtype=k_b.dtype)
            denom = torch.zeros(ns, device=device, dtype=k_b.dtype).index_add_(0, pix2seg, one)
            denom = denom.clamp_min(1.0).unsqueeze(0).unsqueeze(-1)

            k_sum = torch.zeros(heads, ns, dk, device=device, dtype=k_b.dtype)
            v_sum = torch.zeros(heads, ns, dv, device=device, dtype=v_b.dtype)
            for hi in range(heads):
                k_sum[hi].index_add_(0, pix2seg, k_b[hi].transpose(0, 1))
                v_sum[hi].index_add_(0, pix2seg, v_b[hi].transpose(0, 1))

            k_means_per_b.append(k_sum / denom)
            v_means_per_b.append(v_sum / denom)

        nmax = max(num_segments)
        k_sp = torch.zeros(bsz, heads, nmax, dk, device=device, dtype=k.dtype)
        v_sp = torch.zeros(bsz, heads, nmax, dv, device=device, dtype=v.dtype)

        for bi in range(bsz):
            ns = num_segments[bi]
            k_sp[bi, :, :ns, :] = k_means_per_b[bi]
            v_sp[bi, :, :ns, :] = v_means_per_b[bi]

        return k_sp, v_sp, num_segments

    def Region_Convergence_Branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: int,
        w: int,
        sp_labels: torch.Tensor,
    ):
        bsz, heads, n, _ = q.shape
        k_sp, v_sp, num_segments = self._sp_aggregate_kv(k, v, h, w, sp_labels)

        out_cmp = q.new_zeros(bsz, heads, n, self.dv)
        attn_cmp_list = []
        for bi in range(bsz):
            ns = num_segments[bi]
            q_b = q[bi:bi + 1]
            k_b = k_sp[bi:bi + 1, :, :ns, :]
            v_b = v_sp[bi:bi + 1, :, :ns, :]
            out_b, attn_b = self._attn_q_k_v(q_b, k_b, v_b)
            out_cmp[bi] = out_b[0]
            attn_cmp_list.append(attn_b)

        return out_cmp, attn_cmp_list, (k_sp, v_sp, num_segments)

    @staticmethod
    def _resolve_sp_topn(
        sp_top_n: Optional[Union[torch.Tensor, int]],
        batch_idx: int,
    ) -> Optional[int]:
        if sp_top_n is None:
            return None
        if isinstance(sp_top_n, int):
            return int(sp_top_n)
        if not torch.is_tensor(sp_top_n):
            return None
        if sp_top_n.numel() == 1:
            return int(sp_top_n.view(-1)[0].item())
        if sp_top_n.dim() == 2 and sp_top_n.shape[1] == 1:
            return int(sp_top_n[batch_idx, 0].item())
        return int(sp_top_n[batch_idx].item())

    def Sparse_Selection_Branch(
        self,
        q: torch.Tensor,
        attn_cmp_list,
        kv_sp_tuple,
        sp_top_n: Optional[Union[torch.Tensor, int]] = None,
    ):
        bsz, heads, n, _ = q.shape
        k_sp, v_sp, num_segments = kv_sp_tuple
        out_sel = q.new_zeros(bsz, heads, n, self.dv)

        for bi in range(bsz):
            ns = num_segments[bi]

            if attn_cmp_list is None:
                q_b = q[bi:bi + 1]
                k_b = k_sp[bi:bi + 1, :, :ns, :]
                _, prob = self._attn_q_k_v(q_b, k_b, v_sp[bi:bi + 1, :, :ns, :])
            else:
                prob = attn_cmp_list[bi]

            k_eff = self._resolve_sp_topn(sp_top_n, bi)
            if k_eff is None or k_eff <= 0:
                k_eff = self.top_n
            k_eff = max(1, min(int(k_eff), int(ns)))

            if self.use_gqa_group_share:
                prob_group = prob.sum(dim=1)
                idx = prob_group.topk(k=k_eff, dim=-1).indices
                idx = idx.unsqueeze(1).expand(1, heads, n, idx.shape[-1])
            else:
                idx = prob.topk(k=k_eff, dim=-1).indices

            v_b = v_sp[bi, :, :ns, :]
            v_sel = v_b.unsqueeze(0).unsqueeze(2).expand(
                1, heads, n, ns, self.dv
            ).gather(
                3, idx.unsqueeze(-1).expand(1, heads, n, idx.shape[-1], self.dv)
            )

            weight = prob.gather(dim=-1, index=idx)
            weight = torch.softmax(weight, dim=-1)
            out_b = (weight.unsqueeze(-1) * v_sel).sum(dim=-2)
            out_sel[bi] = out_b[0]

        return out_sel

    def Local_Refinement_Branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: int,
        w: int,
    ):
        bsz, heads, n, _ = q.shape
        dv = self.dv

        win_h, win_w = self.win_ks
        win_h = win_h if win_h % 2 == 1 else win_h + 1
        win_w = win_w if win_w % 2 == 1 else win_w + 1
        pad_h, pad_w = win_h // 2, win_w // 2

        k_2d = k.transpose(2, 3).reshape(bsz * heads, self.dk, h, w)
        v_2d = v.transpose(2, 3).reshape(bsz * heads, dv, h, w)

        k_unfold = F.unfold(k_2d, kernel_size=(win_h, win_w), padding=(pad_h, pad_w), stride=1)
        v_unfold = F.unfold(v_2d, kernel_size=(win_h, win_w), padding=(pad_h, pad_w), stride=1)
        window_size = win_h * win_w

        k_unfold = k_unfold.view(bsz, heads, self.dk, window_size, n)
        k_unfold = k_unfold.permute(0, 1, 4, 3, 2).contiguous()

        v_unfold = v_unfold.view(bsz, heads, dv, window_size, n)
        v_unfold = v_unfold.permute(0, 1, 4, 3, 2).contiguous()

        attn = torch.sum(q.unsqueeze(-2) * k_unfold, dim=-1) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.sum(attn.unsqueeze(-1) * v_unfold, dim=-2)
        return out

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        sp_labels: torch.Tensor,
        sp_top_n: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        bsz, n, c = x.shape
        h, w = x_size

        assert c == self.dim, f"Expected dim={self.dim}, got {c}"
        assert h * w == n, "x_size must match sequence length N"
        assert sp_labels is not None, "SGFAttention requires sp_labels"
        assert sp_labels.shape[0] == bsz and sp_labels.shape[-2:] == (h, w), (
            f"sp_labels shape must be [B,H,W] with B={bsz}, H={h}, W={w}"
        )

        if sp_top_n is not None and torch.is_tensor(sp_top_n) and sp_top_n.numel() != 1:
            assert sp_top_n.shape[0] == bsz, (
                f"sp_top_n must have B={bsz} in dim0, got {tuple(sp_top_n.shape)}"
            )

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)

        gate = self.gate(x)
        g_cmp, g_slc, g_win = gate.unbind(dim=-1)
        g_cmp = g_cmp.unsqueeze(1).unsqueeze(-1)
        g_slc = g_slc.unsqueeze(1).unsqueeze(-1)
        g_win = g_win.unsqueeze(1).unsqueeze(-1)

        out_cmp, attn_cmp_list, kv_sp = self.Region_Convergence_Branch(
            q, k, v, h, w, sp_labels
        )
        attn_for_sel = None if self.recompute_attn_cmp else [a.detach() for a in attn_cmp_list]
        out_slc = self.Sparse_Selection_Branch(
            q, attn_for_sel, kv_sp, sp_top_n=sp_top_n
        )
        out_win = self.Local_Refinement_Branch(q, k, v, h, w)

        y = g_cmp * out_cmp + g_slc * out_slc + g_win * out_win
        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class SGF(nn.Module):
    def __init__(self, dim: int, qk_dim: int, mlp_dim: int, heads: int, top_n: int = 5):
        super().__init__()
        self.attn = PreNorm(
            dim,
            SGFAttention(
                dim=dim,
                heads=heads,
                qk_dim=qk_dim,
                top_n=top_n,
                win_ks=(3, 3),
                use_gqa_group_share=False,
                recompute_attn_cmp=False,
                max_sp=4048,
            ),
        )
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.ffn = PreNorm(
            dim,
            ConvFFN(
                in_features=dim,
                hidden_features=mlp_dim,
                out_features=dim,
                kernel_size=5,
                act_layer=nn.GELU,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        sp_labels: torch.Tensor,
        sp_top_n: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        assert sp_labels is not None, "SGF requires sp_labels"
        assert sp_labels.shape[0] == b and sp_labels.shape[-2:] == (h, w), (
            f"sp_labels shape must be [B,H,W] with B={b}, H={h}, W={w}"
        )

        x_seq = rearrange(x, "b c h w -> b (h w) c")

        y_seq = self.attn(x_seq, x_size=(h, w), sp_labels=sp_labels, sp_top_n=sp_top_n)
        y_2d = rearrange(y_seq, "b (h w) c -> b c h w", h=h)
        y_2d = self.conv1x1(y_2d)

        x_seq = x_seq + rearrange(y_2d, "b c h w -> b (h w) c")
        x_seq = x_seq + self.ffn(x_seq, x_size=(h, w))
        x = rearrange(x_seq, "b (h w) c -> b c h w", h=h)
        return x


def pixelshuffle_block(
    in_channels: int,
    out_channels: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    bias: bool = True,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.PixelShuffle(upscale_factor),
    )


@ARCH_REGISTRY.register()
class SGFANet(nn.Module):
    setting = dict(
        dim=48,
        block_num=8,
        qk_dim=36,
        mlp_dim=96,
        heads=3,
        patch_size=[20, 24, 28, 30, 20, 24, 28, 30],
    )

    def __init__(self, in_chans: int = 3, upscale: int = 4, top_n: int = 5):
        super().__init__()

        self.dim = self.setting["dim"]
        self.block_num = self.setting["block_num"]
        self.patch_size = self.setting["patch_size"]
        self.qk_dim = self.setting["qk_dim"]
        self.mlp_dim = self.setting["mlp_dim"]
        self.heads = self.setting["heads"]
        self.upscale = upscale
        self.top_n = int(top_n)

        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for _ in range(self.block_num):
            self.blocks.append(
                nn.ModuleList(
                    [
                        SGF(self.dim, self.qk_dim, self.mlp_dim, self.heads, top_n=self.top_n),
                        HAFA(self.dim, self.qk_dim, self.mlp_dim, self.heads),
                    ]
                )
            )
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        self.recon_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=True)

        if upscale == 1:
            self.upsampler = nn.Conv2d(self.dim, in_chans, 3, 1, 1, bias=True)
        else:
            self.upsampler = pixelshuffle_block(
                in_channels=self.dim,
                out_channels=in_chans,
                upscale_factor=upscale,
                kernel_size=3,
                bias=True,
            )

        self.lrelu = (
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            if upscale != 1 else nn.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _pick_ms_labels(sp_labels_ms: torch.Tensor, scale_id: int) -> torch.Tensor:
        return sp_labels_ms[:, scale_id, :, :]

    @staticmethod
    def _pick_ms_topn(
        sp_top_n_ms: Optional[Union[torch.Tensor, int]],
        scale_id: int,
    ):
        if sp_top_n_ms is None:
            return None
        if isinstance(sp_top_n_ms, int):
            return int(sp_top_n_ms)
        if not torch.is_tensor(sp_top_n_ms):
            return None
        if sp_top_n_ms.numel() == 1:
            return sp_top_n_ms
        if sp_top_n_ms.dim() == 2:
            return sp_top_n_ms[:, scale_id]
        return sp_top_n_ms

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        sp_labels_ms: torch.Tensor,
        sp_top_n_ms: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape

        assert torch.is_tensor(sp_labels_ms) and sp_labels_ms.dim() == 4, (
            f"sp_labels_ms must be [B,S,H,W], got "
            f"{None if sp_labels_ms is None else tuple(sp_labels_ms.shape)}"
        )
        assert sp_labels_ms.shape[0] == b and sp_labels_ms.shape[-2:] == (h, w), (
            f"sp_labels_ms shape must be [B,S,H,W] with B={b}, H={h}, W={w}, "
            f"got {tuple(sp_labels_ms.shape)}"
        )

        num_scales = int(sp_labels_ms.shape[1])
        assert num_scales >= 1, "sp_labels_ms must have S >= 1"

        for i in range(self.block_num):
            residual = x
            sgf_block, hafa_block = self.blocks[i]
            patch_size = self.patch_size[i]

            scale_id = i % num_scales
            sp_labels = self._pick_ms_labels(sp_labels_ms, scale_id)
            sp_top_n = self._pick_ms_topn(sp_top_n_ms, scale_id)

            x = sgf_block(x, sp_labels=sp_labels, sp_top_n=sp_top_n)
            x = hafa_block(x, patch_size)
            x = residual + self.mid_convs[i](x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        sp_labels_ms: torch.Tensor,
        sp_top_n_ms: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        base = x if self.upscale == 1 else F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )

        feat = self.first_conv(x)
        feat = self.forward_features(
            feat,
            sp_labels_ms=sp_labels_ms,
            sp_top_n_ms=sp_top_n_ms,
        ) + feat

        if self.upscale == 1:
            out = self.upsampler(feat) + base
        else:
            feat = self.lrelu(self.recon_conv(feat))
            out = self.upsampler(feat) + base
        return out

    def __repr__(self) -> str:
        num_parameters = sum(p.numel() for p in self.parameters())
        return f"#Params of {self._get_name()}: {num_parameters / 10 ** 3:<.4f} [K]"