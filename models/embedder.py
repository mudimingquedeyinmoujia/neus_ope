import torch
import torch.nn as nn
import numpy as np
import math


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


def learnableEmbed_fn(coords, q_freq, q_coef, q_phase):
    """
    q_freq: 3*feats
    q_coef: 2*feats
    q_phase: feats
    coords: n,3
    return: n, 2*feats
    """
    q_freq = q_freq.unsqueeze(0)
    q_coef = q_coef.unsqueeze(0)
    q_phase = q_phase.unsqueeze(0)
    q_freq = torch.stack(torch.split(q_freq, coords.shape[-1], dim=-1), dim=-1)  # n,3,feats
    q_freq = torch.mul(q_freq, coords.unsqueeze(-1))  # n,3,feats
    q_freq = torch.sum(q_freq, dim=-2)  # n,feats
    q_freq += q_phase  # n,feats
    q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)  # n,2*feats
    ret = torch.mul(q_coef, q_freq)  # n,2*feats
    return ret


def learnableEmbed_fn_v1(coords, q_freq, q_phase):
    """
    q_freq: 3*feats
    q_phase: feats
    coords: n,3
    return: n, 2*feats
    version: 1: no coef / 2: no coef, freq to 2^w
    """
    q_freq = q_freq.unsqueeze(0)
    q_phase = q_phase.unsqueeze(0)
    q_freq = torch.stack(torch.split(q_freq, coords.shape[-1], dim=-1), dim=-1)  # n,3,feats
    q_freq = torch.mul(q_freq, coords.unsqueeze(-1))  # n,3,feats
    q_freq = torch.sum(q_freq, dim=-2)  # n,feats
    q_freq += q_phase  # n,feats
    q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)  # n,2*feats
    return q_freq


def learnableEmbed_fn_v2(coords, q_freq, q_phase):
    """
    q_freq: 3*feats
    q_phase: feats
    coords: n,3
    return: n, 2*feats
    version: 1: no coef / 2: no coef, freq to 2^w
    """
    q_freq = q_freq.unsqueeze(0)
    q_phase = q_phase.unsqueeze(0)
    q_freq = torch.stack(torch.split(q_freq, coords.shape[-1], dim=-1), dim=-1)  # n,3,feats
    q_freq = torch.mul(2 ** q_freq, coords.unsqueeze(-1))  # n,3,feats
    q_freq = torch.sum(q_freq, dim=-2)  # n,feats
    q_freq += q_phase  # n,feats
    q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)  # n,2*feats

    return q_freq


def learnableEmbed_fn_v3(coords, q_freq, q_phase):
    """
    q_freq: 3*feats
    q_phase: feats
    coords: n,3
    return: n, 2*feats
    version: 1: no coef / 2: no coef, freq to 2^w
    """
    q_freq = q_freq.unsqueeze(0)
    q_phase = q_phase.unsqueeze(0)
    q_freq = torch.stack(torch.split(q_freq, coords.shape[-1], dim=-1), dim=-1)  # n,3,feats
    q_freq = torch.mul(2 ** q_freq, coords.unsqueeze(-1))  # n,3,feats
    q_freq = torch.sum(q_freq, dim=-2)  # n,feats
    q_freq += q_phase  # n,feats
    q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)  # n,2*feats

    return q_freq


def get_learnableEmbed_fn(feats):
    return learnableEmbed_fn, feats * 2


def get_learnableEmbed_fn_modify(feats, version):
    if version == 1:
        return learnableEmbed_fn_v1, feats * 2
    elif version == 2:
        return learnableEmbed_fn_v2, feats * 2
    elif version == 3:
        return learnableEmbed_fn_v3, feats * 2
    else:
        return None


def get_opeSins_fns(max_freq, freq_up):
    """
    N,bsize,1 ---> N,bsize,2n+1
    """
    embed_fns = []
    embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))  # x: N,bsize,1
    if freq_up:
        for i in range(1, max_freq + 1):
            embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x[:, :, 0] * 2**freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x[:, :, 0] * 2**freq).unsqueeze(-1))
    else:
        for i in range(1, max_freq + 1):
            embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
    return embed_fns


class OPE3D(nn.Module):
    def __init__(self, max_freq, omega=math.pi, freq_up=False):
        super(OPE3D, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.embed_fns = get_opeSins_fns(self.max_freq, freq_up)
        self.out_dim = (2 * max_freq + 1) ** 3

    def embed(self, inputs):
        """
        N,bsize,1 ---> N,bsize,1,2n+1
        """
        res = torch.cat([fn(inputs * self.omega).to(inputs.device) for fn in self.embed_fns], -1)
        return res.unsqueeze(-2)

    def forward(self, coords):
        """
        coords: N, 3
        N,bsize,2 ---> N,bsize,(2n+1)^2
        return: N, (2*n+1)^3
        """
        coords = coords.unsqueeze(0)  # 1,N,3
        x_coord = coords[:, :, 0].unsqueeze(-1)  # 1,N,1
        y_coord = coords[:, :, 1].unsqueeze(-1)  # 1,N,1
        z_coord = coords[:, :, 2].unsqueeze(-1)  # 1,N,1
        X = self.embed(x_coord)
        Y = self.embed(y_coord)
        Z = self.embed(z_coord)
        xy_mat = torch.matmul(X.transpose(2, 3), Y)  # 1,N,2n+1,2n+1
        xy_flat = xy_mat.view(xy_mat.shape[0], xy_mat.shape[1], -1, 1)  # 1,N,(2n+1)^2,1
        ope_mat = torch.matmul(xy_flat, Z)  # 1,N,(2n+1)^2, 2n+1
        ope_flat = ope_mat.view(ope_mat.shape[0], ope_mat.shape[1], -1)  # 1,N,(2n+1)^3
        return ope_flat.squeeze(0)  # N,(2n+1)^3
