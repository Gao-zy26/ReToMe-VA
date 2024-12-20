import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def bipartite_soft_matching_randframe(metric: torch.Tensor, 
                                      F: int, ratio: float, unm_pre: int, generator: torch.Generator,
                                      target_stride: int = 4, align_batch: bool = False,
                                      merge_mode: str = "replace") -> Tuple[Callable, Callable, dict]:
    """
    Partitions the multi-frame tokens into src and dst and merges ratio of src tokens from src to dst.

    Args:
        - metric [B, N, C]: metric to use for similarity.
        - F: frame number.
        - ratio: ratio of src tokens to be removed (by merging).
        - unm_pre: number of src tokens not merged at previous ToMe. Pre-sequence: [unm_pre|F_0|F_1|...]
        - generator: random number generator
        - target_stride: stride of target frame.
        - align_batch: whether to align similarity matching maps of samples in the batch. 
        - merge_mode: how to merge tokens. "mean": tokens -> Mean(src_token, dst_token); "replace": tokens -> dst_token.

    Returns:
        Merge and unmerge operation according to the matching result. Return a dict including other values.
    """
    B, N, _ = metric.shape

    tnum = (N - unm_pre) // F # token num

    if ratio <= 0:
        return do_nothing, do_nothing, {"unm_num": tnum}

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():

        idx_buffer = torch.arange(
            N - unm_pre, device=metric.device, dtype=torch.int64) 

        # Select the random target frame.
        target_stride = min(target_stride, F)
        randf = torch.randint(0, target_stride, torch.Size(
            [1]), generator=generator, device=generator.device)
        dst_select = ((torch.div(idx_buffer, tnum, rounding_mode='floor')) %
                      target_stride == randf).to(torch.bool) 


        a_idx = idx_buffer[None, ~dst_select, None] + unm_pre 
        b_idx = idx_buffer[None, dst_select, None] + unm_pre

        # Add unmerged tokens to dst.
        unm_buffer = torch.arange(unm_pre, device=metric.device, dtype=torch.int64)[
            None, :, None]
        b_idx = torch.cat([b_idx, unm_buffer], dim=1)

        del idx_buffer, unm_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            # Split src, dst tokens
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        # Cosine similarity between src and dst tokens
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], int(a.shape[1] * ratio))


        if align_batch:
            # Find the most similar greedily among all samples.
            scores = torch.cat([*scores], dim=-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None],
                             dim=-2, index=src_idx) % num_dst # Map index to (0, num_dst - 1)
            
            # Use the same matching result for all samples
            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:
            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode=None) -> torch.Tensor:
        # Merge tokens according to matching result.
        src, dst = split(x)
        n, t1, c = src.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c),
                                     src, reduce=mode, include_self=True)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, **kwarg) -> torch.Tensor:
        # Unmerge tokens to original size according to matching result.
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        # Restored src tokens take value from dst tokens
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        # Scatter dst tokens
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        # Scatter unmerged tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        # Scatter src tokens
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=s_idx).expand(-1, -1, c), src=src)

        return out

    # Return number of tokens not merged.
    ret_dict = {"unm_num": unm_idx.shape[1] if unm_idx.shape[1] is not None else 0}
    return merge, unmerge, ret_dict