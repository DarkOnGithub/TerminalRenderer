from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .utils import resize_frame_keep_aspect

_QUADRANT_MASKS = (
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 0),
    (0, 0, 1, 0),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (1, 1, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (1, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
)

_quadrant_mask_cache: dict[
    torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_quadrant_masks(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = _quadrant_mask_cache.get(device)
    if cached is None:
        masks = torch.tensor(_QUADRANT_MASKS, dtype=torch.float32, device=device)
        fg_counts = masks.sum(dim=1).view(1, len(_QUADRANT_MASKS), 1).clamp_min(1.0)
        bg_counts = (1.0 - masks).sum(dim=1).view(1, len(_QUADRANT_MASKS), 1)
        cached = (masks, fg_counts, bg_counts.clamp_min(1.0))
        _quadrant_mask_cache[device] = cached
    return cached


def _encode_quadrant_cells(frame: torch.Tensor) -> torch.Tensor:
    if frame.shape[0] % 2 == 1:
        frame = torch.cat([frame, frame[-1:, :, :]], dim=0)
    if frame.shape[1] % 2 == 1:
        frame = torch.cat([frame, frame[:, -1:, :]], dim=1)

    tl = frame[0::2, 0::2]
    tr = frame[0::2, 1::2]
    bl = frame[1::2, 0::2]
    br = frame[1::2, 1::2]

    cells = torch.stack((tl, tr, bl, br), dim=2).to(torch.float32)
    height, width = cells.shape[:2]
    flat_cells = cells.reshape(-1, 4, 3)
    masks, fg_counts, bg_counts = _get_quadrant_masks(frame.device)

    fg_sums = torch.einsum("nqc,mq->nmc", flat_cells, masks)
    total_sums = flat_cells.sum(dim=1, keepdim=True)
    bg_sums = total_sums - fg_sums

    cell_sq_sums = (flat_cells * flat_cells).sum(dim=2)
    fg_sq_sums = torch.einsum("nq,mq->nm", cell_sq_sums, masks)
    total_sq_sums = cell_sq_sums.sum(dim=1, keepdim=True)
    bg_sq_sums = total_sq_sums - fg_sq_sums

    fg_means = fg_sums / fg_counts
    bg_means = bg_sums / bg_counts
    fg_errors = fg_sq_sums - (fg_sums * fg_sums).sum(dim=2) / fg_counts.squeeze(-1)
    bg_errors = bg_sq_sums - (bg_sums * bg_sums).sum(dim=2) / bg_counts.squeeze(-1)

    flat_glyph_idx = (fg_errors + bg_errors).argmin(dim=1)
    cell_indices = torch.arange(flat_cells.size(0), device=frame.device)
    fg_best = fg_means[cell_indices, flat_glyph_idx]
    bg_best = bg_means[cell_indices, flat_glyph_idx]

    styles = torch.empty(
        (flat_cells.size(0), 7), dtype=torch.uint8, device=frame.device
    )
    styles[:, 0:3] = fg_best.round().clamp(0, 255).to(torch.uint8)
    styles[:, 3:6] = bg_best.round().clamp(0, 255).to(torch.uint8)
    styles[:, 6] = flat_glyph_idx.to(torch.uint8)
    return styles.view(height, width, 7)


def _set_block_source_cache(
    config: Config, render_mode: str, frame: torch.Tensor
) -> None:
    cached = getattr(config, "_block_source_cache_frame", None)
    if cached is None or cached.shape != frame.shape or cached.device != frame.device:
        setattr(config, "_block_source_cache_frame", frame.clone())
    else:
        cached.copy_(frame)
    setattr(config, "_block_source_cache_mode", render_mode)


def _empty_block_update(
    previous_frame: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty((0, 7), device=device, dtype=torch.uint8),
        previous_frame,
    )


def pre_process_frame(
    previous_frame: Optional[torch.Tensor],
    frame: torch.Tensor,
    config: Config,
    quant_mask: Optional[int] = None,
    diff_thresh_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    render_mode = str(getattr(config, "render_mode", "pixel")).lower()

    if render_mode not in ("pixel", "quadrant"):
        raise ValueError(
            f"Unsupported render mode '{render_mode}'. Supported modes: pixel, quadrant"
        )

    if render_mode == "quadrant":
        quadrant_cell_divisor = max(1, int(getattr(config, "quadrant_cell_divisor", 2)))
        cell_height = max(1, int(config.height) // quadrant_cell_divisor)
        cell_width = max(1, int(config.width) // quadrant_cell_divisor)
        target_height = max(1, cell_height * 2)
        target_width = max(1, cell_width * 2)
    else:
        target_height = int(config.height)
        target_width = int(config.width)

    resized_frame = resize_frame_keep_aspect(frame, target_height, target_width)

    quant_mask_value = config.quant_mask if quant_mask is None else int(quant_mask)
    quant_mask_value = max(0, min(quant_mask_value, 0xFF))
    if quant_mask_value != 0xFF:
        resized_frame = resized_frame & quant_mask_value

    if render_mode == "quadrant":
        source_cell_height = 2
        source_cell_width = 2
        device = resized_frame.device
        cell_shape = (
            (resized_frame.shape[0] + source_cell_height - 1) // source_cell_height,
            (resized_frame.shape[1] + source_cell_width - 1) // source_cell_width,
            7,
        )
        cached_source = getattr(config, "_block_source_cache_frame", None)
        cache_matches = (
            cached_source is not None
            and getattr(config, "_block_source_cache_mode", None) == render_mode
            and cached_source.shape == resized_frame.shape
            and cached_source.device == resized_frame.device
        )

        if (
            previous_frame is None
            or previous_frame.shape != cell_shape
            or not cache_matches
        ):
            cell_styles = _encode_quadrant_cells(resized_frame)
            _set_block_source_cache(config, render_mode, resized_frame)
            height, width = cell_styles.shape[:2]
            ys = torch.arange(height, device=device).repeat_interleave(width)
            xs = torch.arange(width, device=device).repeat(height)
            styles = cell_styles[ys, xs]
            return xs, ys, styles, cell_styles

        source_diff = (resized_frame != cached_source).any(dim=-1)
        if not source_diff.any():
            return _empty_block_update(previous_frame, device)

        dirty_cell_mask = (
            F.max_pool2d(
                source_diff.to(torch.float32).unsqueeze(0).unsqueeze(0),
                kernel_size=(source_cell_height, source_cell_width),
                stride=(source_cell_height, source_cell_width),
                ceil_mode=True,
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.bool)
        )
        dirty_cell_ys, dirty_cell_xs = dirty_cell_mask.nonzero(as_tuple=True)
        cell_y0 = int(dirty_cell_ys.min().item())
        cell_y1 = int(dirty_cell_ys.max().item()) + 1
        cell_x0 = int(dirty_cell_xs.min().item())
        cell_x1 = int(dirty_cell_xs.max().item()) + 1

        src_y0 = cell_y0 * source_cell_height
        src_y1 = min(cell_y1 * source_cell_height, resized_frame.shape[0])
        src_x0 = cell_x0 * source_cell_width
        src_x1 = min(cell_x1 * source_cell_width, resized_frame.shape[1])

        cell_styles = _encode_quadrant_cells(resized_frame[src_y0:src_y1, src_x0:src_x1])
        previous_slice = previous_frame[cell_y0:cell_y1, cell_x0:cell_x1]

        diff_thresh = (
            config.diff_thresh
            if diff_thresh_override is None
            else int(diff_thresh_override)
        )

        glyph_diff = cell_styles[..., 6] != previous_slice[..., 6]
        if diff_thresh <= 0:
            color_diff = (cell_styles[..., :6] != previous_slice[..., :6]).any(dim=-1)
        else:
            color_diff = torch.abs(
                cell_styles[..., :6].to(torch.int16)
                - previous_slice[..., :6].to(torch.int16)
            ).amax(dim=-1) > int(diff_thresh)
        diff_mask = glyph_diff | color_diff

        _set_block_source_cache(config, render_mode, resized_frame)

        if not diff_mask.any():
            return _empty_block_update(previous_frame, device)

        ys, xs = diff_mask.nonzero(as_tuple=True)
        ys = ys + cell_y0
        xs = xs + cell_x0
        styles = cell_styles[ys - cell_y0, xs - cell_x0]
        previous_frame[ys, xs] = styles
        return xs, ys, styles, previous_frame

    device = resized_frame.device

    if previous_frame is None or previous_frame.shape != resized_frame.shape:
        height, width = resized_frame.shape[:2]
        ys = torch.arange(height, device=device).repeat_interleave(width)
        xs = torch.arange(width, device=device).repeat(height)
        colors_rgb = resized_frame[ys, xs]
        return xs, ys, colors_rgb, resized_frame

    diff_thresh = (
        config.diff_thresh
        if diff_thresh_override is None
        else int(diff_thresh_override)
    )
    if diff_thresh <= 0:
        diff_mask = (resized_frame != previous_frame).any(dim=-1)
    else:
        thresh = int(diff_thresh)
        diff_mask = torch.any(
            torch.abs(resized_frame.to(torch.int16) - previous_frame.to(torch.int16))
            > thresh,
            dim=-1,
        )

    if not diff_mask.any():
        return (
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.uint8),
            previous_frame,
        )

    ys, xs = diff_mask.nonzero(as_tuple=True)
    colors_rgb = resized_frame[ys, xs]
    previous_frame[ys, xs] = colors_rgb
    return xs, ys, colors_rgb, previous_frame
