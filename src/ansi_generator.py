# pyright: reportGeneralTypeIssues=false

import torch
import triton
import triton.language as tl
from .config import (
    COL_PREF_VALS,
    CUR_END_VAL,
    ESC_VALS,
    FG_COL_PREF_VALS,
    QUADRANT_GLYPHS,
    RESET_VALS,
    SEP_VAL,
    Config,
)


@triton.jit
def ansi_run_kernel(
    run_r_ptr,
    run_g_ptr,
    run_b_ptr,
    run_lengths_ptr,
    offsets_ptr,
    move_kind_ptr,
    move_v_ptr,
    move_h_ptr,
    needs_color_change_ptr,
    use_rep_run_ptr,
    num_lookup_bytes_ptr,
    num_lookup_lens_ptr,
    r_lookup_bytes_ptr,
    r_lookup_lens_ptr,
    g_lookup_bytes_ptr,
    g_lookup_lens_ptr,
    b_lookup_bytes_ptr,
    b_lookup_lens_ptr,
    out_ptr,
    N,
    LOOKUP_MAX_LEN: tl.constexpr,
    ESC0: tl.constexpr,
    ESC1: tl.constexpr,
    SEP: tl.constexpr,
    CUR_END: tl.constexpr,
    CUR_UP: tl.constexpr,
    CUR_DOWN: tl.constexpr,
    CUR_RIGHT: tl.constexpr,
    CUR_LEFT: tl.constexpr,
    REP_SUFFIX: tl.constexpr,
    COL_PREF0: tl.constexpr,
    COL_PREF1: tl.constexpr,
    COL_PREF2: tl.constexpr,
    COL_PREF3: tl.constexpr,
    COL_PREF4: tl.constexpr,
    COL_PREF5: tl.constexpr,
    COL_PREF6: tl.constexpr,
    SPACE: tl.constexpr,
    MAX_RUN_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    r_val = tl.load(run_r_ptr + pid)
    g_val = tl.load(run_g_ptr + pid)
    b_val = tl.load(run_b_ptr + pid)
    length = tl.load(run_lengths_ptr + pid)
    offset = tl.load(offsets_ptr + pid)
    move_kind = tl.load(move_kind_ptr + pid)
    move_v = tl.load(move_v_ptr + pid)
    move_h = tl.load(move_h_ptr + pid)
    needs_color_change = tl.load(needs_color_change_ptr + pid)
    use_rep_run = tl.load(use_rep_run_ptr + pid)

    ptr = out_ptr + offset

    if move_kind != 0:
        if move_kind == 1:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            row_len = tl.load(num_lookup_lens_ptr + move_v)
            row_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < row_len:
                    byte = tl.load(row_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += row_len

            tl.store(ptr, SEP)
            ptr += 1

            col_len = tl.load(num_lookup_lens_ptr + move_h)
            col_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < col_len:
                    byte = tl.load(col_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += col_len

            tl.store(ptr, CUR_END)
            ptr += 1
        elif move_kind == 2 or move_kind == 3:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            h_len = tl.load(num_lookup_lens_ptr + move_h)
            h_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < h_len:
                    byte = tl.load(h_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += h_len

            if move_kind == 2:
                tl.store(ptr, CUR_RIGHT)
            else:
                tl.store(ptr, CUR_LEFT)
            ptr += 1
        elif move_kind == 4 or move_kind == 5:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            v_len = tl.load(num_lookup_lens_ptr + move_v)
            v_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < v_len:
                    byte = tl.load(v_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += v_len

            if move_kind == 4:
                tl.store(ptr, CUR_DOWN)
            else:
                tl.store(ptr, CUR_UP)
            ptr += 1
        else:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            v_len = tl.load(num_lookup_lens_ptr + move_v)
            v_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < v_len:
                    byte = tl.load(v_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += v_len

            if move_kind == 6 or move_kind == 7:
                tl.store(ptr, CUR_DOWN)
            else:
                tl.store(ptr, CUR_UP)
            ptr += 1

            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            h_len = tl.load(num_lookup_lens_ptr + move_h)
            h_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < h_len:
                    byte = tl.load(h_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += h_len

            if move_kind == 6 or move_kind == 8:
                tl.store(ptr, CUR_RIGHT)
            else:
                tl.store(ptr, CUR_LEFT)
            ptr += 1

    if needs_color_change:
        tl.store(ptr + 0, COL_PREF0)
        tl.store(ptr + 1, COL_PREF1)
        tl.store(ptr + 2, COL_PREF2)
        tl.store(ptr + 3, COL_PREF3)
        tl.store(ptr + 4, COL_PREF4)
        tl.store(ptr + 5, COL_PREF5)
        tl.store(ptr + 6, COL_PREF6)
        ptr += 7

        r_len = tl.load(r_lookup_lens_ptr + r_val)
        r_bytes_base = r_lookup_bytes_ptr + r_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < r_len:
                byte = tl.load(r_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += r_len
        tl.store(ptr, SEP)
        ptr += 1

        g_len = tl.load(g_lookup_lens_ptr + g_val)
        g_bytes_base = g_lookup_bytes_ptr + g_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < g_len:
                byte = tl.load(g_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += g_len
        tl.store(ptr, SEP)
        ptr += 1

        b_len = tl.load(b_lookup_lens_ptr + b_val)
        b_bytes_base = b_lookup_bytes_ptr + b_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < b_len:
                byte = tl.load(b_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += b_len
        tl.store(ptr, 109)
        ptr += 1

    if use_rep_run:
        tl.store(ptr, SPACE)
        ptr += 1
        tl.store(ptr, ESC0)
        ptr += 1
        tl.store(ptr, ESC1)
        ptr += 1

        rep_count = length - 1
        rep_len = tl.load(num_lookup_lens_ptr + rep_count)
        rep_bytes_base = num_lookup_bytes_ptr + rep_count * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < rep_len:
                byte = tl.load(rep_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += rep_len

        tl.store(ptr, REP_SUFFIX)
    else:
        for i in tl.static_range(0, MAX_RUN_LENGTH, 32):
            if i < length:
                off = i + tl.arange(0, 32)
                tl.store(ptr + off, SPACE, mask=off < length)


@triton.jit
def ansi_quadrant_kernel(
    run_fg_r_ptr,
    run_fg_g_ptr,
    run_fg_b_ptr,
    run_bg_r_ptr,
    run_bg_g_ptr,
    run_bg_b_ptr,
    glyph_idx_ptr,
    run_lengths_ptr,
    offsets_ptr,
    move_kind_ptr,
    move_v_ptr,
    move_h_ptr,
    needs_style_change_ptr,
    use_rep_run_ptr,
    num_lookup_bytes_ptr,
    num_lookup_lens_ptr,
    fg_r_lookup_bytes_ptr,
    fg_r_lookup_lens_ptr,
    fg_g_lookup_bytes_ptr,
    fg_g_lookup_lens_ptr,
    fg_b_lookup_bytes_ptr,
    fg_b_lookup_lens_ptr,
    bg_r_lookup_bytes_ptr,
    bg_r_lookup_lens_ptr,
    bg_g_lookup_bytes_ptr,
    bg_g_lookup_lens_ptr,
    bg_b_lookup_bytes_ptr,
    bg_b_lookup_lens_ptr,
    glyph_lookup_bytes_ptr,
    glyph_lookup_lens_ptr,
    out_ptr,
    N,
    LOOKUP_MAX_LEN: tl.constexpr,
    GLYPH_LOOKUP_MAX_LEN: tl.constexpr,
    ESC0: tl.constexpr,
    ESC1: tl.constexpr,
    SEP: tl.constexpr,
    CUR_END: tl.constexpr,
    CUR_UP: tl.constexpr,
    CUR_DOWN: tl.constexpr,
    CUR_RIGHT: tl.constexpr,
    CUR_LEFT: tl.constexpr,
    REP_SUFFIX: tl.constexpr,
    FG_PREF0: tl.constexpr,
    FG_PREF1: tl.constexpr,
    FG_PREF2: tl.constexpr,
    FG_PREF3: tl.constexpr,
    FG_PREF4: tl.constexpr,
    FG_PREF5: tl.constexpr,
    FG_PREF6: tl.constexpr,
    BG_PREF0: tl.constexpr,
    BG_PREF1: tl.constexpr,
    BG_PREF2: tl.constexpr,
    BG_PREF3: tl.constexpr,
    BG_PREF4: tl.constexpr,
    BG_PREF5: tl.constexpr,
    BG_PREF6: tl.constexpr,
    MAX_RUN_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GLYPH_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    fg_r_val = tl.load(run_fg_r_ptr + pid)
    fg_g_val = tl.load(run_fg_g_ptr + pid)
    fg_b_val = tl.load(run_fg_b_ptr + pid)
    bg_r_val = tl.load(run_bg_r_ptr + pid)
    bg_g_val = tl.load(run_bg_g_ptr + pid)
    bg_b_val = tl.load(run_bg_b_ptr + pid)
    glyph_idx = tl.load(glyph_idx_ptr + pid)
    length = tl.load(run_lengths_ptr + pid)
    offset = tl.load(offsets_ptr + pid)
    move_kind = tl.load(move_kind_ptr + pid)
    move_v = tl.load(move_v_ptr + pid)
    move_h = tl.load(move_h_ptr + pid)
    needs_style_change = tl.load(needs_style_change_ptr + pid)
    use_rep_run = tl.load(use_rep_run_ptr + pid)

    ptr = out_ptr + offset

    if move_kind != 0:
        if move_kind == 1:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            row_len = tl.load(num_lookup_lens_ptr + move_v)
            row_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < row_len:
                    byte = tl.load(row_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += row_len

            tl.store(ptr, SEP)
            ptr += 1

            col_len = tl.load(num_lookup_lens_ptr + move_h)
            col_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < col_len:
                    byte = tl.load(col_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += col_len

            tl.store(ptr, CUR_END)
            ptr += 1
        elif move_kind == 2 or move_kind == 3:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            h_len = tl.load(num_lookup_lens_ptr + move_h)
            h_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < h_len:
                    byte = tl.load(h_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += h_len

            if move_kind == 2:
                tl.store(ptr, CUR_RIGHT)
            else:
                tl.store(ptr, CUR_LEFT)
            ptr += 1
        elif move_kind == 4 or move_kind == 5:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            v_len = tl.load(num_lookup_lens_ptr + move_v)
            v_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < v_len:
                    byte = tl.load(v_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += v_len

            if move_kind == 4:
                tl.store(ptr, CUR_DOWN)
            else:
                tl.store(ptr, CUR_UP)
            ptr += 1
        else:
            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            v_len = tl.load(num_lookup_lens_ptr + move_v)
            v_bytes_base = num_lookup_bytes_ptr + move_v * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < v_len:
                    byte = tl.load(v_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += v_len

            if move_kind == 6 or move_kind == 7:
                tl.store(ptr, CUR_DOWN)
            else:
                tl.store(ptr, CUR_UP)
            ptr += 1

            tl.store(ptr, ESC0)
            ptr += 1
            tl.store(ptr, ESC1)
            ptr += 1

            h_len = tl.load(num_lookup_lens_ptr + move_h)
            h_bytes_base = num_lookup_bytes_ptr + move_h * BLOCK_SIZE
            for i in tl.static_range(BLOCK_SIZE):
                if i < h_len:
                    byte = tl.load(h_bytes_base + i)
                    tl.store(ptr + i, byte)
            ptr += h_len

            if move_kind == 6 or move_kind == 8:
                tl.store(ptr, CUR_RIGHT)
            else:
                tl.store(ptr, CUR_LEFT)
            ptr += 1

    if needs_style_change:
        tl.store(ptr + 0, FG_PREF0)
        tl.store(ptr + 1, FG_PREF1)
        tl.store(ptr + 2, FG_PREF2)
        tl.store(ptr + 3, FG_PREF3)
        tl.store(ptr + 4, FG_PREF4)
        tl.store(ptr + 5, FG_PREF5)
        tl.store(ptr + 6, FG_PREF6)
        ptr += 7

        fg_r_len = tl.load(fg_r_lookup_lens_ptr + fg_r_val)
        fg_r_bytes_base = fg_r_lookup_bytes_ptr + fg_r_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < fg_r_len:
                byte = tl.load(fg_r_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += fg_r_len
        tl.store(ptr, SEP)
        ptr += 1

        fg_g_len = tl.load(fg_g_lookup_lens_ptr + fg_g_val)
        fg_g_bytes_base = fg_g_lookup_bytes_ptr + fg_g_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < fg_g_len:
                byte = tl.load(fg_g_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += fg_g_len
        tl.store(ptr, SEP)
        ptr += 1

        fg_b_len = tl.load(fg_b_lookup_lens_ptr + fg_b_val)
        fg_b_bytes_base = fg_b_lookup_bytes_ptr + fg_b_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < fg_b_len:
                byte = tl.load(fg_b_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += fg_b_len
        tl.store(ptr, 109)
        ptr += 1

        tl.store(ptr + 0, BG_PREF0)
        tl.store(ptr + 1, BG_PREF1)
        tl.store(ptr + 2, BG_PREF2)
        tl.store(ptr + 3, BG_PREF3)
        tl.store(ptr + 4, BG_PREF4)
        tl.store(ptr + 5, BG_PREF5)
        tl.store(ptr + 6, BG_PREF6)
        ptr += 7

        bg_r_len = tl.load(bg_r_lookup_lens_ptr + bg_r_val)
        bg_r_bytes_base = bg_r_lookup_bytes_ptr + bg_r_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < bg_r_len:
                byte = tl.load(bg_r_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += bg_r_len
        tl.store(ptr, SEP)
        ptr += 1

        bg_g_len = tl.load(bg_g_lookup_lens_ptr + bg_g_val)
        bg_g_bytes_base = bg_g_lookup_bytes_ptr + bg_g_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < bg_g_len:
                byte = tl.load(bg_g_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += bg_g_len
        tl.store(ptr, SEP)
        ptr += 1

        bg_b_len = tl.load(bg_b_lookup_lens_ptr + bg_b_val)
        bg_b_bytes_base = bg_b_lookup_bytes_ptr + bg_b_val * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < bg_b_len:
                byte = tl.load(bg_b_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += bg_b_len
        tl.store(ptr, 109)
        ptr += 1

    glyph_len = tl.load(glyph_lookup_lens_ptr + glyph_idx)
    glyph_bytes_base = glyph_lookup_bytes_ptr + glyph_idx * GLYPH_BLOCK_SIZE
    if use_rep_run:
        for i in tl.static_range(GLYPH_BLOCK_SIZE):
            if i < glyph_len:
                byte = tl.load(glyph_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += glyph_len

        tl.store(ptr, ESC0)
        ptr += 1
        tl.store(ptr, ESC1)
        ptr += 1

        rep_count = length - 1
        rep_len = tl.load(num_lookup_lens_ptr + rep_count)
        rep_bytes_base = num_lookup_bytes_ptr + rep_count * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < rep_len:
                byte = tl.load(rep_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += rep_len

        tl.store(ptr, REP_SUFFIX)
    else:
        for i in tl.static_range(GLYPH_BLOCK_SIZE):
            if i < glyph_len:
                byte = tl.load(glyph_bytes_base + i)
                for j in tl.static_range(0, MAX_RUN_LENGTH, 32):
                    if j < length:
                        off = j + tl.arange(0, 32)
                        tl.store(ptr + off * glyph_len + i, byte, mask=off < length)


def _max_run_length_for_config(cfg: Config) -> int:
    render_mode = str(getattr(cfg, "render_mode", "pixel")).lower()
    width = max(1, int(cfg.width))

    if render_mode == "quadrant":
        divisor = max(1, int(getattr(cfg, "quadrant_cell_divisor", 2)))
        width = (width + divisor - 1) // divisor

    return ((width + 31) // 32) * 32


def ansi_generate(
    xs: torch.Tensor,
    ys: torch.Tensor,
    colors: torch.Tensor,
    byte_vals: torch.Tensor,
    byte_lens: torch.Tensor,
    cfg: Config,
    run_color_diff_thresh_override: int | None = None,
) -> torch.Tensor:
    mode = str(getattr(cfg, "render_mode", "pixel")).lower()

    if mode == "quadrant":
        return ansi_generate_quadrant(
            xs,
            ys,
            colors,
            byte_vals,
            byte_lens,
            cfg,
            run_color_diff_thresh_override=run_color_diff_thresh_override,
        )
    if mode != "pixel":
        raise ValueError(
            f"Unsupported render mode '{mode}'. Supported modes: pixel, quadrant"
        )

    return ansi_generate_rgb(
        xs,
        ys,
        colors,
        byte_vals,
        byte_lens,
        cfg,
        run_color_diff_thresh_override=run_color_diff_thresh_override,
    )


def ansi_generate_rgb(
    xs: torch.Tensor,
    ys: torch.Tensor,
    colors: torch.Tensor,
    byte_vals: torch.Tensor,
    byte_lens: torch.Tensor,
    cfg: Config,
    run_color_diff_thresh_override: int | None = None,
) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device

    if not hasattr(cfg, "_ansi_out_buffer"):
        cfg._ansi_out_buffer = torch.empty(
            10 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    if not hasattr(cfg, "_ansi_reset_tensor"):
        cfg._ansi_reset_tensor = torch.tensor(
            RESET_VALS, dtype=torch.uint8, device=device
        )

    xs_sorted = xs
    ys_sorted = ys
    colors_sorted = colors
    run_color_diff_thresh = max(
        0,
        int(
            cfg.run_color_diff_thresh
            if run_color_diff_thresh_override is None
            else run_color_diff_thresh_override
        ),
    )

    y_diff = ys_sorted[1:] != ys_sorted[:-1]
    x_diff = xs_sorted[1:] - xs_sorted[:-1]
    if run_color_diff_thresh <= 0:
        color_diff = (colors_sorted[1:] != colors_sorted[:-1]).any(dim=1)
    else:
        color_delta = torch.abs(
            colors_sorted[1:].to(torch.int16) - colors_sorted[:-1].to(torch.int16)
        )
        color_diff = color_delta.amax(dim=1) > run_color_diff_thresh

    is_new_run = torch.zeros(N, dtype=torch.bool, device=device)
    is_new_run[0] = True
    is_new_run[1:] = y_diff | (x_diff != 1) | color_diff
    run_start_indices = torch.where(is_new_run)[0]

    num_runs = run_start_indices.size(0)
    if num_runs == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)

    run_lengths = torch.empty(num_runs, dtype=torch.int64, device=device)
    if num_runs > 1:
        run_lengths[:-1] = run_start_indices[1:] - run_start_indices[:-1]
    run_lengths[-1] = N - run_start_indices[-1]

    run_xs = xs_sorted[run_start_indices]
    run_ys = ys_sorted[run_start_indices]

    needs_move = torch.ones(num_runs, dtype=torch.bool, device=device)
    if num_runs > 1:
        same_row = run_ys[1:] == run_ys[:-1]
        continuous_x = run_xs[1:] == (run_xs[:-1] + run_lengths[:-1])
        needs_move[1:] = ~(same_row & continuous_x)

    run_colors = colors_sorted[run_start_indices]

    needs_color_change = torch.ones(num_runs, dtype=torch.bool, device=device)
    if num_runs > 1:
        if run_color_diff_thresh <= 0:
            color_diff_runs = (run_colors[1:] != run_colors[:-1]).any(dim=1)
        else:
            run_color_delta = torch.abs(
                run_colors[1:].to(torch.int16) - run_colors[:-1].to(torch.int16)
            )
            color_diff_runs = run_color_delta.amax(dim=1) > run_color_diff_thresh
        needs_color_change[1:] = color_diff_runs

    run_r = run_colors[:, 0].to(torch.int64)
    run_g = run_colors[:, 1].to(torch.int64)
    run_b = run_colors[:, 2].to(torch.int64)

    row_idx = run_ys + 1
    col_idx = run_xs + 1
    row_lens = byte_lens[row_idx].to(torch.int64)
    col_lens = byte_lens[col_idx].to(torch.int64)

    r_lens = byte_lens[run_r].to(torch.int64)
    g_lens = byte_lens[run_g].to(torch.int64)
    b_lens = byte_lens[run_b].to(torch.int64)

    relative_cursor_moves = bool(getattr(cfg, "relative_cursor_moves", True))
    if not relative_cursor_moves:
        move_kind = needs_move.to(torch.int8)
        move_v = row_idx
        move_h = col_idx
        move_lens = (2 + row_lens + 1 + col_lens + 1) * needs_move.to(torch.int64)
    else:
        move_kind = torch.zeros(num_runs, dtype=torch.int8, device=device)
        move_v = torch.zeros(num_runs, dtype=torch.int64, device=device)
        move_h = torch.zeros(num_runs, dtype=torch.int64, device=device)

        move_kind[0] = 1
        move_v[0] = row_idx[0]
        move_h[0] = col_idx[0]

        if num_runs > 1:
            move_indices = torch.where(needs_move[1:])[0] + 1
            move_kind[move_indices] = 1
            move_v[move_indices] = row_idx[move_indices]
            move_h[move_indices] = col_idx[move_indices]

            prev_rows = run_ys[move_indices - 1]
            prev_cols_after = run_xs[move_indices - 1] + run_lengths[move_indices - 1]
            dy = run_ys[move_indices] - prev_rows
            dx = run_xs[move_indices] - prev_cols_after

            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            dx_lens = byte_lens[abs_dx.to(torch.long)].to(torch.int64)
            dy_lens = byte_lens[abs_dy.to(torch.long)].to(torch.int64)

            abs_lens = 2 + row_lens[move_indices] + 1 + col_lens[move_indices] + 1
            large = torch.full_like(abs_lens, 1 << 30)
            h_lens = torch.where(dy == 0, 2 + dx_lens + 1, large)
            v_lens = torch.where(dx == 0, 2 + dy_lens + 1, large)
            combo_lens = torch.where(
                (dx != 0) & (dy != 0),
                (2 + dy_lens + 1) + (2 + dx_lens + 1),
                large,
            )

            best = torch.argmin(
                torch.stack((abs_lens, h_lens, v_lens, combo_lens), dim=1),
                dim=1,
            )

            h_choice = best == 1
            h_indices = move_indices[h_choice]
            h_dx = dx[h_choice]
            move_h[h_indices] = torch.abs(h_dx)
            move_kind[h_indices[h_dx > 0]] = 2
            move_kind[h_indices[h_dx < 0]] = 3

            v_choice = best == 2
            v_indices = move_indices[v_choice]
            v_dy = dy[v_choice]
            move_v[v_indices] = torch.abs(v_dy)
            move_kind[v_indices[v_dy > 0]] = 4
            move_kind[v_indices[v_dy < 0]] = 5

            combo_choice = best == 3
            combo_indices = move_indices[combo_choice]
            combo_dx = dx[combo_choice]
            combo_dy = dy[combo_choice]
            move_h[combo_indices] = torch.abs(combo_dx)
            move_v[combo_indices] = torch.abs(combo_dy)

            move_kind[combo_indices[(combo_dy > 0) & (combo_dx > 0)]] = 6
            move_kind[combo_indices[(combo_dy > 0) & (combo_dx < 0)]] = 7
            move_kind[combo_indices[(combo_dy < 0) & (combo_dx > 0)]] = 8
            move_kind[combo_indices[(combo_dy < 0) & (combo_dx < 0)]] = 9

        move_lens = torch.zeros(num_runs, dtype=torch.int64, device=device)

        abs_mask = move_kind == 1
        move_lens[abs_mask] = 2 + row_lens[abs_mask] + 1 + col_lens[abs_mask] + 1

        h_mask = (move_kind == 2) | (move_kind == 3)
        h_lens = byte_lens[move_h[h_mask].to(torch.long)].to(torch.int64)
        move_lens[h_mask] = 2 + h_lens + 1

        v_mask = (move_kind == 4) | (move_kind == 5)
        v_lens = byte_lens[move_v[v_mask].to(torch.long)].to(torch.int64)
        move_lens[v_mask] = 2 + v_lens + 1

        combo_mask = move_kind >= 6
        combo_v_lens = byte_lens[move_v[combo_mask].to(torch.long)].to(torch.int64)
        combo_h_lens = byte_lens[move_h[combo_mask].to(torch.long)].to(torch.int64)
        move_lens[combo_mask] = (2 + combo_v_lens + 1) + (2 + combo_h_lens + 1)

    color_len_full = 7 + r_lens + 1 + g_lens + 1 + b_lens + 1
    color_part = color_len_full * needs_color_change.to(torch.int64)

    use_rep = bool(getattr(cfg, "use_rep", False))
    rep_min_run = max(2, int(getattr(cfg, "rep_min_run", 12)))
    if use_rep:
        use_rep_run = run_lengths >= rep_min_run
        rep_counts = (run_lengths - 1).clamp_min(0).to(torch.long)
        rep_count_lens = byte_lens[rep_counts].to(torch.int64)
        payload_lens = torch.where(use_rep_run, 1 + 2 + rep_count_lens + 1, run_lengths)
    else:
        use_rep_run = torch.zeros(num_runs, dtype=torch.bool, device=device)
        payload_lens = run_lengths

    total_lens = move_lens + color_part + payload_lens

    total_len_all = total_lens.sum(dtype=torch.int64)
    offsets = torch.empty(num_runs, dtype=torch.int64, device=device)
    offsets[0] = 0
    if num_runs > 1:
        offsets[1:] = total_lens.cumsum(dim=0)[:-1]
    reset_len = len(RESET_VALS)

    total_len_all_val = int(total_len_all.item())
    total_size_needed_val = total_len_all_val + reset_len
    if cfg._ansi_out_buffer.size(0) < total_size_needed_val:
        cfg._ansi_out_buffer = torch.empty(
            int(total_size_needed_val * 1.2), dtype=torch.uint8, device=device
        )

    out_buffer = cfg._ansi_out_buffer[:total_size_needed_val]

    LOOKUP_MAX_LEN = byte_vals.size(1)
    max_run_length = _max_run_length_for_config(cfg)
    grid = (num_runs,)
    block_size = min(LOOKUP_MAX_LEN, 64)

    num_warps_raw = min(8, max(1, num_runs // 32))
    num_warps = 1
    if num_warps_raw >= 8:
        num_warps = 8
    elif num_warps_raw >= 4:
        num_warps = 4
    elif num_warps_raw >= 2:
        num_warps = 2

    ansi_run_kernel[grid](
        run_r,
        run_g,
        run_b,
        run_lengths,
        offsets,
        move_kind,
        move_v,
        move_h,
        needs_color_change,
        use_rep_run,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        out_buffer,
        num_runs,
        LOOKUP_MAX_LEN,
        ESC_VALS[0],
        ESC_VALS[1],
        SEP_VAL,
        CUR_END_VAL,
        65,
        66,
        67,
        68,
        98,
        COL_PREF_VALS[0],
        COL_PREF_VALS[1],
        COL_PREF_VALS[2],
        COL_PREF_VALS[3],
        COL_PREF_VALS[4],
        COL_PREF_VALS[5],
        COL_PREF_VALS[6],
        32,
        MAX_RUN_LENGTH=max_run_length,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    out_buffer[total_len_all_val:] = cfg._ansi_reset_tensor

    return out_buffer


def _ensure_common_buffers(cfg: Config, device: torch.device) -> None:
    if not hasattr(cfg, "_ansi_out_buffer"):
        cfg._ansi_out_buffer = torch.empty(
            10 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    if not hasattr(cfg, "_ansi_reset_tensor"):
        cfg._ansi_reset_tensor = torch.tensor(
            RESET_VALS, dtype=torch.uint8, device=device
        )


def _ensure_quadrant_lookup(cfg: Config, device: torch.device) -> None:
    if (
        hasattr(cfg, "_quadrant_lookup_bytes")
        and hasattr(cfg, "_quadrant_lookup_lens")
        and getattr(cfg, "_quadrant_lookup_device", None) == device
    ):
        return

    encoded = [glyph.encode("utf-8") for glyph in QUADRANT_GLYPHS]
    max_len = max(len(bs) for bs in encoded)

    lookup_bytes = torch.zeros(
        (len(encoded), max_len), dtype=torch.uint8, device=device
    )
    lookup_lens = torch.empty(len(encoded), dtype=torch.int64, device=device)

    for idx, bs in enumerate(encoded):
        if bs:
            lookup_bytes[idx, : len(bs)] = torch.tensor(
                list(bs), dtype=torch.uint8, device=device
            )
        lookup_lens[idx] = len(bs)

    cfg._quadrant_lookup_bytes = lookup_bytes
    cfg._quadrant_lookup_lens = lookup_lens
    cfg._quadrant_lookup_device = device


def _build_block_runs(
    xs_sorted: torch.Tensor,
    ys_sorted: torch.Tensor,
    styles_sorted: torch.Tensor,
    run_color_diff_thresh: int,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = xs_sorted.device
    n = xs_sorted.numel()

    fg_all = styles_sorted[:, 0:3]
    bg_all = styles_sorted[:, 3:6]
    glyph_all = styles_sorted[:, 6].to(torch.int64)

    is_new_run = torch.zeros(n, dtype=torch.bool, device=device)
    is_new_run[0] = True

    if n > 1:
        y_diff = ys_sorted[1:] != ys_sorted[:-1]
        x_diff = xs_sorted[1:] - xs_sorted[:-1]
        if run_color_diff_thresh <= 0:
            fg_diff = (fg_all[1:] != fg_all[:-1]).any(dim=1)
            bg_diff = (bg_all[1:] != bg_all[:-1]).any(dim=1)
        else:
            fg_delta = torch.abs(
                fg_all[1:].to(torch.int16) - fg_all[:-1].to(torch.int16)
            )
            bg_delta = torch.abs(
                bg_all[1:].to(torch.int16) - bg_all[:-1].to(torch.int16)
            )
            fg_diff = fg_delta.amax(dim=1) > run_color_diff_thresh
            bg_diff = bg_delta.amax(dim=1) > run_color_diff_thresh
        glyph_diff = glyph_all[1:] != glyph_all[:-1]
        is_new_run[1:] = y_diff | (x_diff != 1) | fg_diff | bg_diff | glyph_diff

    run_start_indices = torch.where(is_new_run)[0]
    num_runs = int(run_start_indices.size(0))

    run_lengths = torch.empty(num_runs, dtype=torch.int64, device=device)
    if num_runs > 1:
        run_lengths[:-1] = run_start_indices[1:] - run_start_indices[:-1]
    run_lengths[-1] = n - run_start_indices[-1]

    run_xs = xs_sorted[run_start_indices]
    run_ys = ys_sorted[run_start_indices]
    run_fg = fg_all[run_start_indices]
    run_bg = bg_all[run_start_indices]
    run_glyph_idx = glyph_all[run_start_indices]

    needs_style_change = torch.ones(num_runs, dtype=torch.bool, device=device)
    if num_runs > 1:
        if run_color_diff_thresh <= 0:
            fg_run_diff = (run_fg[1:] != run_fg[:-1]).any(dim=1)
            bg_run_diff = (run_bg[1:] != run_bg[:-1]).any(dim=1)
        else:
            fg_run_delta = torch.abs(
                run_fg[1:].to(torch.int16) - run_fg[:-1].to(torch.int16)
            )
            bg_run_delta = torch.abs(
                run_bg[1:].to(torch.int16) - run_bg[:-1].to(torch.int16)
            )
            fg_run_diff = fg_run_delta.amax(dim=1) > run_color_diff_thresh
            bg_run_diff = bg_run_delta.amax(dim=1) > run_color_diff_thresh
        needs_style_change[1:] = fg_run_diff | bg_run_diff

    return (
        num_runs,
        run_lengths,
        run_xs,
        run_ys,
        run_fg,
        run_bg,
        run_glyph_idx,
        needs_style_change,
    )


def ansi_generate_quadrant(
    xs: torch.Tensor,
    ys: torch.Tensor,
    styles: torch.Tensor,
    byte_vals: torch.Tensor,
    byte_lens: torch.Tensor,
    cfg: Config,
    run_color_diff_thresh_override: int | None = None,
) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device
    _ensure_common_buffers(cfg, device)
    _ensure_quadrant_lookup(cfg, device)

    xs_sorted = xs
    ys_sorted = ys
    styles_sorted = styles

    run_color_diff_thresh = max(
        0,
        int(
            cfg.run_color_diff_thresh
            if run_color_diff_thresh_override is None
            else run_color_diff_thresh_override
        ),
    )
    (
        num_runs,
        run_lengths,
        run_xs,
        run_ys,
        run_fg,
        run_bg,
        run_glyph_idx,
        needs_style_change,
    ) = _build_block_runs(xs_sorted, ys_sorted, styles_sorted, run_color_diff_thresh)

    needs_move = torch.ones(num_runs, dtype=torch.bool, device=device)
    if num_runs > 1:
        same_row = run_ys[1:] == run_ys[:-1]
        continuous_x = run_xs[1:] == (run_xs[:-1] + run_lengths[:-1])
        needs_move[1:] = ~(same_row & continuous_x)

    run_fg_r = run_fg[:, 0].to(torch.int64)
    run_fg_g = run_fg[:, 1].to(torch.int64)
    run_fg_b = run_fg[:, 2].to(torch.int64)
    run_bg_r = run_bg[:, 0].to(torch.int64)
    run_bg_g = run_bg[:, 1].to(torch.int64)
    run_bg_b = run_bg[:, 2].to(torch.int64)

    row_idx = run_ys + 1
    col_idx = run_xs + 1
    row_lens = byte_lens[row_idx].to(torch.int64)
    col_lens = byte_lens[col_idx].to(torch.int64)

    fg_r_lens = byte_lens[run_fg_r].to(torch.int64)
    fg_g_lens = byte_lens[run_fg_g].to(torch.int64)
    fg_b_lens = byte_lens[run_fg_b].to(torch.int64)
    bg_r_lens = byte_lens[run_bg_r].to(torch.int64)
    bg_g_lens = byte_lens[run_bg_g].to(torch.int64)
    bg_b_lens = byte_lens[run_bg_b].to(torch.int64)

    relative_cursor_moves = bool(getattr(cfg, "relative_cursor_moves", True))
    if not relative_cursor_moves:
        move_kind = needs_move.to(torch.int8)
        move_v = row_idx
        move_h = col_idx
        move_lens = (2 + row_lens + 1 + col_lens + 1) * needs_move.to(torch.int64)
    else:
        move_kind = torch.zeros(num_runs, dtype=torch.int8, device=device)
        move_v = torch.zeros(num_runs, dtype=torch.int64, device=device)
        move_h = torch.zeros(num_runs, dtype=torch.int64, device=device)

        move_kind[0] = 1
        move_v[0] = row_idx[0]
        move_h[0] = col_idx[0]

        if num_runs > 1:
            move_indices = torch.where(needs_move[1:])[0] + 1
            move_kind[move_indices] = 1
            move_v[move_indices] = row_idx[move_indices]
            move_h[move_indices] = col_idx[move_indices]

            prev_rows = run_ys[move_indices - 1]
            prev_cols_after = run_xs[move_indices - 1] + run_lengths[move_indices - 1]
            dy = run_ys[move_indices] - prev_rows
            dx = run_xs[move_indices] - prev_cols_after

            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            dx_lens = byte_lens[abs_dx.to(torch.long)].to(torch.int64)
            dy_lens = byte_lens[abs_dy.to(torch.long)].to(torch.int64)

            abs_lens = 2 + row_lens[move_indices] + 1 + col_lens[move_indices] + 1
            large = torch.full_like(abs_lens, 1 << 30)
            h_lens = torch.where(dy == 0, 2 + dx_lens + 1, large)
            v_lens = torch.where(dx == 0, 2 + dy_lens + 1, large)
            combo_lens = torch.where(
                (dx != 0) & (dy != 0),
                (2 + dy_lens + 1) + (2 + dx_lens + 1),
                large,
            )

            best = torch.argmin(
                torch.stack((abs_lens, h_lens, v_lens, combo_lens), dim=1),
                dim=1,
            )

            h_choice = best == 1
            h_indices = move_indices[h_choice]
            h_dx = dx[h_choice]
            move_h[h_indices] = torch.abs(h_dx)
            move_kind[h_indices[h_dx > 0]] = 2
            move_kind[h_indices[h_dx < 0]] = 3

            v_choice = best == 2
            v_indices = move_indices[v_choice]
            v_dy = dy[v_choice]
            move_v[v_indices] = torch.abs(v_dy)
            move_kind[v_indices[v_dy > 0]] = 4
            move_kind[v_indices[v_dy < 0]] = 5

            combo_choice = best == 3
            combo_indices = move_indices[combo_choice]
            combo_dx = dx[combo_choice]
            combo_dy = dy[combo_choice]
            move_h[combo_indices] = torch.abs(combo_dx)
            move_v[combo_indices] = torch.abs(combo_dy)

            move_kind[combo_indices[(combo_dy > 0) & (combo_dx > 0)]] = 6
            move_kind[combo_indices[(combo_dy > 0) & (combo_dx < 0)]] = 7
            move_kind[combo_indices[(combo_dy < 0) & (combo_dx > 0)]] = 8
            move_kind[combo_indices[(combo_dy < 0) & (combo_dx < 0)]] = 9

        move_lens = torch.zeros(num_runs, dtype=torch.int64, device=device)

        abs_mask = move_kind == 1
        move_lens[abs_mask] = 2 + row_lens[abs_mask] + 1 + col_lens[abs_mask] + 1

        h_mask = (move_kind == 2) | (move_kind == 3)
        h_lens = byte_lens[move_h[h_mask].to(torch.long)].to(torch.int64)
        move_lens[h_mask] = 2 + h_lens + 1

        v_mask = (move_kind == 4) | (move_kind == 5)
        v_lens = byte_lens[move_v[v_mask].to(torch.long)].to(torch.int64)
        move_lens[v_mask] = 2 + v_lens + 1

        combo_mask = move_kind >= 6
        combo_v_lens = byte_lens[move_v[combo_mask].to(torch.long)].to(torch.int64)
        combo_h_lens = byte_lens[move_h[combo_mask].to(torch.long)].to(torch.int64)
        move_lens[combo_mask] = (2 + combo_v_lens + 1) + (2 + combo_h_lens + 1)

    fg_len_full = 7 + fg_r_lens + 1 + fg_g_lens + 1 + fg_b_lens + 1
    bg_len_full = 7 + bg_r_lens + 1 + bg_g_lens + 1 + bg_b_lens + 1
    style_part = (fg_len_full + bg_len_full) * needs_style_change.to(torch.int64)

    glyph_lens = cfg._quadrant_lookup_lens[run_glyph_idx].to(torch.int64)
    use_rep = bool(getattr(cfg, "use_rep", False))
    rep_min_run = max(2, int(getattr(cfg, "rep_min_run", 12)))
    if use_rep:
        use_rep_run = run_lengths >= rep_min_run
        rep_counts = (run_lengths - 1).clamp_min(0).to(torch.long)
        rep_count_lens = byte_lens[rep_counts].to(torch.int64)
        payload_lens = torch.where(
            use_rep_run, glyph_lens + 2 + rep_count_lens + 1, glyph_lens * run_lengths
        )
    else:
        use_rep_run = torch.zeros(num_runs, dtype=torch.bool, device=device)
        payload_lens = glyph_lens * run_lengths

    total_lens = move_lens + style_part + payload_lens

    total_len_all = total_lens.sum(dtype=torch.int64)
    offsets = torch.empty(num_runs, dtype=torch.int64, device=device)
    offsets[0] = 0
    if num_runs > 1:
        offsets[1:] = total_lens.cumsum(dim=0)[:-1]

    reset_len = len(RESET_VALS)
    total_len_all_val = int(total_len_all.item())
    total_size_needed_val = total_len_all_val + reset_len
    if cfg._ansi_out_buffer.size(0) < total_size_needed_val:
        cfg._ansi_out_buffer = torch.empty(
            int(total_size_needed_val * 1.2), dtype=torch.uint8, device=device
        )

    out_buffer = cfg._ansi_out_buffer[:total_size_needed_val]

    lookup_max_len = byte_vals.size(1)
    glyph_lookup_max_len = cfg._quadrant_lookup_bytes.size(1)
    max_run_length = _max_run_length_for_config(cfg)
    grid = (num_runs,)
    block_size = min(lookup_max_len, 64)

    num_warps_raw = min(8, max(1, num_runs // 32))
    num_warps = 1
    if num_warps_raw >= 8:
        num_warps = 8
    elif num_warps_raw >= 4:
        num_warps = 4
    elif num_warps_raw >= 2:
        num_warps = 2

    ansi_quadrant_kernel[grid](
        run_fg_r,
        run_fg_g,
        run_fg_b,
        run_bg_r,
        run_bg_g,
        run_bg_b,
        run_glyph_idx,
        run_lengths,
        offsets,
        move_kind,
        move_v,
        move_h,
        needs_style_change,
        use_rep_run,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        byte_vals,
        byte_lens,
        cfg._quadrant_lookup_bytes,
        cfg._quadrant_lookup_lens,
        out_buffer,
        num_runs,
        lookup_max_len,
        glyph_lookup_max_len,
        ESC_VALS[0],
        ESC_VALS[1],
        SEP_VAL,
        CUR_END_VAL,
        65,
        66,
        67,
        68,
        98,
        FG_COL_PREF_VALS[0],
        FG_COL_PREF_VALS[1],
        FG_COL_PREF_VALS[2],
        FG_COL_PREF_VALS[3],
        FG_COL_PREF_VALS[4],
        FG_COL_PREF_VALS[5],
        FG_COL_PREF_VALS[6],
        COL_PREF_VALS[0],
        COL_PREF_VALS[1],
        COL_PREF_VALS[2],
        COL_PREF_VALS[3],
        COL_PREF_VALS[4],
        COL_PREF_VALS[5],
        COL_PREF_VALS[6],
        MAX_RUN_LENGTH=max_run_length,
        BLOCK_SIZE=block_size,
        GLYPH_BLOCK_SIZE=glyph_lookup_max_len,
        num_warps=num_warps,
    )

    out_buffer[total_len_all_val:] = cfg._ansi_reset_tensor

    return out_buffer
