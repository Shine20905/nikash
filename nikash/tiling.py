"""
Scale-adaptive tiled detection.

Why: the detector was trained on stock photos where an onion fills ~200 px of its 640 input.
In a real spread photographed from standing height, onions are ~60 px even at imgsz 1280,
so a third of them are missed. Tiling cuts the image into overlapping tiles sized so each
onion fills a similar share of a tile as in training.

How:
  size    : typical onion size from the COLOUR MASK (distance-transform peaks = onion radii).
            Not from pass-1 detections: pass 1 only sees the big onions, so its size estimate is
            biased large (survivorship) and would produce tiles too big for the small ones.
  pass 1  : whole image at a moderate imgsz - kept only for onions too big for a tile
  tile    : side = TILE_RATIO x typical onion size, overlap >= the largest onion seen
  pass 2  : every tile resized to the detector's training size
  merge   : drop boxes touching an interior tile edge (they are cut; the uncut copy exists
            in a neighbour because overlap >= largest onion), then keep each box only in
            the tile whose centre is nearest its centre, then union with large pass-1
            boxes and NMS.
Pure numpy + cv2: the predictor is injected, so this is testable without the model.
"""
import cv2, numpy as np


def _grid(length, tile, step):
    if length <= tile: return [0]
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] + tile < length: xs.append(length - tile)
    return xs


def _iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9), inter / (np.minimum(aa[:, None], bb[None, :]) + 1e-9)


def nms(boxes, iou_thr=0.5, contain_thr=0.8):
    """Greedy NMS by confidence; also suppresses a box mostly contained in a kept one."""
    if not boxes: return []
    B = np.array([b[:4] for b in boxes], np.float64); S = np.array([b[4] for b in boxes])
    order = np.argsort(-S); keep = []
    for i in order:
        if keep:
            iou, cont = _iou_matrix(B[i:i + 1], B[keep])
            if (iou > iou_thr).any() or (cont > contain_thr).any(): continue
        keep.append(i)
    return [boxes[i] for i in keep]


def tiled_detect(img, predict, typical_px, max_px, tile_ratio=3.3, min_tile=160, edge_px=2, max_tiles=48):
    """predict(list_of_bgr_tiles) -> list of (N_i, 5) arrays [x1,y1,x2,y2,conf] in TILE coords.
    Overlap covers onions up to max_px; bigger ones are left to pass 1. If the grid would exceed
    max_tiles, tiles grow (less upscaling, some recall traded for speed)."""
    H, W = img.shape[:2]
    tile = int(max(min_tile, tile_ratio * typical_px))
    while True:
        tile = min(tile, max(H, W))
        overlap = int(np.clip(1.15 * max_px, 0.25 * tile, 0.5 * tile))
        step = max(16, tile - overlap)
        xs, ys = _grid(W, tile, step), _grid(H, tile, step)
        if len(xs) * len(ys) <= max_tiles or tile >= max(H, W): break
        tile = int(tile * 1.1)
    tiles, offs = [], []
    for y0 in ys:
        for x0 in xs:
            tiles.append(img[y0:y0 + tile, x0:x0 + tile]); offs.append((x0, y0))
    preds = predict(tiles)
    centres = np.array([(x0 + min(tile, W - x0) / 2, y0 + min(tile, H - y0) / 2) for x0, y0 in offs])
    out = []
    for t, ((x0, y0), P) in enumerate(zip(offs, preds)):
        tw, th = min(tile, W - x0), min(tile, H - y0)
        for x1, y1, x2, y2, c in np.asarray(P).reshape(-1, 5):
            # cut by an interior tile edge?  (edges that are image borders don't count)
            if (x1 <= edge_px and x0 > 0) or (y1 <= edge_px and y0 > 0) or \
               (x2 >= tw - edge_px and x0 + tw < W) or (y2 >= th - edge_px and y0 + th < H):
                continue
            gx1, gy1, gx2, gy2 = x1 + x0, y1 + y0, x2 + x0, y2 + y0
            cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            if int(np.argmin(np.hypot(centres[:, 0] - cx, centres[:, 1] - cy))) != t:
                continue                                  # another tile owns this onion
            out.append((float(gx1), float(gy1), float(gx2), float(gy2), float(c)))
    return nms(out), {"tile_px": tile, "overlap_px": overlap, "n_tiles": len(tiles)}


def estimate_onion_sizes(img):
    """Diameters (px) of every blob in the colour mask via distance-transform peaks."""
    from .pipeline import foreground_mask
    m, _ = foreground_mask(img)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peaks = ((dt == cv2.dilate(dt, np.ones((9, 9), np.uint8))) & (dt > 4)).astype(np.uint8)
    n, lab = cv2.connectedComponents(peaks)
    radii = [float(dt[lab == i].max()) for i in range(1, n)]
    return 2.0 * np.array(radii)


def adaptive_detect(img, predict_full, predict_tiles, tile_ratio=3.3, max_tiles=48):
    """Tiles sized from the colour mask; pass 1 contributes onions too big for the tile overlap."""
    p1 = [tuple(map(float, b)) for b in np.asarray(predict_full(img)).reshape(-1, 5)]
    sizes = estimate_onion_sizes(img)
    p1_sides = np.array([np.sqrt((b[2] - b[0]) * (b[3] - b[1])) for b in p1]) if p1 else np.zeros(0)
    if len(sizes) >= 5:
        typical, biggest, source = float(np.median(sizes)), float(np.percentile(sizes, 90)), "colour-mask"
    elif len(p1_sides):
        typical, biggest, source = float(np.median(p1_sides)), float(np.percentile(p1_sides, 90)), "pass1"
    else:
        typical = max(img.shape[:2]) / 12.0; biggest = typical * 2.0; source = "default"
    tiled, info = tiled_detect(img, predict_tiles, typical, biggest, tile_ratio, max_tiles=max_tiles)
    if len(tiled) >= 5:                        # tiles revealed smaller onions than estimated -> shrink once
        t_med = float(np.median([np.sqrt((b[2] - b[0]) * (b[3] - b[1])) for b in tiled]))
        if t_med < 0.8 * typical:
            typical = t_med
            tiled, info = tiled_detect(img, predict_tiles, typical, biggest, tile_ratio, max_tiles=max_tiles)
            source += "+refined"
    big = [b for b in p1 if np.sqrt((b[2] - b[0]) * (b[3] - b[1])) > info["overlap_px"] / 1.15]
    merged = nms(tiled + big)
    info.update(pass1=len(p1), tiled=len(tiled), big_from_pass1=len(big), final=len(merged),
                typical_px=round(typical, 1), size_source=source)
    return merged, info
