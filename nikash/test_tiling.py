import sys, pathlib, cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nikash.tiling import adaptive_detect, nms

rng = np.random.default_rng(0)
MIN_NET_PX = 110       # mock detector: only sees objects >=110 px in its network input (trained on big onions)

def make_scene(W=1040, H=780, n=75):
    gt = []
    while len(gt) < n:
        r = rng.uniform(18, 40) if rng.random() < 0.9 else rng.uniform(50, 70)      # mostly small, a few big
        x, y = rng.uniform(r, W - r), rng.uniform(r, H - r)
        if all(np.hypot(x - g[0], y - g[1]) > (r + g[2]) * 0.92 for g in gt):     # allow slight touching
            gt.append((x, y, r))
    img = np.full((H, W, 3), (190, 205, 215), np.uint8)                              # beige floor
    for x, y, r in gt:
        cv2.circle(img, (int(x), int(y)), int(r), (95, 55, 150), -1)                    # red onion
    return img, gt

def mock(gt, net):
    def predict_region(x0, y0, w, h):
        s = net / max(w, h); out = []
        for x, y, r in gt:
            bx1, by1, bx2, by2 = max(x - r, x0), max(y - r, y0), min(x + r, x0 + w), min(y + r, y0 + h)
            if bx2 <= bx1 or by2 <= by1: continue
            vis = (bx2 - bx1) * (by2 - by1) / (4 * r * r)
            if vis < 0.3 or 2 * r * s < MIN_NET_PX: continue                         # partials fire too
            out.append([bx1 - x0, by1 - y0, bx2 - x0, by2 - y0, 0.9 * vis])
        return np.array(out).reshape(-1, 5)
    return predict_region

def match(boxes, gt):
    hits = [0] * len(gt); fp = 0
    for b in boxes:
        best, bi = 0, -1
        for i, (x, y, r) in enumerate(gt):
            g = (x - r, y - r, x + r, y + r)
            ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
            u = (b[2] - b[0]) * (b[3] - b[1]) + 4 * r * r - ix * iy
            if ix * iy / u > best: best, bi = ix * iy / u, i
        if best >= 0.5: hits[bi] += 1
        else: fp += 1
    return sum(h >= 1 for h in hits), sum(max(0, h - 1) for h in hits), fp

fails = 0
for trial in range(5):
    img, gt = make_scene()
    region = mock(gt, 640)
    full1280 = lambda im: mock(gt, 1280)(0, 0, im.shape[1], im.shape[0])
    # tiles: the predictor sees each tile; we need tile offsets -> wrap via closure over tiled_detect's slicing
    import nikash.tiling as T
    captured = {}
    orig = T.tiled_detect
    def predict_tiles(tiles, _gt=gt):
        # recover each tile's offset from its memory position inside img
        res = []
        for t in tiles:
            off = (t.__array_interface__["data"][0] - img.__array_interface__["data"][0])
            y0, x0 = divmod(off // 3, img.shape[1])
            res.append(mock(_gt, 640)(x0, y0, t.shape[1], t.shape[0]))
        return res
    single_found, _, _ = match([tuple(b) for b in full1280(img)], gt)
    boxes, info = adaptive_detect(img, full1280, predict_tiles)
    found, dups, fp = match(boxes, gt)
    ok = found >= 0.95 * len(gt) and dups == 0 and fp == 0
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} trial {trial}: GT {len(gt)} | single-pass @1280 found {single_found} "
          f"| tiled found {found}, duplicates {dups}, false boxes {fp} | {info}")

# nms containment check
b = [(0, 0, 100, 100, .9), (10, 10, 60, 60, .8), (200, 200, 250, 250, .7)]
ok = len(nms(b)) == 2; fails += not ok
print(("PASS" if ok else "FAIL"), "nms suppresses a box contained in a kept box")
print("ALL PASSED" if not fails else f"{fails} FAILED")
