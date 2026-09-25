"""
Nikash grading pipeline - Stages 2-7 (SIH 2026, PS 26031).

  Stage 2  capture gate      : ArUco -> homography -> rectified sheet; blur/exposure; circle self-check
  Stage 3  detection         : injected detector (YOLO11n) on the capture area
  Stage 4a geometry + mass   : ellipse fit inside each box -> mm -> mass (variety calibration)
  Stage 4b defects           : injected classifier (MobileNetV3-Small, 3 heads)
  Stage 5  confidence router : uncertainty measured around each head's own threshold
  Stage 6  rules engine      : config-driven, mass-weighted % Grade A / % URS
  Stage 7  report            : annotated image + signed JSON + PDF

Detector and classifier are injected callables, so geometry/grading/report are testable
without the models. Sheet geometry below MUST match make_sheet.py (A3 portrait, v2).
"""
import cv2, numpy as np, math, json, hashlib, hmac, datetime, io

# --------------------------------------------------------------------------------------
# Sheet geometry, top-left origin, millimetres. Derived from make_sheet.py v2.
# --------------------------------------------------------------------------------------
SHEET_W, SHEET_H = 297.0, 420.0
MARGIN, MARKER = 15.0, 40.0
MARKERS = {0: (MARGIN, MARGIN),                                   # TL
           1: (SHEET_W - MARGIN - MARKER, MARGIN),                # TR
           2: (MARGIN, SHEET_H - MARGIN - MARKER),                # BL
           3: (SHEET_W - MARGIN - MARKER, SHEET_H - MARGIN - MARKER)}  # BR
CAPTURE = (MARGIN, SHEET_H - 335.0, SHEET_W - MARGIN, SHEET_H - 150.0)  # x0,y0,x1,y1 = 15,85,282,270

def _circle_layout():
    dias = [35.0, 45.0, 55.0, 70.0]
    gap = (SHEET_W - 2 * MARGIN - sum(dias)) / (len(dias) + 1)
    x, out = MARGIN + gap, []
    for d in dias:
        out.append((x + d / 2, SHEET_H - 105.0, d))            # cx, cy, nominal diameter
        x += d + gap
    return out
CIRCLES = _circle_layout()

def marker_corners_mm(mid):
    x, y = MARKERS[mid]
    return np.float32([[x, y], [x + MARKER, y], [x + MARKER, y + MARKER], [x, y + MARKER]])

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
_DET = cv2.aruco.ArucoDetector(_DICT, cv2.aruco.DetectorParameters())

DEFAULT_CONFIG = {
    "px_per_mm": 5.0,
    "min_markers": 2,                 # full precision needs 4; 2-3 is accepted and flagged
    "gate": {"min_sharpness": 60.0,            # PROVISIONAL - marker Laplacian var at 5 px/mm; calibrate on real photos
             "min_paper_brightness": 90, "max_saturated_frac": 0.35,
             "max_tilt_deg": 20.0,
             "max_hidden_outline_frac": 0.40,        # PROVISIONAL - share of onions whose outline is mostly hidden
             "min_onions_for_crowd_check": 4,        # by neighbours -> heap, not a single layer -> retake
             "max_touching_frac": 0.70},             # PROVISIONAL - real: spreads 0.2-0.6, packed photo 0.8 (1 photo)                  # PROVISIONAL - oblique views stretch 3-D onions (parallax)
    "camera": {"focal_px": None,                     # None -> 0.6 x image diagonal (26 mm-equiv phone main camera);
               "parallax_correction": True},         # checked vs self-calibration on 2 tilted real photos: within ~5-10%
    "white_balance": True,
    "variety": "N-53",
    "varieties": {"N-53": {"density_g_cm3": 0.8997, "axis_ratio_k": 1.0766}},
    "grading": {
        "size_min_mm": 45.0, "size_max_mm": 70.0,
        "diameter_definition": "max_equatorial",   # PROVISIONAL - confirm DoCA/AGMARK
        "urs_defects": ["sprouted"],               # other_defect = inspector review only (v0.7): the catch-all head
                                                  # false-alarms on sharp real photos; it must not auto-downgrade
        "unfit_defects": ["rotten"],
        "urs_reporting_enabled": True,
    },
    "router": {"logit_band": 1.0,
               "reject_coverage": 0.12,     # box with <12% onion colour ...
               "reject_conf": 0.60,         # ... and detector conf below this -> not an onion (shadow/floor mark)
               "uncertain_heads": ["rotten", "sprouted"],
               "foreign_green_frac": 0.03},  # v0.7.2: >=3% leaf-green pixels in the box and NOT a sprout -> not an onion?
                                             # real sheet: brinjal 5-7%, onions 0-0.8% (5 photos). Needs a not-onion class later.   # band only where the GRADE can flip (v0.7.1)
    "threshold_overrides": {"binary_bad": 0.95},  # v0.7.1: real-sheet sweep - public bad caught 86%, healthy sheet
                                                  # flagged 55% -> 18% (chosen on the same 38 crops: optimistic)
    "thresholds": {"rotten": 0.11, "sprouted": 0.93, "binary_bad": 0.68},
    "merge_tol_frac": 0.45,
    "signing_key": "NIKASH-DEMO-KEY",
}

# --------------------------------------------------------------------------------------
# Stage 2: capture gate + rectification
# --------------------------------------------------------------------------------------
def _camera_pose(src_px, dst_mm, img_shape, focal_px=None):
    """Camera tilt (deg from straight-down) and height above the sheet (mm) from the marker
    homography. Pinhole model, principal point at image centre, focal from config or
    0.6 x diagonal. Height drives the parallax correction: an onion's outline sits ~D/2 above
    the paper, so it projects onto the sheet plane magnified by Z / (Z - D/2)."""
    G, _ = cv2.findHomography(dst_mm.astype(np.float64), src_px.astype(np.float64))
    if G is None: return None
    h, w = img_shape[:2]
    f = float(focal_px or 0.6 * math.hypot(w, h))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    M = np.linalg.inv(K) @ G
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if t[2] < 0: r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2); r3 /= np.linalg.norm(r3)
    R = np.stack([r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2), r3], 1)
    C = -R.T @ t
    return {"tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(r3[2]))))),
            "camera_height_mm": float(abs(C[2])), "focal_px": f}

def rectify(img_bgr, cfg):
    ppm = cfg["px_per_mm"]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DET.detectMarkers(gray)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in MARKERS: found[int(i)] = c[0]
    gate = {"markers_found": sorted(found), "ok": False, "warnings": []}
    if len(found) < cfg["min_markers"]:
        gate["reason"] = f"only {len(found)} reference marker(s) visible - reframe so the sheet corners show"
        return None, gate
    src = np.vstack([found[i] for i in sorted(found)]).astype(np.float32)
    dst = np.vstack([marker_corners_mm(i) * ppm for i in sorted(found)]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        gate["reason"] = "homography failed"; return None, gate
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    gate["reprojection_mm"] = float(np.mean(np.linalg.norm(proj - dst, axis=1)) / ppm)
    if len(found) < 4:
        gate["warnings"].append(f"{len(found)}/4 markers - reduced precision")
    pose = _camera_pose(src, dst / ppm, img_bgr.shape, cfg.get("camera", {}).get("focal_px"))
    if pose:
        gate["tilt_deg"] = round(pose["tilt_deg"], 1)
        gate["camera_height_mm"] = round(pose["camera_height_mm"], 1)
        if pose["tilt_deg"] > cfg["gate"].get("max_tilt_deg", 90):
            gate["reason"] = (f"phone tilted {pose['tilt_deg']:.0f} deg - hold it flat above the sheet "
                              f"(max {cfg['gate']['max_tilt_deg']:.0f} deg)")
            return None, gate
    rect = cv2.warpPerspective(img_bgr, H, (int(SHEET_W * ppm), int(SHEET_H * ppm)),
                               flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    x0, y0, x1, y1 = [int(v * ppm) for v in CAPTURE]
    cap_bgr = rect[y0:y1, x0:x1]
    cap = cv2.cvtColor(cap_bgr, cv2.COLOR_BGR2GRAY)
    # Sharpness measured on the markers, not the scene: a mostly-blank sheet has low
    # Laplacian variance even when perfectly focused. Markers have known edges in every photo.
    rg = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY); sv = []
    for i in found:
        mx, my = MARKERS[i]
        roi = rg[int(my * ppm):int((my + MARKER) * ppm), int(mx * ppm):int((mx + MARKER) * ppm)]
        if roi.size: sv.append(cv2.Laplacian(roi, cv2.CV_64F).var())
    gate["sharpness"] = float(np.median(sv)) if sv else 0.0
    ring = _white_ring(rect, ppm)
    gate["paper_brightness"] = float(np.median(ring.mean(1))) if len(ring) else float(np.percentile(cap, 90))
    gate["saturated_frac"] = float((cap_bgr.max(axis=2) >= 254).mean())   # clipped, not merely bright
    g = cfg["gate"]
    if gate["sharpness"] < g["min_sharpness"]:
        gate["reason"] = f"image too blurred (sharpness {gate['sharpness']:.0f} < {g['min_sharpness']})"
        return None, gate
    if gate["paper_brightness"] < g["min_paper_brightness"]:
        gate["reason"] = "too dark - move to better light"; return None, gate
    if gate["saturated_frac"] > g["max_saturated_frac"]:
        gate["reason"] = "overexposed - avoid direct glare on the sheet"; return None, gate
    if cfg.get("white_balance", True):
        rect = _white_balance(rect, cfg)
    gate["ok"] = True
    return rect, gate

def _white_ring(rect, ppm, inner=1.0, outer=6.0):
    """Pixels of the white quiet zone around each marker: known white paper in every capture,
    whatever colour the capture area is printed in."""
    H, W = rect.shape[:2]; m = np.zeros((H, W), np.uint8)
    for x, y in MARKERS.values():
        cv2.rectangle(m, (int((x - outer) * ppm), int((y - outer) * ppm)),
                      (int((x + MARKER + outer) * ppm), int((y + MARKER + outer) * ppm)), 255, -1)
        cv2.rectangle(m, (int((x - inner) * ppm), int((y - inner) * ppm)),
                      (int((x + MARKER + inner) * ppm), int((y + MARKER + inner) * ppm)), 0, -1)
    return rect[m > 0]

def _white_balance(rect, cfg):
    px = _white_ring(rect, cfg["px_per_mm"]).astype(np.float32)
    if len(px) < 200: return rect
    ref = np.median(px, axis=0)
    gains = np.clip(ref.mean() / np.maximum(ref, 1), 0.7, 1.4)
    return np.clip(rect.astype(np.float32) * gains, 0, 255).astype(np.uint8)

def check_circles(rect, cfg):
    """Measures the 4 printed circles on every capture. Catches perspective, lens and
    detection faults. Does NOT catch uniform print scaling (markers scale with circles) -
    that needs a one-time check of the printed ruler against a real ruler."""
    ppm = cfg["px_per_mm"]
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    out = []
    for cx, cy, d in CIRCLES:
        R = d / 2; r = R + 4
        x0, y0 = int((cx - r) * ppm), int((cy - r) * ppm)
        roi = gray[y0:int((cy + r) * ppm), x0:int((cx + r) * ppm)]
        if roi.size == 0: out.append({"nominal_mm": d, "measured_mm": None}); continue
        # thin dark ring on paper -> black-hat isolates it from shading and nearby shadows
        bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        ys, xs = np.nonzero(bh > max(12.0, 0.4 * float(np.percentile(bh, 99))))
        rad = np.hypot(xs + x0 - cx * ppm, ys + y0 - cy * ppm) / ppm
        keep = (rad > 0.8 * R) & (rad < 1.2 * R)             # the ring only, not the label text
        if keep.sum() < 20: out.append({"nominal_mm": d, "measured_mm": None}); continue
        pts = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
        for _ in range(3):                                   # refit on inliers (drop label slips, debris)
            if len(pts) < 20: break
            (ex, ey), (w, h), _ = cv2.fitEllipse(pts)
            rr = np.hypot(pts[:, 0] - ex, pts[:, 1] - ey) / ppm
            pts = pts[np.abs(rr - (w + h) / 4 / ppm) < 1.5]
        m = (w + h) / 2 / ppm
        out.append({"nominal_mm": d, "measured_mm": float(m), "error_mm": float(m - d)})
    errs = [abs(c["error_mm"]) for c in out if c.get("error_mm") is not None]
    return {"circles": out,
            "mean_abs_error_mm": float(np.mean(errs)) if errs else None,
            "max_abs_error_mm": float(np.max(errs)) if errs else None,
            "n_measured": len(errs)}

# --------------------------------------------------------------------------------------
# Stage 3: detection (injected)
# --------------------------------------------------------------------------------------
def detect(rect, detector, cfg):
    ppm = cfg["px_per_mm"]
    x0, y0, x1, y1 = [int(v * ppm) for v in CAPTURE]
    boxes = detector(rect[y0:y1, x0:x1])
    out = []
    for (bx1, by1, bx2, by2, conf) in boxes:
        bx1 += x0; bx2 += x0; by1 += y0; by2 += y0
        cxm, cym = (bx1 + bx2) / 2 / ppm, (by1 + by2) / 2 / ppm
        if CAPTURE[0] <= cxm <= CAPTURE[2] and CAPTURE[1] <= cym <= CAPTURE[3]:
            out.append((float(bx1), float(by1), float(bx2), float(by2), float(conf)))
    return out

# --------------------------------------------------------------------------------------
# Stage 4a: geometry + mass
# --------------------------------------------------------------------------------------
def _ellipse_axes(d1, d2, ang):
    """cv2.fitEllipse's angle belongs to its FIRST axis, which is not always the longer one.
    Return (major, minor, angle-of-major in [0,180))."""
    if d1 >= d2: return d1, d2, ang % 180.0
    return d2, d1, (ang + 90.0) % 180.0

def foreground_mask(region_bgr, min_blob_px=150):
    """Onion-vs-background mask from COLOUR difference only (Lab a*, b*), ignoring lightness.
    - background colour = the dominant (a*,b*) in the region (paper or floor dominates the frame)
    - shadows change lightness, not colour  -> stay background
    - pale / pinkish skin differs in colour -> foreground
    - highlight holes inside onions are filled afterwards
    On real photos this removes the edge 'bites' a saturation mask leaves on shiny red onions."""
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    H, ae, be = np.histogram2d(a.ravel(), b.ravel(), bins=64, range=[[0, 256], [0, 256]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    a0, b0 = (ae[i] + ae[i + 1]) / 2, (be[j] + be[j + 1]) / 2
    d = np.sqrt((a - a0) ** 2 + (b - b0) ** 2)
    t, _ = cv2.threshold(np.clip(d * 4, 0, 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t / 4, 6.0)                       # guard: never treat tiny colour noise as onion
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    for c in cs:
        if cv2.contourArea(c) >= min_blob_px:
            cv2.drawContours(filled, [c], -1, 255, -1)
    return filled, {"background_ab": [round(float(a0), 1), round(float(b0), 1)],
                    "chroma_threshold": round(float(thr), 2)}

def _box_geom(lb, ppm):
    bw, bh = lb[2] - lb[0], lb[3] - lb[1]
    return {"cx_mm": (lb[0] + lb[2]) / 2 / ppm, "cy_mm": (lb[1] + lb[3]) / 2 / ppm,
            "major_mm": max(bw, bh) / ppm, "minor_mm": min(bw, bh) / ppm,
            "equiv_fallback_mm": math.sqrt(bw * bh) / ppm,
            "angle": 0.0, "method": "box-fallback", "touching": False, "edge_touch": False}

def _grabcut_geom(region, lb, ppm, pad=0.2, iters=4):
    """Low-contrast onions (pale dry skin): colour model learnt locally from the detector box."""
    x1, y1, x2, y2 = lb; bw, bh = x2 - x1, y2 - y1
    X0, Y0 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    X1, Y1 = int(min(region.shape[1], x2 + pad * bw)), int(min(region.shape[0], y2 + pad * bh))
    roi = region[Y0:Y1, X0:X1].copy()
    if roi.shape[0] < 10 or roi.shape[1] < 10: return None
    gm = np.zeros(roi.shape[:2], np.uint8)
    rx, ry = max(0, int(x1 - X0)), max(0, int(y1 - Y0))
    rect = (rx, ry, max(2, min(int(bw), roi.shape[1] - rx - 1)), max(2, min(int(bh), roi.shape[0] - ry - 1)))
    try:
        cv2.grabCut(roi, gm, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = max(3, int(0.08 * min(bw, bh))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.5 * (math.pi / 4) * bw * bh or len(c) < 20: return None
    (ex, ey), (d1, d2), ang = cv2.fitEllipse(cv2.convexHull(c))
    mj, mn, ang = _ellipse_axes(d1, d2, ang)
    return {"cx_mm": (ex + X0) / ppm, "cy_mm": (ey + Y0) / ppm, "major_mm": mj / ppm, "minor_mm": mn / ppm,
            "angle": float(ang), "method": "grabcut", "touching": False, "edge_touch": False}

def measure_region(region, boxes_local, ppm, confs=None, reject_coverage=0.12, reject_conf=0.60):
    """Stage 4a on a region (rectified capture area in mm, or a raw photo with ppm=1 -> pixels).
    1. Colour-difference foreground mask (see foreground_mask).
    2. Marker-controlled watershed on the distance transform, seeded at each detector box
       centre -> splits touching/overlapping onions along the neck.
    3. Size-relative opening removes thin protrusions (sprouts, root hairs, loose skin).
    4. Convex hull fills residual edge bites (onion bulbs are convex).
    5. Ellipse fitted to the FREE boundary only - the edge against background - excluding the
       cut line shared with a neighbour, which would otherwise shrink the major axis.
    Onions the mask covers poorly (pale dry skin on a light background) -> GrabCut from the box,
    then box equivalent diameter as last resort. Both are flagged for review."""
    if not boxes_local: return [], {}
    mask, info = foreground_mask(region)
    seeds = np.zeros(mask.shape, np.int32)
    for i, lb in enumerate(boxes_local):
        c = (int((lb[0] + lb[2]) / 2), int((lb[1] + lb[3]) / 2))
        r = max(2, int(0.15 * min(lb[2] - lb[0], lb[3] - lb[1])))
        cv2.circle(seeds, c, r, i + 1, -1)
    seeds[mask == 0] = 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    try:
        from skimage.segmentation import watershed
        labels = watershed(-dist, seeds, mask=mask > 0)
    except Exception:                      # fallback: size-weighted nearest-centre assignment
        ys, xs = np.nonzero(mask); labels = np.zeros(mask.shape, np.int32)
        if len(xs):
            C = np.array([[(l[0] + l[2]) / 2, (l[1] + l[3]) / 2, max(l[2] - l[0], l[3] - l[1]) / 2]
                          for l in boxes_local])
            d = np.hypot(xs[:, None] - C[:, 0], ys[:, None] - C[:, 1]) / C[:, 2]
            labels[ys, xs] = np.argmin(d, 1) + 1

    out = []
    for i, lb in enumerate(boxes_local):
        bw, bh = lb[2] - lb[0], lb[3] - lb[1]
        p = 0.15
        x0, y0 = int(max(0, lb[0] - p * bw)), int(max(0, lb[1] - p * bh))
        x1, y1 = int(min(mask.shape[1], lb[2] + p * bw)), int(min(mask.shape[0], lb[3] + p * bh))
        L = labels[y0:y1, x0:x1]
        reg = (L == i + 1).astype(np.uint8) * 255
        k = max(3, int(0.08 * min(bw, bh))) | 1
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea) if cs else None
        coverage = (cv2.contourArea(c) / ((math.pi / 4) * bw * bh)) if c is not None and bw * bh > 0 else 0.0
        conf = confs[i] if confs is not None else 1.0
        if coverage < reject_coverage and conf < reject_conf:
            g = _box_geom(lb, ppm)                      # a shadow or floor mark, not an onion
            g.update(reject=True, reject_reason="no onion colour in box (shadow / floor mark)",
                     mask_coverage=round(float(coverage), 2), conf=round(float(conf), 3))
            out.append(g); continue
        if c is None or len(c) < 20 or coverage < 0.55:
            g = _grabcut_geom(region, lb, ppm) or _box_geom(lb, ppm)
            g["mask_coverage"] = round(float(coverage), 2)
            out.append(g); continue
        hull = np.zeros_like(reg)
        cv2.fillConvexPoly(hull, cv2.convexHull(c), 255)
        hc, _ = cv2.findContours(hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(hc, key=cv2.contourArea).reshape(-1, 2)
        other = cv2.dilate(((L > 0) & (L != i + 1)).astype(np.uint8), np.ones((5, 5), np.uint8))
        free = pts[other[pts[:, 1], pts[:, 0]] == 0]
        touching = len(free) < len(pts)
        reliable = len(free) >= max(20, 0.40 * len(pts))
        (ex, ey), (d1, d2), ang = cv2.fitEllipse((free if reliable else pts).astype(np.float32))
        mj, mn, ang = _ellipse_axes(d1, d2, ang)
        out.append({"cx_mm": (ex + x0) / ppm, "cy_mm": (ey + y0) / ppm,
                    "major_mm": mj / ppm, "minor_mm": mn / ppm, "angle": float(ang),
                    "method": "watershed-free-boundary" if touching else "colour-hull",
                    "touching": bool(touching),
                    "edge_touch": bool(touching and not reliable),     # router: geometry unreliable
                    "free_boundary_frac": round(len(free) / len(pts), 2),
                    "mask_coverage": round(float(coverage), 2)})
    return out, info

def measure_all(rect, boxes, cfg):
    """Stage 4a on the rectified sheet: measure_region on the capture area, in mm."""
    ppm = cfg["px_per_mm"]
    X0, Y0, X1, Y1 = [int(v * ppm) for v in CAPTURE]
    local = [(b[0] - X0, b[1] - Y0, b[2] - X0, b[3] - Y0) for b in boxes]
    r = cfg.get("router", {})
    geoms, info = measure_region(rect[Y0:Y1, X0:X1], local, ppm, confs=[b[4] for b in boxes],
                                 reject_coverage=r.get("reject_coverage", 0.12), reject_conf=r.get("reject_conf", 0.60))
    Z = cfg.get("_camera_height_mm")
    for g in geoms:
        g["cx_mm"] += CAPTURE[0]; g["cy_mm"] += CAPTURE[1]
        if Z and cfg.get("camera", {}).get("parallax_correction", True):
            # outline ~D/2 above paper: D_proj = D * Z / (Z - D/2)  ->  D = D_proj / (1 + D_proj / (2Z))
            s = 1.0 / (1.0 + g["major_mm"] / (2.0 * Z))
            g["parallax_scale"] = round(s, 4)
            for k in ("major_mm", "minor_mm", "equiv_fallback_mm"):
                if g.get(k) is not None: g[k] = float(g[k]) * s
    return geoms, info

def diameter(g, cfg):
    if g["method"] == "box-fallback": return g["equiv_fallback_mm"]
    if cfg["grading"]["diameter_definition"] == "equivalent":
        return math.sqrt(g["major_mm"] * g["minor_mm"])
    return g["major_mm"]

def mass_g(major, minor, cfg):
    v = cfg["varieties"][cfg["variety"]]
    return (math.pi / 6) * major * minor * (v["axis_ratio_k"] * major) / 1000 * v["density_g_cm3"]

# --------------------------------------------------------------------------------------
# Stage 4b: defect crops (classifier injected)
# --------------------------------------------------------------------------------------
def crop_for_classifier(rect, box):
    bx1, by1, bx2, by2, _ = box
    bw, bh = bx2 - bx1, by2 - by1; p = 0.10          # same padding as training crops
    x0, y0 = int(max(0, bx1 - p * bw)), int(max(0, by1 - p * bh))
    x1, y1 = int(min(rect.shape[1], bx2 + p * bw)), int(min(rect.shape[0], by2 + p * bh))
    return cv2.cvtColor(rect[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

# --------------------------------------------------------------------------------------
# Stage 5 + 6: router and rules
# --------------------------------------------------------------------------------------
def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4); return math.log(p / (1 - p))

def judge(o, cfg, calibrated=True):
    thr, band, gr = cfg["thresholds"], cfg["router"]["logit_band"], cfg["grading"]
    pr = o["probs"]
    if o.get("green_frac", 0.0) >= cfg["router"].get("foreign_green_frac", 1.0) and pr["sprouted"] < thr["sprouted"]:
        o.update(defects=[], size_status="not_measured", grade="FOREIGN",
                 review=["not an onion? (leafy green part, no sprout) - excluded from lot %, inspector to confirm"])
        return o
    rotten = pr["rotten"] >= thr["rotten"]
    sprouted = pr["sprouted"] >= thr["sprouted"]
    other = pr["binary_bad"] >= thr["binary_bad"] and not rotten and not sprouted
    defects = [n for n, f in (("rotten", rotten), ("sprouted", sprouted), ("other_defect", other)) if f]

    review = []
    for h in cfg["router"].get("uncertain_heads", ("rotten", "sprouted", "binary_bad")):
        if abs(_logit(pr[h]) - _logit(thr[h])) < band: review.append(f"uncertain:{h}")
    if other: review.append("other defect - possible damage, inspector to confirm")
    if o["geom"]["method"] == "box-fallback": review.append("size from box (outline not isolated)")
    if o["geom"]["method"] == "grabcut": review.append("low-contrast skin - size estimated")
    if o["geom"]["edge_touch"]: review.append("touching neighbour - outline mostly hidden")

    if calibrated:
        d = o["diameter_mm"]
        size = "undersized" if d < gr["size_min_mm"] else ("oversized" if d > gr["size_max_mm"] else "in_band")
    else:
        size = "not_measured"
    if any(x in gr["unfit_defects"] for x in defects): grade = "UNFIT"
    elif any(x in gr["urs_defects"] for x in defects) or size in ("undersized", "oversized"): grade = "URS"
    else: grade = "A" if calibrated else "DEFECT_FREE"
    o.update(defects=defects, size_status=size, grade=grade, review=review)
    return o

# --------------------------------------------------------------------------------------
# Multi-view merge: rectified views share sheet coordinates
# --------------------------------------------------------------------------------------
def merge_views(views, cfg):
    merged = []
    for vi, onions in enumerate(views):
        for o in onions:
            best, bd = None, 1e9
            for m in merged:
                if vi in m["_views"]: continue
                dd = math.hypot(o["geom"]["cx_mm"] - m["_cx"], o["geom"]["cy_mm"] - m["_cy"])
                if dd < cfg["merge_tol_frac"] * m["_major"] and dd < bd: best, bd = m, dd
            if best is None:
                merged.append({"_views": {vi}, "_cx": o["geom"]["cx_mm"], "_cy": o["geom"]["cy_mm"],
                               "_major": o["geom"]["major_mm"], "obs": [o]})
            else:
                best["_views"].add(vi); best["obs"].append(o)
    out = []
    for i, m in enumerate(merged):
        obs = m["obs"]
        good = [x for x in obs if x["geom"]["method"] != "box-fallback"] or obs
        g = dict(good[0]["geom"])
        g["major_mm"] = float(np.median([x["geom"]["major_mm"] for x in good]))
        g["minor_mm"] = float(np.median([x["geom"]["minor_mm"] for x in good]))
        g["cx_mm"] = float(np.median([x["geom"]["cx_mm"] for x in obs]))
        g["cy_mm"] = float(np.median([x["geom"]["cy_mm"] for x in obs]))
        g["edge_touch"] = any(x["geom"]["edge_touch"] for x in obs)
        probs = {h: float(max(x["probs"][h] for x in obs)) for h in obs[0]["probs"]}   # worst view wins
        out.append({"id": i + 1, "views": sorted(m["_views"]), "geom": g, "probs": probs,
                    "box": obs[0]["box"], "view0_box": next((x["box"] for x in obs if x["view"] == 0), None),
                    "green_frac": float(max(x.get("green_frac", 0.0) for x in obs))})
    return out

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def _sha256(b): return hashlib.sha256(b).hexdigest()

def grade_lot(images_bgr, detector, classifier, cfg=None, meta=None, model_info=None):
    cfg = cfg or DEFAULT_CONFIG
    meta = dict(meta or {})
    t0 = datetime.datetime.now()
    views, gates, rects, calib = [], [], [], []
    for vi, img in enumerate(images_bgr):
        rect, gate = rectify(img, cfg)
        gate["view"] = vi; gates.append(gate)
        if rect is None: continue
        rects.append((vi, rect))
        calib.append(check_circles(rect, cfg))
        boxes = detect(rect, detector, cfg)
        crops = [crop_for_classifier(rect, b) for b in boxes]
        P = classifier(crops) if crops else np.zeros((0, 3))
        geoms, mask_info = measure_all(rect, boxes, dict(cfg, _camera_height_mm=gate.get("camera_height_mm")))
        gate["mask"] = mask_info
        gate["n_rejected_not_onion"] = sum(1 for g in geoms if g.get("reject"))
        kept = [g for g in geoms if not g.get("reject")]
        gate["hidden_outline_frac"] = round(sum(1 for g in kept if g.get("edge_touch")) / max(1, len(kept)), 2)
        gate["touching_frac"] = round(sum(1 for g in kept if g.get("touching")) / max(1, len(kept)), 2)
        gg = cfg["gate"]
        if len(kept) >= gg.get("min_onions_for_crowd_check", 4) and \
                (gate["hidden_outline_frac"] > gg.get("max_hidden_outline_frac", 1.0) or
                 gate["touching_frac"] > gg.get("max_touching_frac", 1.0)):
            gate["ok"] = False
            gate["reason"] = (f"onions packed together ({gate['touching_frac']*100:.0f}% touching, "
                              f"{gate['hidden_outline_frac']*100:.0f}% hidden outlines) - "
                              "spread them in a single layer with small gaps and retake")
            rects.pop(); calib.pop(); continue
        onions = []
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        leaf = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 60))
        for b, p, g in zip(boxes, P, geoms):
            if g.get("reject"): continue
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            roi = leaf[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            green = float(roi.mean()) if roi.size else 0.0
            onions.append({"view": vi, "box": b, "geom": g, "green_frac": round(green, 4),
                           "probs": {"rotten": float(p[0]), "sprouted": float(p[1]), "binary_bad": float(p[2])}})
        views.append(onions)

    result = {"meta": meta, "timestamp": t0.isoformat(timespec="seconds"),
              "gates": gates, "views_accepted": [vi for vi, _ in rects]}
    if not rects:
        result["status"] = "REJECTED"
        result["reason"] = "; ".join(g.get("reason", "") for g in gates)
        return result, None

    onions = merge_views(views, cfg)
    for o in onions:
        o["diameter_mm"] = diameter(o["geom"], cfg)
        o["mass_g"] = mass_g(o["geom"]["major_mm"], o["geom"]["minor_mm"], cfg)
        judge(o, cfg)

    graded = [o for o in onions if o["grade"] != "FOREIGN"]          # foreign objects never enter the lot %
    n_foreign = len(onions) - len(graded)
    tot_m = sum(o["mass_g"] for o in graded) or 1.0
    by = lambda gr: sum(o["mass_g"] for o in graded if o["grade"] == gr)
    cnt = lambda gr: sum(1 for o in graded if o["grade"] == gr)
    n = len(graded) or 1
    urs_on = cfg["grading"]["urs_reporting_enabled"]
    lot = {"n_onions": len(graded), "n_foreign_excluded": n_foreign,
           "n_review": sum(1 for o in onions if o["review"]),
           "n_other_defect_inspector_check": sum(1 for o in onions if "other_defect" in o["defects"]),
           "total_mass_g_est": round(tot_m, 1),
           "pct_gradeA_by_weight": round(100 * by("A") / tot_m, 2),
           "pct_URS_by_weight": round(100 * (by("URS") + by("UNFIT")) / tot_m, 2),
           "pct_unfit_by_weight": round(100 * by("UNFIT") / tot_m, 2),
           "pct_gradeA_by_count": round(100 * cnt("A") / n, 2),
           "pct_URS_by_count": round(100 * (cnt("URS") + cnt("UNFIT")) / n, 2),
           "second_bucket_label": "URS" if urs_on else "Below Grade A (URS not procured under current circular)"}
    ce = [c["max_abs_error_mm"] for c in calib if c["max_abs_error_mm"] is not None]
    calib_summary = {"per_view": calib, "worst_circle_error_mm": max(ce) if ce else None,
                     "status": ("PASS" if ce and max(ce) <= 1.5 else ("WARN" if ce else "NOT MEASURED"))}

    cfg_public = {k: v for k, v in cfg.items() if k != "signing_key"}
    result.update(status="GRADED", lot=lot, onions=onions, calibration_check=calib_summary,
                  config=cfg_public, config_sha256=_sha256(json.dumps(cfg_public, sort_keys=True).encode()),
                  models=model_info or {},
                  image_sha256=[_sha256(cv2.imencode(".png", im)[1].tobytes()) for im in images_bgr],
                  processing_s=round((datetime.datetime.now() - t0).total_seconds(), 2))
    annotated = annotate(rects[0][1], onions, cfg, view=rects[0][0])
    result["annotated_sha256"] = _sha256(cv2.imencode(".png", annotated)[1].tobytes())
    return sign(result, cfg), annotated

def sign(result, cfg):
    body = {k: v for k, v in result.items() if k != "signature"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=float).encode()
    result["signature"] = {"alg": "HMAC-SHA256", "payload_sha256": _sha256(canon),
                           "hmac": hmac.new(cfg["signing_key"].encode(), canon, hashlib.sha256).hexdigest(),
                           "key_id": "demo" if cfg["signing_key"] == "NIKASH-DEMO-KEY" else "device"}
    return result

def verify(result, key):
    body = {k: v for k, v in result.items() if k != "signature"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=float).encode()
    return hmac.compare_digest(hmac.new(key.encode(), canon, hashlib.sha256).hexdigest(),
                               result["signature"]["hmac"])

# --------------------------------------------------------------------------------------
# Stage 7: annotation + PDF
# --------------------------------------------------------------------------------------
COLORS = {"A": (60, 170, 60), "DEFECT_FREE": (60, 170, 60), "URS": (0, 160, 240), "UNFIT": (40, 40, 220),
          "FOREIGN": (150, 150, 150)}  # BGR

def draw(img, onions, ppm, offset_mm=(0.0, 0.0)):
    majors = [o["geom"]["major_mm"] * ppm for o in onions] or [60]
    fs = float(np.clip(np.median(majors) / 90.0, 0.35, 0.9))
    th = max(1, int(round(fs * 4)))
    for o in onions:
        g = o["geom"]; col = COLORS[o["grade"]]
        c = (int((g["cx_mm"] - offset_mm[0]) * ppm), int((g["cy_mm"] - offset_mm[1]) * ppm))
        ax = (max(1, int(g["major_mm"] * ppm / 2)), max(1, int(g["minor_mm"] * ppm / 2)))
        cv2.ellipse(img, c, ax, g["angle"], 0, 360, col, th)
        if o["review"]:
            cv2.ellipse(img, c, (ax[0] + 2 * th, ax[1] + 2 * th), g["angle"], 0, 360, (200, 60, 200), max(1, th // 2))
        t = str(o["id"]); (tw, tht), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        org = (c[0] - tw // 2, c[1] + tht // 2)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 2)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
    return img

def annotate(rect, onions, cfg, view=0):
    ppm = cfg["px_per_mm"]
    x0, y0, x1, y1 = [int(v * ppm) for v in CAPTURE]
    return draw(rect.copy(), onions, ppm)[y0:y1, x0:x1]

def analyze_uncalibrated(img_bgr, detector, classifier, cfg=None, meta=None):
    """NO reference sheet in frame. Runs detection, defect classification and outline fitting
    on the raw photo. Sizes are in PIXELS and perspective is uncorrected, so there is no size
    criterion, no mass and no Grade-A-by-weight - only defect counts and relative sizes.
    For demos and model checks on field photos; not a grading result."""
    cfg = cfg or DEFAULT_CONFIG
    t0 = datetime.datetime.now()
    boxes = detector(img_bgr)
    crops = [crop_for_classifier(img_bgr, b) for b in boxes]
    Pr = classifier(crops) if crops else np.zeros((0, 3))
    r = cfg.get("router", {})
    geoms, info = measure_region(img_bgr, [b[:4] for b in boxes], 1.0, confs=[b[4] for b in boxes],
                                 reject_coverage=r.get("reject_coverage", 0.12), reject_conf=r.get("reject_conf", 0.60))
    n_rej = sum(1 for g in geoms if g.get("reject"))
    kept = [(b, p, g) for b, p, g in zip(boxes, Pr, geoms) if not g.get("reject")]
    onions = []
    for k, (b, p, g) in enumerate(kept):
        o = {"id": k + 1, "box": b, "geom": g,
             "probs": {"rotten": float(p[0]), "sprouted": float(p[1]), "binary_bad": float(p[2])}}
        o["diameter_px"] = diameter(g, cfg); o["diameter_mm"] = None
        onions.append(judge(o, cfg, calibrated=False))
    n = len(onions)
    cnt = lambda f: sum(1 for o in onions if f(o))
    d = np.array([o["diameter_px"] for o in onions]) if onions else np.zeros(0)
    summary = {"mode": "UNCALIBRATED - no reference sheet, sizes in pixels, perspective uncorrected",
               "n_onions": n,
               "n_defect_free": cnt(lambda o: o["grade"] == "DEFECT_FREE"),
               "n_rotten": cnt(lambda o: "rotten" in o["defects"]),
               "n_sprouted": cnt(lambda o: "sprouted" in o["defects"]),
               "n_other_defect": cnt(lambda o: "other_defect" in o["defects"]),
               "n_review": cnt(lambda o: bool(o["review"])),
               "n_rejected_not_onion": n_rej,
               "diameter_px_quantiles": {q: round(float(np.percentile(d, q)), 1) for q in (10, 50, 90)} if n else {},
               "methods": {m: cnt(lambda o, m=m: o["geom"]["method"] == m) for m in
                           sorted({o["geom"]["method"] for o in onions})}}
    result = {"meta": dict(meta or {}), "timestamp": t0.isoformat(timespec="seconds"),
              "summary": summary, "onions": onions, "mask": info,
              "processing_s": round((datetime.datetime.now() - t0).total_seconds(), 2)}
    return result, draw(img_bgr.copy(), onions, 1.0)

def report_pdf(result, annotated_bgr, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    ss = getSampleStyleSheet(); sm = ss["BodyText"].clone("sm", fontSize=7.5, leading=9)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=14*mm, rightMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm)
    lot, meta, cal = result["lot"], result["meta"], result["calibration_check"]
    el = [Paragraph("<b>Nikash - Onion Quality Report</b>", ss["Title"]),
          Paragraph(f"Lot <b>{meta.get('lot_id','-')}</b> &nbsp; Farmer {meta.get('farmer_id','-')} &nbsp; "
                    f"Centre {meta.get('centre','-')} &nbsp; GPS {meta.get('gps','not captured')} &nbsp; "
                    f"{result['timestamp']}", sm), Spacer(1, 4*mm)]
    head = Table([["% Grade A (by weight)", f"% {lot['second_bucket_label']} (by weight)"],
                  [f"{lot['pct_gradeA_by_weight']:.1f}%", f"{lot['pct_URS_by_weight']:.1f}%"]],
                 colWidths=[88*mm, 88*mm])
    head.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, 0), 9), ("FONTSIZE", (0, 1), (-1, 1), 22),
                              ("LEADING", (0, 1), (-1, 1), 26), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                              ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#2E7D32")),
                              ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#E65100")),
                              ("BOX", (0, 0), (-1, -1), 0.5, colors.grey), ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey)]))
    el += [head, Spacer(1, 2*mm),
           Paragraph(f"of which unfit (rotten): {lot['pct_unfit_by_weight']:.1f}% by weight &nbsp;|&nbsp; "
                     f"by count: Grade A {lot['pct_gradeA_by_count']:.1f}%, URS {lot['pct_URS_by_count']:.1f}% &nbsp;|&nbsp; "
                     f"{lot['n_onions']} onions assessed, est. {lot['total_mass_g_est']:.0f} g, "
                     f"<b>{lot['n_review']} flagged for inspector review</b>", sm),
           Paragraph(f"Calibration self-check: <b>{cal['status']}</b> (worst printed-circle error "
                     f"{cal['worst_circle_error_mm'] if cal['worst_circle_error_mm'] is None else round(cal['worst_circle_error_mm'],2)} mm) &nbsp;|&nbsp; "
                     f"views accepted {result['views_accepted']} &nbsp;|&nbsp; processing {result['processing_s']} s", sm),
           Spacer(1, 3*mm)]
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    h, w = annotated_bgr.shape[:2]; W = 182*mm
    el += [Image(io.BytesIO(buf.tobytes()), width=W, height=W * h / w),
           Paragraph("Outline colour: green = Grade A, orange = URS, red = unfit. Magenta ring = inspector review.", sm),
           Spacer(1, 3*mm)]
    rows = [["#", "diam mm", "mass g", "size", "defects", "grade", "review"]]
    for o in result["onions"]:
        rows.append([o["id"], f"{o['diameter_mm']:.1f}", f"{o['mass_g']:.0f}", o["size_status"],
                     ", ".join(o["defects"]) or "-", o["grade"], "; ".join(o["review"])[:60] or "-"])
    t = Table(rows, colWidths=[8*mm, 16*mm, 14*mm, 20*mm, 30*mm, 14*mm, 80*mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 6.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE"))]))
    el += [t, Spacer(1, 3*mm)]
    c = result["config"]; g = c["grading"]
    el += [Paragraph(f"Rules: size band {g['size_min_mm']}-{g['size_max_mm']} mm "
                     f"({g['diameter_definition']} diameter, provisional); variety {c['variety']}; "
                     f"thresholds {c['thresholds']}. Config SHA-256 {result['config_sha256'][:16]}...", sm),
           Paragraph(f"Models: {result.get('models', {})}", sm),
           Paragraph(f"Signature {result['signature']['alg']} ({result['signature']['key_id']} key): "
                     f"{result['signature']['hmac']}", sm),
           Paragraph("Advisory measurement. The inspector retains the final grading decision and signs below. "
                     "External defects only - internal rot is not visible to a camera.", sm),
           Spacer(1, 8*mm),
           Paragraph("Inspector signature: ______________________ &nbsp;&nbsp;&nbsp; Farmer acknowledgement: ______________________", sm)]
    doc.build(el)
    return path

# --------------------------------------------------------------------------------------
# Test utilities: raster of the printed sheet
# --------------------------------------------------------------------------------------
FIELD_BLUE_BGR = (215, 150, 70)          # make_sheet.py --blue : RGB (70, 150, 215)

def render_sheet(ppm=5.0, field=None):
    """Raster of the printed sheet for tests. field=None (white) or 'blue'."""
    W, H = int(SHEET_W * ppm), int(SHEET_H * ppm)
    img = np.full((H, W, 3), 250, np.uint8)
    if field == "blue":
        x0, y0, x1, y1 = [int(v * ppm) for v in CAPTURE]
        img[y0:y1, x0:x1] = FIELD_BLUE_BGR
    for i, (x, y) in MARKERS.items():
        s = int(MARKER * ppm)
        m = cv2.aruco.generateImageMarker(_DICT, i, s)
        img[int(y * ppm):int(y * ppm) + s, int(x * ppm):int(x * ppm) + s] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    for cx, cy, d in CIRCLES:
        cv2.circle(img, (int(cx * ppm), int(cy * ppm)), int(d / 2 * ppm), (20, 20, 20), max(1, int(0.25 * ppm)), cv2.LINE_AA)
    return img
