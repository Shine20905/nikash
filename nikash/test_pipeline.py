import sys, math, json, copy, numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from nikash import pipeline as P

rng = np.random.default_rng(3)
CFG = copy.deepcopy(P.DEFAULT_CONFIG)
CFG["camera"]["parallax_correction"] = False   # synthetic onions are painted FLAT on the sheet - no height, no parallax
PPM = CFG["px_per_mm"]

def draw_onion(img, cx, cy, maj, mnr, ang, kind, shadow=True):
    c = (int(cx*PPM), int(cy*PPM)); ax = (int(maj/2*PPM), int(mnr/2*PPM))
    if shadow:   # a shadow DARKENS what it falls on (shadow on blue = darker blue); must not be measured
        sm = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(sm, (c[0]+int(3*PPM), c[1]+int(3*PPM)), ax, ang, 0, 360, 255, -1)
        sm = cv2.GaussianBlur(sm, (15, 15), 0).astype(np.float32) / 255 * 0.40
        img[:] = (img.astype(np.float32) * (1 - sm[..., None])).astype(np.uint8)
    base = {"healthy": (70,55,150), "rotten": (40,55,75), "sprouted": (70,55,150),
            "pale": (196,204,226)}[kind]
    cv2.ellipse(img, c, ax, ang, 0, 360, base, -1)
    for k in range(-3, 4):   # skin stripes
        cv2.ellipse(img, c, (max(2, ax[0]-abs(k)*6), ax[1]), ang, 90+k*8, 270+k*8,
                    tuple(int(v*0.8) for v in base), 1)
    if kind == "sprouted":
        tip = (int(c[0] + ax[0]*0.6), int(c[1] - ax[1]*0.6))
        cv2.line(img, tip, (tip[0]+int(6*PPM), tip[1]-int(10*PPM)), (60,180,90), int(2*PPM))

def make_lot(spec, field=None):
    img = P.render_sheet(PPM, field); gt = []
    for (cx, cy, maj, ratio, ang, kind) in spec:
        mnr = maj*ratio
        draw_onion(img, cx, cy, maj, mnr, ang, kind)
        gt.append(dict(cx=cx, cy=cy, major=maj, minor=mnr, angle=ang, kind=kind))
    return img, gt

def camera(img, strength, seed):
    r = np.random.default_rng(seed); h, w = img.shape[:2]; pad = 0.2
    W2, H2 = int(w*(1+2*pad)), int(h*(1+2*pad))
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = src + np.float32([w*pad, h*pad]) + r.uniform(-strength, strength, (4,2)).astype(np.float32)*np.float32([w,h])
    Hm = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, Hm, (W2, H2), borderValue=(120,110,100))
    out = cv2.GaussianBlur(out, (5,5), 0.8)
    out = np.clip(out.astype(np.int16) + r.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)

def mock_models(gt, jitter=0.08):
    """Detector returns GT bounding boxes (in capture-crop px) with jitter, like a loose YOLO box.
       Classifier returns label-driven probabilities, in the same box order."""
    order = []
    def det(cap_img):
        order.clear(); boxes = []
        x0, y0 = P.CAPTURE[0]*PPM, P.CAPTURE[1]*PPM
        for g in gt:
            a, b, t = g["major"]/2, g["minor"]/2, math.radians(g["angle"])
            hw = math.sqrt((a*math.cos(t))**2 + (b*math.sin(t))**2)
            hh = math.sqrt((a*math.sin(t))**2 + (b*math.cos(t))**2)
            j = lambda: 1 + rng.uniform(-jitter, jitter)
            cx, cy = g["cx"]*PPM - x0, g["cy"]*PPM - y0
            boxes.append((cx-hw*PPM*j(), cy-hh*PPM*j(), cx+hw*PPM*j(), cy+hh*PPM*j(), 0.9))
            order.append(g["kind"])
        return boxes
    def cls(crops):
        table = {"healthy": [0.02, 0.05, 0.10], "rotten": [0.90, 0.10, 0.95], "sprouted": [0.05, 0.98, 0.90],
                 "pale": [0.02, 0.05, 0.10]}
        return np.array([table[k] for k in order[:len(crops)]])
    return det, cls

def gt_grades(gt, cfg):
    g = cfg["grading"]; out = []
    for o in gt:
        d = o["major"] if g["diameter_definition"] == "max_equatorial" else math.sqrt(o["major"]*o["minor"])
        if o["kind"] == "rotten": gr = "UNFIT"
        elif o["kind"] == "sprouted" or not (g["size_min_mm"] <= d <= g["size_max_mm"]): gr = "URS"
        else: gr = "A"
        out.append((gr, P.mass_g(o["major"], o["minor"], cfg)))
    tot = sum(m for _, m in out)
    return round(100*sum(m for gr, m in out if gr == "A")/tot, 2)

# ------------------------------------------------------------------ lot spec
SPEC = [ # cx, cy (mm, inside capture 15-282 x 85-270), major mm, minor/major, angle, kind
    (45, 115, 52, 0.80, 20, "healthy"), (105, 112, 41, 0.85, 70, "healthy"),   # 41 -> undersized
    (165, 118, 58, 0.78, 45, "healthy"), (230, 115, 48, 0.82, 10, "rotten"),
    (50, 185, 62, 0.76, 35, "healthy"), (113, 180, 46, 0.88, 80, "sprouted"),
    (170, 185, 55, 0.80, 0, "healthy"),
    (218, 188, 44, 0.90, 30, "healthy"), (255, 188, 42, 0.85, 60, "healthy"),  # touching pair
    (60, 245, 50, 0.80, 50, "healthy"), (130, 243, 73, 0.75, 15, "healthy"),   # 73 -> oversized
    (205, 245, 47, 0.83, 40, "healthy"),
]
fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    fails += (not cond)

sheet, gt = make_lot(SPEC)
det, cls = mock_models(gt)

# ---- 1. single flat view
res, ann = P.grade_lot([camera(sheet, 0.0, 1)], det, cls, CFG, meta={"lot_id": "T1"})
check("single view graded", res["status"] == "GRADED")
check("all onions found", res["lot"]["n_onions"] == len(gt), f"{res['lot']['n_onions']} vs {len(gt)}")
errs = []
for o in res["onions"]:
    g = min(gt, key=lambda g: math.hypot(g["cx"]-o["geom"]["cx_mm"], g["cy"]-o["geom"]["cy_mm"]))
    errs.append((abs(o["geom"]["major_mm"]-g["major"]), abs(o["geom"]["minor_mm"]-g["minor"]), o["geom"]["method"], g["kind"]))
maj_err = np.mean([e[0] for e in errs]); mnr_err = np.mean([e[1] for e in errs])
check("sizing MAE < 1.0 mm (flat)", maj_err < 1.0 and mnr_err < 1.0, f"major {maj_err:.2f}, minor {mnr_err:.2f}")
print("   methods:", {m: sum(1 for e in errs if e[2] == m) for m in set(e[2] for e in errs)})
check("calibration self-check PASS", res["calibration_check"]["status"] == "PASS",
      f"worst {res['calibration_check']['worst_circle_error_mm']:.3f} mm")
gtA = gt_grades(gt, CFG)
check("lot %A matches ground truth within 2 pp", abs(res["lot"]["pct_gradeA_by_weight"]-gtA) < 2.0,
      f"pred {res['lot']['pct_gradeA_by_weight']} vs gt {gtA}")
check("signature verifies", P.verify(res, CFG["signing_key"]))
tampered = json.loads(json.dumps(res, default=float)); tampered["lot"]["pct_gradeA_by_weight"] += 5
check("tampered result fails verification", not P.verify(tampered, CFG["signing_key"]))
cv2.imwrite("t1_annotated.jpg", ann)
P.report_pdf(res, ann, "t1_report.pdf"); print("   pdf written")

# ---- 2. three tilted views -> merge
views = [camera(sheet, s, k) for k, s in enumerate([0.03, 0.07, 0.10], start=10)]
res3, ann3 = P.grade_lot(views, det, cls, CFG, meta={"lot_id": "T3"})
check("3 views accepted", res3["views_accepted"] == [0, 1, 2], str(res3["views_accepted"]))
check("merge: no duplicate onions", res3["lot"]["n_onions"] == len(gt), f"{res3['lot']['n_onions']} vs {len(gt)}")
e3 = []
for o in res3["onions"]:
    g = min(gt, key=lambda g: math.hypot(g["cx"]-o["geom"]["cx_mm"], g["cy"]-o["geom"]["cy_mm"]))
    e3.append(abs(o["geom"]["major_mm"]-g["major"]))
check("3-view sizing MAE < 1.0 mm", np.mean(e3) < 1.0, f"{np.mean(e3):.2f}")
check("every merged onion seen in 3 views", all(len(o["views"]) == 3 for o in res3["onions"]),
      str([len(o["views"]) for o in res3["onions"]]))
check("3-view lot %A within 2 pp", abs(res3["lot"]["pct_gradeA_by_weight"]-gtA) < 2.0,
      f"pred {res3['lot']['pct_gradeA_by_weight']} vs gt {gtA}")

# ---- 3. grading specifics
by_kind = {}
for o in res["onions"]:
    g = min(gt, key=lambda g: math.hypot(g["cx"]-o["geom"]["cx_mm"], g["cy"]-o["geom"]["cy_mm"]))
    by_kind.setdefault((g["kind"], round(g["major"])), o)
check("rotten -> UNFIT", by_kind[("rotten", 48)]["grade"] == "UNFIT")
check("sprouted -> URS", by_kind[("sprouted", 46)]["grade"] == "URS")
check("41 mm -> undersized URS", by_kind[("healthy", 41)]["size_status"] == "undersized")
check("73 mm -> oversized URS", by_kind[("healthy", 73)]["size_status"] == "oversized")
check("52 mm healthy -> A", by_kind[("healthy", 52)]["grade"] == "A")

# ---- 4. circular change: URS withdrawn -> config only
cfg2 = copy.deepcopy(CFG); cfg2["grading"]["urs_reporting_enabled"] = False
r2, _ = P.grade_lot([camera(sheet, 0.0, 1)], det, cls, cfg2)
check("URS withdrawn -> label switches, no code change", r2["lot"]["second_bucket_label"].startswith("Below Grade A"))
cfg3 = copy.deepcopy(CFG); cfg3["grading"]["size_min_mm"] = 55.0
r3, _ = P.grade_lot([camera(sheet, 0.0, 1)], det, cls, cfg3)
check("size norm 45->55 mm lowers %A, config only", r3["lot"]["pct_gradeA_by_weight"] < res["lot"]["pct_gradeA_by_weight"],
      f"{res['lot']['pct_gradeA_by_weight']} -> {r3['lot']['pct_gradeA_by_weight']}")

# ---- 5. router
o = {"probs": {"rotten": 0.12, "sprouted": 0.01, "binary_bad": 0.2}, "geom": {"method": "ellipse-saturation", "edge_touch": False}, "diameter_mm": 50}
P.judge(o, CFG); check("rotten p=0.12 near its 0.11 threshold -> review", any("uncertain:rotten" in r for r in o["review"]), str(o["review"]))
o = {"probs": {"rotten": 0.001, "sprouted": 0.001, "binary_bad": 0.01}, "geom": {"method": "ellipse-saturation", "edge_touch": False}, "diameter_mm": 50}
P.judge(o, CFG); check("confident healthy -> no review", o["review"] == [], str(o["review"]))
o = {"probs": {"rotten": 0.01, "sprouted": 0.02, "binary_bad": 0.97}, "geom": {"method": "ellipse-saturation", "edge_touch": False}, "diameter_mm": 50}
P.judge(o, CFG); check("bad but not rotten/sprouted -> other_defect + review", "other_defect" in o["defects"] and o["review"], str(o["review"]))

# ---- 6. gate failures
blur = cv2.GaussianBlur(camera(sheet, 0.0, 1), (0, 0), 12)
rb, _ = P.grade_lot([blur], det, cls, CFG)
check("heavy blur rejected", rb["status"] == "REJECTED", rb.get("reason", ""))
nomark = camera(sheet, 0.0, 1).copy(); nomark[:, :] = cv2.GaussianBlur(nomark, (0, 0), 0.1)
h, w = nomark.shape[:2]; nomark[: h//2, :] = 120; nomark[:, : w//2] = 120      # hide 3 markers
rn, _ = P.grade_lot([nomark], det, cls, CFG)
check("<2 markers rejected with reason", rn["status"] == "REJECTED" and "marker" in rn.get("reason", ""), rn.get("reason", ""))

# ---- 7. no-sheet mode (field photos without the reference sheet)
flat = camera(sheet, 0.0, 1)
def det_full(img):     # same GT boxes, but in full-image pixel coords of the unwarped photo
    x0, y0 = P.CAPTURE[0]*PPM, P.CAPTURE[1]*PPM
    pad = (flat.shape[1] - sheet.shape[1]) / 2, (flat.shape[0] - sheet.shape[0]) / 2
    return [(b[0]+x0+pad[0], b[1]+y0+pad[1], b[2]+x0+pad[0], b[3]+y0+pad[1], b[4]) for b in det(None)]
ru, au = P.analyze_uncalibrated(flat, det_full, cls)
check("no-sheet mode: all onions found", ru["summary"]["n_onions"] == len(gt), str(ru["summary"]["n_onions"]))
check("no-sheet mode: rotten/sprouted counted, no size grading",
      ru["summary"]["n_rotten"] == 1 and ru["summary"]["n_sprouted"] == 1
      and all(o["size_status"] == "not_measured" for o in ru["onions"]), str(ru["summary"]))

# ---- 8. blue capture field (make_sheet.py --blue) - same lot, plus pale papery onions
SPEC_B = SPEC + [(70, 150, 50, 0.90, 10, "pale"), (245, 245, 54, 0.88, 70, "pale")]
SPEC_B = [s for s in SPEC_B if not (s[0] == 60 and s[1] == 245)]            # make room
sheet_b, gt_b = make_lot(SPEC_B, field="blue"); det_b, cls_b = mock_models(gt_b)
rb2, _ = P.grade_lot([camera(sheet_b, 0.0, 1)], det_b, cls_b, CFG)
check("blue: graded, all onions found", rb2["status"] == "GRADED" and rb2["lot"]["n_onions"] == len(gt_b),
      f"{rb2.get('lot', {}).get('n_onions')} vs {len(gt_b)}")
eb = []
for o in rb2["onions"]:
    g = min(gt_b, key=lambda g: math.hypot(g["cx"]-o["geom"]["cx_mm"], g["cy"]-o["geom"]["cy_mm"]))
    eb.append((abs(o["geom"]["major_mm"]-g["major"]), g["kind"], o["geom"]["method"]))
check("blue: sizing MAE < 1.0 mm", np.mean([e[0] for e in eb]) < 1.0, f"{np.mean([e[0] for e in eb]):.2f}")
check("blue: calibration self-check PASS", rb2["calibration_check"]["status"] == "PASS")
check("blue: white reference read from marker quiet zones", rb2["gates"][0]["paper_brightness"] > 200,
      f"{rb2['gates'][0]['paper_brightness']:.0f}")
pale_b = [e for e in eb if e[1] == "pale"]
check("blue: pale onions segmented by colour (no GrabCut/box fallback)",
      all(e[2] in ("colour-hull", "watershed-free-boundary") for e in pale_b), str(pale_b))
check("blue: pale onions sized within 1 mm", all(e[0] < 1.0 for e in pale_b), str([round(e[0], 2) for e in pale_b]))
# same pale onions on WHITE paper, for comparison (expected to need fallbacks)
sheet_w, gt_w = make_lot(SPEC_B, field=None); det_w, cls_w = mock_models(gt_w)
rw, _ = P.grade_lot([camera(sheet_w, 0.0, 1)], det_w, cls_w, CFG)
pale_w = []
for o in rw["onions"]:
    g = min(gt_w, key=lambda g: math.hypot(g["cx"]-o["geom"]["cx_mm"], g["cy"]-o["geom"]["cy_mm"]))
    if g["kind"] == "pale": pale_w.append((round(abs(o["geom"]["major_mm"]-g["major"]), 2), o["geom"]["method"]))
print(f"   info - same pale onions on WHITE paper: {pale_w}")

# ---- 9. a detector box on a bare shadow must not become an onion
sheet_s, gt_s = make_lot([(60, 130, 50, 0.85, 20, "healthy"), (150, 130, 48, 0.9, 0, "pale"),
                          (230, 200, 52, 0.8, 60, "healthy")])
sm = np.zeros(sheet_s.shape[:2], np.uint8)                          # a lone shadow patch, no onion
cv2.ellipse(sm, (int(120*PPM), int(225*PPM)), (int(24*PPM), int(18*PPM)), 30, 0, 360, 255, -1)
sm = cv2.GaussianBlur(sm, (21, 21), 0).astype(np.float32) / 255 * 0.45
sheet_s = (sheet_s.astype(np.float32) * (1 - sm[..., None])).astype(np.uint8)
det_s, cls_s = mock_models(gt_s)
def det_with_shadow(cap):
    b = det_s(cap); x0, y0 = P.CAPTURE[0]*PPM, P.CAPTURE[1]*PPM
    b.append(((120-26)*PPM-x0, (225-20)*PPM-y0, (120+26)*PPM-x0, (225+20)*PPM-y0, 0.35)); return b
def cls_shadow(crops): return np.vstack([cls_s(crops[:len(gt_s)]), np.array([[0.05, 0.05, 0.5]])])[:len(crops)]
rs, _ = P.grade_lot([camera(sheet_s, 0.0, 1)], det_with_shadow, cls_shadow, CFG)
check("shadow box rejected, real onions kept", rs["lot"]["n_onions"] == 3 and rs["gates"][0]["n_rejected_not_onion"] == 1,
      f"onions {rs['lot']['n_onions']}, rejected {rs['gates'][0].get('n_rejected_not_onion')}")
pale_kept = [o for o in rs["onions"] if abs(o["geom"]["cx_mm"] - 150) < 10]
check("pale onion on white paper NOT rejected", len(pale_kept) == 1,
      str([(o["geom"]["method"], o["geom"].get("mask_coverage")) for o in pale_kept]))

print(json.dumps(res["lot"], indent=1))

# ---------------- v0.6: parallax correction + tilt gate ----------------
print("\n--- v0.6 parallax + tilt ---")
Z = 400.0; D = 60.0
d_proj = D * Z / (Z - D / 2)                          # what the sheet-plane homography sees
g = [{"major_mm": d_proj, "minor_mm": d_proj, "cx_mm": 0.0, "cy_mm": 0.0}]
cfgp = copy.deepcopy(P.DEFAULT_CONFIG); cfgp["_camera_height_mm"] = Z
s_ = 1.0 / (1.0 + d_proj / (2 * Z))
check("parallax: 60 mm onion at 40 cm reads %.1f mm uncorrected" % d_proj, d_proj > 64)
check("parallax: corrected back to 60.0 mm", abs(d_proj * s_ - D) < 0.05, round(d_proj * s_, 3))

sheet = P.render_sheet(PPM)
Hs, Ws = sheet.shape[:2]
def view(tilt_deg, f=1800.0, dist=480.0):
    """Render the sheet through a pinhole camera tilted about the x axis."""
    t = math.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    W, H = 1500, 2000
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    cxm, cym = P.SHEET_W / 2, P.SHEET_H / 2
    Tsheet = np.array([[1 / PPM, 0, -cxm], [0, 1 / PPM, -cym], [0, 0, 1]])   # rect px -> mm centred
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], np.array([0, 0, dist])])
    return cv2.warpPerspective(sheet, Hm @ Tsheet, (W, H), borderValue=(90, 70, 60)), f
for tilt, want_ok in ((5, True), (15, True), (30, False)):
    img, f = view(tilt)
    c2 = copy.deepcopy(P.DEFAULT_CONFIG); c2["camera"]["focal_px"] = f
    r, gt = P.rectify(img, c2)
    ok = r is not None
    check(f"tilt gate: {tilt} deg -> {'accepted' if want_ok else 'rejected'}", ok == want_ok,
          f"est tilt {gt.get('tilt_deg')}, height {gt.get('camera_height_mm')} mm, reason {gt.get('reason','-')}")
img, f = view(10); c2 = copy.deepcopy(P.DEFAULT_CONFIG); c2["camera"]["focal_px"] = f
_, gt = P.rectify(img, c2)
check("pose: camera height recovered within 3%", abs(gt["camera_height_mm"] - 480) / 480 < 0.03, gt["camera_height_mm"])

# ---------------- v0.7: other_defect = inspector review only; crowded heap rejected ----------------
print("\n--- v0.7 policy + crowd gate ---")
o = {"probs": {"rotten": 0.01, "sprouted": 0.02, "binary_bad": 0.97}, "diameter_mm": 55.0,
     "geom": {"method": "colour-hull", "edge_touch": False}}
P.judge(o, CFG)
check("other_defect alone keeps Grade A but goes to inspector review",
      o["grade"] == "A" and any("inspector" in r for r in o["review"]), (o["grade"], o["review"]))
heap = [(40 + 44 * i + (22 if j % 2 else 0), 110 + 38 * j, 46, 0.95, 0, "healthy")
        for j in range(4) for i in range(5) if 40 + 44 * i + (22 if j % 2 else 0) < 270]
himg, hgt = make_lot(heap); hdet, hcls = mock_models(hgt, jitter=0.03)
hres, _ = P.grade_lot([himg], hdet, hcls, CFG)
g0 = hres["gates"][0]
check("packed heap rejected with retake advice", hres["status"] == "REJECTED" and "packed" in g0.get("reason", ""),
      f"hidden {g0.get('hidden_outline_frac')} touching {g0.get('touching_frac')} | {g0.get('reason', '-')}")
simg, sgt = make_lot(SPEC); sdet, scls = mock_models(sgt)
sres, _ = P.grade_lot([simg], sdet, scls, CFG)
check("normal spread (one touching pair) NOT rejected", sres["status"] == "GRADED",
      f"hidden {sres['gates'][0].get('hidden_outline_frac')} touching {sres['gates'][0].get('touching_frac')}")
o = {"probs": {"rotten": 0.01, "sprouted": 0.02, "binary_bad": 0.90}, "diameter_mm": 55.0,
     "geom": {"method": "colour-hull", "edge_touch": False}}
c3 = copy.deepcopy(CFG); c3["thresholds"]["binary_bad"] = 0.95; P.judge(o, c3)
check("v0.7.1: binary_bad just below threshold -> no review (band only on grade-changing heads)",
      o["grade"] == "A" and not o["review"], o["review"])

# ---------------- v0.7.2: leafy-green foreign object excluded ----------------
fimg, fgt = make_lot(SPEC)
cx, cy = SPEC[0][0], SPEC[0][1]                        # paint a green calyx on object 0 (a 'brinjal')
cv2.ellipse(fimg, (int((cx + 14) * PPM), int((cy - 14) * PPM)), (int(12 * PPM), int(7 * PPM)), 30, 0, 360, (60, 170, 90), -1)
fdet, fcls = mock_models(fgt)
fres, _ = P.grade_lot([fimg], fdet, fcls, CFG)
kinds = sorted(o["grade"] for o in fres["onions"])
check("v0.7.2: green-calyx object -> FOREIGN, excluded from lot count",
      kinds.count("FOREIGN") == 1 and fres["lot"]["n_onions"] == 11 and fres["lot"]["n_foreign_excluded"] == 1,
      f"{fres['lot']['n_onions']} onions, foreign {fres['lot']['n_foreign_excluded']}")
check("v0.7.2: sprouted onion (green shoot) stays SPROUTED, not foreign",
      any("sprouted" in o["defects"] for o in fres["onions"]))

# ---------------- v0.7.3: borderline sizes -> range; bad calibration -> retake ----------------
bspec = [(60, 120, 69.2, 0.95, 0, "healthy"), (150, 120, 55, 0.9, 0, "healthy"), (230, 120, 58, 0.9, 0, "healthy"),
         (60, 200, 52, 0.9, 0, "healthy"), (150, 200, 60, 0.9, 0, "healthy")]
bimg, bgt = make_lot(bspec); bdet, bcls = mock_models(bgt, jitter=0.03)
bres, _ = P.grade_lot([bimg], bdet, bcls, CFG)
rng_ = bres["lot"]["pct_gradeA_range_by_weight"]
check("v0.7.3: 69 mm onion flagged borderline, lot % given as a range",
      bres["lot"]["n_borderline"] == 1 and rng_[0] < rng_[1] and any("borderline" in r for o in bres["onions"] for r in o["review"]),
      f"range {rng_}, A {bres['lot']['pct_gradeA_by_weight']}")
c4 = copy.deepcopy(CFG); c4["gate"]["max_circle_error_mm"] = 0.01          # force the check to trip
cres, _ = P.grade_lot([bimg], bdet, bcls, c4)
check("v0.7.3: calibration circles off -> retake", cres["status"] == "REJECTED" and "circles" in cres["gates"][0].get("reason", ""),
      cres["gates"][0].get("reason", "-"))

# ---------------- v0.7.4: inspector decisions ----------------
ires, _ = P.grade_lot([bimg], bdet, bcls, CFG)                  # the 5-onion lot with one borderline 69 mm onion
a0 = ires["lot"]["pct_gradeA_by_weight"]
target = next(o for o in ires["onions"] if o.get("borderline"))
P.apply_inspector_decision(ires, target["id"], "URS", "Shalma")
ires = P.sign(ires, CFG)
check("v0.7.4: inspector marks borderline onion URS -> lot %A drops, AI grade kept, review cleared",
      ires["lot"]["pct_gradeA_by_weight"] < a0 and target["ai_grade"] == "A" and target["grade"] == "URS"
      and ires["lot"]["n_inspector_decisions"] == 1 and ires["lot"]["n_borderline"] == 0,
      f"{a0} -> {ires['lot']['pct_gradeA_by_weight']}")
check("v0.7.4: decision is inside the signed record", P.verify(ires, CFG["signing_key"]))
import os, tempfile
_pdf = os.path.join(tempfile.gettempdir(), "nikash_decision_test.pdf")          # works on Windows too
P.report_pdf(ires, P.annotate(P.rectify(bimg, CFG)[0], ires["onions"], CFG), _pdf)
check("v0.7.4: report with inspector decision renders", os.path.getsize(_pdf) > 5000)
print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")