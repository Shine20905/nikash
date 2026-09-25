"""
Real-model loaders for the Nikash pipeline, plus config assembly from the banked
checkpoints. Kept separate from pipeline.py so the pipeline stays testable without models.
"""
import copy, json, pathlib, numpy as np, cv2, yaml
from . import pipeline as P


def load_detector(weights, imgsz=1344, conf=0.25, tiled=False, tile_imgsz=640, max_tiles=48):
    """YOLO11n single-class onion detector -> callable(img_bgr) -> [(x1,y1,x2,y2,conf), ...]

    tiled=False : one pass. Right for the SHEET path: rectification fixes the scale at 5 px/mm,
                  so a 45 mm onion is ~225 px - the size the detector was trained on - and
                  imgsz 1344 covers the 1335 px capture area at ~1:1.
    tiled=True  : scale-adaptive tiling for photos WITHOUT the sheet, where onions can be
                  small in frame (see tiling.py). Last run's tiling info: detector.last_info"""
    from ultralytics import YOLO
    from .tiling import adaptive_detect
    model = YOLO(str(weights))

    def _run(imgs, sz):
        rs = model.predict(imgs, imgsz=sz, conf=conf, verbose=False)
        out = []
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 5), np.float32)); continue
            out.append(np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]))
        return out

    def single(img_bgr):
        return [tuple(map(float, b)) for b in _run([img_bgr], imgsz)[0]]

    def tiled_fn(img_bgr):
        boxes, info = adaptive_detect(img_bgr,
                                      predict_full=lambda im: _run([im], 1280)[0],
                                      predict_tiles=lambda tiles: _run(tiles, tile_imgsz),
                                      max_tiles=max_tiles)
        tiled_fn.last_info = info
        return boxes

    return tiled_fn if tiled else single


def load_classifier(ckpt_path, device="cpu"):
    """MobileNetV3-Small, 3 sigmoid heads -> callable(list of RGB crops) -> (n,3) probs.
    Preprocessing mirrors B3 eval: crop -> 256x256 (INTER_AREA) -> 224x224 -> ImageNet norm."""
    import torch, torch.nn as nn
    from torchvision import models
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    heads = ck["heads"]
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(heads))
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def classifier(crops_rgb):
        if not crops_rgb:
            return np.zeros((0, len(heads)), np.float32)
        batch = []
        for c in crops_rgb:
            x = cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)
            x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            batch.append(((x - mean) / std).transpose(2, 0, 1))
        with torch.no_grad():
            out = torch.sigmoid(m(torch.from_numpy(np.stack(batch)).to(device))).cpu().numpy()
        return out
    return classifier, heads, {"epoch": int(ck.get("epoch", -1))}


def build(det_dir, def_dir, cal_dir, variety="N-53", overrides=None):
    """Assemble detector, classifier and config from the three banked Kaggle datasets."""
    det_dir, def_dir, cal_dir = map(pathlib.Path, (det_dir, def_dir, cal_dir))
    cfg = copy.deepcopy(P.DEFAULT_CONFIG)

    metrics = json.loads((def_dir / "metrics.json").read_text())
    cfg["thresholds"] = {k: round(float(v), 4) for k, v in metrics["thresholds"].items()}
    cfg["thresholds"].update(cfg.get("threshold_overrides", {}))       # real-domain recalibration wins

    cal = yaml.safe_load((cal_dir / "calibration.yaml").read_text())
    cfg["varieties"] = {k: {"density_g_cm3": float(v["density_g_cm3"]), "axis_ratio_k": float(v["axis_ratio_k"])}
                        for k, v in cal["varieties"].items()}
    cfg["variety"] = variety
    g = cal["grading"]
    cfg["grading"]["size_min_mm"] = float(g["size_min_mm"])
    cfg["grading"]["size_max_mm"] = float(g["size_max_mm"])
    d = str(g.get("diameter_definition", ""))
    if d in ("max_equatorial", "equivalent"):
        cfg["grading"]["diameter_definition"] = d
    else:
        cfg["grading"]["diameter_definition"] = "max_equatorial"
        cfg["grading"]["diameter_definition_note"] = "provisional default - calibration.yaml marks this OPEN"

    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    detector = load_detector(det_dir / "best.pt", imgsz=1344)          # sheet path: fixed scale
    classifier, heads, cinfo = load_classifier(def_dir / "best.pt")
    assert heads == ["rotten", "sprouted", "binary_bad"], f"unexpected head order {heads}"
    det_manifest = json.loads((det_dir / "MANIFEST.json").read_text())
    info = {"detector": f"yolo11n mAP50 {det_manifest.get('metrics_val', {}).get('mAP50')}",
            "defect": f"mobilenet_v3_small epoch {cinfo['epoch']}",
            "calibration": f"{variety} (literature)"}
    return detector, classifier, cfg, info
