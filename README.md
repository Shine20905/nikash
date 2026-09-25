# Nikash (निकष) — AI onion quality grading

Smart India Hackathon 2026 · PS 26031 · Department of Consumer Affairs

A phone photo of onions on a printed reference sheet → per-onion size, estimated weight,
defect check, and a lot-level **% Grade A / % URS by weight**, with a signed PDF report in seconds.

## How it works
1. **Capture gate** – 4 ArUco markers on an A3 sheet: rectify to 5 px/mm, reject blur, glare,
   tilt > 20°, and packed heaps; printed calibration circles self-check every photo.
2. **Detection** – YOLO11n (single class: onion).
3. **Size** – colour segmentation + watershed for touching onions, ellipse fit,
   parallax correction from the camera height (onions stand ~D/2 above the paper).
4. **Defects** – MobileNetV3-Small, 3 heads: rotten → UNFIT, sprouted → URS,
   other anomaly → inspector review (never auto-downgrades).
5. **Rules** – configurable size band (default 45–70 mm), mass-weighted lot percentages.
6. **Report** – annotated image, per-onion table, HMAC-signed JSON record, PDF.

## Results so far (prototype, one kitchen lot)
| Check | Result |
|---|---|
| Printed-circle calibration error | ≤ 0.61 mm on every accepted photo |
| Diameter vs ruler (6 onions × 5 photos, verified readings) | MAE 1.85 mm, bias +0.13 mm |
| Diameter incl. 3 disputed ruler readings | MAE 3.9 mm |
| Parallax correction | halves the error (7.4 → 3.9 mm) |
| Lot % Grade A (truth 100 %) | 100 % on all 5 labelled photos |
| Packed heap / tilted phone | rejected with retake advice |
| Processing time | ≈ 3–4 s per photo on a laptop CPU |

Limitations: one variety and one lot tested; foreign objects are only caught when they have
leafy-green parts; black mould is not a trained class; size band pending DoCA/AGMARK confirmation.

## Run it
```
nikash-app/
  app.py
  nikash/            code
  models/detector/   best.pt, MANIFEST.json       (Kaggle: nikash-b2-det-ckpt)
  models/defect/     best.pt, metrics.json        (Kaggle: nikash-b3-defect-ckpt-ft)
  models/calibration/calibration.yaml             (Kaggle: nikash-b4-calibration)
```
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Mac/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m nikash.test_pipeline    # 46 checks, should end with ALL PASSED
python app.py                     # prints laptop link, phone link and a QR code
```
Print the reference sheet: `python nikash/make_sheet.py` → A3, 100 % scale, check the 150 mm ruler.

Model weights are not in this repository.