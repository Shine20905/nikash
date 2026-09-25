"""Nikash A3 reference sheet generator (v3).

    python make_sheet.py            -> nikash_sheet_A3.pdf        (white capture area, B/W printers)
    python make_sheet.py --blue     -> nikash_sheet_A3_blue.pdf   (blue capture area, colour printers)

Geometry is IDENTICAL in both variants and must stay identical to nikash/pipeline.py:
markers, circles, ruler and capture-area rectangle do not move.

Why blue: onions are red / pink / purple / pale / yellow. Cyan-blue sits far from all of them in
colour space, so the colour-difference mask separates every onion type - including pale, papery
skins and shiny highlights that vanish into white paper. Shadows on blue stay blue, so they are
still excluded. Green is avoided on purpose: it would hide sprouts.

v3 changes over v2: optional blue field; everything else unchanged.
v2 fixes over v1: 15 mm margins, ruler clear of markers, 7 mm quiet zone, title below markers.
"""
import sys, pathlib, cv2
from reportlab.lib.pagesizes import A3
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

BLUE = "--blue" in sys.argv
FIELD_RGB = (70 / 255, 150 / 255, 215 / 255)      # mid cyan-blue; a tint, not a solid primary, to save ink

OUT = pathlib.Path(".")
MK = OUT / "_markers"; MK.mkdir(exist_ok=True)
try:
    D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    gen = lambda i: cv2.aruco.generateImageMarker(D, i, 600)
except AttributeError:
    D = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    gen = lambda i: cv2.aruco.drawMarker(D, i, 600)
for i in range(4):
    cv2.imwrite(str(MK / f"m{i}.png"), gen(i))

PW, PH = 297, 420
MARGIN, MS, QZ = 15, 40, 7
POS = {0: (MARGIN, PH - MARGIN - MS), 1: (PW - MARGIN - MS, PH - MARGIN - MS),
       2: (MARGIN, MARGIN), 3: (PW - MARGIN - MS, MARGIN)}
CA = dict(x=MARGIN, y=150, w=PW - 2 * MARGIN, h=185)

PDF = str(OUT / ("nikash_sheet_A3_blue.pdf" if BLUE else "nikash_sheet_A3.pdf"))
c = canvas.Canvas(PDF, pagesize=A3)

# capture area first (so nothing else is painted over by it)
if BLUE:
    c.setFillColorRGB(*FIELD_RGB); c.setStrokeColorRGB(*FIELD_RGB)
    c.rect(CA["x"] * mm, CA["y"] * mm, CA["w"] * mm, CA["h"] * mm, fill=1, stroke=0)
else:
    c.setDash(3, 3); c.setStrokeColorRGB(.65, .65, .65); c.setLineWidth(0.5)
    c.rect(CA["x"] * mm, CA["y"] * mm, CA["w"] * mm, CA["h"] * mm, fill=0, stroke=1)
    c.setDash()

for i, (x, y) in POS.items():
    c.setFillColorRGB(1, 1, 1); c.setStrokeColorRGB(1, 1, 1)
    c.rect((x - QZ) * mm, (y - QZ) * mm, (MS + 2 * QZ) * mm, (MS + 2 * QZ) * mm, fill=1, stroke=0)
    c.drawImage(str(MK / f"m{i}.png"), x * mm, y * mm, MS * mm, MS * mm)
c.setFillColorRGB(.45, .45, .45); c.setFont("Helvetica", 6)
c.drawString(MARGIN * mm, (PH - MARGIN + 2) * mm, "id 0")
c.drawRightString((PW - MARGIN) * mm, (PH - MARGIN + 2) * mm, "id 1")
c.drawString(MARGIN * mm, (MARGIN - 4.5) * mm, "id 2")
c.drawRightString((PW - MARGIN) * mm, (MARGIN - 4.5) * mm, "id 3")

c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 14)
c.drawCentredString(PW / 2 * mm, 352 * mm, "NIKASH  -  onion grading reference sheet")
c.setFont("Helvetica", 8.5)
c.drawCentredString(PW / 2 * mm, 345 * mm,
                    "Print A3 at 100%  -  'Fit to page' OFF  -  check the 150 mm ruler with a real ruler before use")
c.setFillColorRGB(.45, .45, .45); c.setFont("Helvetica", 8.5)
c.drawCentredString(PW / 2 * mm, 338 * mm,
                    "place onions on the " + ("blue area" if BLUE else "dashed area") +
                    "  -  SINGLE LAYER, keep the corner markers and circles uncovered")

c.setFillColorRGB(0, 0, 0); c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(0.7)
CIRCLES, CY = [35, 45, 55, 70], 105
gap = (PW - 2 * MARGIN - sum(CIRCLES)) / (len(CIRCLES) + 1)
x = MARGIN + gap
for dia in CIRCLES:
    cx = x + dia / 2
    c.circle(cx * mm, CY * mm, (dia / 2) * mm, fill=0, stroke=1)
    c.setFont("Helvetica-Bold", 9); c.drawCentredString(cx * mm, (CY - 1.5) * mm, f"{dia} mm")
    x += dia + gap
c.setFillColorRGB(.45, .45, .45); c.setFont("Helvetica", 7.5)
c.drawCentredString(PW / 2 * mm, 66 * mm, "calibration circles  -  keep visible, do not cover with onions")

RX, RY, RL = 68, 38, 150
c.setFillColorRGB(0, 0, 0); c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(0.5)
c.line(RX * mm, RY * mm, (RX + RL) * mm, RY * mm)
for i in range(RL + 1):
    t = 5.0 if i % 10 == 0 else (3.0 if i % 5 == 0 else 1.6)
    c.setLineWidth(0.5 if i % 10 == 0 else 0.3)
    c.line((RX + i) * mm, RY * mm, (RX + i) * mm, (RY + t) * mm)
    if i % 10 == 0:
        c.setFont("Helvetica", 6); c.drawCentredString((RX + i) * mm, (RY + 6.5) * mm, str(i))
c.setFillColorRGB(.45, .45, .45); c.setFont("Helvetica", 7.5)
c.drawCentredString((RX + RL / 2) * mm, (RY - 5.5) * mm, "millimetres  -  lay an onion alongside to read its diameter")

c.showPage(); c.save()
print("wrote", PDF)
