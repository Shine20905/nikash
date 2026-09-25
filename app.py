"""
Nikash - onion quality grading app (SIH 2026, PS 26031).

    python app.py            start the app (laptop + phone links + QR code)
    python app.py --check    grade test.jpg / test.dng in this folder once and exit


The AI grades every onion (size band, rot, sprouting, damage, dark-patch %) and the inspector can
confirm or overrule any call; the AI grade stays beside the decision in the signed record.

Folder:  app.py, nikash/ (code v0.8+), models/detector, models/defect, models/calibration
"""
import sys, time, json, base64, pathlib, datetime, socket, uuid, threading
import cv2, numpy as np
from nikash import pipeline as P, models as M, __version__ as CODE_VERSION

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports"; OUT.mkdir(exist_ok=True)
MAX_SIDE = 2560          # A3 sheet at 5 px/mm needs ~2100 px; 2560 keeps margin, 2.5x less to upload/decode
SCANS, LOCK = {}, threading.Lock()        # scan_id -> {res, rect, stem}; kept in memory for inspector decisions


# ------------------------------------------------------------------ models
def load_models():
    t = time.time()
    det, cls, cfg, info = M.build(ROOT / "models" / "detector", ROOT / "models" / "defect",
                                  ROOT / "models" / "calibration")
    print(f"models loaded in {time.time() - t:.1f} s | code {CODE_VERSION}")
    t = time.time()                                   # warm-up: first inference is always slow (lazy init)
    dummy = np.full((1335, 925, 3), 235, np.uint8)
    det(dummy); cls([dummy[:224, :224, ::-1].copy()])
    print(f"warm-up done in {time.time() - t:.1f} s")
    return det, cls, cfg, info


# ------------------------------------------------------------------ photo reading (JPEG/PNG/HEIC-as-JPEG/iPhone DNG)
def _largest_embedded_jpeg(buf):
    best, off = None, 0
    while True:
        off = buf.find(b"\xff\xd8\xff", off)
        if off < 0: return best
        im = cv2.imdecode(np.frombuffer(buf, np.uint8, offset=off), cv2.IMREAD_COLOR)
        if im is not None and (best is None or im.size > best.size): best = im
        off += 3

def decode_photo(buf, name="photo"):
    im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if im is None or name.lower().endswith(".dng"):
        emb = _largest_embedded_jpeg(buf)
        im = emb if emb is not None else im
    if im is None:
        raise ValueError(f"Cannot read {name}. Use JPEG (iPhone: Settings > Camera > Formats > Most Compatible).")
    s = MAX_SIDE / max(im.shape[:2])
    if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


# ------------------------------------------------------------------ result -> phone JSON
def view(scan_id, S):
    res, cfg = S["res"], S["cfg"]
    ann = P.annotate(S["rect"], res["onions"], cfg, view=S["view"])
    h, w = ann.shape[:2]; s = min(1.0, 1400 / w)
    small = cv2.resize(ann, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    S["ann"] = ann
    L, cal = res["lot"], res["calibration_check"]
    return {
        "scan_id": scan_id, "lot_id": res["meta"]["lot_id"], "status": "GRADED", "photos": S["photos"],
        "seconds": S["seconds"],
        "pct_a": L["pct_gradeA_by_weight"], "pct_urs": L["pct_URS_by_weight"], "pct_unfit": L["pct_unfit_by_weight"],
        "bucket": L["second_bucket_label"], "n": L["n_onions"], "mass_g": round(L["total_mass_g_est"]),
        "n_review": L["n_review"], "n_decided": L.get("n_inspector_decisions", 0),
        "n_foreign": L.get("n_foreign_excluded", 0), "n_borderline": L.get("n_borderline", 0),
        "range_a": L.get("pct_gradeA_range_by_weight"), "calib": cal["status"],
        "calib_err": round(cal["worst_circle_error_mm"], 2) if cal["worst_circle_error_mm"] is not None else None,
        "signature": res["signature"]["payload_sha256"][:16],
        "image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode(),
        "pdf": f"/r/{S['stem']}.pdf?v={S['version']}", "json": f"/r/{S['stem']}_signed.json?v={S['version']}",
        "onions": [{"id": o["id"], "d": round(o["diameter_mm"], 1), "m": round(o["mass_g"]), "size": o["size_status"],
                    "grade": o["grade"], "ai_grade": o.get("ai_grade", o["grade"]), "defects": o["defects"],
                    "review": o["review"], "decision": o.get("inspector_decision"),
                    "dark": o.get("dark_pct", 0.0)} for o in res["onions"]],
    }

def save(S):
    """Write the signed JSON now; the PDF is (re)built only when someone opens it."""
    (OUT / f"{S['stem']}_signed.json").write_text(json.dumps(S["res"], default=float, indent=1))
    pdf = OUT / f"{S['stem']}.pdf"
    if pdf.exists(): pdf.unlink()


# ------------------------------------------------------------------ grading
def grade(imgs, lot_id, inspector, models):
    det, cls, cfg, info = models
    lot_id = "".join(c for c in (lot_id or "").strip() if c.isalnum() or c in "-_ ")[:40].strip() \
        or datetime.datetime.now().strftime("LOT-%Y%m%d-%H%M%S")
    meta = {"lot_id": lot_id, "inspector": (inspector or "-").strip()[:40] or "-", "centre": "demo",
            "app": f"nikash {CODE_VERSION}"}
    t = time.time()
    res, ann = P.grade_lot(imgs, det, cls, cfg, meta=meta, model_info=info)
    sec = round(time.time() - t, 1)
    photos = [{"n": g.get("view", 0) + 1, "ok": bool(g.get("ok")), "reason": g.get("reason", ""),
               "tilt": g.get("tilt_deg"),
               "height_cm": round(g["camera_height_mm"] / 10) if g.get("camera_height_mm") else None}
              for g in res.get("gates", [])]
    if res["status"] != "GRADED":
        return {"status": res["status"], "photos": photos, "seconds": sec}
    vi = res["views_accepted"][0]
    rect, _ = P.rectify(imgs[vi], cfg)                   # kept so the image can be redrawn after decisions
    scan_id = uuid.uuid4().hex[:12]
    S = {"res": res, "rect": rect, "view": vi, "cfg": cfg, "photos": photos, "seconds": sec, "version": 0,
         "stem": f"{lot_id}_{datetime.datetime.now():%H%M%S}_{scan_id[:4]}".replace(" ", "_")}
    with LOCK: SCANS[scan_id] = S
    save(S)
    L = res["lot"]
    print(f"graded {lot_id}: {L['n_onions']} onions, A {L['pct_gradeA_by_weight']:.1f}%, "
          f"{L['n_review']} for inspector, {sec} s")
    return view(scan_id, S)

def decide(scan_id, onion_id, decision, inspector, cfg):
    with LOCK:
        S = SCANS.get(scan_id)
        if S is None: raise ValueError("scan not found - grade again")
        o = P.apply_inspector_decision(S["res"], onion_id, decision, inspector)
        S["res"] = P.sign(S["res"], cfg)                  # decision is part of the signed record
        S["version"] += 1
        save(S)
    print(f"inspector {inspector or '-'}: onion {onion_id} {o['ai_grade']} -> {decision}")
    return view(scan_id, S)


# ------------------------------------------------------------------ web page
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#5b1f45"><title>Nikash</title>
<style>
:root{--brand:#5b1f45;--brand2:#8a2f63;--bg:#f6f3f1;--card:#fff;--ink:#1f1a1d;--mute:#6f6570;--line:#e7e0e3;
      --a:#2e7d32;--urs:#e65100;--unfit:#c62828;--rev:#a2379a;--r:18px}
@media (prefers-color-scheme:dark){:root{--bg:#141013;--card:#1f191d;--ink:#f3eef1;--mute:#a79ca4;--line:#342a31}}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;padding:calc(18px + env(safe-area-inset-top)) 20px 22px}
header .row{display:flex;align-items:baseline;gap:10px;max-width:760px;margin:auto}
header h1{margin:0;font-size:26px;letter-spacing:.3px}
header .hi{opacity:.8;font-size:18px}
header p{margin:4px auto 0;max-width:760px;opacity:.85;font-size:14px}
main{max-width:760px;margin:auto;padding:16px 16px 40px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:16px;margin-bottom:14px}
h2{font-size:17px;margin:0 0 10px}
label{font-size:13px;color:var(--mute);display:block;margin:0 0 4px}
input[type=text]{width:100%;font-size:16px;padding:11px 12px;border:1px solid var(--line);border-radius:12px;background:transparent;color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;border:0;border-radius:14px;font-size:17px;font-weight:600;padding:15px;cursor:pointer;text-decoration:none}
.primary{background:var(--brand);color:#fff}.primary:disabled{opacity:.4}
.ghost{background:transparent;color:var(--brand2);border:1.5px solid var(--brand2)}
@media (prefers-color-scheme:dark){.ghost{color:#e59ac8;border-color:#e59ac8}}
.stack>*+*{margin-top:10px}
.tips{margin:0;padding-left:18px;color:var(--mute);font-size:14px}
.thumbs{display:flex;gap:10px;flex-wrap:wrap}
.thumb{position:relative;width:92px;height:92px;border-radius:12px;overflow:hidden;background:#ddd;display:flex;align-items:center;justify-content:center;font-size:12px;color:#555}
.thumb img{width:100%;height:100%;object-fit:cover}
.thumb button{position:absolute;top:4px;right:4px;width:26px;height:26px;border-radius:50%;border:0;background:rgba(0,0,0,.6);color:#fff;font-size:15px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.stat{border-radius:16px;padding:14px;color:#fff}
.stat .v{font-size:34px;font-weight:700;line-height:1.1}.stat .k{font-size:13px;opacity:.9}
.sa{background:var(--a)}.su{background:var(--urs)}
.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}
.mini div{border:1px solid var(--line);border-radius:12px;padding:8px 10px}.mini b{display:block;font-size:19px}.mini span{font-size:12px;color:var(--mute)}
.res img{width:100%;border-radius:12px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--mute);margin-top:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
.onion{padding:10px 0;border-top:1px solid var(--line)}
.onion:first-child{border-top:0}
.orow{display:flex;align-items:center;gap:10px}
.badge{min-width:34px;height:34px;border-radius:50%;color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:14px}
.orow .t{flex:1}.orow .t small{color:var(--mute);display:block;font-size:12px}
.pill{font-size:12px;font-weight:600;padding:3px 9px;border-radius:99px;color:#fff;white-space:nowrap}
.flag{color:var(--rev);font-size:12px}
.decide{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 0 44px}
.decide button{border:0;border-radius:10px;padding:10px 4px;font-size:14px;font-weight:700;color:#fff;cursor:pointer}
.decide .dA{background:var(--a)}.decide .dU{background:var(--urs)}.decide .dX{background:var(--unfit)}
.done{margin:6px 0 0 44px;font-size:12px;font-weight:600;color:var(--brand2)}
@media (prefers-color-scheme:dark){.done{color:#e59ac8}}
.todo{border-radius:12px;padding:10px 12px;font-weight:600;font-size:14px;background:rgba(162,55,154,.12);color:var(--rev)}
.bad{border-color:var(--unfit)} .bad h2{color:var(--unfit)}
.meta{font-size:12px;color:var(--mute)}
.busy{display:none;text-align:center;padding:30px 10px}.spin{width:44px;height:44px;border:4px solid var(--line);border-top-color:var(--brand2);border-radius:50%;animation:s 1s linear infinite;margin:0 auto 12px}
@keyframes s{to{transform:rotate(360deg)}}
.hide{display:none}
</style></head><body>
<header><div class="row"><h1>Nikash</h1><span class="hi">निकष</span></div>
<p>Onion quality grading — Grade A / URS in seconds</p></header>
<main>
 <section id="scan">
  <div class="card stack">
   <div class="two"><div><label>Lot ID</label><input id="lot" type="text" placeholder="auto"></div>
   <div><label>Inspector</label><input id="insp" type="text" placeholder="your name"></div></div>
  </div>
  <div class="card stack">
   <h2>Photograph the lot</h2>
   <ol class="tips"><li>Onions in <b>one layer</b> on the sheet, small gaps</li>
   <li>All <b>4 corner markers</b> and the circles visible</li><li>Hold the phone <b>flat</b>, straight above</li></ol>
   <button class="btn primary" id="camBtn">📷 Scan onions</button>
   <button class="btn ghost hide" id="galBtn">Choose from photos (test mode)</button>
   <input id="cam" type="file" accept="image/*" capture="environment" hidden>
   <input id="gal" type="file" accept="image/*,.dng,.DNG" multiple hidden>
   <div class="thumbs" id="thumbs"></div>
   <div class="meta" id="count">Live capture only — gallery photos are not accepted.</div>
   <button class="btn primary" id="go" disabled>Grade lot</button>
  </div>
 </section>
 <div class="card busy" id="busy"><div class="spin"></div><div id="busyTxt">Checking photo…</div></div>
 <section id="result" class="hide"></section>
</main>
<script>
const $=s=>document.querySelector(s); let photos=[], J=null;
const GC={A:"var(--a)",URS:"var(--urs)",UNFIT:"var(--unfit)",FOREIGN:"#8a8a8a"};
const NICE={"uncertain:rotten":"borderline rotten score","uncertain:sprouted":"borderline sprout score","uncertain:binary_bad":"borderline defect score"};
if(new URLSearchParams(location.search).get("dev")==="1") $("#galBtn").classList.remove("hide");
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function nice(a){return a.map(x=>NICE[x]||x).join("; ")}
function label(g,b){return g==="URS"?b:(g==="FOREIGN"?"Not onion":(g==="UNFIT"?"Unfit":g))}
$("#camBtn").onclick=()=>$("#cam").click(); $("#galBtn").onclick=()=>$("#gal").click();
$("#cam").onchange=e=>addFiles(e.target.files); $("#gal").onchange=e=>addFiles(e.target.files);
async function shrink(file){
  if(/\.dng$/i.test(file.name)) return file;
  try{const url=URL.createObjectURL(file); const img=new Image(); img.src=url; await img.decode();
      const s=Math.min(1,2560/Math.max(img.naturalWidth,img.naturalHeight));
      const c=document.createElement("canvas"); c.width=Math.round(img.naturalWidth*s); c.height=Math.round(img.naturalHeight*s);
      c.getContext("2d").drawImage(img,0,0,c.width,c.height); URL.revokeObjectURL(url);
      const b=await new Promise(r=>c.toBlob(r,"image/jpeg",0.92)); return new File([b],file.name.replace(/\.\w+$/,"")+".jpg",{type:"image/jpeg"});
  }catch(e){return file}
}
async function addFiles(list){ for(const f of list){ if(photos.length>=3) break; photos.push({file:await shrink(f),name:f.name}); }
  $("#cam").value=""; $("#gal").value=""; draw(); }
function draw(){
  $("#thumbs").innerHTML=""; photos.forEach((p,i)=>{const d=document.createElement("div"); d.className="thumb";
    if(!/\.dng$/i.test(p.name)){const im=document.createElement("img"); im.src=URL.createObjectURL(p.file); d.appendChild(im);} else d.textContent="RAW";
    const x=document.createElement("button"); x.textContent="×"; x.onclick=()=>{photos.splice(i,1);draw()}; d.appendChild(x); $("#thumbs").appendChild(d);});
  $("#go").disabled=!photos.length;
  $("#count").textContent=photos.length?`${photos.length}/3 photo${photos.length>1?"s":""} ready`:"Live capture only — gallery photos are not accepted.";
}
$("#go").onclick=async()=>{
  const fd=new FormData(); photos.forEach(p=>fd.append("files",p.file,p.file.name||p.name));
  fd.append("lot_id",$("#lot").value); fd.append("inspector",$("#insp").value);
  $("#scan").classList.add("hide"); $("#result").classList.add("hide"); $("#busy").style.display="block";
  const steps=["Uploading…","Checking photo & markers…","Finding onions…","Measuring sizes…","Checking defects…","Writing report…"]; let k=0;
  const tm=setInterval(()=>{$("#busyTxt").textContent=steps[Math.min(++k,steps.length-1)]},900); $("#busyTxt").textContent=steps[0];
  try{const r=await fetch("api/grade",{method:"POST",body:fd}); show(await r.json());}
  catch(e){show({status:"ERROR",error:String(e)})}
  clearInterval(tm); $("#busy").style.display="none";
};
async function decide(id,g){
  const fd=new FormData(); fd.append("scan_id",J.scan_id); fd.append("onion_id",id); fd.append("decision",g); fd.append("inspector",$("#insp").value);
  document.querySelectorAll(`#o${id} .decide button`).forEach(b=>b.disabled=true);
  try{const r=await fetch("api/decide",{method:"POST",body:fd}); const j=await r.json(); if(j.status==="GRADED") show(j,true); else alert(j.error||"error");}
  catch(e){alert(String(e))}
}
function show(j,keepScroll){
  J=j; const R=$("#result"); R.classList.remove("hide");
  if(j.status!=="GRADED"){
    const why=j.error?`<p>${esc(j.error)}</p>`:(j.photos||[]).map(p=>`<p><b>Photo ${p.n}:</b> ${p.ok?"accepted":esc(p.reason)}</p>`).join("");
    R.innerHTML=`<div class="card bad stack"><h2>Retake needed</h2>${why}
      <ol class="tips"><li>Spread onions in one layer with gaps</li><li>Keep all 4 corner markers in view</li><li>Hold the phone flat above the sheet</li></ol>
      <button class="btn primary" onclick="again(true)">Retake photos</button></div>`; window.scrollTo(0,0); return;}
  const on=j.onions.map(o=>{
    const d=o.decision, flagged=!d&&o.grade!=="FOREIGN"&&(o.review.length>0||o.defects.length>0);   // clear size-only calls need no buttons
    return `<div class="onion" id="o${o.id}"><div class="orow"><div class="badge" style="background:${GC[o.grade]}">${o.id}</div>
     <div class="t"><b>${o.d} mm</b> · ${o.m} g<small>${o.grade==="FOREIGN"?"excluded from lot %":o.size.replace("_"," ")}${o.defects.length?" · "+o.defects.map(x=>x.replace("_"," ")).join(", "):""}${o.dark>=1?` · dark ${o.dark.toFixed(0)}%`:""}</small>
     ${o.review.length&&!d?`<div class="flag">⚑ ${esc(nice(o.review))}</div>`:""}</div>
     <span class="pill" style="background:${GC[o.grade]}">${label(o.grade,j.bucket)}</span></div>
     ${flagged?`<div class="decide"><button class="dA" onclick="decide(${o.id},'A')">${o.grade==="A"?"Keep A":"Make A"}</button>
        <button class="dU" onclick="decide(${o.id},'URS')">${esc(j.bucket==="URS"?"URS":"Below A")}</button>
        <button class="dX" onclick="decide(${o.id},'UNFIT')">Unfit</button></div>`:""}
     ${d?`<div class="done">✓ Inspector${d.by&&d.by!=="-"?" "+esc(d.by):""}: ${label(d.ai_grade,j.bucket)} → ${label(d.final_grade,j.bucket)}</div>`:""}</div>`}).join("");
  const ok=j.photos.filter(p=>p.ok).map(p=>`tilt ${Math.round(p.tilt)}°, ${p.height_cm} cm`).join(" · ");
  const todo=j.n_review>0?`<div class="todo">⚑ ${j.n_review} AI call${j.n_review>1?"s":""} to confirm or overrule — see "Per onion"</div>`
    :(j.n_decided?`<div class="meta">✓ All flags resolved · ${j.n_decided} inspector decision${j.n_decided>1?"s":""} recorded in the signed report</div>`:"");
  R.innerHTML=`
  <div class="card stack"><div class="meta">Lot <b>${esc(j.lot_id)}</b></div>
    <div class="stats"><div class="stat sa"><div class="v">${j.pct_a.toFixed(1)}%</div><div class="k">Grade A · by weight</div></div>
    <div class="stat su"><div class="v">${j.pct_urs.toFixed(1)}%</div><div class="k">${esc(j.bucket)} · by weight</div></div></div>
    <div class="mini"><div><b>${j.n}</b><span>onions</span></div><div><b>${j.mass_g} g</b><span>est. weight</span></div>
    <div><b style="color:var(--rev)">${j.n_review}</b><span>inspector checks</span></div></div>
    ${(j.range_a&&j.range_a[1]-j.range_a[0]>=0.5)?`<div class="meta" style="color:var(--rev)"><b>Grade A ${j.range_a[0].toFixed(0)}–${j.range_a[1].toFixed(0)}%</b> depending on inspector checks${j.n_borderline?` · ${j.n_borderline} within 2 mm of a size limit`:""}</div>`:""}
    ${j.n_foreign>0?`<div class="meta">${j.n_foreign} non-onion object${j.n_foreign>1?"s":""} excluded from the lot</div>`:""}
    ${j.pct_unfit>0?`<div class="meta" style="color:var(--unfit)">of which unfit: ${j.pct_unfit.toFixed(1)}%</div>`:""}
    ${todo}
  </div>
  <div class="card res"><img src="${j.image}" alt="annotated lot">
    <div class="legend"><span><i class="dot" style="background:var(--a)"></i>Grade A</span><span><i class="dot" style="background:var(--urs)"></i>${esc(j.bucket)}</span>
    <span><i class="dot" style="background:var(--unfit)"></i>Unfit</span><span><i class="dot" style="background:#8a8a8a"></i>Not onion</span><span><i class="dot" style="background:var(--rev)"></i>Inspector check</span></div></div>
  <div class="card"><h2>Per onion</h2>${on}</div>
  <div class="card stack"><a class="btn primary" href="${j.pdf}" target="_blank">⬇ Report (PDF)</a>
    <a class="btn ghost" href="${j.json}" target="_blank">Signed record (JSON)</a>
    <div class="meta">Calibration ${j.calib} (worst circle error ${j.calib_err} mm) · ${ok} · ${j.seconds} s · signature ${j.signature}…</div>
    <button class="btn ghost" onclick="again(false)">New lot</button></div>`;
  if(!keepScroll) window.scrollTo(0,0);
}
function again(keepLot){photos=[];draw(); if(!keepLot) $("#lot").value=""; $("#result").classList.add("hide"); $("#scan").classList.remove("hide"); window.scrollTo(0,0);}
</script></body></html>"""


# ------------------------------------------------------------------ server
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except OSError:
        return "127.0.0.1"

def serve(models, port=7860):
    import gradio as gr
    from fastapi import UploadFile, File, Form
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.routing import APIRoute
    from typing import List
    cfg = models[2]

    def page():
        return HTMLResponse(PAGE)

    def api_grade(files: List[UploadFile] = File(...), lot_id: str = Form(""), inspector: str = Form("")):
        try:
            t0 = time.time()
            imgs = [decode_photo(f.file.read(), f.filename or "photo") for f in files[:3]]
            t1 = time.time()
            out = grade(imgs, lot_id, inspector, models)
            print(f"   timing: receive+decode {t1 - t0:.1f} s | grade+draw {time.time() - t1:.1f} s | "
                  f"photo {imgs[0].shape[1]}x{imgs[0].shape[0]}")
            return JSONResponse(out)
        except Exception as e:                                   # never leave the phone hanging
            print("ERROR:", repr(e))
            return JSONResponse({"status": "ERROR", "error": str(e)})

    def api_decide(scan_id: str = Form(...), onion_id: int = Form(...), decision: str = Form(...),
                   inspector: str = Form("")):
        try:
            return JSONResponse(decide(scan_id, onion_id, decision, inspector.strip()[:40], cfg))
        except Exception as e:
            print("ERROR:", repr(e))
            return JSONResponse({"status": "ERROR", "error": str(e)})

    def report(name: str):
        p = (OUT / name).resolve()
        if p.parent == OUT.resolve() and not p.exists() and p.suffix == ".pdf":
            S = next((S for S in SCANS.values() if f"{S['stem']}.pdf" == name), None)
            if S: P.report_pdf(S["res"], S["ann"], str(p))      # built on first open, rebuilt after decisions
        if p.parent != OUT.resolve() or not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, filename=p.name)

    with gr.Blocks(title="Nikash") as stub:                       # Gradio only provides the server + share tunnel
        gr.Markdown("Nikash")
    stub.launch(server_name="0.0.0.0", server_port=port, share=True, prevent_thread_lock=True, quiet=True)
    for r in (APIRoute("/", page, methods=["GET"]),
              APIRoute("/api/grade", api_grade, methods=["POST"]),
              APIRoute("/api/decide", api_decide, methods=["POST"]),
              APIRoute("/r/{name}", report, methods=["GET"])):
        stub.app.router.routes.insert(0, r)
    print("\n" + "=" * 60)
    print(f"  Laptop            : http://127.0.0.1:{port}")
    print(f"  Phone, same Wi-Fi : http://{lan_ip()}:{port}")
    print(f"  Phone, anywhere   : {stub.share_url or '(share link not available - use the Wi-Fi address)'}")
    print("=" * 60)
    phone = stub.share_url or f"http://{lan_ip()}:{port}"
    try:
        import qrcode, os, tempfile
        q = qrcode.QRCode(border=1); q.add_data(phone); q.make()
        qr_path = pathlib.Path(tempfile.gettempdir()) / f"nikash_qr_{int(time.time())}.png"   # fresh file each run
        q.make_image().save(qr_path)
        print(f"  Phone link: {phone}")
        print("  QR code opening now - scan it with the iPhone camera")
        if hasattr(os, "startfile"): os.startfile(str(qr_path))
    except ImportError:
        print("  tip: pip install qrcode[pil]  -> makes a QR code to open the app on the phone")
    except Exception as e:
        print("  (could not make QR code:", e, ")")
    print("\n  Ctrl+C to stop\n")
    stub.block_thread()


if __name__ == "__main__":
    models = load_models()
    if "--check" in sys.argv:
        photo = next((p for p in ROOT.iterdir() if p.stem.lower() == "test"), None)
        assert photo, "put a sheet photo named test.jpg / test.dng in this folder"
        out = grade([decode_photo(photo.read_bytes(), photo.name)], "CHECK", "self-test", models)
        out.pop("image", None); out.pop("onions", None)
        print(json.dumps(out, indent=1, default=float)); sys.exit(0)
    serve(models)