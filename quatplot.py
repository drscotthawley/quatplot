"""quatplot — quaternion-based interactive 3D plots for Jupyter.

Plotly-style usage:

    import numpy as np
    import quatplot as qp

    x = np.linspace(-3, 3, 41)
    y = np.linspace(-3, 3, 41)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    qp.plot(qp.Surface(x, y, Z), theme="dark", height=520)

Rotation uses quaternions (ganja.js Algebra(0,2) when the CDN is reachable,
otherwise a built-in fallback), so there is no gimbal lock.
Everything renders inside an <iframe srcdoc=...> so multiple plots per
notebook can't collide on IDs, CSS, or globals.
"""

import html as _html
import json as _json

try:
    import numpy as _np
except ImportError:  # numpy optional; plain lists work too
    _np = None


# --------------------------------------------------------------------------
# trace classes
# --------------------------------------------------------------------------

def _as_list(a, dtype=float):
    if _np is not None:
        return _np.asarray(a, dtype=dtype).tolist()
    return [list(map(dtype, row)) if hasattr(row, "__iter__") else dtype(row)
            for row in a]


class Surface:
    """Surface plot from mesh data.

    x : (N,) axis values
    y : (M,) axis values
    z : (M, N) heights — z[j][i] corresponds to (x[i], y[j])
    """

    def __init__(self, x, y, z):
        self.x = _as_list(x)
        self.y = _as_list(y)
        self.z = _as_list(z)
        if any(len(row) != len(self.x) for row in self.z) or len(self.z) != len(self.y):
            raise ValueError(
                f"shape mismatch: z must be (len(y)={len(self.y)}, len(x)={len(self.x)}); "
                f"got {len(self.z)} rows of length {len(self.z[0]) if self.z else 0}"
            )

    def _config(self):
        return {"type": "surface", "x": self.x, "y": self.y, "z": self.z}


class Scatter3:
    """3D scatter with colored markers and hover tooltips.

    x, y, z : (K,) coordinates
    v       : (K,) optional values for marker color; defaults to z
    """

    def __init__(self, x, y, z, v=None):
        self.x = _as_list(x)
        self.y = _as_list(y)
        self.z = _as_list(z)
        self.v = _as_list(v) if v is not None else list(self.z)
        n = len(self.x)
        if not (len(self.y) == len(self.z) == len(self.v) == n):
            raise ValueError("x, y, z (and v if given) must have equal length")

    def _config(self):
        return {"type": "scatter3", "x": self.x, "y": self.y,
                "z": self.z, "v": self.v}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

_VALID_CMAPS = ("Viridis", "Plasma", "Jet", "Coolwarm", "Grayscale")


def to_html(*traces, theme="dark", proj="iso", cmap="Viridis",
            fill=True, wire=True, colorbar=False, title="QuatPlot"):
    """Return a complete standalone HTML document for the given traces."""
    if not traces:
        raise ValueError("at least one trace (Surface or Scatter3) required")
    if cmap not in _VALID_CMAPS:
        raise ValueError(f"cmap must be one of {_VALID_CMAPS}")
    config = {
        "traces": [t._config() for t in traces],
        "layout": {"theme": theme, "proj": proj, "cmap": cmap,
                   "fill": bool(fill), "wire": bool(wire),
                   "cbar": bool(colorbar), "title": str(title)},
    }
    return _TEMPLATE.replace("__QUATPLOT_CONFIG__",
                             _json.dumps(config, allow_nan=False))


class _IFramePlot:
    """Displays via _repr_html_; avoids IPython.display.HTML's iframe warning
    (IFrame proper wants a src URL, but we deliberately use srcdoc)."""

    def __init__(self, doc, height):
        self._doc = doc
        self._height = int(height)

    def _repr_html_(self):
        # CSS-resizable wrapper: drag the bottom-right corner to resize; the
        # plot's internal ResizeObserver redraws to fit. The bottom padding
        # strip keeps the resize grip outside the iframe (which eats mouse
        # events). No scripts needed, so it survives notebook HTML sanitizers.
        return (
            f'<div style="resize:both;overflow:hidden;width:100%;'
            f'height:{self._height}px;min-width:280px;min-height:180px;'
            f'padding:0 0 12px 0;box-sizing:border-box;'
            f'border:1px solid #8884;border-radius:4px;">'
            f'<iframe srcdoc="{_html.escape(self._doc)}" '
            f'style="width:100%;height:100%;border:none;display:block;" '
            f'sandbox="allow-scripts allow-downloads"></iframe>'
            f'</div>')


def plot(*traces, height=520, **kwargs):
    """Render in a Jupyter notebook (returns an object Jupyter displays as an iframe)."""
    return _IFramePlot(to_html(*traces, **kwargs), height)


def save(path, *traces, **kwargs):
    """Write a standalone HTML file (screenshot button downloads directly)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_html(*traces, **kwargs))
    return path


# --------------------------------------------------------------------------
# embedded HTML/JS template (single source of truth for notebook + files)
# --------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuatPlot</title>
<script src="https://cdn.jsdelivr.net/npm/ganja.js"></script>
<style>
  :root{
    --bg:#f5f6f8; --panel:#ffffff; --ink:#1c2430; --muted:#68737f;
    --edge:#d8dde3; --accent:#2563b0; --grid:#c9cfd6; --cube:#b6bdc6;
    --tip-bg:#1c2430; --tip-ink:#f5f6f8;
  }
  [data-theme="dark"]{
    --bg:#14181d; --panel:#1d232a; --ink:#e8ecf1; --muted:#93a0ad;
    --edge:#313a44; --accent:#5aa2e8; --grid:#3a444f; --cube:#4a5560;
    --tip-bg:#e8ecf1; --tip-ink:#14181d;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{
    background:var(--bg); color:var(--ink);
    font:13px/1.4 "Segoe UI", system-ui, sans-serif;
    display:flex; flex-direction:column;
  }
  header{
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    padding:6px 12px; background:var(--panel); border-bottom:1px solid var(--edge);
  }
  .ctl{display:flex; align-items:center; gap:6px; color:var(--muted)}
  .ctl label{white-space:nowrap}
  select,button{
    font:inherit; color:var(--ink); background:var(--bg);
    border:1px solid var(--edge); border-radius:6px; padding:3px 8px; cursor:pointer;
  }
  select:focus-visible,button:focus-visible{outline:2px solid var(--accent); outline-offset:1px}
  button:hover{border-color:var(--accent)}
  .chk{display:flex; align-items:center; gap:4px; color:var(--muted); cursor:pointer; user-select:none}
  .chk input{accent-color:var(--accent)}
  #stage{position:relative; flex:1; min-height:0}
  canvas{position:absolute; inset:0; width:100%; height:100%; display:block; cursor:grab}
  canvas.dragging{cursor:grabbing}
  #tip{
    position:absolute; pointer-events:none; display:none;
    background:var(--tip-bg); color:var(--tip-ink);
    border-radius:4px; padding:5px 8px; font-size:12px; white-space:pre;
    box-shadow:0 2px 8px rgba(0,0,0,.25);
  }
  #status{position:absolute; left:10px; bottom:6px; color:var(--muted); font-size:11px; pointer-events:none}
  #shotOverlay{
    position:absolute; inset:0; display:none; z-index:10;
    background:rgba(0,0,0,.55); align-items:center; justify-content:center; flex-direction:column; gap:10px;
  }
  #shotOverlay img{max-width:85%; max-height:75%; border:1px solid var(--edge); box-shadow:0 6px 24px rgba(0,0,0,.4)}
  #shotOverlay .bar{display:flex; gap:10px; align-items:center; background:var(--panel); color:var(--muted);
    border:1px solid var(--edge); border-radius:8px; padding:8px 12px}
</style>
</head>
<body>
<header>
  <div class="ctl"><label for="proj">Projection</label>
    <select id="proj">
      <option value="iso">Isometric</option>
      <option value="perspective">Perspective</option>
    </select>
  </div>
  <div class="ctl"><label for="cmap">Colormap</label>
    <select id="cmap">
      <option>Viridis</option><option>Plasma</option><option>Jet</option>
      <option>Coolwarm</option><option>Grayscale</option>
    </select>
  </div>
  <label class="chk" id="fillCtl"><input type="checkbox" id="showFill">Fill</label>
  <label class="chk" id="wireCtl"><input type="checkbox" id="showWire">Wireframe</label>
  <label class="chk" id="cbarCtl"><input type="checkbox" id="showCbar">Color bar</label>
  <button id="themeBtn" title="Toggle light/dark">Theme</button>
  <button id="shotBtn" title="Download PNG">Screenshot</button>
  <button id="resetBtn" title="Reset view">Reset view</button>
  <button id="topBtn" title="Look straight down (x right, y up)">Top view</button>
</header>
<div id="stage">
  <canvas id="cv"></canvas>
  <div id="tip"></div>
  <div id="status">drag: rotate · shift-drag / right-drag: pan · wheel: zoom</div>
  <div id="shotOverlay">
    <img id="shotImg" alt="Screenshot">
    <div class="bar">
      <span>Right-click (or long-press) the image → "Save image as…"</span>
      <button id="shotClose">Close</button>
    </div>
  </div>
</div>

<script>
"use strict";
const CONFIG = __QUATPLOT_CONFIG__;
const SURFACES = CONFIG.traces.filter(t=>t.type==="surface");
const SCATTERS = CONFIG.traces.filter(t=>t.type==="scatter3");

/* ---------- quaternions via ganja.js (Algebra(0,2) ≅ ℍ), with fallback ---------- */
let H = null;
try { if (typeof Algebra === "function") H = Algebra(0,2); } catch(e){ H = null; }
function qmul(a,b){
  if (H){ const r = new H(a).Mul(new H(b)); return [r[0],r[1],r[2],r[3]]; }
  const aw=a[0],ax=a[1],ay=a[2],az=a[3], bw=b[0],bx=b[1],by=b[2],bz=b[3];
  return [aw*bw-ax*bx-ay*by-az*bz,
          aw*bx+ax*bw+ay*bz-az*by,
          aw*by-ax*bz+ay*bw+az*bx,
          aw*bz+ax*by-ay*bx+az*bw];
}
function qconj(a){ return [a[0],-a[1],-a[2],-a[3]]; }
function qnormize(a){ const n=Math.hypot(a[0],a[1],a[2],a[3])||1; return [a[0]/n,a[1]/n,a[2]/n,a[3]/n]; }
function qaxis(ax,ay,az,ang){
  const n=Math.hypot(ax,ay,az)||1, s=Math.sin(ang/2);
  return [Math.cos(ang/2), ax/n*s, ay/n*s, az/n*s];
}
function qrot(q,v){
  const p=[0,v[0],v[1],v[2]];
  const r=qmul(qmul(q,p),qconj(q));
  return [r[1],r[2],r[3]];
}

/* ---------- colormaps ---------- */
const CMAPS = {
  Viridis:[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
  Plasma:[[13,8,135],[126,3,168],[204,71,120],[248,149,64],[240,249,33]],
  Jet:[[0,0,143],[0,0,255],[0,255,255],[255,255,0],[255,0,0],[128,0,0]],
  Coolwarm:[[59,76,192],[144,178,254],[221,221,221],[245,156,125],[180,4,38]],
  Grayscale:[[30,30,30],[230,230,230]]
};
function cmapColor(name,t){
  const s=CMAPS[name]; t=Math.min(1,Math.max(0,t));
  const x=t*(s.length-1), i=Math.min(s.length-2,Math.floor(x)), f=x-i;
  const c=[0,1,2].map(k=>Math.round(s[i][k]+(s[i+1][k]-s[i][k])*f));
  return "rgb("+c[0]+","+c[1]+","+c[2]+")";
}

/* ---------- state ---------- */
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
const tip=document.getElementById("tip");
const L=CONFIG.layout;
const Q0=qnormize(qmul(qaxis(1,0,0,-1.1), qaxis(0,0,1,0.6)));
const S={
  q:Q0.slice(), zoom:1, panX:0, panY:0,
  proj:L.proj||"iso", cmap:L.cmap||"Viridis",
  fill:!!L.fill, wire:!!L.wire, cbar:!!L.cbar,
  theme:L.theme||"dark"
};
function resetView(){ S.q=Q0.slice(); S.zoom=1; S.panX=0; S.panY=0; draw(); }

/* ---------- bounds & ticks ---------- */
function bounds(){
  const xs=[Infinity,-Infinity], ys=[Infinity,-Infinity], zs=[Infinity,-Infinity];
  const acc=(x,y,z)=>{ xs[0]=Math.min(xs[0],x);xs[1]=Math.max(xs[1],x);
    ys[0]=Math.min(ys[0],y);ys[1]=Math.max(ys[1],y);
    zs[0]=Math.min(zs[0],z);zs[1]=Math.max(zs[1],z); };
  for(const s of SURFACES)
    for(let j=0;j<s.y.length;j++)for(let i=0;i<s.x.length;i++)acc(s.x[i],s.y[j],s.z[j][i]);
  for(const t of SCATTERS)
    for(let k=0;k<t.x.length;k++)acc(t.x[k],t.y[k],t.z[k]);
  return {xs,ys,zs};
}
function niceTicks(a,b,n){
  n=n||5;
  const span=b-a||1, raw=span/n, mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const r=raw/mag, step=(r<1.5?1:r<3?2:r<7?5:10)*mag;
  const t=[], start=Math.ceil(a/step)*step;
  for(let v=start; v<=b+1e-9; v+=step) t.push(+v.toFixed(10));
  return t;
}

/* ---------- projection ---------- */
let B=null;
function toUnit(x,y,z){
  const cx=(B.xs[0]+B.xs[1])/2, cy=(B.ys[0]+B.ys[1])/2, cz=(B.zs[0]+B.zs[1])/2;
  const sx=(B.xs[1]-B.xs[0])/2||1, sy=(B.ys[1]-B.ys[0])/2||1, sz=(B.zs[1]-B.zs[0])/2||1;
  return [(x-cx)/sx,(y-cy)/sy,(z-cz)/sz];
}
function project(x,y,z){
  const v=qrot(S.q, toUnit(x,y,z));
  const w=cv.width/devicePixelRatio, h=cv.height/devicePixelRatio;
  const base=Math.min(w,h)*0.32*S.zoom;
  let px,py, depth=v[2];
  if(S.proj==="perspective"){
    const d=4.2, f=d/(d-v[2]);
    px=v[0]*base*f; py=-v[1]*base*f;
  } else { px=v[0]*base; py=-v[1]*base; }
  return [w/2+px+S.panX, h/2+py+S.panY, depth];
}

/* ---------- drawing ---------- */
function css(name){ return getComputedStyle(document.body).getPropertyValue(name).trim(); }
let projScatter=[];

// global color scales
function surfaceZRange(){
  let lo=Infinity, hi=-Infinity;
  for(const s of SURFACES)for(const row of s.z)for(const z of row){ lo=Math.min(lo,z); hi=Math.max(hi,z); }
  return [lo,hi];
}
function scatterVRange(){
  let lo=Infinity, hi=-Infinity;
  for(const t of SCATTERS)for(const v of t.v){ lo=Math.min(lo,v); hi=Math.max(hi,v); }
  return [lo,hi];
}

function draw(){
  B=bounds();
  const w=cv.clientWidth, h=cv.clientHeight;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  ctx.clearRect(0,0,w,h);
  drawAxes();
  const items=[];
  if(S.fill||S.wire) collectSurfaces(items);
  collectScatters(items);
  items.sort((a,b)=>a.depth-b.depth);
  const wireC=css("--ink");
  for(const it of items){
    if(it.kind==="quad"){
      ctx.beginPath();
      ctx.moveTo(it.p[0][0],it.p[0][1]);
      for(let k=1;k<4;k++) ctx.lineTo(it.p[k][0],it.p[k][1]);
      ctx.closePath();
      if(S.fill){ ctx.fillStyle=it.col; ctx.fill(); }
      if(S.wire){ ctx.strokeStyle=S.fill?"rgba(0,0,0,.35)":wireC; ctx.lineWidth=.6; ctx.stroke(); }
    } else {
      ctx.beginPath(); ctx.arc(it.x,it.y,it.r,0,7);
      ctx.fillStyle=it.col; ctx.fill();
      ctx.lineWidth=1; ctx.strokeStyle=css("--panel"); ctx.stroke();
    }
  }
  if(S.cbar && SCATTERS.length) drawColorbar(w,h);
}
function collectSurfaces(items){
  const zr=surfaceZRange(), zmin=zr[0], zspan=(zr[1]-zr[0])||1;
  for(const s of SURFACES){
    const NX=s.x.length, NY=s.y.length, P=[];
    for(let j=0;j<NY;j++){ P.push([]);
      for(let i=0;i<NX;i++) P[j].push(project(s.x[i],s.y[j],s.z[j][i]));
    }
    for(let j=0;j<NY-1;j++)for(let i=0;i<NX-1;i++){
      const p=[P[j][i],P[j][i+1],P[j+1][i+1],P[j+1][i]];
      const zavg=(s.z[j][i]+s.z[j][i+1]+s.z[j+1][i+1]+s.z[j+1][i])/4;
      items.push({kind:"quad", p,
        depth:(p[0][2]+p[1][2]+p[2][2]+p[3][2])/4,
        col:cmapColor(S.cmap,(zavg-zmin)/zspan)});
    }
  }
}
function collectScatters(items){
  if(!SCATTERS.length){ projScatter=[]; return; }
  const vr=scatterVRange(), vmin=vr[0], vspan=(vr[1]-vr[0])||1;
  projScatter=[];
  for(const t of SCATTERS){
    for(let k=0;k<t.x.length;k++){
      const q=project(t.x[k],t.y[k],t.z[k]);
      const r=S.proj==="perspective"? 4.5*(4.2/(4.2-q[2]))*0.9 : 4.5;
      const it={kind:"pt", x:q[0], y:q[1], depth:q[2], r:Math.max(2,r),
        col:cmapColor(S.cmap,(t.v[k]-vmin)/vspan),
        src:{x:t.x[k],y:t.y[k],z:t.z[k]}};
      items.push(it); projScatter.push(it);
    }
  }
}
function drawColorbar(w,h){
  const vr=scatterVRange(), vmin=vr[0], vmax=vr[1];
  const bw=16, bh=Math.min(260,h*0.55), bx=w-bw-52, by=(h-bh)/2;
  for(let i=0;i<bh;i++){
    ctx.fillStyle=cmapColor(S.cmap,1-i/(bh-1));
    ctx.fillRect(bx,by+i,bw,1);
  }
  ctx.strokeStyle=css("--edge"); ctx.lineWidth=1; ctx.strokeRect(bx-.5,by-.5,bw+1,bh+1);
  ctx.fillStyle=css("--muted"); ctx.font="11px system-ui";
  ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.strokeStyle=css("--muted");
  for(const t of niceTicks(vmin,vmax,5)){
    const f=(t-vmin)/((vmax-vmin)||1), ty=by+bh-f*bh;
    ctx.beginPath(); ctx.moveTo(bx+bw,ty); ctx.lineTo(bx+bw+4,ty); ctx.stroke();
    ctx.fillText(String(t), bx+bw+7, ty);
  }
  ctx.save();
  ctx.translate(bx-8,by+bh/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign="center"; ctx.textBaseline="middle";
  ctx.fillStyle=css("--ink"); ctx.font="bold 11px system-ui";
  ctx.fillText("value",0,0);
  ctx.restore();
}
function drawAxes(){
  const xs=B.xs, ys=B.ys, zs=B.zs;
  const corners=[];
  for(const x of xs)for(const y of ys)for(const z of zs) corners.push([x,y,z]);
  const pc=corners.map(c=>project(c[0],c[1],c[2]));
  ctx.strokeStyle=css("--cube"); ctx.lineWidth=1;
  const idx=(xi,yi,zi)=>xi*4+yi*2+zi;
  const edges=[];
  for(let xi=0;xi<2;xi++)for(let yi=0;yi<2;yi++)for(let zi=0;zi<2;zi++){
    if(xi===0) edges.push([idx(0,yi,zi),idx(1,yi,zi),"x"]);
    if(yi===0) edges.push([idx(xi,0,zi),idx(xi,1,zi),"y"]);
    if(zi===0) edges.push([idx(xi,yi,0),idx(xi,yi,1),"z"]);
  }
  ctx.beginPath();
  for(const e of edges){ ctx.moveTo(pc[e[0]][0],pc[e[0]][1]); ctx.lineTo(pc[e[1]][0],pc[e[1]][1]); }
  ctx.stroke();
  const w=cv.clientWidth, h=cv.clientHeight, cx=w/2, cy=h/2;
  ctx.fillStyle=css("--muted"); ctx.font="11px system-ui"; ctx.strokeStyle=css("--muted");
  for(const ax of ["x","y","z"]){
    let best=null, bd=-1, bmy=-Infinity;
    for(const e of edges){ if(e[2]!==ax) continue;
      const mx=(pc[e[0]][0]+pc[e[1]][0])/2, my=(pc[e[0]][1]+pc[e[1]][1])/2;
      const d=(mx-cx)*(mx-cx)+(my-cy)*(my-cy);
      if(d>bd*1.02 || (d>bd*0.98 && my>bmy)){ bd=Math.max(bd,d); bmy=my; best=e; }
    }
    const A=pc[best[0]], Bp=pc[best[1]];
    if(Math.hypot(Bp[0]-A[0],Bp[1]-A[1])<10) continue;
    const rng = ax==="x"?xs: ax==="y"?ys: zs;
    const ticks=niceTicks(rng[0],rng[1]);
    const mx=(A[0]+Bp[0])/2, my=(A[1]+Bp[1])/2;
    let ox=mx-cx, oy=my-cy; const on=Math.hypot(ox,oy)||1; ox/=on; oy/=on;
    for(const t of ticks){
      const f=(t-rng[0])/((rng[1]-rng[0])||1);
      const px=A[0]+(Bp[0]-A[0])*f, py=A[1]+(Bp[1]-A[1])*f;
      ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+ox*5,py+oy*5); ctx.stroke();
      ctx.textAlign = ox>0.3?"left":ox<-0.3?"right":"center";
      ctx.textBaseline = oy>0.3?"top":oy<-0.3?"bottom":"middle";
      ctx.fillText(String(t), px+ox*9, py+oy*9);
    }
    ctx.font="bold 12px system-ui"; ctx.fillStyle=css("--ink");
    ctx.fillText(ax.toUpperCase(), mx+ox*34, my+oy*34);
    ctx.font="11px system-ui"; ctx.fillStyle=css("--muted");
  }
}

/* ---------- interaction ---------- */
let drag=null;
cv.addEventListener("contextmenu",e=>e.preventDefault());
cv.addEventListener("pointerdown",e=>{
  cv.setPointerCapture(e.pointerId); cv.classList.add("dragging");
  drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};
});
cv.addEventListener("pointerup",e=>{ drag=null; cv.classList.remove("dragging"); });
cv.addEventListener("pointermove",e=>{
  if(drag){
    const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    drag.x=e.clientX; drag.y=e.clientY;
    if(drag.pan){ S.panX+=dx; S.panY+=dy; }
    else{
      const k=0.008;
      const dq=qmul(qaxis(1,0,0,dy*k), qaxis(0,1,0,dx*k));
      S.q=qnormize(qmul(dq,S.q));
    }
    tip.style.display="none"; draw(); return;
  }
  if(!SCATTERS.length){ return; }
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=null, bd=100;
  for(const it of projScatter){
    const d=(it.x-mx)*(it.x-mx)+(it.y-my)*(it.y-my);
    if(d<bd && d<(it.r+4)*(it.r+4)){ bd=d; best=it; }
  }
  if(best){
    tip.textContent="x: "+best.src.x.toFixed(3)+"\ny: "+best.src.y.toFixed(3)+"\nz: "+best.src.z.toFixed(3);
    tip.style.left=(best.x+12)+"px"; tip.style.top=(best.y+12)+"px";
    tip.style.display="block";
  } else tip.style.display="none";
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  S.zoom*=Math.exp(-e.deltaY*0.0012);
  S.zoom=Math.min(20,Math.max(.1,S.zoom));
  draw();
},{passive:false});

/* ---------- controls ---------- */
const $=id=>document.getElementById(id);
const on=(id,ev,fn)=>$(id).addEventListener(ev,fn);
function syncControls(){
  document.body.dataset.theme=S.theme;
  $("themeBtn").textContent = S.theme==="light" ? "Dark mode" : "Light mode";
  $("proj").value=S.proj; $("cmap").value=S.cmap;
  $("showFill").checked=S.fill; $("showWire").checked=S.wire; $("showCbar").checked=S.cbar;
  $("fillCtl").style.display = SURFACES.length ? "flex" : "none";
  $("wireCtl").style.display = SURFACES.length ? "flex" : "none";
  $("cbarCtl").style.display = SCATTERS.length ? "flex" : "none";
}
on("proj","change",e=>{S.proj=e.target.value; draw();});
on("cmap","change",e=>{S.cmap=e.target.value; draw();});
on("showFill","change",e=>{S.fill=e.target.checked; draw();});
on("showWire","change",e=>{S.wire=e.target.checked; draw();});
on("showCbar","change",e=>{S.cbar=e.target.checked; draw();});
on("resetBtn","click",resetView);
on("topBtn","click",()=>{ S.q=[1,0,0,0]; tip.style.display="none"; draw(); });
on("themeBtn","click",()=>{
  S.theme=S.theme==="light"?"dark":"light";
  syncControls(); draw();
});
on("shotBtn","click",()=>{
  const out=document.createElement("canvas");
  out.width=cv.width; out.height=cv.height;
  const c=out.getContext("2d");
  c.fillStyle=css("--bg"); c.fillRect(0,0,out.width,out.height);
  c.drawImage(cv,0,0);
  const url=out.toDataURL("image/png");
  let downloaded=false;
  try{ const a=document.createElement("a"); a.download="quatplot.png"; a.href=url; a.click(); downloaded=true; }catch(e){}
  // in sandboxed iframes without allow-downloads the click is silently ignored;
  // show the overlay fallback unless we're clearly a standalone top-level page
  if(!downloaded || window.self!==window.top){
    $("shotImg").src=url;
    $("shotOverlay").style.display="flex";
  }
});
on("shotClose","click",()=>{ $("shotOverlay").style.display="none"; });

/* ---------- resize & boot ---------- */
function resize(){
  const r=cv.parentElement.getBoundingClientRect();
  cv.width=Math.round(r.width*devicePixelRatio);
  cv.height=Math.round(r.height*devicePixelRatio);
  draw();
}
new ResizeObserver(resize).observe($("stage"));
syncControls();
resize();
if(!H) $("status").textContent+="  ·  (ganja.js CDN unavailable — using built-in quaternion fallback)";
</script>
</body>
</html>
"""
