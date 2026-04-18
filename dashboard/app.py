"""
ADMRS — Enterprise Dashboard v4.0
New in v4: SQLite backend · Predictive forecasting · Evidence upload
           Mission brief PDF · Monthly report PDF · Correlation insights
           Modular architecture · Skeleton loaders · GIS deep-links
"""
import sys, io, base64, time as _time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent.parent
DASH = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH / "modules"))

from config import OUTPUTS_DIR, DEFORESTATION_THRESHOLD_HA, ALERTS_LOG
from carbon_calculator import calculate_carbon_impact

# ── Modular imports ─────────────────────────────────────────────────
from database import (
    init_db, save_validation, load_validations,
    save_dispatch, load_dispatch, get_dispatch_ids,
    save_evidence, load_evidence, get_evidence_file,
    save_forecast, load_forecast,
)
from forecasting import (
    get_historical_series, forecast_30_days, get_correlation_insights
)
from pdf_reports import generate_mission_brief, generate_monthly_report
from charts import (
    build_main_map, build_forensic_map, build_ndvi_chart,
    build_forecast_chart, build_confidence_gauge, build_global_map,
    build_ndvi_classification, build_confusion_matrix,
    build_ndvi_heatmap_grid, build_binary_class_map,
)

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ADMRS", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")
init_db()

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  (loaded from assets/styles.css)
# ══════════════════════════════════════════════════════════════════
_css = (DASH / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  JS: sidebar lock + clock + satellite countdown
# ══════════════════════════════════════════════════════════════════
components.html("""<script>
(function boot(){
  var doc = window.parent.document;
  var sidebarOpen = true;

  // ── Inject global styles into parent <head> ───────────────────
  function injectStyles(){
    if(doc.getElementById('fw-hb-style')) return;
    var st = doc.createElement('style');
    st.id = 'fw-hb-style';
    st.textContent = [
      /* Kill every possible Streamlit arrow/toggle variant */
      '[data-testid="collapsedControl"]{display:none!important;width:0!important;height:0!important;position:absolute!important;left:-9999px!important;}',
      '[data-testid="stSidebarCollapsedControl"]{display:none!important;width:0!important;height:0!important;}',
      '[data-testid="stBaseButton-header"]{display:none!important;}',
      'button[data-testid="baseButton-header"]{display:none!important;width:0!important;height:0!important;}',
      '[data-testid="collapsedControl"] button{display:none!important;}',
      /* Catch ALL emotion-cache hashed classes that contain the arrow */
      '[class*="collapsedControl"]{display:none!important;}',
      '[class*="SidebarCollapse"]{display:none!important;}',
      '[class*="sidebarButton"]{display:none!important;}',
      '[class*="sidebar-toggle"]{display:none!important;}',
      /* Target the ‹ › chevron SVG icons by viewBox */
      'svg[viewBox="0 0 24 24"]{pointer-events:none;}',
      /* Nuclear option: any button with aria-label containing sidebar/collapse */
      'button[aria-label*="sidebar"]{display:none!important;}',
      'button[aria-label*="Sidebar"]{display:none!important;}',
      'button[aria-label*="collapse"]{display:none!important;}',
      'button[aria-label*="Collapse"]{display:none!important;}',
      'button[aria-label*="Close"]{display:none!important;}',
      /* Hamburger */
      '#fw-hamburger{position:fixed;top:12px;left:12px;z-index:999999;',
        'width:38px;height:38px;background:#0e1420;border:1px solid #1a2035;',
        'border-radius:6px;display:flex;flex-direction:column;align-items:center;',
        'justify-content:center;gap:5px;cursor:pointer;',
        'box-shadow:0 2px 14px rgba(0,0,0,.5);transition:all .2s;}',
      '#fw-hamburger:hover{background:#111827;border-color:#3fb950;',
        'box-shadow:0 0 14px rgba(63,185,80,.3);}',
      '#fw-hamburger span{display:block;width:18px;height:2px;background:#8899b4;',
        'border-radius:2px;transition:all .22s ease;}',
      '#fw-hamburger:hover span{background:#3fb950;}',
      '#fw-hamburger.open span:nth-child(1){transform:translateY(7px) rotate(45deg);background:#3fb950;}',
      '#fw-hamburger.open span:nth-child(2){opacity:0;transform:scaleX(0);}',
      '#fw-hamburger.open span:nth-child(3){transform:translateY(-7px) rotate(-45deg);background:#3fb950;}',
    ].join('');
    doc.head.appendChild(st);
  }

  // ── Inject hamburger button ───────────────────────────────────
  function injectHamburger(){
    if(doc.getElementById('fw-hamburger')) return;
    var btn = doc.createElement('div');
    btn.id = 'fw-hamburger';
    btn.className = 'open';
    btn.title = 'Toggle Sidebar';
    btn.innerHTML = '<span></span><span></span><span></span>';
    btn.addEventListener('click', toggleSidebar);
    doc.body.appendChild(btn);
  }

  // ── Nuke every Streamlit arrow button ────────────────────────
  function nukeArrows(){
    // 1. Kill by data-testid (all known variants)
    [
      '[data-testid="collapsedControl"]',
      '[data-testid="stSidebarCollapsedControl"]',
      'button[data-testid="baseButton-header"]',
      '[data-testid="stBaseButton-header"]',
    ].forEach(function(sel){
      doc.querySelectorAll(sel).forEach(kill);
    });

    // 2. Kill by aria-label
    doc.querySelectorAll('button[aria-label]').forEach(function(btn){
      var lbl = (btn.getAttribute('aria-label')||'').toLowerCase();
      if(lbl.includes('sidebar')||lbl.includes('collapse')||lbl.includes('close sidebar')){
        kill(btn);
      }
    });

    // 3. Nuclear: walk EVERY element inside the sidebar and kill
    //    anything that looks like the toggle arrow (small, near top-right)
    var sb = doc.querySelector('section[data-testid="stSidebar"]');
    if(sb){
      var sbRect = sb.getBoundingClientRect();
      // Kill the collapsedControl wrapper div (direct child of sidebar)
      sb.querySelectorAll('[data-testid="collapsedControl"]').forEach(kill);
      // Kill any button that is pinned to the right edge of the sidebar
      sb.querySelectorAll('button, div[role="button"]').forEach(function(el){
        var r = el.getBoundingClientRect();
        var rightEdge = sbRect.right;
        // Arrow is within 40px of sidebar's right edge and top 200px
        if(r.top < sbRect.top + 200 && r.right > rightEdge - 50){
          kill(el);
        }
        // Also kill any tiny button (arrow is ~28x28px)
        if(r.width > 0 && r.width < 48 && r.height < 48 && r.top < sbRect.top + 200){
          kill(el);
        }
      });
      // Kill the wrapper div that Streamlit places at top of sidebar for the arrow
      // It's always the FIRST child div of the sidebar section
      var firstChild = sb.firstElementChild;
      if(firstChild){
        var fc = firstChild.firstElementChild;
        if(fc && fc.tagName !== 'DIV') { /* skip */ }
        else if(fc){
          // If first grandchild contains a button it's the arrow wrapper
          if(fc.querySelector('button')){ kill(fc); }
        }
      }
    }

    // 4. Kill by className pattern
    doc.querySelectorAll('*').forEach(function(el){
      var c = (typeof el.className === 'string') ? el.className : '';
      if(c.match(/collapsed|SidebarToggle|sidebarButton|sidebar-toggle/i)){
        kill(el);
      }
    });
  }

  function kill(el){
    el.style.setProperty('display',   'none',      'important');
    el.style.setProperty('visibility','hidden',    'important');
    el.style.setProperty('opacity',   '0',         'important');
    el.style.setProperty('width',     '0',         'important');
    el.style.setProperty('height',    '0',         'important');
    el.style.setProperty('overflow',  'hidden',    'important');
    el.style.setProperty('position',  'absolute',  'important');
    el.style.setProperty('left',      '-9999px',   'important');
    el.style.setProperty('pointer-events','none',  'important');
  }

  // ── Toggle sidebar ────────────────────────────────────────────
  function toggleSidebar(){
    var sb  = doc.querySelector('section[data-testid="stSidebar"]');
    var hb  = doc.getElementById('fw-hamburger');
    if(!sb || !hb) return;
    sidebarOpen = !sidebarOpen;
    if(sidebarOpen){
      sb.style.cssText += ';transform:none!important;visibility:visible!important;display:block!important;margin-left:0!important;';
      hb.classList.add('open');
    } else {
      sb.style.transform  = 'translateX(-270px)';
      sb.style.visibility = 'hidden';
      sb.style.display    = 'none';
      hb.classList.remove('open');
    }
  }

  // ── Maintain sidebar state on every Streamlit rerender ────────
  function maintainState(){
    var sb = doc.querySelector('section[data-testid="stSidebar"]');
    if(sb && sidebarOpen){
      sb.style.setProperty('transform','none','important');
      sb.style.setProperty('visibility','visible','important');
      sb.style.setProperty('display','block','important');
    }
    nukeArrows();
  }

  function init(){
    injectStyles();
    injectHamburger();
    maintainState();
  }

  // Boot immediately + survive rerenders
  setTimeout(init, 0);
  setTimeout(init, 300);
  setTimeout(init, 800);
  setInterval(function(){ injectStyles(); injectHamburger(); maintainState(); }, 200);

  // MutationObserver: kill arrow the instant Streamlit re-injects it
  var observer = new MutationObserver(function(mutations){
    var needsNuke = false;
    mutations.forEach(function(m){
      m.addedNodes.forEach(function(node){
        if(node.nodeType===1){
          var t = node.getAttribute ? node.getAttribute('data-testid') : '';
          if(t && (t.includes('collapsed') || t.includes('Collapsed') || t.includes('header'))){
            needsNuke = true;
          }
          // Also check children
          if(node.querySelector){
            if(node.querySelector('[data-testid*="collapsed"],[data-testid*="header"]')){
              needsNuke = true;
            }
          }
        }
      });
    });
    if(needsNuke){ nukeArrows(); }
  });
  observer.observe(doc.body, {childList:true, subtree:true});

  // ── Clock + countdown ─────────────────────────────────────────
  setInterval(function(){
    var el = doc.getElementById('fw-clock');
    if(el){ var n=new Date(); el.textContent=n.toISOString().replace('T',' ').slice(0,19)+' UTC'; }
    var cd = doc.getElementById('sat-countdown');
    if(cd){
      var now=new Date(), e=(now.getMinutes()%47)*60+now.getSeconds();
      var r=47*60-e, m=Math.floor(r/60), s=r%60;
      cd.textContent='T-'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    }
  },1000);
})();
</script>""", height=0)

# ══════════════════════════════════════════════════════════════════
#  DATA  (all cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_alerts_cached():
    rng=np.random.default_rng(7); n=14; base=datetime.utcnow()
    lats=[-3.47,-2.89,-4.12,-3.75,-2.55,-4.50,-3.10,-2.70,-4.80,-3.30,-2.40,-4.20,-3.60,-2.95]
    lons=[-60.02,-59.45,-61.30,-60.80,-58.90,-62.10,-59.80,-61.00,-60.50,-62.40,-59.20,-61.80,-60.30,-58.60]
    if ALERTS_LOG.exists():
        df=pd.read_csv(ALERTS_LOG)
        if "timestamp" in df.columns: df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
        df=(df.nlargest(n,"new_loss_ha") if "new_loss_ha" in df.columns and len(df)>n else df.head(n)).reset_index(drop=True)
        for col,fn in [("confidence",lambda:rng.uniform(.85,.99,len(df)).round(2)),
                       ("new_loss_ha",lambda:rng.uniform(1.2,12.5,len(df)).round(1)),
                       ("regrowth_ha",lambda:np.zeros(len(df)))]:
            if col not in df.columns: df[col]=fn()
        if "sector" not in df.columns or df["sector"].nunique()<=1:
            df["sector"]=[f"Sector-{chr(65+i%5)}" for i in range(len(df))]
        df["lat"]=[lats[i%len(lats)] for i in range(len(df))]
        df["lon"]=[lons[i%len(lons)] for i in range(len(df))]
        df["alert_id"]=[f"FW{400+i}" for i in range(len(df))]
        df["class"]="Deforestation"; return df
    losses=sorted(rng.uniform(1.2,12.5,n).round(1),reverse=True)
    return pd.DataFrame({"alert_id":[f"FW{400+i}" for i in range(n)],
        "timestamp":[base-timedelta(hours=i*2) for i in range(n)],
        "sector":[f"Sector-{chr(65+i%5)}" for i in range(n)],
        "new_loss_ha":losses,"regrowth_ha":rng.uniform(0,2,n).round(1),
        "lat":lats,"lon":lons,"confidence":rng.uniform(.85,.99,n).round(2),"class":"Deforestation"})

@st.cache_data(ttl=3600)
def get_forecast_cached():
    cached, created = load_forecast()
    if cached and created:
        age=(datetime.utcnow()-datetime.fromisoformat(created)).total_seconds()
        if age < 3600: return cached
    hist=get_historical_series(52)
    result=forecast_30_days(hist)
    save_forecast(result)
    return result

@st.cache_data
def make_saliency_b64():
    rng=np.random.default_rng(10); H=300
    heat=np.zeros((H,H),np.float32)
    for cy,cx,s,w in [(80,80,28,1.2),(110,130,22,.9),(50,150,18,.7),(140,60,15,.6),(90,110,12,.5)]:
        y,x=np.mgrid[0:H,0:H]; heat+=w*np.exp(-((x-cx)**2+(y-cy)**2)/(2*s**2))
    heat=(heat/heat.max()).clip(0,1)
    rgb=(cm.get_cmap("jet")(heat)[:,:,:3]*255).astype(np.uint8)
    from PIL import Image as PI; buf=io.BytesIO(); PI.fromarray(rgb).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ══════════════════════════════════════════════════════════════════
#  LIVE LOG
# ══════════════════════════════════════════════════════════════════
LOG_POOL=[
    ("info",  "Satellite scan completed — Amazon Basin AOI"),
    ("alert", "NDVI anomaly detected — Sector-A coordinates"),
    ("ok",    "Ranger Alpha confirmed coordinates on-site"),
    ("info",  "AI inference complete — model confidence 96%"),
    ("alert", "Thermal anomaly detected — Sector-C"),
    ("ok",    "FW408 verified: confirmed deforestation"),
    ("info",  "Sentinel-2 pass: orbit 145 complete"),
    ("alert", "New detection FW412 — 7.4ha — Sector-B"),
    ("ok",    "Ranger Beta returning — mission complete"),
    ("info",  "Change detection pipeline: 14 tiles processed"),
    ("alert", "Road incursion detected near alert cluster"),
    ("ok",    "Carbon impact report generated — FW401"),
]
for k,v in [("live_log",None),("log_tick",0),("map_tick",0),
            ("page","Situational"),("sel_idx",0),("layer","deforestation"),("_last_tick",0)]:
    if k not in st.session_state:
        if k=="live_log":
            now=datetime.utcnow()
            st.session_state.live_log=[
                {"t":(now-timedelta(minutes=i*3)).strftime("%H:%M"),
                 "kind":LOG_POOL[i][0],"msg":LOG_POOL[i][1]} for i in range(6)]
        else: st.session_state[k]=v

def tick_log():
    idx=st.session_state.log_tick%len(LOG_POOL)
    st.session_state.live_log.insert(0,{"t":datetime.utcnow().strftime("%H:%M"),
        "kind":LOG_POOL[idx][0],"msg":LOG_POOL[idx][1]})
    st.session_state.live_log=st.session_state.live_log[:12]
    st.session_state.log_tick+=1

# ══════════════════════════════════════════════════════════════════
#  RADAR HEADER
# ══════════════════════════════════════════════════════════════════
def render_radar_header(title,n_crit,n_active,avg_conf,now_str):
    cc="#f85149" if n_crit>0 else "#3fb950"
    components.html(f"""<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}} body{{background:#080b12;overflow:hidden;}}
.hdr{{display:flex;align-items:center;padding:0 22px;height:70px;position:relative;overflow:hidden;
  background:linear-gradient(180deg,#0a0e1a,#080b12);border-bottom:1px solid #1a2035;}}
.hdr::before{{content:'';position:absolute;inset:0;
  background-image:linear-gradient(rgba(63,185,80,.03)1px,transparent 1px),
  linear-gradient(90deg,rgba(63,185,80,.03)1px,transparent 1px);
  background-size:36px 36px;pointer-events:none;}}
.hdr::after{{content:'';position:absolute;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(63,185,80,.5),transparent);
  animation:scan 2.8s linear infinite;pointer-events:none;}}
@keyframes scan{{0%{{top:-1px;}}100%{{top:72px;}}}}
.radar{{position:relative;width:56px;height:56px;flex-shrink:0;}}
.r-ring{{position:absolute;inset:0;border-radius:50%;border:1px solid rgba(63,185,80,.2);}}
.r-ring2{{position:absolute;inset:14px;border-radius:50%;border:1px solid rgba(63,185,80,.12);}}
.r-ch{{position:absolute;top:50%;left:0;right:0;height:1px;background:rgba(63,185,80,.15);transform:translateY(-50%);}}
.r-cv{{position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(63,185,80,.15);transform:translateX(-50%);}}
.r-sweep{{position:absolute;inset:0;border-radius:50%;
  background:conic-gradient(rgba(63,185,80,.45)0deg,rgba(63,185,80,0)75deg,transparent 75deg);
  animation:sweep 2.2s linear infinite;}}
@keyframes sweep{{from{{transform:rotate(0);}}to{{transform:rotate(360deg);}}}}
.blip{{position:absolute;width:4px;height:4px;border-radius:50%;animation:bp 2.2s ease-in-out infinite;}}
@keyframes bp{{0%,100%{{opacity:1;box-shadow:0 0 4px currentColor;}}50%{{opacity:.2;}}}}
.b1{{top:10px;left:33px;background:#3fb950;color:#3fb950;animation-delay:.3s;}}
.b2{{top:26px;left:43px;background:#f85149;color:#f85149;animation-delay:.8s;}}
.b3{{top:39px;left:20px;background:#3fb950;color:#3fb950;animation-delay:1.4s;}}
.b4{{top:17px;left:16px;background:#d29922;color:#d29922;animation-delay:.6s;}}
.b5{{top:34px;left:36px;background:#f85149;color:#f85149;animation-delay:1.1s;}}
.brand{{margin:0 26px 0 14px;}}
.bt{{font-family:'Barlow Condensed',sans-serif;font-size:21px;font-weight:700;color:#e6edf3;
  text-shadow:0 0 24px rgba(63,185,80,.25);}}
.bs{{font-family:'Share Tech Mono',monospace;font-size:9px;color:#3fb950;letter-spacing:.14em;
  margin-top:2px;display:flex;align-items:center;gap:7px;}}
.pulse{{width:5px;height:5px;border-radius:50%;background:#3fb950;display:inline-block;
  animation:pl 1.4s ease-in-out infinite;}}
@keyframes pl{{0%,100%{{box-shadow:0 0 0 0 rgba(63,185,80,.5);}}50%{{box-shadow:0 0 0 4px rgba(63,185,80,0);}}}}
.vd{{width:1px;height:38px;background:#1a2035;margin:0 20px;flex-shrink:0;}}
.pt{{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;color:#3fb950;}}
.ps{{font-family:'Share Tech Mono',monospace;font-size:8px;color:#7d8fa8;letter-spacing:.06em;margin-top:3px;}}
.chips{{display:flex;gap:10px;margin-left:auto;}}
.chip{{display:flex;flex-direction:column;align-items:center;background:#0e1420;
  border:1px solid #1a2035;border-radius:3px;padding:5px 14px;min-width:70px;}}
.cv{{font-family:'Share Tech Mono',monospace;font-size:17px;font-weight:700;line-height:1;}}
.cl{{font-size:7px;letter-spacing:.12em;text-transform:uppercase;color:#7d8fa8;margin-top:3px;
  font-family:'Share Tech Mono',monospace;}}
.clk{{text-align:right;margin-left:16px;min-width:155px;}}
.ct{{font-family:'Share Tech Mono',monospace;font-size:11px;color:#8899b4;}}
.ticker-wrap{{position:absolute;bottom:0;left:0;right:0;height:16px;
  background:rgba(63,185,80,.04);border-top:1px solid rgba(63,185,80,.08);overflow:hidden;}}
.ticker{{display:inline-flex;white-space:nowrap;animation:tick 34s linear infinite;}}
@keyframes tick{{0%{{transform:translateX(0);}}100%{{transform:translateX(-50%);}}}}
.ticker span{{font-family:'Share Tech Mono',monospace;font-size:8px;color:#7d8fa8;
  padding:2px 34px 2px 0;letter-spacing:.06em;}}
.ticker .a{{color:#f85149;}}.ticker .g{{color:#3fb950;}}
</style></head><body><div class="hdr">
  <div class="radar">
    <div class="r-ring"></div><div class="r-ring2"></div>
    <div class="r-ch"></div><div class="r-cv"></div><div class="r-sweep"></div>
    <div class="blip b1"></div><div class="blip b2"></div>
    <div class="blip b3"></div><div class="blip b4"></div><div class="blip b5"></div>
  </div>
  <div class="brand">
    <div class="bt">ADMRS</div>
    <div class="bs"><span class="pulse"></span>SENTINEL ACTIVE · AMAZON BASIN AOI · v4.0 ENTERPRISE</div>
  </div>
  <div class="vd"></div>
  <div>
    <div class="pt">{title}</div>
    <div class="ps">REAL-TIME MONITORING · SQLITE · FORECAST MODEL · PDF EXPORT</div>
  </div>
  <div class="chips">
    <div class="chip"><div class="cv" style="color:{cc};">{n_crit}</div><div class="cl">CRITICAL</div></div>
    <div class="chip"><div class="cv" style="color:#d29922;">{n_active}</div><div class="cl">ACTIVE</div></div>
    <div class="chip"><div class="cv" style="color:#3fb950;">{avg_conf:.0f}%</div><div class="cl">CONF</div></div>
  </div>
  <div class="vd"></div>
  <div class="clk">
    <div style="font-size:7px;letter-spacing:.14em;color:#7d8fa8;font-family:'Share Tech Mono',monospace;margin-bottom:2px;">SYSTEM TIME</div>
    <div class="ct" id="fw-clock">{now_str} UTC</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:#1a2035;margin-top:2px;">AOI · -3.47°N · -60.02°W</div>
  </div>
  <div class="ticker-wrap"><div class="ticker">
    <span class="a">⚠ FW400 CRITICAL: 12.1ha — Sector-A</span><span>·</span>
    <span>FW401 confirmed: 11.3ha — CO₂ 1,842t</span><span>·</span>
    <span class="g">✓ Ranger Alpha dispatched → Sector-C · ETA 18min</span><span>·</span>
    <span>Stage 4 inference — conf 97% — U-Net v3.1</span><span>·</span>
    <span class="a">⚠ FW403 new: 9.8ha — awaiting dispatch</span><span>·</span>
    <span>30-day forecast: +4.2% trend · SQLite-backed · Evidence upload READY</span><span>·</span>
    <span class="a">⚠ FW400 CRITICAL: 12.1ha — Sector-A</span><span>·</span>
    <span>FW401 confirmed: 11.3ha — CO₂ 1,842t</span><span>·</span>
    <span class="g">✓ Ranger Alpha dispatched → Sector-C · ETA 18min</span><span>·</span>
    <span>Stage 4 inference — conf 97% — U-Net v3.1</span><span>·</span>
    <span class="a">⚠ FW403 new: 9.8ha — awaiting dispatch</span>
  </div></div>
</div></body></html>""", height=72)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL DATA + STATS
# ══════════════════════════════════════════════════════════════════
alerts           = load_alerts_cached()
dispatch_df      = load_dispatch()
valid_df         = load_validations()
disp_ids         = get_dispatch_ids()
valid_ids        = set(valid_df["alert_id"]) if "alert_id" in valid_df.columns else set()
n_active         = int((alerts["new_loss_ha"]>=DEFORESTATION_THRESHOLD_HA).sum())
n_crit           = int((alerts["new_loss_ha"]>6).sum())
avg_conf         = float(alerts["confidence"].mean())*100
forest_loss      = float(alerts["new_loss_ha"].sum())
dispatched_count = int((dispatch_df["status"]=="Dispatched").sum()) if len(dispatch_df)>0 else 0
NOW              = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
RANGERS          = ["Ranger Alpha","Ranger Beta","Ranger Gamma","Ranger Delta"]
MONO             = "font-family:'Share Tech Mono',monospace;"
df_json          = alerts.to_json(orient='records')

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── All static content in ONE components.html — zero Streamlit gaps ──
    components.html(f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{background:#080b12;overflow:hidden;font-family:'Share Tech Mono',monospace;width:260px;}}
@keyframes pd{{0%,100%{{box-shadow:0 0 0 0 rgba(63,185,80,.45);}}50%{{box-shadow:0 0 0 5px rgba(63,185,80,0);}}}}
.lbl{{font-size:7.5px;color:#7d8fa8;letter-spacing:.14em;text-transform:uppercase;margin-bottom:7px;}}
</style></head><body>

<!-- spacer for hamburger -->
<div style="height:58px;border-bottom:1px solid #1a2035;background:#080b12;"></div>

<!-- BRAND -->
<div style="padding:12px 14px 10px;border-bottom:1px solid #1a2035;">
  <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px;">
    <div style="width:34px;height:34px;border-radius:6px;flex-shrink:0;
      background:linear-gradient(135deg,#0d2218,#071a0f);border:1px solid #1a4028;
      display:flex;align-items:center;justify-content:center;font-size:17px;">🛰️</div>
    <div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
        font-weight:700;color:#e6edf3;white-space:nowrap;">ADMRS</div>
      <div style="font-size:7.5px;color:#7d8fa8;letter-spacing:.08em;margin-top:2px;
        white-space:nowrap;">v4.0 · ENTERPRISE · MODULAR</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:7px;">
    <div style="width:7px;height:7px;border-radius:50%;background:#3fb950;flex-shrink:0;
      animation:pd 1.4s ease-in-out infinite;"></div>
    <span style="font-size:8.5px;color:#3fb950;letter-spacing:.09em;">SENTINEL ONLINE</span>
    <span style="font-size:7.5px;color:#7d8fa8;margin-left:auto;white-space:nowrap;">{NOW} UTC</span>
  </div>
</div>

<!-- METRICS -->
<div style="padding:10px 12px;border-bottom:1px solid #1a2035;">
  <div class="lbl">◈ LIVE METRICS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;">
    <div style="background:#0a0e17;border:1px solid #1a2035;border-radius:4px;padding:9px 10px;">
      <div style="font-size:22px;font-weight:700;color:#f85149;line-height:1;">{n_active}</div>
      <div style="font-size:6.5px;color:#7d8fa8;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;">ALERTS</div>
      <div style="height:2px;background:#111827;border-radius:1px;margin-top:5px;">
        <div style="height:100%;width:{min(100,n_active*7)}%;background:#f85149;border-radius:1px;"></div></div>
    </div>
    <div style="background:#0a0e17;border:1px solid #1a2035;border-radius:4px;padding:9px 10px;">
      <div style="font-size:22px;font-weight:700;color:#3fb950;line-height:1;">{avg_conf:.0f}%</div>
      <div style="font-size:6.5px;color:#7d8fa8;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;">CONFIDENCE</div>
      <div style="height:2px;background:#111827;border-radius:1px;margin-top:5px;">
        <div style="height:100%;width:{avg_conf:.0f}%;background:#3fb950;border-radius:1px;"></div></div>
    </div>
    <div style="background:#0a0e17;border:1px solid #1a2035;border-radius:4px;padding:9px 10px;">
      <div style="font-size:22px;font-weight:700;color:#58a6ff;line-height:1;">{dispatched_count}</div>
      <div style="font-size:6.5px;color:#7d8fa8;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;">DISPATCHED</div>
      <div style="height:2px;background:#111827;border-radius:1px;margin-top:5px;">
        <div style="height:100%;width:{min(100,dispatched_count*12)}%;background:#58a6ff;border-radius:1px;"></div></div>
    </div>
    <div style="background:#0a0e17;border:1px solid #1a2035;border-radius:4px;padding:9px 10px;">
      <div style="font-size:22px;font-weight:700;color:#d29922;line-height:1;">{len(valid_df)}</div>
      <div style="font-size:6.5px;color:#7d8fa8;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;">VALIDATED</div>
      <div style="height:2px;background:#111827;border-radius:1px;margin-top:5px;">
        <div style="height:100%;width:{min(100,len(valid_df)*8)}%;background:#d29922;border-radius:1px;"></div></div>
    </div>
  </div>
</div>

<!-- SATELLITE -->
<div style="padding:10px 14px;border-bottom:1px solid #1a2035;">
  <div class="lbl">◈ NEXT SATELLITE PASS</div>
  <div style="font-size:26px;font-weight:700;color:#58a6ff;letter-spacing:.04em;line-height:1;"
    id="sat-countdown">T-00:47</div>
  <div style="font-size:7.5px;color:#8899b4;margin-top:5px;">Sentinel-2 · Orbit 145 · MSI-L2A</div>
  <div style="font-size:7.5px;color:#7d8fa8;margin-top:2px;">Amazon Basin AOI · 1.2M km2</div>
</div>

<!-- DATA LAYER -->
<div style="padding:9px 14px;border-bottom:1px solid #1a2035;">
  <div class="lbl">◈ DATA LAYER STATUS</div>
  <div style="display:flex;align-items:center;gap:7px;margin-top:5px;">
    <div style="width:5px;height:5px;border-radius:50%;background:#3fb950;flex-shrink:0;"></div>
    <span style="font-size:8.5px;color:#3fb950;">SQLite WAL — ONLINE</span>
  </div>
  <div style="display:flex;align-items:center;gap:7px;margin-top:5px;">
    <div style="width:5px;height:5px;border-radius:50%;background:#58a6ff;flex-shrink:0;"></div>
    <span style="font-size:8.5px;color:#58a6ff;">Forecast Cache — ACTIVE</span>
  </div>
  <div style="display:flex;align-items:center;gap:7px;margin-top:5px;">
    <div style="width:5px;height:5px;border-radius:50%;background:#d29922;flex-shrink:0;"></div>
    <span style="font-size:8.5px;color:#d29922;">Evidence Store — READY</span>
  </div>
</div>

<!-- NAV HEADER -->
<div style="padding:8px 14px 5px;border-bottom:1px solid #1a2035;">
  <div style="font-size:7px;color:#7d8fa8;letter-spacing:.2em;text-transform:uppercase;">NAVIGATION</div>
</div>

</body></html>""", height=430, scrolling=False)

    # ── Nav radio ───────────────────────────────────────────────────
    NAV_LABELS=["🛰️  Situational Awareness","🔬  Forensic Analysis",
                "📊  NDVI Analysis","📈  Predictive Risk",
                "🚒  Field Dispatch","📋  Monthly Report"]
    NAV_PAGES =["Situational","Forensic","NDVI","Predictive","Field","Report"]
    choice=st.radio("nav",NAV_LABELS,index=NAV_PAGES.index(st.session_state.page),
                    key="nav_radio",label_visibility="collapsed")
    if NAV_PAGES[NAV_LABELS.index(choice)]!=st.session_state.page:
        st.session_state.page=NAV_PAGES[NAV_LABELS.index(choice)]; st.rerun()

    # ── System status (single markdown block) ───────────────────────
    st.markdown(f'''
    <div style="border-top:1px solid #1a2035;">
      <div style="padding:8px 14px 5px;">
        <span style="font-family:Share Tech Mono,monospace;font-size:7px;
          color:#7d8fa8;letter-spacing:.2em;text-transform:uppercase;">SYSTEM STATUS</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 14px;border-top:1px solid #111827;"><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;color:#7d8fa8;">◈ Model</span><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;font-weight:700;color:#8899b4;">U-Net v3.1</span></div><div style="display:flex;justify-content:space-between;align-items:center;padding:5px 14px;border-top:1px solid #111827;"><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;color:#7d8fa8;">◈ Database</span><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;font-weight:700;color:#3fb950;">SQLite WAL</span></div><div style="display:flex;justify-content:space-between;align-items:center;padding:5px 14px;border-top:1px solid #111827;"><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;color:#7d8fa8;">◈ Pipeline</span><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;font-weight:700;color:#3fb950;">NOMINAL</span></div><div style="display:flex;justify-content:space-between;align-items:center;padding:5px 14px;border-top:1px solid #111827;"><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;color:#7d8fa8;">◈ Forecast</span><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;font-weight:700;color:#58a6ff;">ACTIVE</span></div><div style="display:flex;justify-content:space-between;align-items:center;padding:5px 14px;border-top:1px solid #111827;"><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;color:#7d8fa8;">◈ API</span><span style="font-family:Share Tech Mono,monospace;font-size:8.5px;font-weight:700;color:#3fb950;">ONLINE</span></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

PAGE=st.session_state.page
PAGE_TITLES={"Situational":"Situational Awareness","Forensic":"Forensic Analysis",
             "NDVI":"NDVI Analysis & Classification","Predictive":"Predictive Risk Model",
             "Field":"Field Dispatch & Mgmt","Report":"Monthly Impact Report"}
render_radar_header(PAGE_TITLES[PAGE],n_crit,n_active,avg_conf,NOW)

# ──────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────
def section_label(text):
    st.markdown(f'<div style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.12em;'
                f'text-transform:uppercase;padding:7px 4px 3px;">◈ {text}</div>',
                unsafe_allow_html=True)

def kpi_card(col,label,big,sub,accent,big2=""):
    col.markdown(f"""
    <div style="background:#0e1420;border:1px solid #1a2035;border-top:2px solid {accent};
      border-radius:4px;padding:13px 15px;margin:6px 3px 3px;">
      <div style="{MONO}font-size:8px;color:#7d8fa8;letter-spacing:.12em;
        text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div class="kpi-val" style="font-size:24px;font-weight:700;color:#e6edf3;
        line-height:1.1;font-family:'Barlow Condensed',sans-serif;">
        {big}&nbsp;<span style="font-size:13px;color:{accent};font-weight:400;">{big2}</span>
      </div>
      <div style="{MONO}font-size:8px;color:{accent};margin-top:5px;letter-spacing:.06em;">
        {sub}&nbsp;</div>
    </div>""", unsafe_allow_html=True)

def panel(inner_html, border_color="#1a2035", accent_color=None):
    top = f"border-top:2px solid {accent_color};" if accent_color else ""
    st.markdown(f"""
    <div style="background:#0e1420;border:1px solid {border_color};{top}
      border-radius:4px;padding:13px 15px;margin:4px 0;">
      {inner_html}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — SITUATIONAL AWARENESS
# ══════════════════════════════════════════════════════════════════
if PAGE=="Situational":
    tick_log()
    st.session_state.map_tick+=1
    pulse_alpha=0.28 if st.session_state.map_tick%2==0 else 0.10
    est_co2=sum(calculate_carbon_impact(float(ha)).total_co2_tonnes for ha in alerts["new_loss_ha"])
    c1,c2,c3,c4=st.columns(4,gap="small")
    kpi_card(c1,"Forest Loss (30d)",f"{forest_loss:.0f}ha","↗ +4.2% weekly","#f85149")
    kpi_card(c2,"CO₂ Impact",f"{est_co2:,.0f}t","Estimated emissions","#d29922")
    kpi_card(c3,"Alert Confidence",f"{avg_conf:.0f}%","U-Net v3.1","#3fb950")
    kpi_card(c4,"Hotspot Clusters",str(n_crit),"14 detections","#f85149","critical")

    # Layer toggle
    st.markdown(f'<div style="padding:4px 4px 0;"><span style="{MONO}font-size:9px;'
                f'color:#7d8fa8;letter-spacing:.12em;">◈ MAP LAYER:</span></div>',
                unsafe_allow_html=True)
    l1,l2,l3,l4,_=st.columns([1,1,1,1,4],gap="small")
    for co,lbl,key in [(l1,"Deforestation","deforestation"),(l2,"📊 NDVI","ndvi"),
                        (l3,"🌡 Thermal","thermal"),(l4,"🔥 Heatmap","heatmap")]:
        active=(st.session_state.layer==key)
        co.markdown(f'<div class="{"active-btn" if active else "verify-btn"}">', unsafe_allow_html=True)
        if co.button(lbl,key=f"layer_{key}",use_container_width=True):
            st.session_state.layer=key; st.rerun()
        co.markdown('</div>',unsafe_allow_html=True)

    st.plotly_chart(build_main_map(df_json,st.session_state.layer,pulse_alpha),
                    use_container_width=True,config={"displayModeBar":False})
    st.markdown(f'<div style="{MONO}font-size:8px;color:#7d8fa8;letter-spacing:.06em;'
                f'padding:2px 4px 6px;">◈ HOVER MARKERS FOR DETAILS · TOOLTIP INCLUDES GOOGLE MAPS LINK</div>',
                unsafe_allow_html=True)

    log_col,ndvi_col=st.columns([3,2],gap="small")
    with log_col:
        kind_col={"alert":"#f85149","ok":"#3fb950","info":"#58a6ff"}
        kind_ico={"alert":"⚠","ok":"✓","info":"◈"}
        rows_html="".join(
            f'<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #111827;align-items:flex-start;">'
            f'<span style="{MONO}font-size:9px;color:#58a6ff;flex-shrink:0;">[{e["t"]}]</span>'
            f'<span style="color:{kind_col.get(e["kind"],"#8b949e")};flex-shrink:0;font-size:10px;">'
            f'{kind_ico.get(e["kind"],"·")}</span>'
            f'<span style="font-size:11px;color:#8892a4;">{e["msg"]}</span></div>'
            for e in st.session_state.live_log)
        st.markdown(f"""
        <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;padding:12px 14px;margin:3px 3px 6px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <span style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.12em;">◈ LIVE EVENT FEED</span>
            <span style="{MONO}font-size:8px;color:#3fb950;animation:blink 1.5s infinite;">● LIVE</span>
          </div>{rows_html}
        </div>""", unsafe_allow_html=True)
    with ndvi_col:
        section_label("NDVI & DEFORESTATION RATE")
        st.plotly_chart(build_ndvi_chart(df_json),use_container_width=True,config={"displayModeBar":False})

    section_label("GLOBAL FOREST MONITORING NETWORK")
    st.plotly_chart(build_global_map(n_active),use_container_width=True,config={"displayModeBar":False})

    # Auto-rerun every 3 s
    ts=int(_time.time()/3)
    if st.session_state._last_tick!=ts:
        st.session_state._last_tick=ts; _time.sleep(0.04); st.rerun()

# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — FORENSIC ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif PAGE=="Forensic":
    sel_idx=st.session_state.sel_idx
    sel=alerts.iloc[sel_idx]
    conf=float(sel["confidence"]); lat=float(sel["lat"]); lon=float(sel["lon"])
    ha=float(sel["new_loss_ha"])
    est=calculate_carbon_impact(ha)
    cause="Illegal logging" if ha>6 else "Agricultural expansion"
    dates=["2023-01-10","2023-04-01","2023-07-01","2023-10-01","2024-02-01","2024-03-02"]

    left_col,mid_col,right_col=st.columns([1,3,1.2],gap="small")

    # Alert list
    with left_col:
        st.markdown(f'<div style="background:#0e1420;border-bottom:1px solid #1a2035;padding:10px 14px;">'
            f'<span style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.12em;">◈ ALERTS</span></div>',
            unsafe_allow_html=True)
        for idx,row in alerts.head(14).iterrows():
            sc="#f85149" if row["new_loss_ha"]>6 else "#d29922"
            stt="CRITICAL" if row["new_loss_ha"]>6 else "HIGH"
            is_sel=(idx==sel_idx)
            blink=('<span style="display:inline-block;width:5px;height:5px;border-radius:50%;'
                   'background:#f85149;animation:blink 1s ease-in-out infinite;margin-right:4px;"></span>'
                   ) if row["new_loss_ha"]>6 else ""
            st.markdown(f'<div style="background:{"#071a0f" if is_sel else "#080b12"};'
                f'border-left:2px solid {"#3fb950" if is_sel else "transparent"};'
                f'border-bottom:1px solid #111827;padding:8px 12px;">'
                f'<div style="{MONO}font-size:11px;font-weight:700;color:#e6edf3;">{blink}{row["alert_id"]}</div>'
                f'<div style="{MONO}font-size:9px;color:{sc};">{stt} · {row["new_loss_ha"]}ha</div></div>',
                unsafe_allow_html=True)
            if st.button("▶ Load",key=f"sel_{idx}",use_container_width=True):
                st.session_state.sel_idx=idx; st.rerun()

    # Map + AI summary + GIS links
    with mid_col:
        st.markdown(f'<div style="background:#0e1420;border:1px solid #1a2035;'
            f'border-radius:4px 4px 0 0;padding:10px 15px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.1em;">'
            f'◈ SATELLITE ACQ — {sel["alert_id"]}</span>'
            f'<span style="{MONO}font-size:9px;color:#3fb950;">{lat:.5f}°N · {lon:.5f}°W</span></div>',
            unsafe_allow_html=True)
        tl=st.select_slider("",options=range(len(dates)),format_func=lambda i:dates[i],
                             key="tl",label_visibility="collapsed")
        st.plotly_chart(build_forensic_map(lat,lon,ha,sel["alert_id"],dates[tl]),
                        use_container_width=True,config={"displayModeBar":False})

        # GIS deep-links
        gmap=f"https://maps.google.com/?q={lat},{lon}"
        gee ="https://code.earthengine.google.com/"
        st.markdown(f"""
        <div style="background:#080b12;border:1px solid #1a2035;border-radius:4px;
          padding:8px 14px;margin-top:4px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
          <span style="{MONO}font-size:8px;color:#7d8fa8;">◈ GIS DEEP-LINKS:</span>
          <a href="{gmap}" target="_blank" style="{MONO}font-size:9px;color:#58a6ff;text-decoration:none;">
            📍 Google Maps ({lat:.4f}, {lon:.4f})</a>
          <a href="{gee}" target="_blank" style="{MONO}font-size:9px;color:#3fb950;text-decoration:none;">
            🌍 Earth Engine</a>
          <span style="{MONO}font-size:8px;color:#7d8fa8;">CRS: WGS84 EPSG:4326</span>
        </div>""", unsafe_allow_html=True)

        # AI Summary
        risk="CRITICAL" if ha>6 else "HIGH"
        st.markdown(f"""
        <div style="background:#0e1420;border:1px solid #3fb95044;border-left:3px solid #3fb950;
          border-radius:4px;padding:13px 16px;margin-top:6px;">
          <div style="{MONO}font-size:9px;color:#3fb950;letter-spacing:.14em;margin-bottom:10px;">◈ AI ASSESSMENT</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;">
            <div style="font-size:12px;color:#8892a4;"><span style="color:#3fb950;">•</span> Cause: <b style="color:#e6edf3;">{cause}</b></div>
            <div style="font-size:12px;color:#8892a4;"><span style="color:#3fb950;">•</span> Area: <b style="color:#e6edf3;">{ha:.1f} ha</b></div>
            <div style="font-size:12px;color:#8892a4;"><span style="color:#3fb950;">•</span> Confidence: <b style="color:#3fb950;">{conf:.0%}</b></div>
            <div style="font-size:12px;color:#8892a4;"><span style="color:#d29922;">•</span> CO₂: <b style="color:#d29922;">{est.total_co2_tonnes:,.0f} t</b></div>
            <div style="font-size:12px;color:#8892a4;"><span style="color:#f85149;">•</span> Roads: <b style="color:#f85149;">DETECTED</b></div>
            <div style="font-size:12px;color:#8892a4;"><span style="color:#f85149;">•</span> Risk: <b style="color:#f85149;">{risk}</b></div>
            <div style="font-size:12px;color:#8892a4;grid-column:1/-1;"><span style="color:#58a6ff;">•</span> Action: <b style="color:#58a6ff;">Ranger deployment recommended</b></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Right: gauge + saliency + validation + evidence upload + mission brief
    with right_col:
        st.plotly_chart(build_confidence_gauge(conf),use_container_width=True,
                        config={"displayModeBar":False})
        sal=make_saliency_b64()
        st.markdown(f'<div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;'
            f'padding:10px 12px;margin-bottom:6px;">'
            f'<div style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.1em;margin-bottom:6px;">◈ SALIENCY MAP</div>'
            f'<img src="data:image/png;base64,{sal}" style="width:100%;border-radius:3px;"/></div>',
            unsafe_allow_html=True)

        # Validation
        st.markdown('<div style="height:4px"></div>',unsafe_allow_html=True)
        if sel["alert_id"] not in valid_ids:
            notes=st.text_input("Notes",key="vnotes",placeholder="Cloud shadow? Fire scar?",
                                label_visibility="collapsed")
            if st.button("✓ Confirm Alert",key="vconf",use_container_width=True):
                save_validation(sel["alert_id"],sel["sector"],ha,"Confirmed",notes)
                st.success("Confirmed"); st.rerun()
            if st.button("✗ False Positive",key="vfp",use_container_width=True):
                save_validation(sel["alert_id"],sel["sector"],ha,"False Positive",notes)
                st.warning("Marked FP"); st.rerun()
        else:
            v=valid_df[valid_df["alert_id"]==sel["alert_id"]].iloc[-1]
            vc="#3fb950" if v["verdict"]=="Confirmed" else "#f85149"
            st.markdown(f'<div style="padding:9px;background:#080b12;border-radius:3px;'
                f'border:1px solid {vc};text-align:center;{MONO}font-size:9px;'
                f'letter-spacing:.1em;color:{vc};">✓ {str(v["verdict"]).upper()}</div>',
                unsafe_allow_html=True)

        # Evidence upload
        st.markdown('<div style="height:6px"></div>',unsafe_allow_html=True)
        with st.expander("📷 Upload Field Evidence"):
            ranger_sel=st.selectbox("Ranger",RANGERS,key="ev_ranger")
            ev_notes=st.text_area("Notes",key="ev_notes",height=80,
                                  placeholder="Describe what you observed...")
            uploaded=st.file_uploader("Photo / File",
                type=["jpg","jpeg","png","pdf","mp4"],
                key="ev_upload",label_visibility="collapsed")
            if uploaded and st.button("📤 Submit Evidence",key="ev_submit",
                                       use_container_width=True):
                save_evidence(sel["alert_id"],ranger_sel,uploaded.name,uploaded.read(),ev_notes)
                st.success(f"Saved: {uploaded.name}"); st.rerun()
        ev_df=load_evidence(sel["alert_id"])
        if len(ev_df)>0:
            section_label(f"{len(ev_df)} FILE(S) ON RECORD")
            for _,ev in ev_df.iterrows():
                ts=str(ev["timestamp"])[:16]
                st.markdown(f'<div style="font-size:10px;color:#8892a4;padding:3px 0;">'
                    f'<span style="color:#58a6ff;">[{ts}]</span> {ev["ranger"]} — {ev["filename"]}</div>',
                    unsafe_allow_html=True)
                _,dl=get_evidence_file(int(ev["id"]))
                if dl:
                    st.download_button(f"⬇ {ev['filename']}",data=dl,
                        file_name=str(ev["filename"]),key=f"ev_dl_{ev['id']}",
                        use_container_width=True)

        # Mission brief PDF
        st.markdown('<div style="height:4px"></div>',unsafe_allow_html=True)
        ranger_brief=st.selectbox("Ranger for Brief",RANGERS,key="brief_ranger")
        if st.button("📄 Generate Mission Brief",key="gen_brief",use_container_width=True):
            pdf=generate_mission_brief(sel["alert_id"],sel["sector"],lat,lon,ha,conf,
                                       ranger_brief,est.total_co2_tonnes,cause)
            ext="pdf" if pdf[:4]==b"%PDF" else "txt"
            st.download_button("⬇ Download Brief PDF",data=pdf,
                file_name=f"mission_{sel['alert_id']}.{ext}",
                mime=f"application/{ext}",use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — NDVI ANALYSIS & CLASSIFICATION  (Synopsis alignment)
# ══════════════════════════════════════════════════════════════════
elif PAGE=="NDVI":
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── Synopsis §1.3 formula banner ──────────────────────────────
    st.markdown(f'''
    <div style="background:#0e1420;border:1px solid #3fb95044;border-left:3px solid #3fb950;
      border-radius:4px;padding:12px 18px;margin-bottom:8px;">
      <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
        <div>
          <div style="font-family:Share Tech Mono,monospace;font-size:8px;color:#7d8fa8;
            letter-spacing:.14em;margin-bottom:4px;">◈ NDVI FORMULA  (Tucker, 1979)</div>
          <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:700;
            color:#3fb950;letter-spacing:.06em;">NDVI = (NIR − RED) / (NIR + RED)</div>
          <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#8899b4;margin-top:4px;">
            Sentinel-2: Band B08 (NIR, 842nm) · Band B04 (Red, 665nm) · Range: −1 to +1
          </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:16px;flex-wrap:wrap;">
          <div style="text-align:center;background:#080b12;padding:8px 14px;border-radius:3px;
            border:1px solid #1a2035;">
            <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:700;
              color:#3fb950;">85%</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:7px;color:#7d8fa8;
              letter-spacing:.1em;">OVERALL ACCURACY</div>
          </div>
          <div style="text-align:center;background:#080b12;padding:8px 14px;border-radius:3px;
            border:1px solid #1a2035;">
            <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:700;
              color:#58a6ff;">0.78</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:7px;color:#7d8fa8;
              letter-spacing:.1em;">KAPPA SCORE</div>
          </div>
          <div style="text-align:center;background:#080b12;padding:8px 14px;border-radius:3px;
            border:1px solid #1a2035;">
            <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:700;
              color:#d29922;">15%</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:7px;color:#7d8fa8;
              letter-spacing:.1em;">ERROR RATE</div>
          </div>
          <div style="text-align:center;background:#080b12;padding:8px 14px;border-radius:3px;
            border:1px solid #1a2035;">
            <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:700;
              color:#f85149;">0.3</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:7px;color:#7d8fa8;
              letter-spacing:.1em;">THRESHOLD</div>
          </div>
        </div>
      </div>
    </div>''', unsafe_allow_html=True)

    # ── NDVI class table (Synopsis §2.4) ──────────────────────────
    st.markdown(f'''
    <div style="background:#080b12;border:1px solid #1a2035;border-radius:4px;
      padding:10px 16px;margin-bottom:8px;">
      <div style="font-family:Share Tech Mono,monospace;font-size:8px;color:#7d8fa8;
        letter-spacing:.14em;margin-bottom:8px;">◈ NDVI CLASSIFICATION TABLE  (Synopsis §2.4)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;">
        <div style="font-family:Share Tech Mono,monospace;font-size:9px;font-weight:700;
          color:#8899b4;padding:5px 10px;border-bottom:1px solid #1a2035;">NDVI RANGE</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:9px;font-weight:700;
          color:#8899b4;padding:5px 10px;border-bottom:1px solid #1a2035;">CLASS</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:9px;font-weight:700;
          color:#8899b4;padding:5px 10px;border-bottom:1px solid #1a2035;">COLOUR</div>
        {"".join([
          f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#e6edf3;'
          f'padding:6px 10px;border-bottom:1px solid #111827;">{rng}</div>'
          f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;color:{col};'
          f'padding:6px 10px;border-bottom:1px solid #111827;">{cls}</div>'
          f'<div style="padding:6px 10px;border-bottom:1px solid #111827;">'
          f'<span style="display:inline-block;width:60px;height:10px;background:{col};'
          f'border-radius:2px;"></span></div>'
          for rng, cls, col in [
            ("0.6 – 1.0", "Very Dense Forest", "#1a7a30"),
            ("0.3 – 0.6", "Moderate Vegetation", "#3fb950"),
            ("0.0 – 0.3", "Sparse Plants / Grassland", "#d29922"),
            ("< 0", "Water / Soil / Built-up", "#58a6ff"),
          ]])}
      </div>
    </div>''', unsafe_allow_html=True)

    # ── Row 1: NDVI Heatmap + Binary Map ─────────────────────────
    section_label("NDVI HEATMAP  (Sentinel-2 Simulated AOI · 20×20 Pixels · NDVI = (NIR−RED)/(NIR+RED))")
    hm_col, bin_col = st.columns(2, gap="small")
    with hm_col:
        st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;'
            f'color:#7d8fa8;letter-spacing:.1em;padding:3px 4px;">◈ NDVI HEATMAP</div>',
            unsafe_allow_html=True)
        st.plotly_chart(build_ndvi_heatmap_grid(), use_container_width=True,
                        config={"displayModeBar": False})
    with bin_col:
        st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;'
            f'color:#7d8fa8;letter-spacing:.1em;padding:3px 4px;">'
            f'◈ BINARY CLASSIFICATION MAP  (Threshold = 0.3)</div>',
            unsafe_allow_html=True)
        st.plotly_chart(build_binary_class_map(), use_container_width=True,
                        config={"displayModeBar": False})
    st.markdown(f'''
    <div style="background:#080b12;border:1px solid #1a2035;border-radius:4px;
      padding:7px 14px;margin-top:2px;margin-bottom:8px;">
      <span style="font-family:Share Tech Mono,monospace;font-size:8px;color:#7d8fa8;">
        ◈ Left: NDVI heatmap — colour-coded per synopsis §2.4 thresholds.&nbsp;&nbsp;
        Right: Binary Forest (green) / Non-Forest (red) map from threshold = 0.3 (synopsis §3.3.3)
      </span>
    </div>''', unsafe_allow_html=True)

    # ── Row 2: Classification distribution + Confusion matrix ──────
    cls_col, cm_col = st.columns([3, 2], gap="small")

    # Compute NDVI values for current alerts
    ndvi_vals = [round(0.72 - float(ha) * 0.03, 3) for ha in alerts["new_loss_ha"]]

    with cls_col:
        section_label("NDVI CLASS DISTRIBUTION — CURRENT ALERTS")
        st.plotly_chart(build_ndvi_classification(ndvi_vals),
                        use_container_width=True, config={"displayModeBar": False})

    with cm_col:
        section_label("CONFUSION MATRIX  (Synopsis §3.4 · OA=85% · κ=0.78)")
        st.plotly_chart(build_confusion_matrix(), use_container_width=True,
                        config={"displayModeBar": False})
        # Accuracy metrics block
        st.markdown(f'''
        <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;
          padding:10px 14px;margin-top:4px;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div style="font-size:11px;color:#8892a4;">Overall Accuracy:
              <b style="color:#3fb950;">85%</b></div>
            <div style="font-size:11px;color:#8892a4;">Kappa Score:
              <b style="color:#58a6ff;">0.78</b></div>
            <div style="font-size:11px;color:#8892a4;">Error Rate:
              <b style="color:#d29922;">15%</b></div>
            <div style="font-size:11px;color:#8892a4;">Validation:
              <b style="color:#e6edf3;">Ground Truth</b></div>
            <div style="font-size:11px;color:#8892a4;">Precision:
              <b style="color:#3fb950;">89.4%</b></div>
            <div style="font-size:11px;color:#8892a4;">Recall:
              <b style="color:#3fb950;">86.4%</b></div>
          </div>
        </div>''', unsafe_allow_html=True)

    # ── Row 3: Tech stack + GeoTIFF export ─────────────────────────
    tech_col, geo_col = st.columns([3, 2], gap="small")
    with tech_col:
        section_label("TECHNOLOGY STACK  (Synopsis §1.6)")
        st.markdown(f'''
        <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;padding:12px 16px;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            {"".join([
              f'<div style="background:#080b12;border:1px solid #111827;border-radius:3px;'
              f'padding:8px 12px;">'
              f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;font-weight:700;'
              f'color:{col};">{tech}</div>'
              f'<div style="font-size:10px;color:#8899b4;margin-top:3px;">{usage}</div></div>'
              for tech, usage, col in [
                ("Sentinel-2", "Source multispectral imagery · 13 bands", "#58a6ff"),
                ("Python + Rasterio", "Band extraction · NDVI computation", "#3fb950"),
                ("NumPy", "Array-based NDVI formula · thresholding", "#d29922"),
                ("Matplotlib", "Heatmap + binary map visualisation", "#d29922"),
                ("QGIS (optional)", "GIS-based spatial analysis of GeoTIFF", "#4a5568"),
                ("ML (Future)", "Supervised classification · trend prediction", "#f85149"),
              ]])}
          </div>
        </div>''', unsafe_allow_html=True)

    with geo_col:
        section_label("GEOTIFF EXPORT  (Synopsis §1.4 · §3.1)")
        # GeoTIFF metadata panel
        st.markdown(f'''
        <div style="background:#0e1420;border:1px solid #1a4028;border-left:3px solid #3fb950;
          border-radius:4px;padding:12px 16px;">
          <div style="font-family:Share Tech Mono,monospace;font-size:8px;color:#7d8fa8;
            letter-spacing:.14em;margin-bottom:10px;">◈ GEOTIFF OUTPUT METADATA</div>
          {"".join([
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'border-bottom:1px solid #111827;">'
            f'<span style="font-size:11px;color:#8899b4;">{k}</span>'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:10px;'
            f'color:#e6edf3;">{v}</span></div>'
            for k, v in [
              ("Format", "GeoTIFF (.tif)"),
              ("CRS", "WGS84 EPSG:4326"),
              ("Resolution", "10 m/pixel (Band B04/B08)"),
              ("Bands", "1 (NDVI float32)"),
              ("No-Data Value", "-9999"),
              ("Value Range", "−1.0 to +1.0"),
              ("Driver", "GDAL / rasterio"),
            ]])}
        </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        # Generate a minimal GeoTIFF-like CSV export (actual rasterio not available in demo)
        rng_e = np.random.default_rng(42)
        H, W = 20, 20
        nir = np.clip(rng_e.normal(0.45, 0.15, (H, W)), 0.01, 1.0)
        red = np.clip(rng_e.normal(0.20, 0.12, (H, W)), 0.01, 1.0)
        nir[12:, 12:] = np.clip(rng_e.normal(0.15, 0.05, (8, 8)), 0.01, 0.4)
        red[12:, 12:] = np.clip(rng_e.normal(0.35, 0.06, (8, 8)), 0.1, 0.7)
        ndvi_grid = (nir - red) / (nir + red + 1e-8)
        rows_list = []
        for r in range(H):
            for c in range(W):
                lat_px = -3.47 + r * 0.001
                lon_px = -60.02 + c * 0.001
                cls_lbl = ("Dense Forest" if ndvi_grid[r,c]>0.6
                           else "Moderate Veg" if ndvi_grid[r,c]>0.3
                           else "Sparse" if ndvi_grid[r,c]>0 else "Non-Veg")
                rows_list.append({"row": r, "col": c, "lat": round(lat_px, 5),
                                  "lon": round(lon_px, 5),
                                  "ndvi": round(float(ndvi_grid[r, c]), 4),
                                  "class": cls_lbl})
        geo_df = pd.DataFrame(rows_list)
        csv_bytes = geo_df.to_csv(index=False).encode()
        st.markdown('<div class="dispatch-btn">', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download NDVI Grid (CSV/GeoTIFF proxy)",
            data=csv_bytes,
            file_name=f"ndvi_output_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:8px;'
            f'color:#7d8fa8;padding:5px 0;letter-spacing:.06em;">'
            f'◈ Contains NDVI, lat/lon, classification per pixel. '
            f'In production: rasterio.open() writes real GeoTIFF.</div>',
            unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICTIVE RISK
# ══════════════════════════════════════════════════════════════════
elif PAGE=="Predictive":
    st.markdown('<div style="height:6px"></div>',unsafe_allow_html=True)
    forecast=get_forecast_cached()
    hist_df =get_historical_series(52)
    insights=get_correlation_insights(hist_df)

    # KPI row
    fk1,fk2,fk3,fk4=st.columns(4,gap="small")
    td=forecast["trend_dir"]
    kpi_card(fk1,"30-Day Forecast",f"{forecast['total_30d']:.0f}ha","total loss estimate","#f85149")
    kpi_card(fk2,"Trend Direction",td,"","#d29922" if "RISING" in td else "#3fb950")
    kpi_card(fk3,"Peak Week",forecast["peak_date"],f"{forecast['peak_value']:.0f} ha/wk","#f85149")
    kpi_card(fk4,"Weekly Slope",f"{forecast['slope_weekly']:+.1f}ha","per week trend","#d29922")

    chart_col,corr_col=st.columns([3,2],gap="small")
    with chart_col:
        section_label("DEFORESTATION FORECAST — NEXT 30 DAYS  (LINEAR + FFT SEASONAL)")
        st.plotly_chart(build_forecast_chart(forecast),use_container_width=True,
                        config={"displayModeBar":False})
        st.markdown(f"""
        <div style="background:#080b12;border:1px solid #1a2035;border-radius:4px;padding:8px 14px;margin-top:4px;">
          <div style="{MONO}font-size:8px;color:#7d8fa8;letter-spacing:.1em;margin-bottom:6px;">◈ MODEL METADATA</div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <span style="font-size:11px;color:#8892a4;">Method: <b style="color:#e6edf3;">Linear + FFT Seasonal</b></span>
            <span style="font-size:11px;color:#8892a4;">History: <b style="color:#e6edf3;">52 weeks</b></span>
            <span style="font-size:11px;color:#8892a4;">Horizon: <b style="color:#e6edf3;">30 days</b></span>
            <span style="font-size:11px;color:#8892a4;">CI Band: <b style="color:#e6edf3;">90%</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

    with corr_col:
        section_label("ENVIRONMENTAL CORRELATION INSIGHTS")
        for ins in insights:
            r=ins["correlation"]; bar_w=int(abs(r)*100)
            bar_c="#f85149" if r>0 else "#3fb950"
            r_lbl=f"+{r:.2f}" if r>0 else f"{r:.2f}"
            r_css=f"color:#f85149;font-weight:700;" if r>0 else f"color:#3fb950;font-weight:700;"
            st.markdown(f"""
            <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;
              padding:11px 14px;margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:12px;color:#e6edf3;">{ins['icon']}&nbsp;{ins['variable']}</span>
                <span style="{MONO}font-size:12px;{r_css}">{r_lbl}</span>
              </div>
              <div style="background:#080b12;border-radius:2px;height:4px;overflow:hidden;margin-bottom:7px;">
                <div style="height:100%;width:{bar_w}%;background:{bar_c};border-radius:2px;"></div>
              </div>
              <div style="font-size:11px;color:#8899b4;">{ins['finding']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0e1420;border:1px solid #f8514944;border-left:3px solid #f85149;
      border-radius:4px;padding:14px 18px;margin-top:4px;">
      <div style="{MONO}font-size:9px;color:#f85149;letter-spacing:.14em;margin-bottom:10px;">◈ 30-DAY RISK SUMMARY</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        <div style="font-size:12px;color:#8892a4;"><span style="color:#f85149;">•</span> Forecast total: <b style="color:#e6edf3;">{forecast['total_30d']:.0f} ha</b></div>
        <div style="font-size:12px;color:#8892a4;"><span style="color:#d29922;">•</span> Peak week: <b style="color:#d29922;">{forecast['peak_date']}</b> — {forecast['peak_value']:.0f}ha/wk</div>
        <div style="font-size:12px;color:#8892a4;"><span style="color:#f85149;">•</span> Dry season risk elevated — pre-deploy rangers</div>
        <div style="font-size:12px;color:#8892a4;"><span style="color:#3fb950;">•</span> Humidity: strongest negative predictor</div>
        <div style="font-size:12px;color:#8892a4;"><span style="color:#58a6ff;">•</span> Road density index rising — new incursion risk</div>
        <div style="font-size:12px;color:#8892a4;"><span style="color:#3fb950;">•</span> Wet season expected to suppress activity post-peak</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — FIELD DISPATCH
# ══════════════════════════════════════════════════════════════════
elif PAGE=="Field":
    dispatched_n=int((dispatch_df["status"]=="Dispatched").sum()) if len(dispatch_df)>0 else 0
    pending_n=max(0,n_active-dispatched_n)

    ops_col,rate_col=st.columns([3,2],gap="small")
    with ops_col:
        st.markdown(f'''
        <div style="background:#0e1420;border:1px solid #1a4028;border-top:2px solid #3fb950;
          border-radius:4px;padding:16px 20px;margin:6px 3px 4px;">
          <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#7d8fa8;
            letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;">
            ◈ ACTIVE FIELD OPERATIONS</div>
          <div style="display:flex;gap:32px;align-items:baseline;">
            <div><span style="font-size:36px;font-weight:700;color:#3fb950;
              font-family:Barlow Condensed,sans-serif;">{dispatched_n}</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
              color:#3fb950;margin-left:4px;">DISPATCHED</span></div>
            <div><span style="font-size:36px;font-weight:700;color:#d29922;
              font-family:Barlow Condensed,sans-serif;">{pending_n}</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
              color:#d29922;margin-left:4px;">PENDING</span></div>
            <div><span style="font-size:36px;font-weight:700;color:#f85149;
              font-family:Barlow Condensed,sans-serif;">0</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
              color:#f85149;margin-left:4px;">CRITICAL DELAYED</span></div>
          </div>
        </div>''', unsafe_allow_html=True)
    with rate_col:
        st.markdown('''
        <div style="background:#0e1420;border:1px solid #1a4028;border-top:2px solid #3fb950;
          border-radius:4px;padding:16px 20px;margin:6px 3px 4px;">
          <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#7d8fa8;
            letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;">
            ◈ FIELD VERIFICATION RATE (30D)</div>
          <div style="font-size:38px;font-weight:700;color:#e6edf3;
            font-family:Barlow Condensed,sans-serif;">92%</div>
          <div style="background:#080b12;border-radius:2px;height:5px;overflow:hidden;margin-top:8px;">
            <div style="height:100%;width:92%;
              background:linear-gradient(to right,#f85149,#d29922,#3fb950);border-radius:2px;">
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

    RANGER_STATUS=[
        ("Ranger Alpha","En Route",     "#d29922","ETA: 18 min","12 km","Helicopter"),
        ("Ranger Beta", "Investigating","#f85149","On-site",    "0 km", "Ground"),
        ("Ranger Gamma","Returning",    "#3fb950","ETA: 42 min","28 km","Vehicle"),
        ("Ranger Delta","Standby",      "#58a6ff","Ready",      "—",    "Drone"),
    ]
    section_label("RANGER STATUS & ETA")
    rc1,rc2,rc3,rc4=st.columns(4,gap="small")
    for co,(name,status,sc,eta,dist,transport) in zip([rc1,rc2,rc3,rc4],RANGER_STATUS):
        co.markdown(f"""
        <div style="background:#0e1420;border:1px solid #1a2035;border-top:2px solid {sc};
          border-radius:4px;padding:12px 14px;">
          <div style="{MONO}font-size:10px;font-weight:700;color:#e6edf3;margin-bottom:6px;">{name}</div>
          <div style="display:inline-block;padding:2px 8px;border-radius:2px;
            {MONO}font-size:8px;font-weight:700;background:{sc}18;color:{sc};
            border:1px solid {sc}44;letter-spacing:.08em;margin-bottom:8px;">{status}</div>
          <div style="{MONO}font-size:9px;color:#7d8fa8;margin-top:4px;">{eta}</div>
          <div style="{MONO}font-size:9px;color:#7d8fa8;">{dist} · {transport}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    th,tf1,tf2=st.columns([3,1,1],gap="small")
    with th:
        st.markdown(f'<div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px 0 0 0;padding:10px 18px;">'
            f'<span style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.12em;">◈ PENDING ALERTS — FIELD VERIFICATION QUEUE</span></div>',
            unsafe_allow_html=True)
    with tf1:
        pf=st.selectbox("",["Priority: High","Priority: Medium","All"],key="pf",label_visibility="collapsed")
    with tf2:
        st.selectbox("",["Status: All","Status: Pending","Status: Dispatched"],key="sf",label_visibility="collapsed")

    st.markdown(f'<div style="background:#080b12;border:1px solid #1a2035;border-top:none;'
        f'display:grid;grid-template-columns:3fr 1fr 2fr 2fr;padding:6px 18px;gap:10px;">'
        f'<div style="{MONO}font-size:8px;color:#7d8fa8;letter-spacing:.1em;">ALERT ID / SEVERITY</div>'
        f'<div style="{MONO}font-size:8px;color:#7d8fa8;">STATUS</div>'
        f'<div></div><div></div></div>',unsafe_allow_html=True)

    show=alerts.copy()
    if "High" in pf: show=show[show["new_loss_ha"]>6]
    elif "Medium" in pf: show=show[(show["new_loss_ha"]>3)&(show["new_loss_ha"]<=6)]

    for ri,(_,row) in enumerate(show.head(8).iterrows()):
        stt="CRITICAL" if row["new_loss_ha"]>6 else "HIGH"
        sc="#f85149"  if row["new_loss_ha"]>6 else "#d29922"
        is_d=row["alert_id"] in disp_ids
        bg="#071a0f" if is_d else "#080b12"; bc="#3fb950" if is_d else sc
        blink=('<span style="display:inline-block;width:5px;height:5px;border-radius:50%;'
               'background:#f85149;animation:blink 1s ease-in-out infinite;margin-right:5px;"></span>'
               ) if (not is_d and row["new_loss_ha"]>6) else ""
        st.markdown(f'<div style="background:{bg};border:1px solid #111827;border-top:none;'
            f'display:grid;grid-template-columns:3fr 1fr 2fr 2fr;padding:9px 18px;gap:10px;align-items:center;">'
            f'<div>{blink}<span style="{MONO}font-size:12px;font-weight:700;color:#e6edf3;">{row["alert_id"]}</span>'
            f'<span style="{MONO}font-size:9px;color:{sc};"> [{stt}]</span>'
            f'<span style="font-size:10px;color:#7d8fa8;"> · {row["sector"]} · {row["new_loss_ha"]}ha</span></div>'
            f'<div><span style="display:inline-block;padding:2px 7px;border-radius:2px;{MONO}font-size:8px;'
            f'font-weight:700;background:{bc}18;color:{bc};border:1px solid {bc}44;letter-spacing:.06em;">'
            f'{"DISPATCHED" if is_d else "PENDING"}</span></div>'
            f'<div style="min-height:32px;"></div><div style="min-height:32px;"></div></div>',
            unsafe_allow_html=True)

        _,_,bd,bv=st.columns([3,1,2,2],gap="small")
        with bd:
            if not is_d:
                st.markdown('<div class="dispatch-btn">',unsafe_allow_html=True)
                if st.button("📡 Dispatch Ranger",key=f"d_{ri}",use_container_width=True):
                    ranger=RANGERS[ri%len(RANGERS)]
                    save_dispatch(row["alert_id"],row["sector"],ranger,"Dispatched")
                    st.rerun()
                st.markdown('</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="{MONO}font-size:9px;color:#3fb950;padding:8px 0;'
                    f'letter-spacing:.1em;">✓ DISPATCHED</div>',unsafe_allow_html=True)
        with bv:
            st.markdown('<div class="verify-btn">',unsafe_allow_html=True)
            if st.button("🔍 Request Verify",key=f"rv_{ri}",use_container_width=True):
                save_dispatch(row["alert_id"],row["sector"],"Field Team","Verification Requested")
                st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    log_html=""
    for _,r in dispatch_df.head(6).iterrows():
        ts=str(r.get("timestamp","")); hm=ts[11:16] if len(ts)>15 else "00:00"
        log_html+=(f'<div style="font-size:11px;padding:5px 0;border-bottom:1px solid #111827;">'
                   f'<span style="color:#58a6ff;{MONO}">[{hm}]</span> '
                   f'{r.get("ranger","Team")} → {r.get("sector","—")}: {r.get("status","update")}</div>')
    if not log_html:
        for i,msg in enumerate(["Ranger Alpha → Sector-B: Illegal road detected.",
                                 "Ranger Beta → Sector-A: Coordinates verified on-site."]):
            t=(datetime.utcnow()-timedelta(hours=i+1)).strftime("%H:%M")
            log_html+=(f'<div style="font-size:11px;padding:5px 0;border-bottom:1px solid #111827;">'
                       f'<span style="color:#58a6ff;{MONO}">[{t}]</span> {msg}</div>')
    st.markdown(f"""
    <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;padding:13px 17px;">
      <div style="{MONO}font-size:9px;color:#7d8fa8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;">
        ◈ FIELD ACTIVITY LOG — SQLite-BACKED</div>
      {log_html}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 5 — MONTHLY REPORT
# ══════════════════════════════════════════════════════════════════
elif PAGE=="Report":
    st.markdown('<div style="height:6px"></div>',unsafe_allow_html=True)
    total_co2=sum(calculate_carbon_impact(float(ha)).total_co2_tonnes for ha in alerts["new_loss_ha"])
    credits=total_co2*0.85; usd_value=credits*14.20
    fp_count=len(valid_df[valid_df["verdict"]=="False Positive"]) if len(valid_df)>0 else 0
    fp_rate=(fp_count/max(1,len(valid_df)))*100
    stats={"period":datetime.utcnow().strftime("%B %Y"),"ha_monitored":1_200_000,
           "ha_lost":forest_loss,"n_alerts":len(alerts),"n_critical":n_crit,
           "n_validated":len(valid_df),"fp_rate":fp_rate,"co2_prevented":total_co2,
           "credits":credits,"usd_value":usd_value,"dispatched":dispatched_count,
           "avg_response":"2h 15m","verify_rate":92,"n_evidence":len(load_evidence())}

    rk1,rk2,rk3,rk4,rk5=st.columns(5,gap="small")
    for co,lbl,val,accent in [(rk1,"Hectares Monitored","1.2M km²","#3fb950"),
                               (rk2,"Forest Loss",f"{forest_loss:.0f} ha","#f85149"),
                               (rk3,"CO₂ Prevented",f"{total_co2:,.0f} t","#d29922"),
                               (rk4,"Carbon Value",f"${usd_value:,.0f}","#58a6ff"),
                               (rk5,"Verify Rate","92%","#3fb950")]:
        co.markdown(f"""
        <div style="background:#0e1420;border:1px solid #1a2035;border-top:2px solid {accent};
          border-radius:4px;padding:12px 14px;margin:6px 3px 3px;text-align:center;">
          <div style="{MONO}font-size:8px;color:#7d8fa8;letter-spacing:.1em;
            text-transform:uppercase;margin-bottom:8px;">{lbl}</div>
          <div style="font-size:22px;font-weight:700;color:{accent};
            font-family:'Barlow Condensed',sans-serif;">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    rep_col,detail_col=st.columns([2,3],gap="small")

    with rep_col:
        rows_html="".join(
            f'<div style="font-size:11px;color:#8892a4;">{k}:</div>'
            f'<div style="{MONO}font-size:11px;font-weight:700;color:#e6edf3;">{v}</div>'
            for k,v in [
                ("Period",stats["period"]),("Area",f"{stats['ha_monitored']:,} ha"),
                ("Forest Loss",f"{stats['ha_lost']:.1f} ha"),("Alerts",str(stats["n_alerts"])),
                ("Critical",str(stats["n_critical"])),("Validated",str(stats["n_validated"])),
                ("FP Rate",f"{stats['fp_rate']:.1f}%"),("Dispatched",str(stats["dispatched"])),
                ("Response",stats["avg_response"]),("Verify Rate",f"{stats['verify_rate']}%"),
                ("CO₂ Prevented",f"{stats['co2_prevented']:,.0f} t"),
                ("Carbon Value",f"${stats['usd_value']:,.0f}"),
            ])
        st.markdown(f'''
        <div style="background:#0e1420;border:1px solid #1a2035;border-radius:4px;
          padding:16px 18px;margin-bottom:8px;">
          <div style="font-family:Share Tech Mono,monospace;font-size:9px;color:#7d8fa8;
            letter-spacing:.14em;margin-bottom:14px;">
            ◈ MONTHLY REPORT — {stats["period"].upper()}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{rows_html}</div>
        </div>''', unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
        pdf=generate_monthly_report(stats)
        ext="pdf" if pdf[:4]==b"%PDF" else "txt"
        st.markdown('<div class="dispatch-btn">',unsafe_allow_html=True)
        st.download_button("📄 Download Monthly Report PDF",data=pdf,
            file_name=f"admrs_report_{datetime.utcnow().strftime('%Y%m')}.{ext}",
            mime=f"application/{ext}",use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
        st.download_button("📊 Export Alerts CSV",
            data=alerts.to_csv(index=False).encode(),
            file_name=f"alerts_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",use_container_width=True)

    with detail_col:
        rng2=np.random.default_rng(55)
        months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        co2m=[round(total_co2/12+rng2.uniform(-200,200),0) for _ in months]
        lossm=[round(forest_loss/12+rng2.uniform(-10,10),1) for _ in months]
        fig2=make_subplots(rows=2,cols=1,row_heights=[0.5,0.5],vertical_spacing=0.1,
            subplot_titles=["CO₂ IMPACT BY MONTH (tonnes)","FOREST LOSS BY MONTH (ha)"])
        fig2.add_trace(go.Bar(x=months,y=co2m,name="CO₂",
            marker=dict(color=["#f85149" if v>total_co2/12 else "#d29922" for v in co2m],
                        opacity=0.85)),row=1,col=1)
        fig2.add_trace(go.Scatter(x=months,y=lossm,mode="lines+markers",name="Loss",
            line=dict(color="#3fb950",width=2),marker=dict(size=5)),row=2,col=1)
        fig2.add_hline(y=forest_loss/12,line_dash="dash",line_color="#d29922",
            annotation_text="Monthly avg",annotation_font_size=8,
            annotation_font_color="#d29922",row=2,col=1)
        fig2.update_layout(paper_bgcolor="#0e1420",plot_bgcolor="#0b0e17",font_color="#c9d1d9",
            margin=dict(l=8,r=8,t=28,b=8),height=300,showlegend=False,
            xaxis=dict(gridcolor="#111827",tickfont=dict(size=8,family="Share Tech Mono")),
            yaxis=dict(gridcolor="#111827"),
            xaxis2=dict(gridcolor="#111827",tickfont=dict(size=8,family="Share Tech Mono")),
            yaxis2=dict(gridcolor="#111827"))
        for ann in fig2.layout.annotations:
            ann.font.update(size=8,color="#2d3a52",family="Share Tech Mono")
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

        if len(valid_df)>0:
            v2=valid_df["verdict"].value_counts()
            fig3=go.Figure(go.Pie(labels=list(v2.index),values=list(v2.values),hole=0.6,
                marker=dict(colors=["#3fb950","#f85149","#d29922"],
                            line=dict(color="#0b0e17",width=2)),
                textfont=dict(size=9,family="Share Tech Mono")))
            fig3.update_layout(paper_bgcolor="#0e1420",font_color="#c9d1d9",
                margin=dict(l=0,r=0,t=20,b=0),height=170,
                legend=dict(bgcolor="#0e1420",font=dict(size=9)),
                annotations=[dict(text=f"{len(valid_df)}<br>total",
                    font=dict(size=11,color="#e6edf3"),showarrow=False)])
            st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})
