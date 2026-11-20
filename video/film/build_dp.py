"""CS Funda — "The End-to-End Data Science Pipeline" (YouTube 16:9 tutorial + one 9:16 Reel under 3 minutes).

Reuses the CS Funda engine and the ML Models helpers.
python build_dp.py        -> this folder (YouTube 1920x1080)
python build_dp.py reel   -> ../reel (1080x1920)
Every number comes from notebooks/pipeline_live.ipynb (-> reports/viz.json) and the raw file (-> reports/examples.json).
"""
import importlib.util, json, math, re, shutil, sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
VID = HERE.parent
ROOT = VID.parent

spec = importlib.util.spec_from_file_location("ml", Path.home() / "Projects/MLModels/video/film/build_ml.py")
ml = importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
b, e, sg = ml.b, ml.e, ml.sg

# ---- point the engine at this project
b.HERE = HERE; e.HERE = HERE; sg.HERE = HERE
b.REC = VID
b.EP1_SFX = False
b.BOXES = json.loads((ROOT / "recording/boxes.json").read_text())
meta = json.loads((VID / "vo/scenes.json").read_text())
b.SCENES = dict(meta)
b.OUTROS = {}
V = json.loads((ROOT / "reports/viz.json").read_text())
EX = json.loads((ROOT / "reports/examples.json").read_text())
INK, AMB, RED, GRN, MUTE, BLU, PUR = ml.INK, ml.AMB, ml.RED, ml.GRN, ml.MUTE, ml.BLU, ml.PUR
at, label, chapter, glass, aura = b.at, b.label, b.chapter, b.glass, b.aura
chart, curve, hbars, layer, card, meters, stamp, countup, capture, seg = e.chart, e.curve, e.hbars, e.layer, sg.card, sg.meters, sg.stamp, sg.countup, sg.capture, sg.seg
tip, formula, frames, pulse = ml.tip, ml.formula, ml.frames, ml.pulse
sfx, EXTRA = e.sfx, e.EXTRA
MONO = "font-family:'JetBrains Mono',monospace;"

STAGES = [("The Question", "DEFINE"), ("Ingest", "LOAD THE RAW FILE"), ("Validate", "PROFILE · RULES"), ("Clean", "FIX · DOCUMENT"), ("Store", "RAW → CLEAN LAYER"),
          ("Explore", "EDA"), ("Features", "ENGINEER"), ("Split", "HONEST TEST"), ("Pipeline", "PREPROCESS"), ("Leakage", "THE TRAP"),
          ("Model", "BASELINE FIRST"), ("Evaluate", "WHERE IT FAILS"), ("Explain", "WHAT MATTERS"), ("Deploy", "SERVE"), ("Monitor", "DRIFT")]
STAGE_OF = {"Y03": 1, "Y04": 2, "Y05": 2, "Y06": 3, "Y07": 3, "Y08": 3, "Y09": 4, "Y10": 4, "Y11": 4, "Y12": 4, "Y13": 4, "Y14": 5, "Y15": 6, "Y16": 6, "Y17": 6,
            "Y18": 7, "Y19": 8, "Y20": 8, "Y21": 9, "Y22": 10, "Y23": 11, "Y24": 11, "Y25": 12, "Y26": 12, "Y27": 13, "Y28": 14, "Y29": 15, "Y30": 15}
FIRST_BEAT = {"Y03", "Y04", "Y06", "Y09", "Y14", "Y15", "Y18", "Y19", "Y21", "Y22", "Y23", "Y25", "Y27", "Y28", "Y29"}
b.HOLD = {"Y01": 0.6, "Y02": 0.8, "Y05": 0.6, "Y08": 1.0, "Y12": 0.8, "Y13": 0.8, "Y17": 0.8, "Y19": 0.8, "Y22": 1.0, "Y24": 1.0, "Y26": 0.8, "Y28": 1.0,
          "Y30": 1.0, "Y31": 1.2, "Y32": 1.6, "Q01": 0.4, "Q03": 0.4, "Q06": 0.4, "Q07": 0.6, "Q08": 0.4, "Q10": 1.2}
b.SFX = {k: ("whoosh-short", 0.0) for k in FIRST_BEAT}
b.SFX.update({"Y31": ("whoosh-cinematic", 0.0), "Q02": ("whoosh-short", 0.0), "Q04": ("whoosh-short", 0.0), "Q06": ("whoosh-short", 0.0), "Q08": ("whoosh-short", 0.0)})
PIPE = ["Question", "Ingest", "Validate", "Clean", "Store", "Explore", "Features", "Split", "Pipeline", "Leakage", "Model", "Evaluate", "Explain", "Deploy", "Monitor"]
f1 = lambda n: f"{n:,.0f}"


# ---------------------------------------------------------------- frame pieces
def flow_label(n):
    """What is flowing through the pipeline at stage n (real counts)."""
    S_, I_ = V["split"], V["ingest"]
    return {1: "a question, no data yet", 2: f"{I_['rows']:,} rows", 3: f"{I_['rows']:,} rows", 4: f"{I_['rows']:,} → {V['store']['rows']:,} rows",
            5: f"{V['store']['rows']:,} rows", 6: f"{V['store']['rows']:,} rows", 7: f"{V['store']['rows']:,} rows · 27 features",
            8: f"{S_['train']:,} train · {S_['test']:,} test", 9: f"{S_['train']:,} train rows", 10: f"{S_['train']:,} train rows", 11: f"{S_['train']:,} train rows",
            12: f"{S_['test']:,} test rows", 13: "2,000 test rows", 14: "1 listing per request", 15: f"{V['monitor']['n_new']:,} new listings"}[n]


def stage_header(sc, n):
    """A live pipeline track: 15 stations, a data packet travels to stage n carrying what flows through it."""
    u, W, H = sc.u, sc.W, sc.H
    name, tag = STAGES[n - 1]
    x0, y0, x1 = W * 0.05, H * 0.05, W * 0.60
    xs = [x0 + i * (x1 - x0) / 14 for i in range(15)]
    first = sc.sid in FIRST_BEAT and n > 1
    plate = sc.id("hplate")
    sc.add(f'<div id="{plate}"></div>', f"#{plate}{{position:absolute;left:0;top:0;width:{W}px;height:{y0 + 110 * u}px;background:linear-gradient(#141517 0,#141517 75%,rgba(20,21,23,0));z-index:8;}}")
    base, fill, pk, ring, chip = (sc.id(k) for k in ("tbase", "tfill", "tpk", "tring", "tchip"))
    nodes = "".join(f'<i style="left:{xx - x0 - (9 if i == n - 1 else 6) * u:.1f}px;width:{(18 if i == n - 1 else 12) * u}px;height:{(18 if i == n - 1 else 12) * u}px;'
                    f'top:{-(9 if i == n - 1 else 6) * u + 2 * u:.1f}px;background:{AMB if i < n - 1 else (INK if i == n - 1 else "#3A3B3F")}"></i>' for i, xx in enumerate(xs))
    sc.add(f'<div id="{base}" class="tb0">{nodes}</div><div id="{fill}" class="tf0"></div><div id="{ring}" class="tr0"></div><div id="{pk}" class="tp0"></div>'
           f'<div id="{chip}" class="tc0">▸ {flow_label(n)}</div>',
           f"#{base}{{left:{x0}px;top:{y0}px;width:{x1 - x0}px;}}#{fill}{{left:{x0}px;top:{y0}px;width:{max(1, xs[n - 1] - x0):.1f}px;}}"
           f"#{pk}{{left:{xs[n - 1] - 11 * u:.1f}px;top:{y0 - 9 * u:.1f}px;}}#{ring}{{left:{xs[n - 1] - 11 * u:.1f}px;top:{y0 - 9 * u:.1f}px;}}#{chip}{{left:{x1 + 34 * u}px;top:{y0 - 15 * u}px;}}")
    sc.css.append(f"#root .tb0{{position:absolute;height:{4 * u}px;background:rgba(243,240,232,.14);border-radius:{2 * u}px;z-index:9;}}#root .tb0 i{{position:absolute;display:block;border-radius:50%;}}"
                  f"#root .tf0{{position:absolute;height:{4 * u}px;background:{AMB};border-radius:{2 * u}px;z-index:9;transform-origin:left center;}}"
                  f"#root .tp0,#root .tr0{{position:absolute;width:{22 * u}px;height:{22 * u}px;border-radius:50%;z-index:10;}}"
                  f"#root .tp0{{background:{AMB};box-shadow:0 0 {18 * u}px {6 * u}px rgba(255,178,36,.55);}}#root .tr0{{border:{3 * u}px solid {AMB};}}"
                  f"#root .tc0{{position:absolute;z-index:9;{MONO}font-size:{22 * u}px;font-weight:700;color:{AMB};white-space:nowrap;}}")
    d = b.dur(sc.sid)
    if first:      # the packet travels from the previous station; the filled track grows with it
        seg_ = xs[n - 1] - xs[n - 2]
        sc.js.append(f'tl.fromTo("#{fill}",{{scaleX:{(xs[n - 2] - x0) / max(1, xs[n - 1] - x0):.3f}}},{{scaleX:1,duration:.8,ease:"power2.inOut"}},.1);'
                     f'tl.fromTo("#{pk}",{{x:{-seg_:.1f}}},{{x:0,duration:.8,ease:"power2.inOut"}},.1);'
                     f'tl.fromTo("#{chip}",{{opacity:0,x:{-14 * u}}},{{opacity:1,x:0,duration:.4}},.7);')
    reps = max(0, int((d - 1.0) / 1.4) - 1)
    sc.js.append(f'tl.fromTo("#{ring}",{{scale:.7,opacity:.9}},{{scale:2.4,opacity:0,duration:1.4,ease:"power2.out",repeat:{reps}}},.9);')
    hd = sc.id("mh")
    sc.add(f'<div id="{hd}" class="mh"><span>STAGE {n:02d}</span><b>{escape(name)}</b><em>{tag}</em></div>', f"#{hd}{{left:{x0}px;top:{y0 + 30 * u}px;}}")
    sc.css.append(f"#root .mh{{position:absolute;display:flex;align-items:baseline;gap:{18 * u}px;white-space:nowrap;z-index:9;}}#root .mh span{{{MONO}color:{AMB};font-size:{26 * u}px;font-weight:700;}}"
                  f"#root .mh b{{font-size:{40 * u}px;font-weight:900;}}#root .mh em{{font-style:normal;font-size:{22 * u}px;font-weight:800;color:#141517;background:{AMB};padding:{4 * u}px {12 * u}px;border-radius:{8 * u}px;}}")
    if sc.sid in FIRST_BEAT:
        sc.js.append(f'tl.fromTo("#{hd} b",{{opacity:0,x:{-30 * u}}},{{opacity:1,x:0,duration:.5,ease:"expo.out"}},0.05);')


# The pipeline agent: the real code of each step, typed live, then the real result it printed.
AGENT = {
    "Y03": ('TARGET, METRIC = "price", "MAE in $"', "ok", "goal set · must beat a baseline"),
    "Y04": ('raw = pd.read_csv("listings.csv.gz")', "ok", "29,918 rows × 90 columns · raw layer is read-only"),
    "Y05": ("raw.head(4)", "warn", "3 of 4 prices are NaN · prices are text"),
    "Y06": ("raw.isna().mean()", "bad", "13 columns 100% empty · price 35.1% missing"),
    "Y07": ("validate(raw, rules)", "warn", "2 of 5 rules fail"),
    "Y08": ("raw.price.isna().groupby(raw.minimum_nights >= 30).mean()", "warn", "11.0% vs 40.9% → not random"),
    "Y09": ('df.dropna(axis=1, how="all")[KEEP]', "ok", "90 → 77 → 25 columns"),
    "Y10": ('df.price.str.replace(r"[$,]", "") → float', "ok", "19,406 rows with a price"),
    "Y11": ("df.amenities.map(json.loads)", "ok", "n_amenities + 5 flags · baths → numbers"),
    "Y12": ("df[(df.minimum_nights <= 365) & (df.price <= 2000)]", "ok", "7 + 194 rows removed · 19,205 left"),
    "Y13": ("raw.duplicated() · df.duplicated(subset=key)", "warn", "0 exact · 2,282 look-alike twins"),
    "Y14": ('df.to_parquet("listings_2026-09-14.parquet")', "ok", "16.0 MB → 1.1 MB · types kept"),
    "Y15": ("np.log(df.price).skew()", "ok", "skew 2.48 → 0.30"),
    "Y16": ("df.groupby([borough, room_type]).price.median()", "ok", "5 boroughs × 4 room types"),
    "Y17": ("df.groupby(df.minimum_nights >= 30).price.median()", "ok", "$370 vs $146"),
    "Y18": ('df["km_to_times_sq"] = haversine(lat, lon, TSQ)', "ok", "corr −0.42 · 27 features"),
    "Y19": ("GroupShuffleSplit(test_size=0.2).split(groups=host_id)", "ok", "twins across the split: 342 → 0"),
    "Y20": ('new = df[df.first_review >= "2026-01-01"]', "ok", "1,510 listings locked away"),
    "Y21": ("ColumnTransformer([imputer, one_hot]).fit(X_train)", "ok", "27 → 145 columns · 0 missing"),
    "Y22": ("model.fit(X_train + price_quote_price_per_night)", "bad", "R² 1.000 → leak · column removed"),
    "Y23": ("LGBMRegressor(n_estimators=800).fit(X_train, log(y))", "ok", "MAE $72.8 · baseline $140.3"),
    "Y24": ("same model · train_test_split(random)", "warn", "MAE $65.6 → too optimistic"),
    "Y25": ("mean_absolute_error(y_test, model.predict(X_test))", "ok", "61.4% within ±25% · worst: Manhattan"),
    "Y26": ("model.predict(test.iloc[[3, 50, 400]])", "ok", "$455 · $109 · $233"),
    "Y27": ("permutation_importance(model, X_test, y_test)", "ok", "top: minimum_nights"),
    "Y28": ('joblib.dump(model) · POST /predict', "ok", '{"price_per_night": 109.29} · 20 ms'),
    "Y29": ("psi(train[col], new[col]) for col in features", "warn", "2 of 6 features shifted"),
    "Y30": ("mean_absolute_error(new.price, model.predict(new))", "bad", "$72.8 → $85.0 · retrain"),
    "Q02": ('pd.read_csv("listings.csv.gz")', "warn", "29,918 × 90 · 13 empty"),
    "Q03": ("price.isna().groupby(long_stay)", "warn", "11.0% vs 40.9%"),
    "Q04": ('price.str.replace("[$,]", "")', "ok", "19,205 clean rows"),
    "Q05": ("np.log(df.price).skew()", "ok", "2.48 → 0.30"),
    "Q06": ("GroupShuffleSplit(groups=host_id)", "ok", "twins across: 0"),
    "Q07": ("+ price_quote_price_per_night", "bad", "R² 1.000 · removed"),
    "Q08": ("POST /predict", "ok", "$109.29 in 20 ms"),
    "Q09": ("mae(new_2026)", "bad", "$85.0 → retrain"),
}


def agent_line(sc, t):
    """Terminal strip: the agent types the real code (cursor + typing sound), then prints the real result."""
    if sc.sid not in AGENT: return
    cmd, kind, out = AGENT[sc.sid]
    u, W, H = sc.u, sc.W, sc.H
    fs = 22 if sc.P else 24
    col, sym = {"ok": (GRN, "✓"), "warn": (AMB, "⚠"), "bad": (RED, "✗")}[kind]
    el = sc.id("ag"); n = len(cmd); D = min(1.5, .035 * n + .25); cw = fs * 0.6 * u      # JetBrains Mono: every glyph is 0.6 em wide
    sc.add(f'<div id="{el}" class="ag"><em>agent ▸</em><span class="cw"><code>{escape(cmd)}</code><i></i></span><b style="color:{col}">{sym} {escape(out)}</b></div>',
           f"#{el}{{left:{W * 0.05}px;top:{H * (0.715 if sc.P else 0.815)}px;max-width:{W * 0.90}px;}}")
    sc.css.append(f"#root .ag{{position:absolute;z-index:7;display:flex;align-items:center;gap:{14 * u}px;{MONO}font-size:{fs * u}px;padding:{8 * u}px {16 * u}px;white-space:nowrap;"
                  f"background:rgba(20,21,23,.94);border:{2 * u}px solid rgba(243,240,232,.14);border-radius:{10 * u}px;{'flex-wrap:wrap;' if sc.P else ''}}}"
                  f"#root .ag em{{font-style:normal;font-weight:800;color:{AMB};}}#root .ag .cw{{position:relative;}}#root .ag code{{color:{INK};font-family:inherit;}}"
                  f"#root .ag .cw i{{position:absolute;left:0;top:{2 * u}px;width:{fs * .55 * u}px;height:{fs * 1.05 * u}px;background:{AMB};}}#root .ag b{{font-weight:800;}}")
    sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{12 * u}}},{{opacity:1,y:0,duration:.3,ease:"power2.out"}},{t:.2f});'
                 f'tl.fromTo("#{el} code",{{clipPath:"inset(0 100% 0 0)"}},{{clipPath:"inset(0 0% 0 0)",duration:{D:.2f},ease:"steps({n})"}},{t + .2:.2f});'
                 f'tl.fromTo("#{el} .cw i",{{x:0}},{{x:{n * cw:.1f},duration:{D:.2f},ease:"steps({n})"}},{t + .2:.2f});'
                 f'tl.to("#{el} .cw i",{{opacity:0,duration:.15,repeat:3,yoyo:true,ease:"steps(1)"}},{t + .2 + D:.2f});tl.to("#{el} .cw i",{{opacity:0,duration:.1}},{t + .9 + D:.2f});'
                 f'tl.fromTo("#{el} b",{{opacity:0,x:{-10 * u}}},{{opacity:1,x:0,duration:.3,ease:"expo.out"}},{t + .35 + D:.2f});')
    sfx(sc.sid, t + .2, "typing", 0.14)
    sfx(sc.sid, t + .35 + D, "click-soft", 0.25)


def cam_then_hide(sc, cell, t0, t1, hl, zoom=2.0):
    """Show the real notebook cell (live run) from t0 to t1, then fade it out for the visual."""
    capture(sc, [seg(cell, t0, t1, hl, zoom=zoom)])
    sc.js.append(f'tl.to("#{sc.id("cam0")}",{{opacity:0,duration:.3}},{t1:.2f});')


def table(sc, name, cols, rows, x, y, w, t, size=24, cellcls=None, stagger=.12, colw=None):
    """A real data table. cellcls(j, value) -> css class: nan (red), txt (amber), ans (purple)."""
    u = sc.u; el = sc.id(name)
    head = "".join(f'<th style="{"width:" + str(colw[j]) + "%;" if colw else ""}">{escape(c)}</th>' for j, c in enumerate(cols))
    body = ""
    for i, r_ in enumerate(rows):
        tds = ""
        for j, v in enumerate(r_):
            cls = cellcls(j, v) if cellcls else ""
            txt = "NaN" if v is None else escape(str(v))
            tds += f'<td class="{cls}"><span>{txt}</span></td>'
        body += f'<tr id="{el}-r{i}">{tds}</tr>'
    sc.add(f'<table id="{el}" class="tb"><tr id="{el}-h">{head}</tr>{body}</table>', f"#{el}{{left:{x}px;top:{y}px;width:{w}px;}}")
    sc.css.append(f"#root .tb{{position:absolute;border-collapse:collapse;table-layout:fixed;{MONO}font-size:{size * u}px;{glass(sc)}}}"
                  f"#root .tb th{{color:{MUTE};text-align:left;font-weight:700;font-size:{(size - 4) * u}px;padding:{12 * u}px {14 * u}px;border-bottom:{2 * u}px solid rgba(243,240,232,.2);"
                  f"font-family:Inter,sans-serif;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}}"
                  f"#root .tb td{{padding:{12 * u}px {14 * u}px;border-bottom:{1 * u}px solid rgba(243,240,232,.08);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#E8E5DD;}}"
                  f"#root .tb td.nan span{{color:{RED};font-weight:800;}}#root .tb td.txt span{{color:{AMB};font-weight:800;}}#root .tb td.ans span{{color:{PUR};font-weight:800;}}#root .tb td.ok span{{color:{GRN};font-weight:800;}}")
    sc.appear(f"{el}-h", t, y=0, d=.3)
    for i in range(len(rows) if t > 0.05 else 0):
        sc.js.append(f'tl.fromTo("#{el}-r{i}",{{opacity:0}},{{opacity:1,duration:.3}},{t + .2 + i * stagger:.2f});')
    return el


def pipe_map(sc, x, y, w, times, cols=5, rowh=120, lit=None, small=False):
    """The 15 stages as numbered chips; chip i appears at times[i] (and turns amber when lit)."""
    u = sc.u; gap = 22 * u; cw = (w - gap * (cols - 1)) / cols
    for i, nm in enumerate(PIPE):
        el = sc.id(f"pm{i}"); r_, c_ = divmod(i, cols)
        sc.add(f'<div id="{el}" class="pm"><span>{i + 1:02d}</span>{nm}</div>', f"#{el}{{left:{x + c_ * (cw + gap)}px;top:{y + r_ * rowh * u}px;width:{cw}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{14 * u},scale:.92}},{{opacity:1,y:0,scale:1,duration:.35,ease:"back.out(2)"}},{times[i]:.2f});'
                     f'tl.to("#{el}",{{borderColor:"{AMB}",backgroundColor:"rgba(255,178,36,.16)",duration:.25}},{(lit or times)[i]:.2f});')
    pk = sc.id("pmk"); cx = lambda i: x + (i % cols) * (cw + gap) + cw - 26 * u; cy = lambda i: y + (i // cols) * rowh * u + 22 * u
    sc.add(f'<div id="{pk}" class="pmk"></div>', f"#{pk}{{left:{cx(0) - 9 * u:.1f}px;top:{cy(0) - 9 * u:.1f}px;}}")
    sc.css.append(f"#root .pmk{{position:absolute;width:{18 * u}px;height:{18 * u}px;border-radius:50%;background:{AMB};box-shadow:0 0 {18 * u}px {6 * u}px rgba(255,178,36,.6);z-index:5;}}")
    sc.js.append(f'tl.fromTo("#{pk}",{{opacity:0}},{{opacity:1,duration:.2}},{(lit or times)[0]:.2f});')
    for i in range(1, 15):        # the data packet hops station to station as each one lights up
        sc.js.append(f'tl.to("#{pk}",{{x:{cx(i) - cx(0):.1f},y:{cy(i) - cy(0):.1f},duration:.3,ease:"power2.inOut"}},{(lit or times)[i] - .3:.2f});')
    fs = 26 if small else 30
    sc.css.append(f"#root .pm{{position:absolute;padding:{16 * u}px {16 * u}px;border:{2 * u}px solid rgba(243,240,232,.25);border-radius:{14 * u}px;background:#1C1D20;"
                  f"font-size:{fs * u}px;font-weight:800;white-space:nowrap;}}#root .pm span{{display:block;{MONO}font-size:{20 * u}px;color:{AMB};margin-bottom:{4 * u}px;}}")


ZONES = [("DATA IN", BLU, "db", [0, 1, 2]), ("PREPARE", GRN, "funnel", [3, 4]), ("UNDERSTAND", AMB, "bars", [5, 6]),
         ("LEARN", PUR, "net", [7, 8, 9, 10]), ("SHIP & WATCH", RED, "server", [11, 12, 13, 14])]


def zone_out():
    """What leaves each block (real numbers)."""
    return [f"{V['ingest']['rows']:,} × {V['ingest']['cols']} · 35.1% no price", f"{V['store']['rows']:,} clean rows · {V['store']['clean_mb']} MB",
            f"27 features · skew {2.48} → {0.30:.2f}", f"MAE ${V['models']['Baseline: median of borough × room type']['mae']:.0f} → ${V['models']['LightGBM']['mae']:.1f}",
            f"{V['deploy']['ms']:.0f} ms · drift → retrain"]


def icon(kind, c, s):
    if kind == "db":
        return (f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/>'
                f'<path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>')
    if kind == "funnel":
        return f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><path d="M3 4h18l-7 8v7l-4 2v-9z"/></svg>'
    if kind == "bars":
        return f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>'
    if kind == "net":
        return (f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/>'
                f'<circle cx="19" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><path d="M7 6l3 5M7 18l3-5M14 12h3"/></svg>')
    return (f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><rect x="3" y="3" width="18" height="7" rx="1"/>'
            f'<rect x="3" y="14" width="18" height="7" rx="1"/><path d="M7 6.5h.01M7 17.5h.01"/></svg>')


def pipeline_blocks(sc, x, y, w, h, t_blocks, t_flow, d, stage_t=None, vertical=False):
    """The pipeline as 3D blocks: raw database → 5 blocks (15 stages) → live API, with data packets streaming along the arrows.
    t_blocks: when each block rises in; stage_t: when each of the 15 stages lights up; t_flow: when the data starts to stream."""
    u = sc.u; dep = 22 * u; outs = zone_out(); n = len(ZONES)
    if not vertical:
        sw, ow, gap = 116 * u, 150 * u, 50 * u
        bw = (w - sw - ow - gap * (n + 1) - dep * n) / n; bh = h
        boxes = [(x + sw + gap + i * (bw + dep + gap), y, bw, bh) for i in range(n)]
        src = (x, y + bh * .5 - 80 * u, sw, 160 * u); out = (x + w - ow, y + bh * .5 - 70 * u, ow, 140 * u)
    else:
        sh, oh, gap = 110 * u, 110 * u, 44 * u
        bh = (h - sh - oh - gap * (n + 1) - dep * n) / n; bw = w - dep
        boxes = [(x, y + sh + gap + dep + i * (bh + dep + gap), bw, bh) for i in range(n)]
        src = (x, y, w * .5, sh); out = (x, y + h - oh, w * .5, oh)
    css = (f"#root .zb{{position:absolute;}}#root .zt,#root .zs{{position:absolute;}}#root .zf{{position:absolute;inset:0;background:linear-gradient(170deg,#25262B,#1A1B1E);"
           f"border:{2 * u}px solid rgba(243,240,232,.12);border-radius:{4 * u}px;padding:{14 * u}px {14 * u}px;overflow:hidden;}}"
           f"#root .zh{{display:flex;align-items:center;gap:{10 * u}px;font-weight:900;font-size:{(24 if not vertical else 26) * u}px;letter-spacing:.06em;margin-bottom:{10 * u}px;white-space:nowrap;}}"
           f"#root .zl{{display:flex;flex-direction:{'column' if not vertical else 'row'};flex-wrap:wrap;gap:{6 * u}px;}}"
           f"#root .zc{{font-size:{(23 if not vertical else 22) * u}px;font-weight:800;padding:{6 * u}px {10 * u}px;border-radius:{6 * u}px;background:rgba(243,240,232,.06);color:#B8B5AE;white-space:nowrap;}}"
           f"#root .zc span{{{MONO}font-size:{16 * u}px;margin-right:{6 * u}px;opacity:.8;}}"
           f"#root .zm{{position:absolute;left:{14 * u}px;right:{14 * u}px;bottom:{12 * u}px;{MONO}font-size:{(19 if not vertical else 21) * u}px;font-weight:700;line-height:1.3;}}"
           f"#root .zio{{position:absolute;text-align:center;font-weight:800;}}#root .zio b{{display:block;{MONO}font-size:{20 * u}px;}}#root .zio span{{display:block;font-size:{18 * u}px;color:#B8B5AE;}}"
           f"#root .zpk{{position:absolute;width:{14 * u}px;height:{14 * u}px;border-radius:{3 * u}px;z-index:4;}}")
    sc.css.append(css)
    for i, ((nm, c_, ic, st), (bx, by, bw_, bh_)) in enumerate(zip(ZONES, boxes)):
        el = sc.id(f"z{i}")
        chips = "".join(f'<div id="{sc.id(f"zc{k}")}" class="zc"><span>{k + 1:02d}</span>{PIPE[k]}</div>' for k in st)
        sc.add(f'<div id="{el}" class="zb"><div class="zt" style="background:{c_}"></div><div class="zs" style="background:{c_};opacity:.45"></div>'
               f'<div class="zf" id="{el}f"><div class="zh" style="color:{c_}">{icon(ic, c_, 30 * u)}{nm}</div><div class="zl">{chips}</div><div class="zm" style="color:{c_}">{outs[i]}</div></div></div>',
               f"#{el}{{left:{bx:.1f}px;top:{by:.1f}px;width:{bw_:.1f}px;height:{bh_:.1f}px;}}"
               f"#{el} .zt{{left:0;top:{-dep:.1f}px;width:{bw_ + dep:.1f}px;height:{dep:.1f}px;clip-path:polygon({dep:.1f}px 0,100% 0,{bw_:.1f}px 100%,0 100%);}}"
               f"#{el} .zs{{left:{bw_:.1f}px;top:{-dep:.1f}px;width:{dep:.1f}px;height:{bh_ + dep:.1f}px;clip-path:polygon(0 {dep:.1f}px,100% 0,100% {bh_:.1f}px,0 100%);}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{40 * u}}},{{opacity:1,y:0,duration:.55,ease:"expo.out"}},{t_blocks[i]:.2f});')
        sc.js.append(f'tl.to("#{el}f",{{borderColor:"{c_}",duration:.4}},{t_blocks[i] + .3:.2f});')
        sfx(sc.sid, t_blocks[i], "click-soft", 0.22)
        for j, k in enumerate(st):
            tk = stage_t[k] if stage_t else t_blocks[i] + .25 + j * .12
            sc.js.append(f'tl.to("#{sc.id(f"zc{k}")}",{{backgroundColor:"{c_}",color:"#141517",duration:.25}},{tk:.2f});')
    # raw source (database) and live output (API)
    s_el, o_el = sc.id("zsrc"), sc.id("zout")
    sc.add(f'<div id="{s_el}" class="zio">{icon("db", BLU, 84 * u)}<b style="color:{BLU}">raw CSV</b><span>{V["ingest"]["rows"]:,} rows</span></div>'
           f'<div id="{o_el}" class="zio">{icon("server", GRN, 72 * u)}<b style="color:{GRN}">POST /predict</b><span>${V["deploy"]["out"]} · {V["deploy"]["ms"]:.0f} ms</span></div>',
           f"#{s_el}{{left:{src[0]:.1f}px;top:{src[1]:.1f}px;width:{src[2]:.1f}px;}}#{o_el}{{left:{out[0]:.1f}px;top:{out[1]:.1f}px;width:{out[2]:.1f}px;}}")
    sc.js.append(f'tl.fromTo("#{s_el}",{{opacity:0,scale:.7}},{{opacity:1,scale:1,duration:.4,ease:"back.out(2)"}},{max(.1, t_blocks[0] - .4):.2f});'
                 f'tl.fromTo("#{o_el}",{{opacity:0,scale:.7}},{{opacity:1,scale:1,duration:.4,ease:"back.out(2)"}},{t_blocks[-1] + .4:.2f});')
    # arrows + streaming packets
    if not vertical:
        ends = [(src[0] + src[2], y + bh * .5)] + [(bx_ + bw_ + dep, y + bh * .5) for bx_, _, bw_, _ in boxes]
        starts = [(bx_, y + bh * .5) for bx_, _, _, _ in boxes] + [(out[0], y + bh * .5)]
    else:
        ends = [(x + w * .5, src[1] + src[3])] + [(x + bw * .5, by_ + bh_) for _, by_, _, bh_ in boxes]
        starts = [(x + bw * .5, by_ - dep) for _, by_, _, _ in boxes] + [(x + bw * .5, out[1])]
    svg = ""
    cols = [BLU] + [z[1] for z in ZONES]
    for i, ((xa, ya), (xb, yb)) in enumerate(zip(ends, starts)):
        if vertical and i == 0: xa = xb
        ang = math.atan2(yb - ya, xb - xa); ah = 12 * u
        hx, hy = xb - 4 * u * math.cos(ang), yb - 4 * u * math.sin(ang)
        p1 = (hx - ah * math.cos(ang - .45), hy - ah * math.sin(ang - .45)); p2 = (hx - ah * math.cos(ang + .45), hy - ah * math.sin(ang + .45))
        svg += (f'<g id="{sc.id(f"za{i}")}"><line x1="{xa:.1f}" y1="{ya:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{cols[i]}" stroke-width="{4 * u}" stroke-dasharray="{8 * u} {6 * u}"/>'
                f'<polygon points="{hx:.1f},{hy:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{cols[i]}"/></g>')
    layer(sc, (0, 0, sc.W, sc.H), svg)
    for i, ((xa, ya), (xb, yb)) in enumerate(zip(ends, starts)):
        if vertical and i == 0: xa = xb
        ta = (t_blocks[i] if i < n else t_blocks[-1] + .4) - .1
        sc.js.append(f'tl.fromTo("#{sc.id(f"za{i}")}",{{opacity:0}},{{opacity:1,duration:.3}},{ta:.2f});'
                     f'tl.fromTo("#{sc.id(f"za{i}")} line",{{strokeDashoffset:0}},{{strokeDashoffset:{-14 * u * 20:.0f},duration:{max(1.0, d - ta):.2f},ease:"none"}},{ta:.2f});')
        for k in range(3):     # three packets per arrow, staggered: data streaming through the pipeline
            pk = sc.id(f"zp{i}_{k}"); per = 1.0
            t0 = (t_flow + i * .18 if t_flow is not None else ta + .5) + k * .33
            reps = max(0, int((d - t0 - .2) / per) - 1)
            sc.add(f'<div id="{pk}" class="zpk" style="background:{cols[i]};box-shadow:0 0 {12 * u}px {3 * u}px {cols[i]}"></div>',
                   f"#{pk}{{left:{xa - 7 * u:.1f}px;top:{ya - 7 * u:.1f}px;}}")
            sc.js.append(f'tl.fromTo("#{pk}",{{opacity:0}},{{opacity:1,duration:.2}},{t0:.2f});'
                         f'tl.fromTo("#{pk}",{{x:0,y:0}},{{x:{xb - xa - 14 * u * math.cos(math.atan2(yb - ya, xb - xa)):.1f},y:{yb - ya - 14 * u * math.sin(math.atan2(yb - ya, xb - xa)):.1f},duration:{per},ease:"none",repeat:{reps}}},{t0:.2f});')
    return boxes


def band(x1, a1, a2, x2, b1, b2):
    mx = (x1 + x2) / 2
    return f"M{x1:.1f},{a1:.1f} C{mx:.1f},{a1:.1f} {mx:.1f},{b1:.1f} {x2:.1f},{b1:.1f} L{x2:.1f},{b2:.1f} C{mx:.1f},{b2:.1f} {mx:.1f},{a2:.1f} {x1:.1f},{a2:.1f} Z"


def sankey(sc, box, t0, dt=.9):
    """Every row's journey (real counts): raw → drops peel off → clean rows → train / test / held back."""
    u = sc.u; x, y, w, h = box; L_ = {s_["step"][:12]: s_["rows"] for s_ in V["clean_log"]}
    raw, priced, stays, final = V["ingest"]["rows"], L_["drop rows wi"], L_["drop impossi"], L_["drop prices "]
    S_ = V["split"]; k = h * 0.62 / raw; nw = 14 * u; narrow = w < 900 * u
    xs = [x + i * (w - nw) / 4 for i in range(5)]
    drops = [(raw - priced, "no price"), (priced - stays, "stay > 365 nights"), (stays - final, "price > $2,000")]
    keeps = [priced, stays, final]
    svg, nodes, labs = "", "", []
    nodes += f'<rect x="{xs[0]:.1f}" y="{y:.1f}" width="{nw}" height="{raw * k:.1f}" fill="{BLU}"/>'
    labs.append((xs[0], y - 34 * u, f"{raw:,}" + ("" if narrow else " raw rows"), BLU, t0))
    for i, ((dn, why), kp) in enumerate(zip(drops, keeps)):
        x1, x2 = xs[i] + nw, xs[i + 1]; tt = t0 + .3 + i * dt
        keep_p = band(x1, y, y + kp * k, x2, y, y + kp * k)
        th = max(3 * u, dn * k); yd = y + h - 24 * u - i * 0 
        drop_p = band(x1, y + kp * k, y + kp * k + th, x1 + (x2 - x1) * .75, yd - th, yd)
        g = sc.id(f"sk{i}")
        svg += (f'<g id="{g}"><path d="{keep_p}" fill="url(#{sc.id("gk")})"/><path d="{drop_p}" fill="{RED}" fill-opacity=".55"/>'
                f'<rect x="{x2:.1f}" y="{y:.1f}" width="{nw}" height="{kp * k:.1f}" fill="{GRN if i == 2 else BLU}"/></g>')
        sc.js.append(f'tl.fromTo("#{g}",{{clipPath:"inset(0 100% 0 0)"}},{{clipPath:"inset(0 0% 0 0)",duration:{dt * .8:.2f},ease:"power2.inOut"}},{tt:.2f});')
        labs.append(((x1 + 6 * u, yd + 8 * u + i * 30 * u) if narrow else (x1 + (x2 - x1) * .5, yd + 8 * u)) + (f"−{dn:,} {why}", RED, tt + .4))
        labs.append((x2, y - 34 * u, f"{kp:,}", GRN if i == 2 else INK, tt + .5))
        sfx(sc.sid, tt + .2, "whoosh-short", 0.15)
    # the final split: train / test / held back (new in 2026)
    x1, x2 = xs[3] + nw, xs[4]; tt = t0 + .3 + 3 * dt; gap = 26 * u
    parts = [(S_["train"], "train", BLU), (S_["test"], "test", GRN), (S_["new"], "new · locked" if narrow else "new in 2026 · locked", PUR)]
    a_, b_ = y, y; g = sc.id("sk3"); inner = ""
    for j, (n_, nm, c_) in enumerate(parts):
        inner += f'<path d="{band(x1, a_, a_ + n_ * k, x2, b_, b_ + n_ * k)}" fill="{c_}" fill-opacity=".5"/><rect x="{x2:.1f}" y="{b_:.1f}" width="{nw}" height="{n_ * k:.1f}" fill="{c_}"/>'
        labs.append((x2 + nw + 12 * u, b_ + n_ * k / 2 - 14 * u, f"{n_:,} {nm}", c_, tt + .5 + j * .2))
        a_ += n_ * k; b_ += n_ * k + gap
    svg += f'<g id="{g}">{inner}</g>'
    sc.js.append(f'tl.fromTo("#{g}",{{clipPath:"inset(0 100% 0 0)"}},{{clipPath:"inset(0 0% 0 0)",duration:{dt * .8:.2f},ease:"power2.inOut"}},{tt:.2f});')
    defs = f'<defs><linearGradient id="{sc.id("gk")}" x1="0" x2="1"><stop offset="0" stop-color="{BLU}" stop-opacity=".55"/><stop offset="1" stop-color="{GRN}" stop-opacity=".55"/></linearGradient></defs>'
    n0 = sc.id("sk_n0"); layer(sc, (0, 0, sc.W, sc.H), f'{defs}<g id="{n0}">{nodes}</g>{svg}')
    sc.js.append(f'tl.fromTo("#{n0}",{{opacity:0}},{{opacity:1,duration:.3}},{t0:.2f});')
    for i, (lx, ly, txt, c_, tt) in enumerate(labs):
        el = sc.id(f"skl{i}")
        sc.add(f'<div id="{el}" class="skl" style="color:{c_}">{txt}</div>', f"#{el}{{left:{lx:.1f}px;top:{ly:.1f}px;}}")
        sc.appear(el, tt, y=8, d=.3)
    sc.css.append(f"#root .skl{{position:absolute;{MONO}font-size:{22 * u}px;font-weight:800;white-space:nowrap;}}")


def parallel(sc, box, t, t_hi):
    """Parallel coordinates: 399 real listings across six features, coloured by price (cheap blue → expensive red)."""
    u = sc.u; x, y, w, h = box; P_ = EX["parallel"]; ax = P_["axes"]; n = len(ax)
    xs = [x + i * w / (n - 1) for i in range(n)]
    Y = lambda j, v: y + h - (v - P_["lo"][j]) / max(1e-9, P_["hi"][j] - P_["lo"][j]) * h
    pr = sorted(r_[-1] for r_ in P_["rows"]); q1, q2 = pr[len(pr) // 3], pr[2 * len(pr) // 3]
    groups = {BLU: "", AMB: "", RED: ""}
    for r_ in P_["rows"]:
        c_ = BLU if r_[-1] < q1 else (AMB if r_[-1] < q2 else RED)
        groups[c_] += f'<path d="M' + " L".join(f"{xs[j]:.1f},{Y(j, v):.1f}" for j, v in enumerate(r_)) + '" pathLength="1"/>'
    axes = "".join(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y}" y2="{y + h}" stroke="#F3F0E8" stroke-opacity=".5" stroke-width="{2 * u}"/>'
                   f'<text x="{xx:.1f}" y="{y + h + 34 * u:.1f}" text-anchor="middle" font-size="{20 * u}" fill="#D9D6CE" font-family="JetBrains Mono" font-weight="700">{ {"accommodates": "guests", "km_to_times_sq": "km to TSQ", "n_amenities": "amenities"}.get(ax[j], ax[j])}</text>'
                   f'<text x="{xx:.1f}" y="{y - 12 * u:.1f}" text-anchor="middle" font-size="{18 * u}" fill="#B8B5AE" font-family="JetBrains Mono">{("$" if ax[j] == "price" else "")}{P_["hi"][j]:g}</text>'
                   f'<text x="{xx:.1f}" y="{y + h + 58 * u:.1f}" text-anchor="middle" font-size="{18 * u}" fill="#8E8B85" font-family="JetBrains Mono">{("$" if ax[j] == "price" else "")}{P_["lo"][j]:g}</text>'
                   for j, xx in enumerate(xs))
    ga = sc.id("pca"); gs = {c_: sc.id(f"pc{i}") for i, c_ in enumerate(groups)}
    layer(sc, (0, 0, sc.W, sc.H), f'<g id="{ga}">{axes}</g>' + "".join(f'<g id="{gs[c_]}" fill="none" stroke="{c_}" stroke-opacity=".32" stroke-width="{1.6 * u}" stroke-dasharray="1 1">{pths}</g>' for c_, pths in groups.items()))
    sc.js.append(f'tl.fromTo("#{ga}",{{opacity:0}},{{opacity:1,duration:.4}},{t - .3:.2f});')
    for i, g in enumerate(gs.values()):
        sc.js.append(f'tl.fromTo("#{g} path",{{strokeDashoffset:1}},{{strokeDashoffset:0,duration:1.2,stagger:.002,ease:"power2.out"}},{t + i * .5:.2f});')
    sc.js.append(f'tl.to("#{gs[BLU]},#{gs[AMB]}",{{opacity:.25,duration:.5}},{t_hi:.2f});tl.to("#{gs[RED]}",{{strokeOpacity:.75,duration:.5}},{t_hi:.2f});')


def learn_curve(sc, box, t, d_draw=3.0):
    """LightGBM learning live: the test error after 1, 10 … 800 trees (same model as the notebook, re-run and checked)."""
    u = sc.u; C_ = EX["curve"]; n = len(C_["trees"]); base = V["models"]["Baseline: median of borough × room type"]["mae"]
    X, Y, bx = chart(sc, "lc", box, (-.3, n - .7), (60, 170), "trees added", "test MAE ($)", ([], [60, 100, 140], lambda v: f"${v:.0f}"), t0=max(.1, t - .4))
    ticks = "".join(f'<text x="{X(i):.1f}" y="{Y(60) + 34 * u:.1f}" text-anchor="middle" font-size="{20 * u}" fill="#B8B5AE" font-family="JetBrains Mono">{tr}</text>' for i, tr in enumerate(C_["trees"]))
    bl = sc.id("lcb")
    layer(sc, bx, f'<g id="{bl}">{ticks}<line x1="{X(-.3):.1f}" x2="{X(n - .7):.1f}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" stroke="#B8B5AE" stroke-width="{3 * u}" stroke-dasharray="{10 * u} {8 * u}"/>'
                 f'<text x="{X(n - .7) - 8 * u:.1f}" y="{Y(base) - 10 * u:.1f}" text-anchor="end" font-size="{20 * u}" fill="#B8B5AE" font-family="Inter" font-weight="700">baseline ${base:.0f}</text></g>')
    sc.js.append(f'tl.fromTo("#{bl}",{{opacity:0}},{{opacity:1,duration:.4}},{t:.2f});')
    curve(sc, bx, "lcc", X, Y, list(range(n)), C_["mae"], GRN, t + .3, d=d_draw, width=5)
    frames(sc, "lcr", [f"{tr:>3} trees → ${m_:.1f}" for tr, m_ in zip(C_["trees"], C_["mae"])], box[0] + 130 * u, box[1] + 20 * u, t + .3, d_draw / n, size=26, color=GRN)


def impute_demo(sc, x, y, w, t_in, t_med, t_hot):
    """The fitted pipeline at work on real training rows: missing beds → the train median; room type → one-hot columns."""
    u = sc.u; I_ = EX["impute"]; cats = I_["cats"]
    before = [[r_[0].replace("Entire home/apt", "Entire home"), None] for r_ in I_["rows"]]
    table(sc, "ib", ["room_type", "beds"], before, x, y, w * .30, t_in, size=22, cellcls=lambda j, v: "nan" if v is None else "")
    ar = sc.id("iar"); sc.add(f'<div id="{ar}" class="iar">→</div>', f"#{ar}{{left:{x + w * .31:.1f}px;top:{y + 90 * u:.1f}px;}}")
    sc.css.append(f"#root .iar{{position:absolute;font-size:{56 * u}px;color:{AMB};font-weight:900;}}")
    sc.appear(ar, t_med - .2, y=0)
    after = [[f'{I_["median_beds"]:g}'] + [int(r_[0] == c_) for c_ in cats] for r_ in I_["rows"]]
    cls = lambda j, v: "ok" if j == 0 else ("txt" if v == 1 else "")
    table(sc, "ia", ["beds"] + [c_.replace("Entire home/apt", "entire").replace(" room", "").lower() for c_ in cats], after, x + w * .38, y, w * .62, t_hot, size=22, cellcls=cls)
    label(sc, "im", f'median learned from TRAIN = {I_["median_beds"]:g}', x, y - 40 * u, size=24, color=GRN, weight=800, t=t_med, mono=True)


def miss_grid(sc, x, y, cell, t, d=2.5):
    """The real missing-value map: 60 random listings × 24 columns. Bright = missing."""
    u = sc.u; M_ = V["missing"]; el = sc.id("mg")
    cells = "".join(f'<i style="background:{"#FFB224" if v else "rgba(243,240,232,.07)"}"></i>' for row in M_["matrix"] for v in row)
    n = len(M_["cols"])
    sc.add(f'<div id="{el}" class="mg">{cells}</div>', f"#{el}{{left:{x}px;top:{y}px;width:{n * cell}px;grid-template-columns:repeat({n},{cell - 2 * u}px);grid-auto-rows:{cell * .24 - 2 * u}px;gap:{2 * u}px;}}")
    sc.css.append(f"#root .mg{{position:absolute;display:grid;}}#root .mg i{{display:block;border-radius:{2 * u}px;}}")
    sc.js.append(f'tl.fromTo("#{el}",{{clipPath:"inset(0 100% 0 0)"}},{{clipPath:"inset(0 0% 0 0)",duration:{d},ease:"none"}},{t:.2f});')
    return n, len(M_["matrix"]) * (cell * .24)


def clean_log(sc, x, y, w, upto, t_new, rowh=58):
    """The cleaning log: one line per step with rows × columns after it. Steps before `upto` are already there."""
    u = sc.u; log = V["clean_log"]; n0 = log[0]["rows"]
    label(sc, "clh", "CLEANING LOG · rows × columns after each step", x, y - 44 * u, size=22, color=MUTE, weight=800, t=.2 if t_new else 0)
    names = {"raw": "raw file", "drop 100%-empty columns": "drop empty columns", "keep 25 useful columns (drop urls, ids, free text)": "keep 25 useful columns",
             "drop rows with no price (no label to learn)": "drop rows with no price", "parse baths, amenities, flags and dates": "parse text → numbers",
             "drop impossible minimum stays (> 365 nights)": "drop stays > 365 nights", "drop prices above $2,000 per night": "drop prices > $2,000"}
    for i, s_ in enumerate(log[:upto + 1]):
        el = sc.id(f"cl{i}"); bw = (w * 0.42) * s_["rows"] / n0; new = i in t_new
        sc.add(f'<div id="{el}" class="clr{" new" if new else ""}"><b>{names.get(s_["step"], s_["step"])}</b><i style="width:{bw:.0f}px"></i><span>{s_["rows"]:,} × {s_["cols"]}</span></div>',
               f"#{el}{{left:{x}px;top:{y + i * rowh * u}px;width:{w}px;}}")
        if new:
            sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,x:{-20 * u}}},{{opacity:1,x:0,duration:.4,ease:"expo.out"}},{t_new[i]:.2f});'
                         f'tl.fromTo("#{el} i",{{scaleX:0}},{{scaleX:1,duration:.6,ease:"power3.out"}},{t_new[i] + .1:.2f});')
            sfx(sc.sid, t_new[i], "click-soft", 0.25)
    sc.css.append(f"#root .clr{{position:absolute;display:flex;align-items:center;gap:{14 * u}px;font-size:{24 * u}px;color:#B8B5AE;}}"
                  f"#root .clr b{{width:{w * 0.38}px;font-weight:700;white-space:nowrap;}}#root .clr i{{display:block;height:{18 * u}px;border-radius:{4 * u}px;background:rgba(243,240,232,.25);transform-origin:left center;}}"
                  f"#root .clr span{{{MONO}font-weight:700;white-space:nowrap;}}#root .clr.new{{color:{INK};}}#root .clr.new i{{background:{AMB};}}#root .clr.new span{{color:{AMB};}}")


def morph(sc, name, a, bb, x, y, t, size=40, ca=AMB, cb=GRN):
    """'before → after' for one real value."""
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="mo"><s>{escape(str(a))}</s><i>→</i><b>{escape(str(bb))}</b></div>', f"#{el}{{left:{x}px;top:{y}px;}}")
    sc.css.append(f"#root .mo{{position:absolute;display:flex;gap:{20 * u}px;align-items:center;{MONO}font-size:{size * u}px;font-weight:800;white-space:nowrap;}}"
                  f"#root .mo s{{text-decoration:none;color:{ca};}}#root .mo i{{font-style:normal;color:{MUTE};}}#root .mo b{{color:{cb};}}")
    sc.js.append(f'tl.fromTo("#{el} s",{{opacity:0}},{{opacity:1,duration:.3}},{t:.2f});tl.fromTo("#{el} i",{{opacity:0,x:{-12 * u}}},{{opacity:1,x:0,duration:.3}},{t + .35:.2f});'
                 f'tl.fromTo("#{el} b",{{opacity:0,scale:1.4}},{{opacity:1,scale:1,duration:.35,ease:"back.out(2)"}},{t + .6:.2f});')


def bigstat(sc, name, value, sub, x, y, t, color=AMB, size=96, dec=0, prefix="", suffix="", w=None):
    """A big number that counts up to its real value, with a one-line meaning under it."""
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="bst"><b style="color:{color}">{prefix}{value:,.{dec}f}{suffix}</b><span>{sub}</span></div>', f"#{el}{{left:{x}px;top:{y}px;{f'width:{w}px;' if w else ''}}}")
    sc.css.append(f"#root .bst{{position:absolute;}}#root .bst b{{display:block;{MONO}font-weight:800;line-height:1;white-space:nowrap;}}#{el} b{{font-size:{size * u}px;}}"
                  f"#root .bst span{{display:block;font-size:{28 * u}px;color:#B8B5AE;margin-top:{10 * u}px;font-weight:700;}}")
    sc.appear(el, t, y=16, d=.35)
    if t > 0.05:                      # on screen from frame 0 = shown at its final value (Reel hook)
        countup(sc, el, value, t, d=1.0, dec=dec, prefix=prefix, suffix=suffix)


def price_strip(sc, box, t, t_cap):
    """Every price on a log axis: the far-right dots are the extreme listings; the cap removes them."""
    u = sc.u; x, y, w, h = box; P_ = EX["prices"]
    lo, hi = math.log10(10), math.log10(40000)
    X = lambda v: x + (math.log10(v) - lo) / (hi - lo) * w
    pts = [(v, i) for i, v in enumerate(P_["sample"])] + [(v, 2000 + i) for i, v in enumerate(P_["top"])]
    inner_ok, inner_out = "", ""
    for v, i in pts:
        cy = y + h * 0.15 + (((i * 7919) % 97) / 97) * h * 0.6
        dot_ = f'<circle cx="{X(v):.1f}" cy="{cy:.1f}" r="{(6 if v > 2000 else 3.5) * u:.1f}"/>'
        if v > 2000: inner_out += dot_
        else: inner_ok += dot_
    ticks = "".join(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y + h * .82}" y2="{y + h * .86}" stroke="#B8B5AE" stroke-width="{2 * u}"/>'
                    f'<text x="{X(v):.1f}" y="{y + h * .97}" text-anchor="middle" font-size="{24 * u}" fill="#B8B5AE" font-family="JetBrains Mono">${v:,}</text>'
                    for v in (10, 100, 1000, 10000))
    g1, g2, cl = sc.id("pok"), sc.id("pout"), sc.id("pcap")
    layer(sc, (0, 0, sc.W, sc.H), f'{ticks}<line x1="{x}" x2="{x + w}" y1="{y + h * .82}" y2="{y + h * .82}" stroke="#B8B5AE" stroke-width="{2 * u}"/>'
          f'<g id="{g1}" fill="{BLU}" fill-opacity=".55">{inner_ok}</g><g id="{g2}" fill="{RED}">{inner_out}</g>'
          f'<g id="{cl}"><line x1="{X(2000):.1f}" x2="{X(2000):.1f}" y1="{y}" y2="{y + h * .82}" stroke="{AMB}" stroke-width="{4 * u}" stroke-dasharray="{10 * u} {8 * u}"/>'
          f'<text x="{X(2000) + 12 * u:.1f}" y="{y + 26 * u}" font-size="{26 * u}" fill="{AMB}" font-weight="800" font-family="Inter">$2,000 cap</text></g>')
    sc.js.append(f'tl.fromTo("#{g1}",{{opacity:0}},{{opacity:1,duration:.6}},{t:.2f});tl.fromTo("#{g2}",{{opacity:0}},{{opacity:1,duration:.4}},{t + .5:.2f});'
                 f'tl.fromTo("#{cl}",{{opacity:0}},{{opacity:1,duration:.3}},{t_cap:.2f});tl.to("#{g2}",{{opacity:.15,duration:.5}},{t_cap + .6:.2f});')
    return X


def hist_morph(sc, box, t_raw, t_log):
    """Real price histogram, then the same prices on a log scale (bars regrow)."""
    u = sc.u; x, y, w, h = box; Hh = V["hist"]
    bx = sc.id("hbox"); sc.add(f'<div id="{bx}"></div>', f"#{bx}{{position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;{glass(sc)}}}")
    sc.appear(bx, max(.1, t_raw - .3))
    pl, pb = 30 * u, 70 * u
    def bars(counts, edges, color, gid, fmt):
        mx = max(counts); n = len(counts); bw = (w - 2 * pl) / n; s_ = ""
        for i, c_ in enumerate(counts):
            bh = (h - pb - 40 * u) * c_ / mx
            s_ += f'<rect x="{x + pl + i * bw:.1f}" y="{y + h - pb - bh:.1f}" width="{bw - 2 * u:.1f}" height="{bh:.1f}" fill="{color}"/>'
        for k in (0, n // 2, n):
            v = edges[k]; s_ += f'<text x="{x + pl + k * bw:.1f}" y="{y + h - pb + 34 * u:.1f}" text-anchor="middle" font-size="{24 * u}" fill="#B8B5AE" font-family="JetBrains Mono">{fmt(v)}</text>'
        return f'<g id="{gid}">{s_}</g>'
    g1, g2 = sc.id("hr"), sc.id("hl")
    layer(sc, (0, 0, sc.W, sc.H), bars(*Hh["raw"], AMB, g1, lambda v: f"${v:,.0f}") + bars(*Hh["log"], GRN, g2, lambda v: f"${math.exp(v):,.0f}"))
    sc.js.append(f'tl.fromTo("#{g1} rect",{{scaleY:0,transformOrigin:"50% 100%"}},{{scaleY:1,duration:.6,stagger:.01,ease:"power3.out"}},{t_raw:.2f});'
                 f'tl.fromTo("#{g1} text",{{opacity:0}},{{opacity:1,duration:.3}},{t_raw:.2f});tl.set("#{g2}",{{opacity:0}},0);'
                 f'tl.to("#{g1}",{{opacity:0,duration:.4}},{t_log:.2f});tl.set("#{g2}",{{opacity:1}},{t_log:.2f});'
                 f'tl.fromTo("#{g2} rect",{{scaleY:0,transformOrigin:"50% 100%"}},{{scaleY:1,duration:.7,stagger:.012,ease:"power3.out"}},{t_log:.2f});')


def nyc_map(sc, box, t, d=2.0, tsq=None, pt=3.0):
    """The real listings (4,000 sample), coloured by price: blue cheap → amber → red expensive."""
    u = sc.u; x, y, w, h = box; M_ = V["map"]
    la0, la1, lo0, lo1 = 40.50, 40.92, -74.26, -73.70
    k = min(w / (lo1 - lo0) / 0.76, h / (la1 - la0))
    X = lambda lon: x + (lon - lo0) * k * 0.76
    Y = lambda lat: y + (la1 - lat) * k
    lp = M_["logp"]; a_, b_ = sorted(lp)[len(lp) // 20], sorted(lp)[-len(lp) // 20]
    def col(v):
        q = min(1, max(0, (v - a_) / (b_ - a_)))
        return BLU if q < .33 else (AMB if q < .7 else RED)
    groups = {}
    for la, lo, v in zip(M_["lat"], M_["lon"], lp):
        groups.setdefault(col(v), []).append(f'<circle cx="{X(lo):.1f}" cy="{Y(la):.1f}" r="{pt * u:.1f}"/>')
    inner = "".join(f'<g class="mp" fill="{c_}" fill-opacity=".7">{"".join(v)}</g>' for c_, v in groups.items())
    g = sc.id("map"); layer(sc, (0, 0, sc.W, sc.H), f'<g id="{g}">{inner}</g>')
    sc.js.append(f'tl.fromTo("#{g} .mp",{{opacity:0}},{{opacity:1,duration:.6,stagger:{d / 3:.2f}}},{t:.2f});')
    if tsq is not None:
        el = sc.id("tsq"); tx, ty = X(-73.9855), Y(40.7580)
        layer(sc, (0, 0, sc.W, sc.H), f'<g id="{el}"><circle cx="{tx:.1f}" cy="{ty:.1f}" r="{14 * u}" fill="none" stroke="#F3F0E8" stroke-width="{4 * u}"/>'
              f'<text x="{tx + 22 * u:.1f}" y="{ty + 8 * u:.1f}" font-size="{26 * u}" fill="#F3F0E8" font-weight="800" font-family="Inter">Times Square</text></g>')
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,scale:.3,transformOrigin:"{tx:.0f}px {ty:.0f}px"}},{{opacity:1,scale:1,duration:.4,ease:"back.out(2)"}},{tsq:.2f});')
    return X, Y


def legend(sc, x, y, t):
    u = sc.u; el = sc.id("leg")
    sc.add(f'<div id="{el}" class="leg"><i style="background:{BLU}"></i>cheaper<i style="background:{AMB}"></i>middle<i style="background:{RED}"></i>expensive</div>', f"#{el}{{left:{x}px;top:{y}px;}}")
    sc.css.append(f"#root .leg{{position:absolute;display:flex;align-items:center;gap:{10 * u}px;font-size:{24 * u}px;color:#B8B5AE;font-weight:700;}}"
                  f"#root .leg i{{display:block;width:{18 * u}px;height:{18 * u}px;border-radius:50%;margin-left:{10 * u}px;}}")
    sc.appear(el, t, y=0)


def split_bar(sc, x, y, w, t, show_new=True, h=110):
    """Train / test / held-back blocks, widths proportional to the real counts."""
    u = sc.u; S_ = V["split"]; tot = S_["train"] + S_["test"] + (S_["new"] if show_new else 0)
    parts = [("TRAIN", S_["train"], BLU), ("TEST", S_["test"], GRN)] + ([("NEW IN 2026 · locked", S_["new"], PUR)] if show_new else [])
    mn = 300 * u; free = w - mn * len(parts); xx = x
    for i, (nm, n, c_) in enumerate(parts):
        el = sc.id(f"sb{i}"); ww = mn + free * n / tot - 8 * u     # shortcut: widths are proportional above a 300px minimum, so small blocks stay readable
        sc.add(f'<div id="{el}" class="sbk" style="border-color:{c_}"><b style="color:{c_}">{n:,}</b><span>{nm}</span></div>', f"#{el}{{left:{xx}px;top:{y}px;width:{ww}px;height:{h * u}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,scaleX:.2,transformOrigin:"0% 50%"}},{{opacity:1,scaleX:1,duration:.5,ease:"expo.out"}},{t + i * .5:.2f});')
        xx += ww + 8 * u
    sc.css.append(f"#root .sbk{{position:absolute;border:{3 * u}px solid;border-radius:{12 * u}px;background:#1C1D20;padding:{10 * u}px {14 * u}px;overflow:hidden;}}"
                  f"#root .sbk b{{display:block;{MONO}font-size:{40 * u}px;font-weight:800;white-space:nowrap;}}#root .sbk span{{display:block;font-size:{22 * u}px;color:#B8B5AE;font-weight:800;white-space:nowrap;}}")


def twin_split(sc, x, y, w, h, t_rand, t_host):
    """Illustration (counts are real): look-alike twins (linked dots) end up on both sides of a random split, never across a split by host."""
    u = sc.u; S_ = V["split"]; mid = x + w * 0.62
    import random; rnd = random.Random(5)
    pairs = []
    for k in range(26):
        yy = y + 30 * u + k * (h - 60 * u) / 26
        cross = k % 3 == 0
        pairs.append((yy, cross))
    lines_r, lines_h = "", ""
    for yy, cross in pairs:
        if cross:
            a_, b_ = x + rnd.uniform(.08, .5) * w, mid + rnd.uniform(.06, .3) * w
        else:
            a_ = x + rnd.uniform(.05, .45) * w; b_ = a_ + rnd.uniform(.04, .1) * w
        lines_r += f'<line x1="{a_:.1f}" y1="{yy:.1f}" x2="{b_:.1f}" y2="{yy + 6 * u:.1f}" stroke="{RED if cross else "rgba(243,240,232,.35)"}" stroke-width="{3 * u}"/>' \
                   f'<circle cx="{a_:.1f}" cy="{yy:.1f}" r="{6 * u}" fill="{BLU if a_ < mid else GRN}"/><circle cx="{b_:.1f}" cy="{yy + 6 * u:.1f}" r="{6 * u}" fill="{BLU if b_ < mid else GRN}"/>'
        side = rnd.random() < .78
        a2 = (x + rnd.uniform(.05, .45) * w) if side else (mid + rnd.uniform(.05, .28) * w); b2 = a2 + rnd.uniform(.03, .07) * w
        c2 = BLU if side else GRN
        lines_h += f'<line x1="{a2:.1f}" y1="{yy:.1f}" x2="{b2:.1f}" y2="{yy + 6 * u:.1f}" stroke="rgba(243,240,232,.35)" stroke-width="{3 * u}"/>' \
                   f'<circle cx="{a2:.1f}" cy="{yy:.1f}" r="{6 * u}" fill="{c2}"/><circle cx="{b2:.1f}" cy="{yy + 6 * u:.1f}" r="{6 * u}" fill="{c2}"/>'
    g1, g2, ln = sc.id("tr"), sc.id("th"), sc.id("tl")
    layer(sc, (0, 0, sc.W, sc.H),
          f'<g id="{ln}"><line x1="{mid:.1f}" x2="{mid:.1f}" y1="{y}" y2="{y + h}" stroke="#F3F0E8" stroke-width="{3 * u}" stroke-dasharray="{10 * u} {8 * u}"/>'
          f'<text x="{x + 10 * u}" y="{y - 14 * u}" font-size="{26 * u}" fill="{BLU}" font-weight="800" font-family="Inter">TRAIN</text>'
          f'<text x="{mid + 14 * u:.1f}" y="{y - 14 * u}" font-size="{26 * u}" fill="{GRN}" font-weight="800" font-family="Inter">TEST</text></g>'
          f'<g id="{g1}">{lines_r}</g><g id="{g2}">{lines_h}</g>')
    sc.js.append(f'tl.fromTo("#{ln}",{{opacity:0}},{{opacity:1,duration:.3}},{t_rand - .3:.2f});tl.fromTo("#{g1}",{{opacity:0}},{{opacity:1,duration:.5}},{t_rand:.2f});'
                 f'tl.set("#{g2}",{{opacity:0}},0);tl.to("#{g1}",{{opacity:0,duration:.4}},{t_host:.2f});tl.to("#{g2}",{{opacity:1,duration:.5}},{t_host + .2:.2f});')
    el = sc.id("tw"); sc.add(f'<div id="{el}" class="twc"><span>twins split across train/test</span><b>{S_["twins_random"]}</b></div>', f"#{el}{{left:{x + w + 40 * u}px;top:{y + h * .25}px;}}")
    sc.css.append(f"#root .twc{{position:absolute;}}#root .twc span{{display:block;font-size:{26 * u}px;color:#B8B5AE;font-weight:700;width:{300 * u}px;}}#root .twc b{{display:block;{MONO}font-size:{110 * u}px;color:{RED};line-height:1.1;}}")
    sc.appear(el, t_rand + .3, y=10)
    countup(sc, el, S_["twins_random"], t_rand + .3, d=1.0)
    sc.js.append(f'const tw_{sc.sid}={{v:{S_["twins_random"]}}};tl.to(tw_{sc.sid},{{v:{S_["twins_group"]},duration:.8,onUpdate:()=>{{document.querySelector("#{el} b").textContent=Math.round(tw_{sc.sid}.v);}}}},{t_host + .2:.2f});'
                 f'tl.to("#{el} b",{{color:"{GRN}",duration:.3}},{t_host + .6:.2f});')
    label(sc, "illus", "illustration · the counts are real", x, y + h + 14 * u, size=20, color=MUTE, t=t_rand)


def flow(sc, steps, x, y, w, t_list, h=150):
    """A left-to-right flow of boxes with arrows: steps [(title, sub, color)]."""
    u = sc.u; n = len(steps); gap = 54 * u; bw = (w - gap * (n - 1)) / n
    for i, (ti, sub, c_) in enumerate(steps):
        el = sc.id(f"fl{i}")
        sc.add(f'<div id="{el}" class="flb" style="border-color:{c_}"><b style="color:{c_}">{ti}</b><span>{sub}</span></div>', f"#{el}{{left:{x + i * (bw + gap)}px;top:{y}px;width:{bw}px;min-height:{h * u}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{20 * u}}},{{opacity:1,y:0,duration:.4,ease:"expo.out"}},{t_list[i]:.2f});')
        if i:
            ar = sc.id(f"fa{i}")
            sc.add(f'<div id="{ar}" class="fla">→</div>', f"#{ar}{{left:{x + i * (bw + gap) - gap + 8 * u}px;top:{y + h * u * .3}px;}}")
            sc.js.append(f'tl.fromTo("#{ar}",{{opacity:0,x:{-10 * u}}},{{opacity:1,x:0,duration:.3}},{t_list[i] - .1:.2f});')
    sc.css.append(f"#root .flb{{position:absolute;border:{3 * u}px solid;border-radius:{14 * u}px;background:#1C1D20;padding:{14 * u}px {16 * u}px;}}"
                  f"#root .flb b{{display:block;font-size:{30 * u}px;font-weight:900;line-height:1.15;}}#root .flb span{{display:block;font-size:{22 * u}px;color:#B8B5AE;margin-top:{8 * u}px;line-height:1.3;}}"
                  f"#root .fla{{position:absolute;font-size:{40 * u}px;color:{MUTE};font-weight:900;}}")


def json_panel(sc, name, title, body, x, y, w, t, color=INK):
    u = sc.u; el = sc.id(name)
    sc.add(f'<div id="{el}" class="jp"><small>{title}</small><pre style="color:{color}">{escape(body)}</pre></div>', f"#{el}{{left:{x}px;top:{y}px;width:{w}px;}}")
    sc.css.append(f"#root .jp{{position:absolute;padding:{16 * u}px {22 * u}px;{glass(sc)}}}#root .jp small{{display:block;font-size:{20 * u}px;color:{MUTE};font-weight:800;letter-spacing:.08em;margin-bottom:{6 * u}px;}}"
                  f"#root .jp pre{{margin:0;{MONO}font-size:{24 * u}px;line-height:1.45;white-space:pre-wrap;}}")
    sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,y:{20 * u}}},{{opacity:1,y:0,duration:.45,ease:"expo.out"}},{t:.2f});')


def psi_bars(sc, x, y, w, t, rowh=64):
    """Drift per feature (PSI). Bars are capped at 1.0 on screen; the number is the real value. Threshold 0.2."""
    u = sc.u; ps = V["monitor"]["psi"]; lw = w * .38; bw = w - lw - 150 * u
    X = lambda v: x + lw + bw * min(v, 1.0)
    for i, (k, v) in enumerate(ps.items()):
        yy = y + i * rowh * u; c_ = RED if v > .2 else GRN
        lab, br, num = sc.id(f"pl{i}"), sc.id(f"pb{i}"), sc.id(f"pn{i}")
        sc.add(f'<div id="{lab}" class="psl">{k}</div><div id="{br}" class="psb" style="background:{c_}"></div><div id="{num}" class="psn" style="color:{c_}">{v:.3f}{" ▶" if v > 1 else ""}</div>',
               f"#{lab}{{left:{x}px;top:{yy}px;width:{lw - 20 * u}px;}}#{br}{{left:{x + lw}px;top:{yy + 10 * u}px;width:{max(4, X(v) - x - lw):.0f}px;}}#{num}{{left:{x + lw + bw + 20 * u:.0f}px;top:{yy}px;}}")
        sc.appear(lab, t + i * .2, y=8, d=.25)
        sc.js.append(f'tl.fromTo("#{br}",{{scaleX:0}},{{scaleX:1,duration:.6,ease:"power3.out"}},{t + i * .2:.2f});')
        sc.appear(num, t + i * .2 + .35, y=0, d=.2)
    th = sc.id("thr")
    sc.add(f'<div id="{th}" class="thr"><span>0.2 = real shift</span></div>', f"#{th}{{left:{X(.2):.0f}px;top:{y - 40 * u}px;height:{len(ps) * rowh * u + 40 * u}px;}}")
    sc.appear(th, t, y=0)
    sc.css.append(f"#root .psl{{position:absolute;{MONO}font-size:{26 * u}px;text-align:right;line-height:{(rowh - 20) * u}px;white-space:nowrap;}}"
                  f"#root .psb{{position:absolute;height:{(rowh - 36) * u}px;border-radius:{6 * u}px;transform-origin:left center;}}"
                  f"#root .psn{{position:absolute;{MONO}font-size:{28 * u}px;font-weight:800;line-height:{(rowh - 20) * u}px;white-space:nowrap;}}"
                  f"#root .thr{{position:absolute;border-left:{3 * u}px dashed {AMB};}}#root .thr span{{position:absolute;left:{8 * u}px;top:0;font-size:{20 * u}px;color:{AMB};font-weight:800;white-space:nowrap;}}")


def pred_scatter(sc, box, t, small=False):
    """Real test listings: real price vs predicted, log axes, with the ±25% band."""
    u = sc.u; x, y, w, h = box; A_, P_ = V["eval"]["scatter"]
    lo, hi = math.log10(20), math.log10(2000)
    X, Y, bx = chart(sc, "ps", box, (lo, hi), (lo, hi), "real price", "predicted",
                     ([math.log10(v) for v in (20, 100, 500, 2000)], [math.log10(v) for v in (20, 100, 500, 2000)], lambda v: f"${10 ** v:,.0f}"), t0=max(.1, t - .4))
    band = f'<polygon points="{X(lo):.1f},{Y(lo + math.log10(1.25)):.1f} {X(hi - math.log10(1.25)):.1f},{Y(hi):.1f} {X(hi):.1f},{Y(hi):.1f} {X(hi):.1f},{Y(hi - math.log10(1.25) * 2 + math.log10(1.25)):.1f} {X(lo + math.log10(1.25)):.1f},{Y(lo):.1f} {X(lo):.1f},{Y(lo):.1f}" fill="{GRN}" fill-opacity=".10"/>'
    pts = "".join(f'<circle cx="{X(math.log10(max(20, min(2000, a_)))):.1f}" cy="{Y(math.log10(max(20, min(2000, p_)))):.1f}" r="{3.2 * u:.1f}"/>' for a_, p_ in zip(A_, P_))
    g = sc.id("psd"); bd = sc.id("psb")
    layer(sc, bx, f'<g id="{bd}">{band}<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" stroke="{GRN}" stroke-width="{3 * u}"/></g><g id="{g}" fill="{AMB}" fill-opacity=".5">{pts}</g>')
    sc.js.append(f'tl.fromTo("#{g}",{{opacity:0}},{{opacity:1,duration:.8}},{t:.2f});tl.fromTo("#{bd}",{{opacity:0}},{{opacity:1,duration:.5}},{t + .8:.2f});')


def rules_list(sc, x, y, w, t_list, rowh=66):
    u = sc.u
    for i, (k, v) in enumerate(V["rules"].items()):
        el = sc.id(f"rl{i}"); bad = v > 0
        sc.add(f'<div id="{el}" class="rl"><i style="color:{RED if bad else GRN}">{"✗" if bad else "✓"}</i><b>{k}</b><span style="color:{RED if bad else GRN}">{v:,} fail</span></div>',
               f"#{el}{{left:{x}px;top:{y + i * rowh * u}px;width:{w}px;}}")
        sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,x:{-20 * u}}},{{opacity:1,x:0,duration:.35,ease:"expo.out"}},{t_list[i]:.2f});')
        sfx(sc.sid, t_list[i], "click-soft", 0.2)
    sc.css.append(f"#root .rl{{position:absolute;display:flex;align-items:center;gap:{18 * u}px;font-size:{32 * u}px;font-weight:700;}}#root .rl i{{font-style:normal;font-weight:900;font-size:{40 * u}px;width:{40 * u}px;}}"
                  f"#root .rl b{{flex:1;font-weight:700;}}#root .rl span{{{MONO}font-weight:800;}}")


def raw_head(sc, x, y, w, t, size=24, quote_t=None):
    H_ = EX["head"]; cols = ["name", "borough", "room type", "price", "bathrooms", "min nights", "price quote"]
    rows = [[(r_[0] or "")[:18] + ("…" if r_[0] and len(r_[0]) > 18 else ""), r_[1], r_[2].replace("Entire home/apt", "Entire home"), r_[3], r_[4],
             None if r_[5] is None else int(float(r_[5])), r_[6]] for r_ in H_["rows"]]
    cls = lambda j, v: "nan" if v is None else ("txt" if j in (3, 4) else ("ans" if j == 6 else ""))
    return table(sc, "raw", cols, rows, x, y, w, t, size=size, cellcls=cls, colw=[20, 12, 14, 12, 16, 11, 15])


# ---------------------------------------------------------------- YouTube scenes
def S_DP(sid, W, H):
    sc = b.Scene(sid, W, H); u, P = sc.u, sc.P
    L = W * 0.05; d = b.dur(sid); top = H * 0.17
    if sid in STAGE_OF:
        stage_header(sc, STAGE_OF[sid])
    agent_line(sc, 0.9 if sid in FIRST_BEAT else 0.6)
    if P:
        S_reel(sc, sid, W, H, L, d); return sc

    if sid == "Y01":      # hook: the real raw rows, problems light up
        aura(sc, (AMB, RED))
        label(sc, "k", "REAL DATA · 29,918 NYC LISTINGS · NOTHING PRE-CLEANED", L, H * 0.08, size=26, color=MUTE, weight=800, t=.1)
        raw_head(sc, L, H * 0.16, W * 0.90, .2, size=26)
        bigstat(sc, "m", 35.1, "of the prices are missing", L, H * 0.52, at(sid, "missing", default=1.5), color=RED, dec=1, suffix="%")
        bigstat(sc, "t", 1, "column secretly contains the answer", W * 0.36, H * 0.52, at(sid, "secretly", default=5), color=PUR)
        label(sc, "txt", "prices stored as text: \"$164.55\"", W * 0.62, H * 0.55, size=30, color=AMB, weight=800, t=at(sid, "text", default=3), mono=True)
    elif sid == "Y02":    # the pipeline map
        chapter(sc, "THE END-TO-END PIPELINE")
        toks = ["question", "ingest", "validate", "clean", "store", "explore", "engineer", "split", "build", "model", "evaluate", "explain", "deploy", "monitor"]
        times = [at(sid, k_, default=1 + i * .6) for i, k_ in enumerate(toks)]
        times.insert(9, times[8] + .3)    # 'Leakage' sits between pipeline and model
        tb = [times[z[3][0]] - .3 for z in ZONES]
        pipeline_blocks(sc, L, H * 0.22, W * 0.90, H * 0.36, tb, None, d, stage_t=times)
        bigstat(sc, "n", 15, "stages · most tutorials skip half of them", L, H * 0.64, at(sid, "fifteen", default=12), size=64)
    elif sid == "Y03":
        card(sc, "q", "How much should a NYC listing charge per night?", "the question", L, top, W * 0.90, at(sid, "much", default=1.5), AMB, big=48)
        card(sc, "m", "MAE in dollars", "the metric: the average miss per listing", L, top + 200 * u, W * 0.43, at(sid, "metric", default=6), INK, big=44)
        card(sc, "r", "Beat a simple baseline", "on listings the model has never seen", W * 0.52, top + 200 * u, W * 0.43, at(sid, "rule", default=9), GRN, big=44)
        formula(sc, "f", "MAE = mean( | real price − predicted price | )", L, top + 420 * u, W * 0.90, at(sid, "average", default=7), size=36)
    elif sid == "Y04":
        I_ = V["ingest"]
        card(sc, "file", "listings.csv.gz", "Inside Airbnb · New York City · snapshot 2026-09-14 · CC BY 4.0", L, top, W * 0.55, .4, INK, big=44)
        bigstat(sc, "r", I_["rows"], "listings (rows)", L, top + 200 * u, at(sid, "twenty", default=3))
        bigstat(sc, "c", I_["cols"], "columns", L + W * 0.30, top + 200 * u, at(sid, "ninety", default=6))
        bigstat(sc, "s", I_["mb"], "megabytes compressed", L + W * 0.50, top + 200 * u, at(sid, "sixteen", default=7), dec=1)
        k_ = I_["kinds"]; tot = sum(k_.values())
        for i, (nm, n) in enumerate(k_.items()):
            el = sc.id(f"k{i}"); x0 = L + (0 if i == 0 else W * 0.62 * k_["number"] / tot)
            sc.add(f'<div id="{el}" class="kd" style="background:{[BLU, AMB][i]}">{n} {nm} columns</div>', f"#{el}{{left:{x0}px;top:{top + 400 * u}px;width:{W * 0.62 * n / tot - 6 * u}px;}}")
            sc.js.append(f'tl.fromTo("#{el}",{{opacity:0,scaleX:0,transformOrigin:"0% 50%"}},{{opacity:1,scaleX:1,duration:.5}},{at(sid, "ninety", default=6) + .6 + i * .3:.2f});')
        sc.css.append(f"#root .kd{{position:absolute;height:{56 * u}px;border-radius:{10 * u}px;color:#141517;font-weight:900;font-size:{26 * u}px;padding-left:{14 * u}px;line-height:{56 * u}px;white-space:nowrap;overflow:hidden;}}")
        stamp(sc, "ro", "RAW · READ-ONLY", W * 0.66, top + 40 * u, AMB, at(sid, "never", default=9), size=48, rot=-6)
        label(sc, "cp", "every change happens on a copy", W * 0.66, top + 520 * u, size=30, color=MUTE, t=at(sid, "copy", default=12))
    elif sid == "Y05":
        cam_then_hide(sc, 6, .2, d - .3, [("$164.55", at(sid, "exists", default=3)), ("1 private bath", at(sid, "Bathrooms", default=5))], zoom=2.2)
    elif sid == "Y06":
        n, gh = miss_grid(sc, L, top + 40 * u, 34 * u, .4, d=3.0)
        label(sc, "gl", "60 random listings × the 24 columns that have gaps · bright = missing", L, top, size=24, color=MUTE, t=.3)
        bigstat(sc, "e", V["missing"]["empty"], "columns are 100% empty", W * 0.62, top + 40 * u, at(sid, "thirteen", default=4))
        bigstat(sc, "p", 35.1, "of the prices are missing", W * 0.62, top + 260 * u, at(sid, "thirty", default=7), color=RED, dec=1, suffix="%")
    elif sid == "Y07":
        cam_then_hide(sc, 9, .2, at(sid, "price", default=4) + .2, [("10,512 rows fail", at(sid, "ten", default=3))], zoom=2.0)
        t0 = at(sid, "price", default=4) + .4
        rules_list(sc, L, top + 30 * u, W * 0.80, [t0, at(sid, "location", default=t0 + 4), at(sid, "location", default=t0 + 4) + .2, at(sid, "minimum", default=t0 + 2), at(sid, "IDs", default=t0 + 5)])
    elif sid == "Y08":
        M_ = V["mnar"]
        label(sc, "q", "Is the price missing at random?", L, top, size=48, weight=900, t=.3)
        hbars(sc, "mb", [("stays under 30 nights", M_["short"] * 100, GRN, at(sid, "short", default=3), False), ("30 nights or more", M_["long"] * 100, RED, at(sid, "thirty", default=5), True)],
              (L, top + 120 * u, W * 0.62, 0), 50, fmt="{:.1f}% missing", rowh=110, lab_w=.36)
        bigstat(sc, "ls", M_["long_share"] * 100, "of all listings require 30+ nights (NYC short-term rental law)", L, top + 400 * u, at(sid, "eighty", default=8), dec=1, suffix="%", size=80, w=W * .8)
        stamp(sc, "nr", "NOT RANDOM", W * 0.66, top + 420 * u, RED, at(sid, "reason", default=12), size=54)
    elif sid in ("Y09", "Y10", "Y11", "Y12", "Y13"):
        S_clean(sc, sid, W, H, L, d, top)
    elif sid == "Y14":
        S_ = V["store"]
        card(sc, "raw", "RAW · listings.csv.gz", f'{S_["raw_mb"]} MB · {V["ingest"]["rows"]:,} rows · {S_["raw_cols"]} columns · price is text · never edited', L, top + 40 * u, W * 0.40, .4, AMB, big=36)
        card(sc, "cln", "CLEAN · listings_2026-09-14.parquet", f'{S_["clean_mb"]} MB · {S_["rows"]:,} rows · {S_["clean_cols"]} columns · price is a number · dated version', W * 0.55, top + 40 * u, W * 0.40, at(sid, "parquet", default=3), GRN, big=36)
        pulse(sc, "p", L + W * 0.40, top + 110 * u, W * 0.55, top + 110 * u, at(sid, "parquet", default=3), d=1.0)
        bigstat(sc, "a", S_["raw_mb"], "MB raw", L, top + 330 * u, at(sid, "sixteen", default=5), dec=1, color=AMB)
        bigstat(sc, "b", S_["clean_mb"], "MB clean", W * 0.55, top + 330 * u, at(sid, "point", default=6), dec=1, color=GRN)
        label(sc, "lay", "raw layer  →  clean layer  →  features  →  model", L, top + 520 * u, size=34, color=MUTE, weight=800, mono=True, t=at(sid, "layer", default=10))
    elif sid == "Y15":
        hist_morph(sc, (L, top, W * 0.56, H * 0.56), .5, at(sid, "log", default=5))
        label(sc, "hr", "nightly price (real listings)", L, top - 40 * u, size=24, color=MUTE, t=.4)
        bigstat(sc, "s1", 2.48, "skew of the price", W * 0.66, top + 40 * u, at(sid, "skewed", default=2), dec=2, color=AMB)
        bigstat(sc, "s2", 0.30, "skew of log(price)", W * 0.66, top + 260 * u, at(sid, "skewness", default=8), dec=2, color=GRN)
    elif sid == "Y16":
        B_ = V["by_borough"]; rows = [[r_] + [("—" if v < 0 else f"${v:,.0f}") for v in vals] for r_, vals in zip(B_["rows"], B_["vals"])]
        cls = lambda j, v: ("txt" if v in ("$251",) else ("ok" if v == "$101" else ""))
        table(sc, "bt", ["median price"] + [c_.replace("Entire home/apt", "Entire home") for c_ in B_["cols"]], rows, L, top, W * 0.66, .3, size=30, cellcls=cls, stagger=.15)
        label(sc, "x", "private room: Manhattan $251 vs Brooklyn $101 → 2.5×", L, top + 480 * u, size=36, color=AMB, weight=900, t=at(sid, "private", default=4))
    elif sid == "Y17":
        nyc_map(sc, (L, top - 10 * u, W * 0.48, H * 0.58), .3, d=2.0, tsq=at(sid, "Midtown", default=2))
        legend(sc, W * 0.52, top + 400 * u, .5)
        S_ = V["stay_price"]
        hbars(sc, "sp", [("under 30 nights", S_["short"], RED, at(sid, "under", default=5), True), ("30 nights or more", S_["long"], BLU, at(sid, "more", default=8), False)],
              (W * 0.52, top + 120 * u, W * 0.44, 0), 400, fmt="${:.0f}", rowh=110, lab_w=.42)
        label(sc, "sh", "median price per night", W * 0.52, top + 40 * u, size=30, color=MUTE, weight=800, t=at(sid, "surprise", default=4))
    elif sid == "Y18":
        parallel(sc, (L + 20 * u, top + 30 * u, W * 0.36, H * 0.48), .5, at(sid, "closer", default=9))
        label(sc, "pl", "399 real listings · one line each · red = most expensive third", L, top - 30 * u, size=22, color=MUTE, weight=700, t=.3)
        formula(sc, "hv", "km_to_times_sq = haversine(lat, lon, Times Square)<small>computed from latitude and longitude</small>", W * 0.46, top, W * 0.50, at(sid, "computed", default=5), size=28)
        bigstat(sc, "c", V["features"]["corr_dist"], "correlation with log(price): closer → pricier", W * 0.46, top + 200 * u, at(sid, "correlation", default=8), dec=2, color=AMB)
        F_ = V["features"]; chips = F_["num"] + F_["cat"]
        el = sc.id("chips")
        sc.add(f'<div id="{el}" class="chp">' + "".join(f'<i class="{"cat" if c_ in F_["cat"] else ""}">{c_}</i>' for c_ in chips) + "</div>", f"#{el}{{left:{W * 0.46}px;top:{top + 400 * u}px;width:{W * 0.50}px;}}")
        sc.css.append(f"#root .chp{{position:absolute;display:flex;flex-wrap:wrap;gap:{6 * u}px;}}#root .chp i{{font-style:normal;{MONO}font-size:{19 * u}px;padding:{4 * u}px {8 * u}px;border-radius:{6 * u}px;background:rgba(110,155,255,.18);color:#C9D7FF;}}"
                      f"#root .chp i.cat{{background:rgba(180,140,255,.22);color:#E1D2FF;}}")
        sc.js.append(f'tl.fromTo("#{el} i",{{opacity:0,scale:.6}},{{opacity:1,scale:1,duration:.2,stagger:.04}},{at(sid, "flag", default=11):.2f});')
        label(sc, "fc", f'{len(chips)} features · blue = numeric · purple = category', W * 0.46, top + 600 * u, size=24, color=MUTE, t=at(sid, "features", default=13))
    elif sid == "Y19":
        cam_then_hide(sc, 27, .2, at(sid, "twins", default=2) + .3, [("random split: 342", at(sid, "random", default=2))], zoom=1.8)
        twin_split(sc, L, top + 60 * u, W * 0.58, H * 0.52, at(sid, "random", default=3) + .4, at(sid, "host", default=9))
    elif sid == "Y20":
        label(sc, "sh", "every row's journey · real counts", L, top, size=26, color=MUTE, weight=800, t=.2)
        sankey(sc, (L, top + 90 * u, W * 0.74, H * 0.50), .4, dt=.8)
        stamp(sc, "lk", "🔒 UNTOUCHED UNTIL MONITORING", W * 0.56, top + 430 * u, PUR, at(sid, "never", default=5), size=34, rot=-4)
    elif sid == "Y21":
        flow(sc, [("27 columns", "the features, with 12,804 missing values in train", INK), ("Imputer", "fills gaps with the median, learned from TRAIN only", AMB),
                  ("One-hot encoder", "a column per category (rare ones grouped)", AMB), ("145 columns", "0 missing values", GRN), ("Model", "same object in training and production", BLU)],
             L, top + 60 * u, W * 0.90, [.4, at(sid, "median", default=3), at(sid, "one", default=7), at(sid, "hundred", default=9), at(sid, "object", default=12)], h=190)
        label(sc, "tr", "fit on TRAIN only → the test set stays truly unseen", L, top + 270 * u, size=32, color=AMB, weight=900, t=at(sid, "only", default=6))
        impute_demo(sc, L, top + 390 * u, W * 0.70, at(sid, "Missing", default=1.5), at(sid, "median", default=3), at(sid, "one", default=7))
    elif sid == "Y22":
        cam_then_hide(sc, 31, .2, at(sid, "perfect", default=6) + .6, [("R² 1.000", at(sid, "jumps", default=5))], zoom=2.0)
        t0 = at(sid, "perfect", default=6) + .7
        card(sc, "col", "price_quote_price_per_night", "an innocent-looking column", L, top + 30 * u, W * 0.48, t0, PUR, big=40)
        bigstat(sc, "r2", 1.000, "R² with this column · too good to be true", L, top + 220 * u, t0 + .2, dec=3, color=RED, size=120)
        bigstat(sc, "co", 1.000, "correlation with the real price → it IS the price", W * 0.56, top + 220 * u, at(sid, "disguise", default=t0 + 2), dec=3, color=PUR, size=90)
        stamp(sc, "rm", "REMOVED", W * 0.60, top + 30 * u, RED, at(sid, "remove", default=t0 + 3), size=60)
    elif sid == "Y23":
        M_ = V["models"]; ks = list(M_)
        label(sc, "h", "average miss per listing (MAE) · lower is better", L, top, size=28, color=MUTE, weight=800, t=.3)
        hbars(sc, "mae", [("Baseline", M_[ks[0]]["mae"], MUTE, at(sid, "baseline", default=2), False),
                          ("Ridge regression", M_[ks[1]]["mae"], BLU, at(sid, "Ridge", default=6), False), ("LightGBM", M_[ks[2]]["mae"], GRN, at(sid, "seventy", default=8), True)],
              (L, top + 80 * u, W * 0.46, 0), 160, fmt="${:.1f}", rowh=120, lab_w=.42)
        label(sc, "hf", "half the baseline's error", L, top + 470 * u, size=44, color=GRN, weight=900, t=at(sid, "Half", default=10))
        learn_curve(sc, (W * 0.54, top, W * 0.42, H * 0.56), at(sid, "Light", default=6) - 1.5, d_draw=3.0)
    elif sid == "Y24":
        cam_then_hide(sc, 34, .2, at(sid, "lie", default=5) - .2, [("random split $65.6", at(sid, "sixty", default=3))], zoom=2.0)
        t0 = at(sid, "lie", default=5)
        hbars(sc, "rs", [("random split (twins on both sides)", V["split"]["mae_random"], RED, t0, False), ("split by host (honest)", V["models"]["LightGBM"]["mae"], GRN, t0 + .6, True)],
              (L, top + 80 * u, W * 0.88, 0), 90, fmt="${:.1f}", rowh=120, lab_w=.40)
        stamp(sc, "lie", "LOOKS BETTER · IS A LIE", W * 0.50, top + 330 * u, RED, t0 + .2, size=48)
    elif sid == "Y25":
        pred_scatter(sc, (L, top - 10 * u, W * 0.44, H * 0.58), .5)
        bigstat(sc, "w", V["eval"]["within25"] * 100, "of test listings within ±25% of the real price", W * 0.54, top, at(sid, "sixty", default=6), dec=1, suffix="%", color=GRN, w=W * .42)
        rows = [(r_[0], r_[1], RED if r_[0] == "Manhattan" else BLU, at(sid, "biggest", default=10) + i * .15, r_[0] == "Manhattan") for i, r_ in enumerate(V["eval"]["by_borough"])]
        hbars(sc, "bb", rows, (W * 0.54, top + 260 * u, W * 0.42, 0), 120, fmt="${:.0f}", rowh=62, lab_w=.42)
    elif sid == "Y26":
        for i, ex in enumerate(V["examples"]):
            nb_, room, acc, real, pred = ex
            tt = [at(sid, "Mill", default=1.5), at(sid, "Midtown", default=8), at(sid, "Midtown", default=8) + 2][[1, 0, 2][i]] if i < 3 else 1
            card(sc, f"e{i}", f"real ${real:,.0f}  ·  predicted ${pred:,.0f}", f"{nb_} · {room} · {acc} guests", L + i * W * 0.305, top + 80 * u, W * 0.29, tt, GRN if abs(pred - real) / real <= .25 else AMB, big=34)
        label(sc, "n", "three real test listings the model never saw", L, top, size=30, color=MUTE, weight=800, t=.3)
    elif sid == "Y27":
        I_ = V["importance"]
        label(sc, "h", "shuffle one column → how much worse does the model get?", L, top, size=30, color=MUTE, weight=800, t=.3)
        hbars(sc, "im", [(k_, v_, AMB if i == 0 else BLU, at(sid, "minimum", default=4) + i * .3, i == 0) for i, (k_, v_) in enumerate(I_)], (L, top + 70 * u, W * 0.80, 0), I_[0][1] * 1.1, fmt="{:.3f}", rowh=66, lab_w=.36)
        label(sc, "law", "↑ minimum stay: the 30-night law, learned from data", L, top + 640 * u, size=30, color=AMB, weight=900, t=at(sid, "law", default=6))
    elif sid == "Y28":
        D_ = V["deploy"]
        card(sc, "f", "price_pipeline.joblib", f'{D_["mb"]} MB · cleaning rules + encoder + model in one file', L, top, W * 0.42, at(sid, "file", default=3), INK, big=38)
        json_panel(sc, "rq", "POST /predict", json.dumps(D_["listing"], indent=1), L, top + 190 * u, W * 0.42, at(sid, "Send", default=7), INK)
        json_panel(sc, "rs", f'RESPONSE · {D_["ms"]:.0f} ms', json.dumps({"price_per_night": D_["out"]}, indent=1), W * 0.52, top + 190 * u, W * 0.42, at(sid, "price", default=9), GRN)
        pulse(sc, "pp", L + W * 0.42, top + 300 * u, W * 0.52, top + 300 * u, at(sid, "price", default=9) - .4, d=.6)
        label(sc, "rl", f'real price: ${D_["real"]:,.0f}', W * 0.52, top + 420 * u, size=40, color=AMB, weight=900, t=at(sid, "real", default=12))
    elif sid == "Y29":
        label(sc, "h", f'drift (PSI): training data vs {V["monitor"]["n_new"]:,} listings new in 2026', L, top, size=28, color=MUTE, weight=800, t=.3)
        psi_bars(sc, L, top + 100 * u, W * 0.80, at(sid, "compare", default=3))
        label(sc, "ex", "expected: new listings have few reviews", W * 0.56, top + 470 * u, size=30, color=AMB, weight=800, t=at(sid, "expected", default=8))
    elif sid == "Y30":
        cam_then_hide(sc, 43, .2, at(sid, "signal", default=4) - .2, [("$85.0", at(sid, "eighty", default=3))], zoom=1.8)
        t0 = at(sid, "signal", default=4)
        hbars(sc, "mn", [("test set (2025 and older)", V["models"]["LightGBM"]["mae"], GRN, t0, False), ("new listings (2026)", V["monitor"]["mae_new"], RED, t0 + .5, True)],
              (L, top + 80 * u, W * 0.80, 0), 100, fmt="${:.1f}", rowh=120, lab_w=.36)
        stamp(sc, "rt", "↻ RETRAIN", W * 0.62, top + 330 * u, AMB, at(sid, "never", default=t0 + 2), size=64)
    elif sid == "Y31":
        aura(sc, (AMB, GRN))
        toks = ["question", "ingest", "validate", "clean", "store", "explore", "engineer", "split", "build", "model", "evaluate", "explain", "deploy", "monitor"]
        times = [at(sid, k_, default=1 + i * .5) for i, k_ in enumerate(toks)]
        times.insert(9, times[8] + .2)
        ta_, tb_ = at(sid, "question", default=2), at(sid, "monitor", default=9)
        pipeline_blocks(sc, L, H * 0.16, W * 0.90, H * 0.36, [.1 + i * .15 for i in range(5)], 1.0, d, stage_t=[ta_ + (tb_ - ta_) * i / 14 for i in range(15)])
        label(sc, "o", "the model is 1 stage of 15", L, H * 0.62, size=56, color=AMB, weight=900, t=at(sid, "stage", default=12))
    elif sid == "Y32":
        aura(sc, (AMB, GRN))
        for i, (t_, tok) in enumerate([("① never edit raw data", "Never"), ("② split before anything learns", "Split"), ("③ always beat a baseline", "always")]):
            label(sc, f"r{i}", t_, L, H * 0.18 + i * 100 * u, size=60, weight=900, t=at(sid, tok, default=1 + i * 2), w=W * .9)
        label(sc, "gh", "Notebook + every result: github.com/reazulzishan", L, H * 0.58, size=44, color=AMB, weight=900, t=at(sid, "GitHub", default=8), w=W * .9)
        label(sc, "src", "Data: Inside Airbnb, New York City, 2026-09-14 · CC BY 4.0", L, H * 0.66, size=28, color="#D9D6CE", weight=700, t=at(sid, "notebook", default=7), w=W * .9)
    return sc


def S_clean(sc, sid, W, H, L, d, top):
    """Stage 4 beats: a persistent cleaning log on the right, the step itself on the left."""
    lx, lw = W * 0.55, W * 0.40
    if sid == "Y09":
        clean_log(sc, lx, top + 60 * u_(sc), lw, 2, {0: .5, 1: at(sid, "thirteen", default=5), 2: at(sid, "twenty", default=7)})
        bigstat(sc, "e", 13, "empty columns dropped", L, top + 40 * u_(sc), at(sid, "thirteen", default=5), color=RED)
        bigstat(sc, "k", 25, "columns kept: the ones that can explain a price", L, top + 260 * u_(sc), at(sid, "twenty", default=7), color=GRN, w=W * .45)
    elif sid == "Y10":
        clean_log(sc, lx, top + 60 * u_(sc), lw, 3, {3: at(sid, "drop", default=6)})
        label(sc, "h", "price: text → number", L, top, size=30, color=MUTE, weight=800, t=.3)
        for i, (a_, b_) in enumerate(zip(V["price_examples"], [164.55, 236.0, 143.47])):
            morph(sc, f"m{i}", f'"{a_}"', b_, L, top + 70 * u_(sc) + i * 80 * u_(sc), at(sid, "dollar", default=2) + i * .3)
        bigstat(sc, "n", 19406, "rows with a real price remain", L, top + 340 * u_(sc), at(sid, "Nineteen", default=9), color=GRN)
        label(sc, "nv", "we never invent the answer we predict", L, top + 520 * u_(sc), size=30, color=AMB, weight=800, t=at(sid, "invent", default=7))
    elif sid == "Y11":
        clean_log(sc, lx, top + 60 * u_(sc), lw, 4, {4: at(sid, "count", default=9)})
        def baths(t_):
            return 0.5 if "half" in t_.lower() else float(re.findall(r"[\d.]+", t_)[0])
        for i, t_ in enumerate(["1.5 shared baths", "Half-bath", "2 private baths"]):
            morph(sc, f"b{i}", f'"{t_}"', f'{baths(t_)}' + ("  + shared=1" if "shared" in t_ else ""), L, top + i * 70 * u_(sc), at(sid, "bathroom", default=1) + i * .5, size=32)
        A_ = EX["amenities"]; am = json.loads(A_["text"])
        json_panel(sc, "am", "amenities · one text cell (real)", A_["text"][:150] + "…", L, top + 240 * u_(sc), W * 0.46, at(sid, "amenities", default=6), AMB)
        flags = [("wifi", "Wifi"), ("kitchen", "Kitchen"), ("aircon", "Air conditioning"), ("washer", "Washer"), ("workspace", "Dedicated workspace")]
        txt = f'n_amenities = {len(am)}   ' + "  ".join(f'{k_}={int(w_ in am)}' for k_, w_ in flags)
        label(sc, "fl", txt, L, top + 430 * u_(sc), size=24, color=GRN, weight=800, mono=True, t=at(sid, "count", default=9), w=W * .46)
    elif sid == "Y12":
        clean_log(sc, lx, top + 60 * u_(sc), lw, 6, {5: at(sid, "Seven", default=1), 6: at(sid, "keep", default=12)})
        price_strip(sc, (L, top + 40 * u_(sc), W * 0.46, H * 0.36), at(sid, "expensive", default=4), at(sid, "Above", default=8))
        label(sc, "mx", f'max ${EX["prices"]["max"]:,.0f} per night · {EX["prices"]["above_2000"]} listings above $2,000', L, top + H * 0.44, size=30, color=RED, weight=800, t=at(sid, "thirty", default=5), w=W * .48)
        label(sc, "h", "every price, log scale (sample + the most extreme)", L, top, size=24, color=MUTE, t=.3)
    elif sid == "Y13":
        clean_log(sc, lx, top + 60 * u_(sc), lw, 6, {})
        T_ = EX["twins"]
        for i, (id_, nm, nb_, room, acc, price) in enumerate(T_):
            card(sc, f"t{i}", f"{nm[:34]}{'…' if len(nm) > 34 else ''}", f"same host · same spot · {nb_} · {room} · {acc} guests · <b style='color:{AMB};font-size:inherit;display:inline'>{price}</b> · id …{str(id_)[-6:]}",
                 L, top + 150 * u_(sc) + i * 190 * u_(sc), W * 0.46, at(sid, "twins", default=4) + i * .4, INK, big=28)
        D_ = V["dupes"]
        label(sc, "ex", f'exact duplicates: {D_["exact"]}   ·   look-alike rows: {D_["twins"]:,}', L, top + 40 * u_(sc), size=34, color=AMB, weight=900, t=at(sid, "Duplicates", default=1), w=W * .48)
        label(sc, "kp", "real, separate apartments → keep, but watch the split", L, top + 560 * u_(sc), size=28, color=GRN, weight=800, t=at(sid, "keep", default=8), w=W * .48)


u_ = lambda sc: sc.u


# ---------------------------------------------------------------- the Reel (9:16)
def S_reel(sc, sid, W, H, L, d):
    u = sc.u; top = H * 0.16; rid = sid.replace("Q", "R")
    if rid == "R01":      # frame 1: the real raw rows with the problems glowing
        aura(sc, (AMB, RED))
        label(sc, "h", "REAL DATA · 29,918 LISTINGS", L, H * 0.10, size=34, color=MUTE, weight=900, t=0)
        H_ = EX["head"]
        rows = [[r_[1], r_[3], r_[4], r_[6]] for r_ in H_["rows"]]
        cls = lambda j, v: "nan" if v is None else ("txt" if j in (1, 2) else ("ans" if j == 3 else ""))
        table(sc, "raw", ["borough", "price", "bathrooms", "price quote"], rows, L, H * 0.15, W * 0.90, 0, size=30, cellcls=cls, stagger=.05)
        bigstat(sc, "m", 35.1, "of the prices are missing", L, H * 0.42, 0, color=RED, dec=1, suffix="%", size=130)
        bigstat(sc, "t", 1, "column secretly contains the answer", L, H * 0.56, at(sid, "secretly", default=3), color=PUR, size=130)
        th = at(sid, "Here", default=4.5)          # the hook gives way to the whole pipeline
        sc.js.append("".join(f'tl.to("#{sc.id(k_)}",{{opacity:0,y:{-30 * u},duration:.35}},{th - .1:.2f});' for k_ in ("h", "raw", "m", "t")))
        pipeline_blocks(sc, L, H * 0.11, W * 0.90, H * 0.66, [th + .1 + i * .3 for i in range(5)], None, d, vertical=True)
    elif rid == "R02":
        label(sc, "h", "INGEST → VALIDATE", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        bigstat(sc, "r", V["ingest"]["rows"], "listings · read-only raw file", L, top + 20 * u, .3, size=100)
        n, gh = miss_grid(sc, L, top + 260 * u, 38 * u, at(sid, "validate", default=4), d=2.5)
        bigstat(sc, "e", 13, "empty columns", L, top + 300 * u + gh, at(sid, "thirteen", default=5), color=RED, size=80)
        bigstat(sc, "p", 10512, "rows without a price", W * 0.48, top + 300 * u + gh, at(sid, "ten", default=6), color=RED, size=80)
    elif rid == "R03":
        M_ = V["mnar"]
        label(sc, "h", "MISSING ≠ RANDOM", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        hbars(sc, "mb", [("short stays", M_["short"] * 100, GRN, at(sid, "short", default=3), False), ("30+ nights", M_["long"] * 100, RED, at(sid, "Forty", default=2), True)],
              (L, top + 60 * u, W * 0.90, 0), 50, fmt="{:.1f}%", rowh=130, lab_w=.34)
        stamp(sc, "nr", "FIND THE REASON", L + 40 * u, top + 420 * u, RED, at(sid, "reason", default=6), size=56)
    elif rid == "R04":
        label(sc, "h", "CLEAN · STEP BY STEP", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        for i, (a_, b_) in enumerate(zip(V["price_examples"][:2], [164.55, 236.0])):
            morph(sc, f"m{i}", f'"{a_}"', b_, L, top + 30 * u + i * 80 * u, at(sid, "text", default=1.5) + i * .3, size=46)
        morph(sc, "bt", '"1.5 shared baths"', "1.5 + flag", L, top + 210 * u, at(sid, "baths", default=4), size=40)
        sankey(sc, (L, top + 400 * u, W * 0.58, 400 * u), at(sid, "Clean", default=.3) + 1.0, dt=1.4)
    elif rid == "R05":
        label(sc, "h", "EXPLORE", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        hist_morph(sc, (L, top + 10 * u, W * 0.90, H * 0.24), .3, at(sid, "log", default=3))
        nyc_map(sc, (L + W * 0.05, top + H * 0.28, W * 0.80, H * 0.25), at(sid, "closer", default=4), d=1.5, tsq=at(sid, "Times", default=5), pt=3.4)
    elif rid == "R06":
        label(sc, "h", "SPLIT BEFORE ANYTHING LEARNS", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        twin_split(sc, L, top + 70 * u, W * 0.50, H * 0.36, at(sid, "random", default=2), at(sid, "host", default=7))
    elif rid == "R07":
        label(sc, "h", "THE LEAKAGE TRAP", L, top - 40 * u, size=34, color=RED, weight=900, t=.1)
        card(sc, "col", "price_quote_price_per_night", "one innocent-looking column", L, top + 20 * u, W * 0.90, .4, PUR, big=44)
        bigstat(sc, "r2", 1.000, "R² · a perfect score", L, top + 230 * u, at(sid, "perfect", default=2), dec=3, color=RED, size=150)
        stamp(sc, "rm", "IT WAS THE PRICE", L + 20 * u, top + 520 * u, RED, at(sid, "itself", default=4), size=58)
    elif rid == "R08":
        M_ = V["models"]; ks = list(M_)
        label(sc, "h", "BASELINE FIRST → MODEL → DEPLOY", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        hbars(sc, "mae", [("Baseline", M_[ks[0]]["mae"], MUTE, at(sid, "Baseline", default=.5), False), ("LightGBM", M_[ks[2]]["mae"], GRN, at(sid, "seventy", default=3), True)],
              (L, top + 40 * u, W * 0.90, 0), 160, fmt="${:.1f}", rowh=130, lab_w=.30)
        D_ = V["deploy"]
        json_panel(sc, "rs", f'POST /predict → {D_["ms"]:.0f} ms', json.dumps({"price_per_night": D_["out"]}), L, top + 360 * u, W * 0.90, at(sid, "deploy", default=5), GRN)
        label(sc, "rl", f'real price ${D_["real"]:,.0f}', L, top + 540 * u, size=44, color=AMB, weight=900, t=at(sid, "milliseconds", default=8))
    elif rid == "R09":
        label(sc, "h", "MONITOR", L, top - 40 * u, size=34, color=AMB, weight=900, t=.1)
        hbars(sc, "mn", [("test set", V["models"]["LightGBM"]["mae"], GRN, .4, False), ("new in 2026", V["monitor"]["mae_new"], RED, at(sid, "eighty", default=4), True)],
              (L, top + 40 * u, W * 0.90, 0), 100, fmt="${:.1f}", rowh=130, lab_w=.32)
        stamp(sc, "rt", "↻ RETRAIN", L + 120 * u, top + 400 * u, AMB, at(sid, "retrain", default=6), size=70, rot=-5)
    elif rid == "R10":
        aura(sc, (AMB, GRN))
        for i, (t_, tok) in enumerate([("① never edit raw data", "Never"), ("② split before anything learns", "Split"), ("③ beat a baseline", "baseline")]):
            card(sc, f"r{i}", t_, "", L, top + i * 150 * u, W * 0.90, at(sid, tok, default=.3 + i * 1.5), AMB, big=44)
        label(sc, "yt", "Full tutorial: YouTube @csfunda.official", L, H * 0.60, size=44, weight=900, color=AMB, t=at(sid, "YouTube", default=6), w=W * .9)
        label(sc, "c", "Which stage do you skip most? Comment ↓", L, H * 0.66, size=44, weight=900, t=at(sid, "Comment", default=9), w=W * .9)


ORDER = [f"Y{i:02d}" for i in range(1, 33)]
REEL = [f"Q{i:02d}" for i in range(1, 11)]
for i in range(1, 11):   # the Reel's voice lines are R01..R10 in the script
    b.SCENES[f"Q{i:02d}"] = {**meta[f"R{i:02d}"], "id": f"Q{i:02d}"}
    for src_, dst_ in [(VID / f"words/R{i:02d}.words.json", VID / f"words/Q{i:02d}.words.json"), (HERE / f"assets/vo/R{i:02d}.wav", HERE / f"assets/vo/Q{i:02d}.wav")]:
        if src_.exists(): dst_.write_bytes(src_.read_bytes())
b.S, b.ORDER = S_DP, ORDER


def badge(out, W, H, reel, total):
    u = min(W, H) / 1080
    (out / "compositions/badge.html").write_text(f"""<template>
<style>#root{{position:absolute;inset:0;}}
#root #bd{{position:absolute;right:{W * 0.04}px;top:{H * 0.025}px;font-family:Inter,sans-serif;font-weight:800;color:#F3F0E8;font-size:{26 * u}px;
 padding:{8 * u}px {16 * u}px;border-radius:{12 * u}px;background:rgba(20,21,23,.8);opacity:0;}}#root #bd b{{color:{AMB}}}</style>
<div id="root" data-composition-id="badge" data-width="{W}" data-height="{H}"><div id="bd">CS Funda · <b>Data Pipeline</b></div></div>
<script>const tl=gsap.timeline({{paused:true}});tl.fromTo("#bd",{{opacity:0}},{{opacity:1,duration:.5}},3.0);window.__timelines["badge"]=tl;</script></template>""")
    return (f'<div id="badge" data-composition-id="badge" data-composition-src="compositions/badge.html" data-start="0" data-duration="{total}" '
            f'data-track-index="6" data-track-kind="graphics" data-width="{W}" data-height="{H}" style="z-index:6"></div>')


b.part_badge = badge


if __name__ == "__main__":
    b.SFXLEN = {f.stem: round(float(__import__("subprocess").run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                capture_output=True, text=True).stdout), 2) for f in (HERE / "assets/sfx").glob("*.mp3")}
    EXTRA.clear()
    if len(sys.argv) > 1 and sys.argv[1] == "reel":
        b.LEAD, b.TAIL = 0.1, 0.3
        out = VID / "reel"; shutil.rmtree(out, ignore_errors=True)
        b.project(out, 1080, 1920, REEL, reel=1)
        ml.add_extra_audio(out)
    else:
        b.project(HERE, 1920, 1080, ORDER)
        ml.add_extra_audio(HERE)
