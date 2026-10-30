"""Design system: palette, CSS, small HTML builders and chart specifications.

Type: Instrument Serif (display), IBM Plex Sans (body), IBM Plex Mono (labels and data). The fonts load from
Google Fonts; when they cannot load (offline) the fallbacks below are used and nothing else changes.
Every piece of dynamic text that goes into HTML passes through esc()."""
import html
from typing import Iterable, List, Optional

import pandas as pd

PAPER, SURFACE, INK, MUTED, ACCENT = "#F5F3EE", "#FFFFFF", "#20231F", "#73766F", "#B64B36"
LINE, LINE_SOFT = "#DDD8CC", "#E9E5DB"
RISK = {"Low": "#4F7A5A", "Medium": "#B7862B", "High": "#A8412F"}
RISK_TINT = {"Low": "#E6EEE6", "Medium": "#F4EBD3", "High": "#F3DFD9"}

SERIF = "'Instrument Serif', Georgia, 'Times New Roman', serif"
SANS = "'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', Consolas, 'SFMono-Regular', Menlo, monospace"


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
:root {{ --paper:{PAPER}; --surface:{SURFACE}; --ink:{INK}; --muted:{MUTED}; --accent:{ACCENT}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --low:{RISK['Low']}; --medium:{RISK['Medium']}; --high:{RISK['High']}; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea, select,
[data-baseweb="select"], [data-baseweb="input"] {{ font-family:{SANS}; }}
.stApp {{ background:var(--paper); color:var(--ink); }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer, [data-testid="stDecoration"], .stDeployButton {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:2.4rem; padding-bottom:4rem; max-width:1320px; }}
h1, h2, h3 {{ font-family:{SERIF}; font-weight:400; letter-spacing:-0.01em; color:var(--ink); }}
p, li {{ line-height:1.55; }}
a {{ color:var(--accent); }}
hr {{ border:none; border-top:1px solid var(--line); margin:1.6rem 0; }}

[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--line); }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] {{ padding-top:1.6rem; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:9px 12px 9px 14px; border-left:2px solid transparent; border-radius:0;
  margin:0; width:100%; cursor:pointer; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size:0.95rem; color:var(--muted); margin:0; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {{ color:var(--ink); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ border-left-color:var(--accent); background:var(--paper); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color:var(--ink); font-weight:600; }}

.ci-brand {{ margin:0 0 1.4rem 0; padding:0 0 1.1rem 0; border-bottom:1px solid var(--line); }}
.ci-brand .name {{ font-family:{SERIF}; font-size:1.75rem; line-height:1.05; }}
.ci-brand .flow {{ font-family:{MONO}; font-size:0.66rem; letter-spacing:0.09em; text-transform:uppercase; color:var(--muted); margin-top:0.5rem; }}
.ci-side-note {{ font-family:{MONO}; font-size:0.7rem; color:var(--muted); line-height:1.6; border-top:1px solid var(--line); margin-top:1.4rem; padding-top:0.9rem; }}

.ci-kicker {{ font-family:{MONO}; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); margin-bottom:0.5rem; }}
.ci-title {{ font-family:{SERIF}; font-size:3.1rem; line-height:1.02; letter-spacing:-0.015em; margin:0 0 0.7rem 0; max-width:900px; }}
.ci-lede {{ color:var(--muted); font-size:1.02rem; max-width:760px; margin:0 0 1.6rem 0; }}
.ci-section {{ font-family:{SERIF}; font-size:1.65rem; line-height:1.15; margin:0 0 0.25rem 0; }}
.ci-section-note {{ color:var(--muted); font-size:0.88rem; margin:0 0 0.9rem 0; max-width:680px; }}
.ci-label {{ font-family:{MONO}; font-size:0.68rem; letter-spacing:0.09em; text-transform:uppercase; color:var(--muted); }}
.ci-mono {{ font-family:{MONO}; }}
.ci-muted {{ color:var(--muted); }}

.ci-panel {{ background:var(--surface); border:1px solid var(--line); padding:1.25rem 1.4rem; }}
.ci-panel.flat {{ background:transparent; }}
.ci-ledger {{ display:grid; border:1px solid var(--line); background:var(--surface); }}
.ci-ledger .cell {{ padding:1.1rem 1.4rem 1.2rem 1.4rem; border-right:1px solid var(--line); }}
.ci-ledger .cell:last-child {{ border-right:none; }}
.ci-big {{ font-family:{SERIF}; line-height:1; letter-spacing:-0.02em; margin:0.45rem 0 0.35rem 0; }}
.ci-sub {{ color:var(--muted); font-size:0.84rem; line-height:1.4; }}
.ci-pill {{ display:inline-block; font-family:{MONO}; font-size:0.7rem; letter-spacing:0.06em; text-transform:uppercase; padding:2px 8px; border:1px solid; }}
.ci-pill.Low {{ color:var(--low); background:{RISK_TINT['Low']}; border-color:var(--low); }}
.ci-pill.Medium {{ color:#8A6414; background:{RISK_TINT['Medium']}; border-color:var(--medium); }}
.ci-pill.High {{ color:var(--high); background:{RISK_TINT['High']}; border-color:var(--high); }}
.ci-bandbar {{ display:flex; height:10px; margin:0.7rem 0 0.4rem 0; border:1px solid var(--line); }}
.ci-bandbar div {{ height:100%; }}
.ci-bandkey {{ display:flex; gap:1.4rem; flex-wrap:wrap; font-family:{MONO}; font-size:0.74rem; color:var(--muted); }}
.ci-bandkey b {{ color:var(--ink); font-weight:500; }}
.ci-dot {{ display:inline-block; width:8px; height:8px; margin-right:6px; }}

.ci-insight {{ padding:0.85rem 0; border-top:1px solid var(--line); font-size:0.93rem; }}
.ci-insight:first-of-type {{ border-top:none; padding-top:0; }}
.ci-insight .f {{ font-family:{MONO}; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); display:block; margin-bottom:2px; }}
.ci-row {{ display:grid; grid-template-columns:1fr auto auto; gap:0.9rem; align-items:center; padding:0.65rem 0; border-top:1px solid var(--line); font-size:0.92rem; }}
.ci-row:first-of-type {{ border-top:none; }}
.ci-row .id {{ font-family:{MONO}; font-size:0.84rem; }}
.ci-row .why {{ color:var(--muted); font-size:0.8rem; display:block; }}
.ci-row .p {{ font-family:{MONO}; font-size:0.9rem; text-align:right; }}
.ci-kv {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:0; border-top:1px solid var(--line); border-left:1px solid var(--line); }}
.ci-kv > div {{ padding:0.65rem 0.9rem; border-right:1px solid var(--line); border-bottom:1px solid var(--line); background:var(--surface); }}
.ci-kv .v {{ font-family:{MONO}; font-size:0.92rem; margin-top:2px; word-break:break-word; }}
.ci-callout {{ border-left:2px solid var(--muted); background:var(--surface); padding:0.8rem 1rem; font-size:0.86rem; color:var(--muted); margin:0.8rem 0; }}
.ci-callout.warn {{ border-left-color:var(--medium); }}
.ci-callout.err {{ border-left-color:var(--high); }}
.ci-action {{ background:var(--surface); border:1px solid var(--line); border-left:2px solid var(--accent); padding:0.85rem 1rem; margin-bottom:0.6rem; }}
.ci-action .t {{ font-weight:500; }}
.ci-action .b {{ font-family:{MONO}; font-size:0.74rem; color:var(--muted); margin-top:0.35rem; line-height:1.5; }}
.ci-cm {{ display:grid; grid-template-columns:auto 1fr 1fr; border:1px solid var(--line); background:var(--surface); font-size:0.85rem; }}
.ci-cm > div {{ padding:0.7rem 0.9rem; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }}
.ci-cm .n {{ font-family:{SERIF}; font-size:1.6rem; line-height:1; }}
.ci-table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); font-size:0.88rem; }}
.ci-table th {{ text-align:left; font-family:{MONO}; font-weight:400; font-size:0.68rem; letter-spacing:0.07em; text-transform:uppercase; color:var(--muted); padding:0.6rem 0.8rem; border-bottom:1px solid var(--line); }}
.ci-table td {{ padding:0.6rem 0.8rem; border-bottom:1px solid var(--line-soft); font-family:{MONO}; font-size:0.84rem; }}
.ci-table td:first-child {{ font-family:{SANS}; font-size:0.9rem; }}
.ci-table tr.sel td {{ background:var(--paper); font-weight:500; }}
.ci-status {{ display:grid; grid-template-columns:auto 1fr auto; gap:1rem; padding:0.7rem 0; border-top:1px solid var(--line); align-items:baseline; font-size:0.9rem; }}
.ci-status:first-of-type {{ border-top:none; }}
.ci-gauge {{ position:relative; height:10px; border:1px solid var(--line); margin:1.4rem 0 0.4rem 0; display:flex; }}
.ci-gauge .mark {{ position:absolute; top:-7px; width:2px; height:22px; background:var(--ink); }}

div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .ci-filterbar) {{ background:var(--surface); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
.stButton > button, .stDownloadButton > button {{ border-radius:0; font-weight:500; border:1px solid var(--ink); }}
.stButton > button[kind="primary"] {{ border-color:var(--accent); }}
[data-testid="stTabs"] [role="tab"] {{ font-family:{MONO}; font-size:0.78rem; letter-spacing:0.05em; text-transform:uppercase; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:0; background:var(--surface); }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ border-radius:0; }}
label p {{ font-size:0.85rem; }}
"""


def inject(st) -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def md(st, html_text: str) -> None:
    st.markdown(html_text, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- HTML builders (single-line strings)
def brand() -> str:
    return ("<div class='ci-brand'><div class='name'>Churn Intelligence</div>"
            "<div class='flow'>Predict · Explain · Recommend · Retain</div></div>")


def header(kicker: str, title: str, lede: str = "") -> str:
    lede_html = f"<div class='ci-lede'>{esc(lede)}</div>" if lede else ""
    return f"<div class='ci-kicker'>{esc(kicker)}</div><div class='ci-title'>{esc(title)}</div>{lede_html}"


def section(title: str, note: str = "") -> str:
    n = f"<div class='ci-section-note'>{esc(note)}</div>" if note else ""
    return f"<div class='ci-section'>{esc(title)}</div>{n}"


def label(text: str) -> str:
    return f"<div class='ci-label'>{esc(text)}</div>"


def pill(level: str) -> str:
    return f"<span class='ci-pill {esc(level)}'>{esc(level)}</span>"


def callout(text: str, tone: str = "") -> str:
    return f"<div class='ci-callout {esc(tone)}'>{esc(text)}</div>"


def big_number(label_text: str, value: str, sub: str = "", size: str = "3.4rem", color: str = INK) -> str:
    return (f"<div class='ci-label'>{esc(label_text)}</div>"
            f"<div class='ci-big' style='font-size:{size};color:{color}'>{esc(value)}</div>"
            f"<div class='ci-sub'>{esc(sub)}</div>")


def ledger(cells: List[str], template: str) -> str:
    """cells: HTML for each cell; template: CSS grid-template-columns, e.g. '5fr 7fr'."""
    inner = "".join(f"<div class='cell'>{c}</div>" for c in cells)
    return f"<div class='ci-ledger' style='grid-template-columns:{template}'>{inner}</div>"


def band_bar(summary: pd.DataFrame) -> str:
    """Stacked bar of customers per risk level. summary: risk_level, customers, share."""
    segs, keys = [], []
    for level in ("Low", "Medium", "High"):
        r = summary[summary["risk_level"] == level]
        share = float(r["share"].iloc[0]) if len(r) else 0.0
        n = int(r["customers"].iloc[0]) if len(r) else 0
        segs.append(f"<div style='width:{share * 100:.2f}%;background:{RISK[level]}'></div>")
        keys.append(f"<span><span class='ci-dot' style='background:{RISK[level]}'></span>{level} <b>{share:.0%}</b> · {n:,}</span>")
    return f"<div class='ci-bandbar'>{''.join(segs)}</div><div class='ci-bandkey'>{''.join(keys)}</div>"


def insight_list(items: Iterable[tuple]) -> str:
    """items: (feature label, sentence)."""
    return "".join(f"<div class='ci-insight'><span class='f'>{esc(f)}</span>{esc(t)}</div>" for f, t in items)


def attention_rows(rows: Iterable[dict]) -> str:
    """rows: customer, probability (0..1), level, why."""
    out = []
    for r in rows:
        out.append(f"<div class='ci-row'><div><span class='id'>{esc(r['customer'])}</span>"
                   f"<span class='why'>{esc(r.get('why', ''))}</span></div>"
                   f"<div class='p'>{r['probability']:.1%}</div><div>{pill(r['level'])}</div></div>")
    return "".join(out)


def kv_grid(pairs: Iterable[tuple]) -> str:
    cells = "".join(f"<div><div class='ci-label'>{esc(k)}</div><div class='v'>{esc(v)}</div></div>" for k, v in pairs)
    return f"<div class='ci-kv'>{cells}</div>"


def action_card(action: str, because: str) -> str:
    return f"<div class='ci-action'><div class='t'>{esc(action)}</div><div class='b'>Because: {esc(because)}</div></div>"


def gauge(prob: float, low: float, high: float) -> str:
    """Probability marker on a 0-100% track coloured by the risk bands."""
    def seg(a, b, level):
        return f"<div style='width:{(b - a) * 100:.2f}%;background:{RISK_TINT[level]}'></div>"
    track = seg(0, low, "Low") + seg(low, high, "Medium") + seg(high, 1, "High")
    pos = min(max(prob, 0.0), 1.0) * 100
    return (f"<div class='ci-gauge'>{track}<div class='mark' style='left:{pos:.2f}%'></div></div>"
            f"<div class='ci-bandkey'><span>0%</span><span>Low below <b>{low:.0%}</b></span>"
            f"<span>High from <b>{high:.0%}</b></span><span>100%</span></div>")


def confusion_matrix(cm: dict, threshold: float) -> str:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return ("<div class='ci-cm'>"
            "<div></div><div class='ci-label'>Predicted: churn</div><div class='ci-label'>Predicted: stay</div>"
            f"<div class='ci-label'>Actually churned</div><div><div class='n'>{tp:,}</div><div class='ci-sub'>caught</div></div>"
            f"<div><div class='n'>{fn:,}</div><div class='ci-sub'>missed</div></div>"
            f"<div class='ci-label'>Actually stayed</div><div><div class='n'>{fp:,}</div><div class='ci-sub'>flagged in error</div></div>"
            f"<div><div class='n'>{tn:,}</div><div class='ci-sub'>correctly left alone</div></div></div>"
            f"<div class='ci-sub' style='margin-top:0.5rem'>Churn is predicted at {threshold:.0%} and above. "
            f"Of customers flagged, {prec:.0%} churned; the flags hold {rec:.0%} of all churners.</div>")


def html_table(df: pd.DataFrame, highlight: Optional[str] = None, fmt: Optional[dict] = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{esc(fmt.get(c, '{}').format(v) if pd.notna(v) else '')}</td>" for c, v in row.items())
        body.append(f"<tr class='{'sel' if idx == highlight else ''}'><td>{esc(idx)}</td>{cells}</tr>")
    return f"<table class='ci-table'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# ----------------------------------------------------------------------------- charts (Vega-Lite specifications)
def _config() -> dict:
    return {
        "background": "transparent", "font": "IBM Plex Sans",
        "view": {"stroke": None},
        "axis": {"labelFont": "IBM Plex Mono", "labelFontSize": 11, "labelColor": MUTED, "titleFont": "IBM Plex Mono",
                 "titleFontSize": 11, "titleFontWeight": "normal", "titleColor": MUTED, "domainColor": LINE,
                 "tickColor": LINE, "gridColor": LINE_SOFT},
        "legend": {"labelFont": "IBM Plex Mono", "titleFont": "IBM Plex Mono", "labelFontSize": 11, "titleFontSize": 11,
                   "labelColor": MUTED, "titleColor": MUTED, "orient": "top", "direction": "horizontal"},
    }


def _spec(body: dict, height: int) -> dict:
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": height,
            "config": _config(), **body}


def spec_group_rates(base: float, height: int = 300) -> dict:
    """Observed churn rate per group. Data columns: group, order, customers, churn_rate, top (bool)."""
    return _spec({"layer": [
        {"mark": {"type": "bar", "size": 34},
         "encoding": {
             "x": {"field": "group", "type": "nominal", "sort": {"field": "order"}, "axis": {"labelAngle": 0, "title": None, "labelLimit": 120}},
             "y": {"field": "churn_rate", "type": "quantitative", "axis": {"format": "%", "title": "Observed churn rate"}},
             "color": {"condition": {"test": "datum.top", "value": RISK["High"]}, "value": INK},
             "tooltip": [{"field": "group", "title": "Group"}, {"field": "customers", "title": "Customers", "format": ","},
                         {"field": "churn_rate", "title": "Churn rate", "format": ".1%"}]}},
        {"data": {"values": [{"churn_rate": base}]}, "mark": {"type": "rule", "strokeDash": [4, 3], "color": MUTED},
         "encoding": {"y": {"field": "churn_rate", "type": "quantitative"}}},
        {"data": {"values": [{"churn_rate": base, "t": f"overall {base:.1%}"}]},
         "mark": {"type": "text", "align": "left", "baseline": "bottom", "dx": 4, "dy": -4, "font": "IBM Plex Mono", "fontSize": 11, "color": MUTED},
         "encoding": {"x": {"value": 0}, "y": {"field": "churn_rate", "type": "quantitative"}, "text": {"field": "t"}}},
    ]}, height)


def spec_risk_histogram(height: int = 220) -> dict:
    """Customers per predicted-probability bin. Data columns: low, high, customers, risk_level."""
    return _spec({"mark": {"type": "bar", "stroke": SURFACE, "strokeWidth": 0.5},
                  "encoding": {
                      "x": {"field": "low", "type": "quantitative", "axis": {"format": "%", "title": "Predicted churn probability"}},
                      "x2": {"field": "high"},
                      "y": {"field": "customers", "type": "quantitative", "axis": {"format": ",", "title": "Customers"}},
                      "color": {"field": "risk_level", "type": "nominal", "title": None,
                                "scale": {"domain": ["Low", "Medium", "High"], "range": [RISK["Low"], RISK["Medium"], RISK["High"]]}},
                      "tooltip": [{"field": "low", "title": "From", "format": ".1%"}, {"field": "high", "title": "To", "format": ".1%"},
                                  {"field": "customers", "title": "Customers", "format": ","}, {"field": "risk_level", "title": "Band"}]}}, height)


def spec_roc(height: int = 320) -> dict:
    """Data columns: model, fpr, tpr, selected (bool)."""
    return _spec({"layer": [
        {"data": {"values": [{"fpr": 0, "tpr": 0}, {"fpr": 1, "tpr": 1}]}, "mark": {"type": "line", "color": LINE, "strokeDash": [4, 4]},
         "encoding": {"x": {"field": "fpr", "type": "quantitative"}, "y": {"field": "tpr", "type": "quantitative"}}},
        {"mark": {"type": "line", "interpolate": "step-after"},
         "encoding": {
             "x": {"field": "fpr", "type": "quantitative", "scale": {"domain": [0, 1]}, "axis": {"title": "False positive rate", "format": "%"}},
             "y": {"field": "tpr", "type": "quantitative", "scale": {"domain": [0, 1]}, "axis": {"title": "True positive rate", "format": "%"}},
             "color": {"field": "model", "type": "nominal", "title": None,
                       "scale": {"range": [INK, MUTED, "#9AA3A0", "#BDB6A5"]}},
             "strokeDash": {"field": "model", "type": "nominal", "legend": None,
                            "scale": {"range": [[1, 0], [6, 3], [2, 2], [8, 2, 2, 2]]}},
             "strokeWidth": {"condition": {"test": "datum.selected", "value": 3}, "value": 1.5},
             "tooltip": [{"field": "model", "title": "Model"}, {"field": "fpr", "title": "FPR", "format": ".1%"}, {"field": "tpr", "title": "TPR", "format": ".1%"}]}},
    ]}, height)


def spec_calibration(height: int = 280) -> dict:
    """Data columns: mean_predicted, observed_rate, customers."""
    return _spec({"layer": [
        {"data": {"values": [{"mean_predicted": 0, "observed_rate": 0}, {"mean_predicted": 1, "observed_rate": 1}]},
         "mark": {"type": "line", "color": LINE, "strokeDash": [4, 4]},
         "encoding": {"x": {"field": "mean_predicted", "type": "quantitative"}, "y": {"field": "observed_rate", "type": "quantitative"}}},
        {"mark": {"type": "line", "point": {"filled": True, "size": 60, "color": INK}, "color": INK},
         "encoding": {
             "x": {"field": "mean_predicted", "type": "quantitative", "axis": {"format": "%", "title": "Average predicted probability"}},
             "y": {"field": "observed_rate", "type": "quantitative", "axis": {"format": "%", "title": "Observed churn rate"}},
             "tooltip": [{"field": "mean_predicted", "title": "Predicted", "format": ".1%"}, {"field": "observed_rate", "title": "Observed", "format": ".1%"},
                         {"field": "customers", "title": "Customers", "format": ","}]}},
    ]}, height)


def spec_bars_horizontal(x_title: str, x_format: str = "%", height: int = 280, signed: bool = False) -> dict:
    """Horizontal bars. Data columns: name, value, order. With signed=True bars above zero use the risk red, below the low green."""
    color = ({"condition": {"test": "datum.value > 0", "value": RISK["High"]}, "value": RISK["Low"]} if signed else {"value": INK})
    return _spec({"mark": {"type": "bar", "size": 16},
                  "encoding": {
                      "y": {"field": "name", "type": "nominal", "sort": {"field": "order"}, "axis": {"title": None, "labelLimit": 200}},
                      "x": {"field": "value", "type": "quantitative", "axis": {"format": x_format, "title": x_title}},
                      "color": color,
                      "tooltip": [{"field": "name", "title": "Input"}, {"field": "value", "title": "Value", "format": ".3f" if x_format != "%" else ".1%"}]}}, height)


def spec_rule_counts(height: int = 220) -> dict:
    """Customers per rule. Data columns: name, value, order."""
    return spec_bars_horizontal("Customers", ",", height)


def vega(st, df: pd.DataFrame, spec: dict) -> None:
    """Render a Vega-Lite spec at full width on any Streamlit version."""
    stretch(st.vega_lite_chart, df, spec, theme=None)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element at full width on any version.
    Newer Streamlit takes width='stretch' (its parameter has a text default); older takes use_container_width=True."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    w = params.get("width")
    if w is not None and isinstance(w.default, str):
        return fn(*args, width="stretch", **kwargs)
    if "use_container_width" in params:
        return fn(*args, use_container_width=True, **kwargs)
    return fn(*args, **kwargs)
