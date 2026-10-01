"""Pabrai Screener - daily Dhandho-style ideas for US + Canada.

Run:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import pickle
from datetime import date, datetime
from pathlib import Path

import re

import pandas as pd
import streamlit as st

from pabrai_screener import market_data as md
from pabrai_screener import card, model, options, profile
from pabrai_screener.indicators import compute_technicals
from pabrai_screener.universe import load_universe

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

st.set_page_config(page_title="Pabrai Screener", layout="wide")

METHODOLOGY = """
### How this thinks like Pabrai (sources: The Dhandho Investor, his talks & checklist interviews)

**The screen** - his 7 questions, automated where possible:
| His question | How it's scored |
|---|---|
| >50% discount to intrinsic value? | MOS score /30 - IV from normalized FCF x quality multiple (Graham EPS cross-check), take the *lower* bound |
| Minimal downside? | /25 - net cash or low leverage, real earnings, low beta, P/B cover, cash vs market cap |
| Does it have a moat? | /25 - gross & operating margins, ROE, scale |
| Able/honest managers? | /20 - insider ownership, disciplined capital returns |
| Circle of competence / bet-big conviction | Not automatable - the Business Card is for that call |

**Intrinsic value** is deliberately conservative: `normalized FCF/share x
(10 + 50 x growth)` capped ~17.5x, cross-checked with Graham's
`EPS x (8.5 + 2g)`, and the *lower* wins. Max buy price = IV x 0.5 -
his literal entry rule. Forward IV shown for the 2-3 year test.

**Dhandho flags**: `distressed` = >25% below 52w high (he hunts distressed
businesses in distressed industries - coal in '21, Turkey in '19).
`UBER CANNIBAL` = share count shrinking >5%/yr - buybacks doing the work.

**His portfolio rules**: few names, big positions (~10% each) only when
odds are overwhelming; patient cash otherwise; sell near ~90% of IV;
clone ideas rather than originate.

**Not automatable**: circle of competence, management honesty, spawners -
the Business Card checklist keeps those in front of you.
"""


def screen_cache_path(watchlist: tuple = ()) -> Path:
    suffix = f"_w{abs(hash(tuple(sorted(watchlist))))}" if watchlist else ""
    return CACHE_DIR / f"screen_{date.today().isoformat()}{suffix}.pkl"


@st.cache_data(show_spinner="Loading index constituents...", ttl=3600)
def get_universe(force: bool) -> pd.DataFrame:
    return load_universe(force_refresh=force)


def run_screen(force: bool,
               watchlist: list[str] | None = None) -> tuple[list[model.Pick], dict]:
    watchlist = watchlist or []
    cache_file = screen_cache_path(tuple(watchlist))
    if not force and cache_file.exists():
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
        return payload["picks"], payload["meta"]

    uni = get_universe(force)
    if watchlist:
        extra = [t for t in watchlist if t not in set(uni["ticker"])]
        if extra:
            uni = pd.concat([uni, pd.DataFrame({
                "ticker": extra,
                "market": ["CA" if t.endswith(".TO") else "US" for t in extra],
                "source": "watchlist",
            })], ignore_index=True)
    tickers = uni["ticker"].tolist()
    market_map = dict(zip(uni["ticker"], uni["market"]))

    with st.status(f"Downloading 1y daily prices for {len(tickers)} tickers...",
                   expanded=True) as status:
        history = md.download_history(tickers, force=force)
        for s in watchlist:
            if s not in history:
                df1 = md.download_one(s)
                if df1 is not None:
                    history[s] = df1
        st.write(f"{len(history)} tickers with usable history")
        spy = history.get("SPY")
        spy_close = spy["Close"] if spy is not None else None

        st.write("Computing technicals...")
        techs = {
            s: compute_technicals(df, spy_close)
            for s, df in history.items()
            if s not in model.BENCH_VALUES
        }
        techs = {s: t for s, t in techs.items() if t}

        ranked = sorted(techs.items(),
                        key=lambda kv: model._prefilter_key(kv[1]),
                        reverse=True)
        shortlist = [s for s, t in ranked
                     if model._prefilter_key(t) >= 0][:model.SHORTLIST_N]
        for s in watchlist:  # force-include watchlist names
            if s in techs and s not in shortlist:
                shortlist.append(s)
        st.write(f"{len(shortlist)} passed pre-filter; fetching fundamentals...")

        funds = md.fetch_fundamentals(shortlist)
        picks = model.build_picks(
            {s: techs[s] for s in shortlist}, funds, market_map)
        status.update(label="Screen complete", state="complete")

    meta = {
        "universe_n": len(tickers),
        "with_history": len(history),
        "shortlist": len(shortlist),
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if picks:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump({"picks": picks, "meta": meta}, f)
    return picks, meta


def pick_row(p: model.Pick) -> dict:
    return {
        "Ticker": p.ticker,
        "Mkt": p.market,
        "Name": p.name,
        "Sector": p.sector,
        "Price": p.last,
        "IV est": None if pd.isna(p.iv) else round(p.iv, 2),
        "MaxBuy": None if pd.isna(p.max_buy) else round(p.max_buy, 2),
        "MOS%": None if pd.isna(p.mos_now) else round(p.mos_now * 100),
        "Pabrai": p.pabrai,
        "MOS_s": p.mos_s, "Down": p.down_s, "Moat": p.moat_s,
        "Own": p.own_s,
        "P/E f": None if pd.isna(p.pe_f) else round(p.pe_f, 1),
        "FCFyld%": None if pd.isna(p.fcf_yield) else round(p.fcf_yield * 100, 1),
        "P/B": None if pd.isna(p.pb) else round(p.pb, 1),
        "OffHigh%": round(p.tech.get("pct_off_high", 0) * 100, 1),
        "Unloved": "Y" if p.distressed else "",
        "Signal": p.signal,
    }


# ---------------------------------------------------------------------------
# Sidebar - his rules + controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Pabrai Screener")
    st.caption("Dhandho-style daily ideas: heads I win, tails I don't lose much.")

    portfolio = st.number_input("Portfolio value ($)", min_value=0.0,
                                value=100000.0, step=10000.0)
    mkt = st.radio("Market", ["Both", "US", "CA"], horizontal=True)
    st.divider()
    st.markdown(
        "**His rules**\n"
        f"- Conviction position: **${portfolio*0.10:,.0f}** (~10%)\n"
        "- Buy only below **50% of IV** (max-buy column)\n"
        "- Few names, lots of patience - hold cash if nothing qualifies\n"
        "- Sell near ~90% of intrinsic value\n"
        "- Clone the greats; don't originate"
    )
    st.divider()
    watch_raw = st.text_input(
        "Watchlist tickers (forced into the screen)",
        value="", placeholder="DFN.TO, CCO.TO, ...")
    watchlist = [md.normalize_symbol(t)
                 for t in re.split(r"[,\s]+", watch_raw) if t.strip()]
    if watchlist:
        st.caption(f"{len(watchlist)} extra tickers will be force-included "
                   "even outside the index / below liquidity floor.")
    force = st.checkbox("Ignore today's cache", value=False)
    run = st.button("Run daily screen", type="primary", width="stretch")
    st.divider()
    st.markdown(
        "**His cadence**\n"
        "- Mostly: do nothing, wait for the fat pitch\n"
        "- When a name hits max-buy: run the card, read the filings\n"
        "- Review holdings vs IV quarterly")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cache_file = screen_cache_path(tuple(watchlist))
picks, meta = [], {}
if cache_file.exists() and not run and not force:
    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    picks, meta = payload["picks"], payload["meta"]
elif run:
    picks, meta = run_screen(force, watchlist)

if mkt != "Both":
    picks = [p for p in picks if p.market == mkt]

if not picks:
    st.title("Pabrai Screener")
    if meta:
        st.warning("No picks under the current filters "
                   f"(market={mkt}).")
    else:
        st.info("Press **Run daily screen** in the sidebar. First run "
                "downloads ~1,100 tickers of price history and takes a few "
                "minutes; it's cached for the rest of the day.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe", meta.get("universe_n", "-"))
c2.metric("With price data", meta.get("with_history", "-"))
c3.metric("Screened names", len(picks))
c4.metric("Screen ran", meta.get("ran_at", "-"))

tab_dv, tab_uc, tab_calls, tab_cards, tab_doc = st.tabs(
    ["Deep Value", "Unloved & Cannibals", "Long Calls",
     "Business Cards", "Methodology"]
)

df_all = pd.DataFrame([pick_row(p) for p in picks])


def _lookup_pick(sym: str) -> model.Pick | None:
    p = next((x for x in picks if x.ticker == sym), None)
    if p is not None:
        return p
    df1 = md.download_one(sym)
    spy = md.download_one("SPY")
    if df1 is None:
        return None
    spy_close = spy["Close"] if spy is not None else None
    tech = compute_technicals(df1, spy_close)
    fund = md.fetch_fundamentals([sym]).get(sym, {})
    return model.build_picks({sym: tech}, {sym: fund}, {sym: "?"})[0]


@st.dialog("Business Card", width="large")
def _card_dialog(sym: str) -> None:
    p = _lookup_pick(sym)
    if p is None:
        st.warning("No data for that ticker (use .TO suffix for TSX names).")
        return
    prof = profile.fetch_business_profile(sym)
    if prof is None:
        st.warning("Profile fetch failed (rate limit?) - try again shortly.")
        return
    card.render_business_card(p, prof, portfolio)


def _clickable_table(df: pd.DataFrame, key: str,
                     sym_col: str = "Ticker") -> None:
    """Table where clicking a row opens that stock's business card."""
    if st.session_state.pop(f"_open_{key}", False):
        st.session_state.pop(key, None)  # consumed: reset selection
    ev = st.dataframe(df, width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key=key)
    if ev.selection.rows:
        st.session_state[f"_open_{key}"] = True
        _card_dialog(df.iloc[ev.selection.rows[0]][sym_col])


with tab_dv:
    st.subheader("Deep value - ranked by Dhandho score")
    st.caption("MaxBuy = 50% of estimated intrinsic value, his entry rule. "
               "MOS% = discount to IV today. **Click a row for the business "
               "card.**")
    _clickable_table(df_all.sort_values("Pabrai", ascending=False).head(30),
                     "dv_tbl")
    in_zone = df_all[df_all["Signal"].str.startswith(("DHANDHO", "BUY ZONE"))]
    if not in_zone.empty:
        st.success(f"{len(in_zone)} names currently at/below the 50%-of-IV "
                   "rule: " + ", ".join(in_zone["Ticker"].tolist()))

with tab_uc:
    st.subheader("Unloved (distressed) names")
    st.caption("Pabrai buys distressed businesses in distressed industries - "
               "the hated and ignored, not the merely cheap.")
    dist = df_all[df_all["Unloved"] == "Y"].sort_values(
        "Pabrai", ascending=False)
    _clickable_table(dist.head(25), "uc_tbl")

    st.divider()
    st.subheader("Uber cannibals - share-count shrinkers")
    st.caption("His favorite breed: companies buying back stock aggressively. "
               "Share history needs a per-ticker profile fetch - run on demand.")
    n_scan = st.slider("Scan top N picks for buybacks", 5, 50, 20)
    if st.button("Scan for cannibals"):
        rows = []
        scan_list = sorted(picks, key=lambda p: p.pabrai, reverse=True)[:n_scan]
        prog = st.progress(0)
        for i, p in enumerate(scan_list):
            prof = profile.fetch_business_profile(p.ticker)
            if prof:
                cagr, flag = model.cannibal_check(prof)
                if not pd.isna(cagr):
                    rows.append({"Ticker": p.ticker, "Name": p.name,
                                 "Shares/yr": round(cagr * 100, 1),
                                 "Flag": flag, "Pabrai": p.pabrai,
                                 "MOS%": None if pd.isna(p.mos_now)
                                 else round(p.mos_now * 100)})
            prog.progress((i + 1) / len(scan_list))
        prog.empty()
        if rows:
            cdf = pd.DataFrame(rows).sort_values("Shares/yr")
            _clickable_table(cdf, "cann_tbl")
        else:
            st.warning("No share-count history fetched (rate limit?).")

with tab_calls:
    st.subheader("Long-call chain scan")
    st.caption("Pabrai doesn't trade options - this is for parity with the "
               "screener format. If used at all, he'd size like a stock "
               "position and want the same margin of safety.")
    candidates = sorted(picks, key=lambda p: p.pabrai, reverse=True)[:40]
    choices = {f"{p.ticker} — {p.name} (score {p.pabrai})": p
               for p in candidates}
    sel = st.multiselect("Scan contracts for:", list(choices)[:20],
                         default=list(choices)[:5])
    if st.button("Scan options chains"):
        rows = []
        prog = st.progress(0)
        for i, lbl in enumerate(sel):
            p = choices[lbl]
            exps = md.get_option_expiries(p.ticker)
            rows += options.scan_long_calls(
                p.ticker, p.tech.get("last", 0), exps,
                lambda e, t=p.ticker: md.get_option_chain(t, e))
            prog.progress((i + 1) / len(sel))
        prog.empty()
        if rows:
            cdf = pd.DataFrame(rows).sort_values("rank", ascending=False)
            _clickable_table(
                cdf[["ticker", "contract", "expiry", "dte", "strike", "mid",
                     "delta", "iv", "oi", "vol", "spread_pct", "breakeven",
                     "be_vs_spot", "rank"]],
                "calls_tbl", sym_col="ticker")
        else:
            st.warning("No contracts passed the filters (liquidity/spread).")


with tab_cards:
    st.subheader("Business cards - answer his questions 1, 4, 7 yourself")
    st.caption("Circle of competence, bet-big conviction, honest managers - "
               "these need a human. The card gives you the material.")

    mode = st.radio("Show", ["Single pick", "Top-N report"], horizontal=True)
    if mode == "Single pick":
        sym = st.selectbox(
            "Pick from the list",
            [p.ticker for p in sorted(picks, key=lambda p: p.pabrai,
                                      reverse=True)],
            index=0)
        custom = st.text_input("…or any ticker (US or .TO)", value="")
        custom = md.normalize_symbol(custom)
        sym = custom or sym
        if sym:
            p = _lookup_pick(sym)
            if p is None:
                st.warning("No data for that ticker "
                           "(use .TO suffix for TSX names).")
            else:
                prof = profile.fetch_business_profile(sym)
                if prof is None:
                    st.warning("Profile fetch failed (rate limit?) - "
                               "try again in a minute.")
                else:
                    card.render_business_card(p, prof, portfolio)
                    if p.reasons:
                        with st.expander("Why it scored this way"):
                            st.markdown("- " + "\n- ".join(p.reasons))
    else:
        n = st.slider("Cards for top N Dhandho picks", 1, 20, 5)
        if st.button("Generate cards"):
            top_picks = sorted(picks, key=lambda p: p.pabrai,
                               reverse=True)[:n]
            prog = st.progress(0)
            for i, p in enumerate(top_picks):
                prof = profile.fetch_business_profile(p.ticker)
                if prof:
                    card.render_business_card(p, prof, portfolio)
                else:
                    st.warning(f"{p.ticker}: profile unavailable "
                               "(rate limit - retry later)")
                prog.progress((i + 1) / len(top_picks))
            prog.empty()

with tab_doc:
    st.markdown(METHODOLOGY)
