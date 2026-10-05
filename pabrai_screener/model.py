"""The Pabrai model: Dhandho-style scoring ("heads I win, tails I don't
lose much").

Sources (The Dhandho Investor, Pabrai talks/checklists):

* The 7 questions: understandable business (circle of competence) -
  knowable intrinsic value - >50% discount to IV today AND in 2-3 years -
  willing to bet big - minimal downside - moat - able/honest managers.
* Dhandho patterns: distressed businesses in distressed industries,
  simple businesses in slow-changing industries, concentrated bets when
  odds are overwhelming, patience ("wait for the fat pitch").
* "Uber cannibals": companies aggressively shrinking share count.
* Cloning: borrow ideas from great investors; steal, don't originate.
* Sell discipline: exit near ~90% of intrinsic value.

What is quantifiable is scored automatically (margin of safety, downside,
moat, management/ownership). What is not (circle of competence, honest
management) is surfaced in the business card for the human call.

Pabrai Score /100 = MOS 30 + Downside 25 + Moat 25 + Ownership 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BENCH_VALUES = {"SPY", "^GSPTSE"}

MIN_DOLLAR_VOL = 5e6      # Pabrai will go smaller/more obscure than most
MIN_PRICE = 3.0
SHORTLIST_N = 140


def _num(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Intrinsic value & margin of safety
# ---------------------------------------------------------------------------

def estimate_intrinsic_value(tech: dict, fund: dict) -> dict:
    """Conservative IV estimate. Three independent estimators, median wins -
    Yahoo's single fields are noisy, so no one broken input can poison it:
      1. FCF/share x multiple (10x at zero growth -> ~17.5x at 15%)
      2. Owner earnings: EBITDA x 0.55 (tax + maintenance-capex haircut) /sh x mult
      3. Graham: normalized EPS x (8.5 + 2g)
    """
    px = _num(tech.get("last"))
    fcf = _num(fund.get("freeCashflow"))
    ebitda = _num(fund.get("ebitda"))
    shares = _num(fund.get("sharesOutstanding"))
    eps = _num(fund.get("trailingEps"))
    rg = _num(fund.get("revenueGrowth"))
    g = float(np.clip(rg if not np.isnan(rg) else 0.0, 0.0, 0.15))
    mult = min(10.0 + 50.0 * g, 17.5)

    iv_fcf = np.nan
    if fcf > 0 and shares > 0:
        iv_fcf = fcf / shares * mult

    iv_ebitda = np.nan
    if ebitda > 0 and shares > 0:
        iv_ebitda = ebitda * 0.55 / shares * mult

    iv_eps = np.nan
    if eps > 0:
        iv_eps = eps * min(8.5 + 2 * g * 100, 38.5)

    usable = [x for x in (iv_fcf, iv_ebitda, iv_eps) if not np.isnan(x)]
    iv = float(np.median(usable)) if usable else np.nan
    spread = (max(usable) / min(usable)) if len(usable) > 1 else 1.0
    iv_fwd = iv * (1 + g) ** 2.5 if not np.isnan(iv) else np.nan

    return {
        "iv": iv,
        "iv_fcf": iv_fcf,
        "iv_ebitda": iv_ebitda,
        "iv_eps": iv_eps,
        "n_estimators": len(usable),
        "iv_fwd_2_5y": iv_fwd,
        "iv_spread": spread,
        "g_used": g,
        "max_buy": iv * 0.5 if not np.isnan(iv) else np.nan,   # his 50% rule
        "mos_now": (1 - px / iv) if not np.isnan(iv) and px else np.nan,
        "mos_fwd": (1 - px / iv_fwd) if not np.isnan(iv_fwd) and px else np.nan,
    }


# ---------------------------------------------------------------------------
# Checklist scores
# ---------------------------------------------------------------------------

def score_mos(mos: float, mos_fwd: float) -> tuple[float, list[str]]:
    """/30 - is it >50% below IV? The heart of Dhandho."""
    pts, why = 0.0, []
    if np.isnan(mos):
        return 0.0, ["IV not estimable - no positive FCF/EPS"]
    if mos >= 0.60:
        pts = 30; why.append(f"{mos:.0%} below IV - screaming buy by his rule")
    elif mos >= 0.50:
        pts = 27; why.append(f"{mos:.0%} below IV - meets the 50% rule")
    elif mos >= 0.35:
        pts = 18; why.append(f"{mos:.0%} below IV - approaching")
    elif mos >= 0.15:
        pts = 8; why.append(f"{mos:.0%} below IV - some discount")
    else:
        why.append("trading at/above IV - no margin of safety")
    if not np.isnan(mos_fwd) and mos_fwd >= 0.5:
        pts = min(30, pts + 3)
        why.append("also >50% below IV 2-3 years out")
    return pts, why


def score_downside(tech: dict, fund: dict) -> tuple[float, list[str]]:
    """/25 - 'tails I don't lose much': net cash/low leverage, real earnings,
    low beta, asset cover, price already washed out."""
    pts, why = 0.0, []
    mc = _num(fund.get("marketCap"))
    debt = _num(fund.get("totalDebt"))
    cash = _num(fund.get("totalCash"))
    de = _num(fund.get("debtToEquity"))
    beta = _num(fund.get("beta"))
    if np.isnan(beta):
        beta = _num(tech.get("beta"))
    pb = _num(fund.get("priceToBook"))
    pm = _num(fund.get("profitMargins"))
    sector = (fund.get("sector") or "").lower()

    if not np.isnan(debt) and not np.isnan(cash) and cash > debt:
        pts += 9; why.append("net cash - balance sheet floor")
    elif "financial" in sector or np.isnan(de):
        pts += 4
    elif de < 30:
        pts += 8; why.append(f"near-zero leverage (D/E {de:.0f})")
    elif de < 80:
        pts += 5
    elif de < 150:
        pts += 2; why.append(f"leverage D/E {de:.0f} - watch downside")
    else:
        why.append(f"heavy leverage D/E {de:.0f} - fails downside test")

    if pm > 0.10:
        pts += 5; why.append(f"real earnings ({pm:.0%} margin)")
    elif pm > 0:
        pts += 3
    else:
        why.append("no current earnings - downside unproven")

    if not np.isnan(beta):
        if beta < 0.8:
            pts += 4; why.append(f"low beta {beta:.2f}")
        elif beta < 1.1:
            pts += 3
        elif beta < 1.5:
            pts += 1

    if not np.isnan(pb):
        if pb < 1.0:
            pts += 4; why.append(f"P/B {pb:.1f} - below book")
        elif pb < 2.0:
            pts += 2
    if not np.isnan(mc) and not np.isnan(cash) and mc > 0 and cash / mc > 0.2:
        pts += 3; why.append(f"cash = {cash/mc:.0%} of market cap")

    return min(25.0, pts), why


def score_moat(fund: dict) -> tuple[float, list[str]]:
    """/25 - durable advantage proxies: fat margins, high ROE, scale,
    still-growing demand."""
    pts, why = 0.0, []
    gm = _num(fund.get("grossMargins"))
    om = _num(fund.get("operatingMargins"))
    roe = _num(fund.get("returnOnEquity"))
    mc = _num(fund.get("marketCap"))
    rg = _num(fund.get("revenueGrowth"))

    if gm > 0.50:
        pts += 8; why.append(f"gross margin {gm:.0%} - pricing power")
    elif gm > 0.35:
        pts += 5; why.append(f"gross margin {gm:.0%}")
    elif gm > 0.20:
        pts += 2

    if om > 0.20:
        pts += 6; why.append(f"operating margin {om:.0%}")
    elif om > 0.10:
        pts += 4
    elif om > 0.05:
        pts += 2

    if roe > 0.25:
        pts += 6; why.append(f"ROE {roe:.0%} - compounding machine")
    elif roe > 0.15:
        pts += 4; why.append(f"ROE {roe:.0%}")
    elif roe > 0.08:
        pts += 2

    if mc > 50e9:
        pts += 3; why.append("scale advantage")
    elif mc > 10e9:
        pts += 2
    if not np.isnan(rg) and rg > 0:
        pts += 2

    return min(25.0, pts), why


def score_ownership(fund: dict, tech: dict) -> tuple[float, list[str]]:
    """/20 - able & honest managers, proxied by insider skin-in-the-game,
    shareholder-friendly capital returns and analyst coverage sanity."""
    pts, why = 0.0, []
    ins = _num(fund.get("heldByInsiders"))
    dy = _num(fund.get("dividendYield")) or 0.0
    payout = _num(fund.get("payoutRatio"))
    rec = (fund.get("recommendationKey") or "").lower()

    if not np.isnan(ins):
        if ins > 0.20:
            pts += 10; why.append(f"insiders own {ins:.0%} - owner-operators")
        elif ins > 0.08:
            pts += 7; why.append(f"insider ownership {ins:.0%}")
        elif ins > 0.02:
            pts += 4
        else:
            pts += 2; why.append("insider ownership <2% - hired hands")

    if dy > 0 and not np.isnan(payout) and 0 < payout < 0.7:
        pts += 4; why.append("disciplined dividend (sustainable payout)")
    elif dy == 0:
        pts += 2  # reinvesting or buybacks - neutral

    if rec in ("strong_buy", "buy"):
        pts += 3
    elif rec in ("hold",):
        pts += 1

    pts += 3  # base credit; the card surfaces governance flags for the human
    return min(20.0, pts), why


# ---------------------------------------------------------------------------
# Pick assembly
# ---------------------------------------------------------------------------

@dataclass
class Pick:
    ticker: str
    market: str
    name: str
    sector: str
    industry: str
    last: float
    pabrai: float          # checklist total /100
    mos_s: float
    down_s: float
    moat_s: float
    own_s: float
    reasons: list[str] = field(default_factory=list)
    iv: float = np.nan
    iv_fwd: float = np.nan
    n_est: int = 0
    mos_now: float = np.nan
    max_buy: float = np.nan
    pe_f: float = np.nan
    fcf_yield: float = np.nan
    pb: float = np.nan
    signal: str = ""
    distressed: bool = False
    tech: dict = field(default_factory=dict)
    fund: dict = field(default_factory=dict)


def _signal(mos: float, down: float) -> str:
    if np.isnan(mos):
        return "N/A - IV not estimable"
    if mos >= 0.60 and down >= 15:
        return "DHANDHO - deep discount, minimal downside"
    if mos >= 0.50:
        return "BUY ZONE - >50% below IV"
    if mos >= 0.35:
        return "APPROACHING - watchlist"
    if mos >= 0.15:
        return "WAIT - thin margin of safety"
    return "PASS - priced near/above IV"


def _prefilter_key(tech: dict) -> float:
    """Liquidity + 'unloved' ranking before fundamentals are fetched.
    Pabrai hunts distress, so deeper drawdowns rank higher."""
    if (tech.get("avg_dollar_vol") or 0) < MIN_DOLLAR_VOL:
        return -1
    if (tech.get("last") or 0) < MIN_PRICE:
        return -1
    off = tech.get("pct_off_high") or 0
    return -off  # bigger discount from 52w high ranks higher


def build_picks(
    techs: dict[str, dict],
    funds: dict[str, dict],
    market_map: dict[str, str],
) -> list[Pick]:
    picks: list[Pick] = []
    for sym, tech in techs.items():
        if sym in BENCH_VALUES:
            continue
        fund = funds.get(sym) or {}
        ivd = estimate_intrinsic_value(tech, fund)
        mos_s, w1 = score_mos(ivd["mos_now"], ivd["mos_fwd"])
        if ivd["iv_spread"] > 4:
            w1 = w1 + ["estimators disagree >4x - IV low confidence"]
        if ivd["n_estimators"] == 1:
            w1 = w1 + ["only 1/3 estimators usable (neg FCF/EPS) "
                       "- IV very low confidence"]
        dn_s, w2 = score_downside(tech, fund)
        mo_s, w3 = score_moat(fund)
        ow_s, w4 = score_ownership(fund, tech)

        mc = _num(fund.get("marketCap"))
        fcf = _num(fund.get("freeCashflow"))
        last = _num(tech.get("last"))
        distressed = (_num(tech.get("pct_off_high")) <= -0.25)

        picks.append(Pick(
            ticker=sym,
            market=market_map.get(sym, "?"),
            name=fund.get("shortName") or sym,
            sector=fund.get("sector") or "Unknown",
            industry=fund.get("industry") or "Unknown",
            last=round(last, 2),
            pabrai=round(mos_s + dn_s + mo_s + ow_s, 1),
            mos_s=round(mos_s, 1), down_s=round(dn_s, 1),
            moat_s=round(mo_s, 1), own_s=round(ow_s, 1),
            reasons=w1 + w2 + w3 + w4,
            iv=ivd["iv"], iv_fwd=ivd["iv_fwd_2_5y"],
            n_est=ivd["n_estimators"],
            mos_now=ivd["mos_now"], max_buy=ivd["max_buy"],
            pe_f=_num(fund.get("forwardPE")),
            fcf_yield=(fcf / mc if mc > 0 and not np.isnan(fcf) else np.nan),
            pb=_num(fund.get("priceToBook")),
            signal=_signal(ivd["mos_now"], dn_s),
            distressed=distressed,
            tech=tech, fund=fund,
        ))
    return picks


def explain_pick(p: "Pick") -> list[str]:
    """Five plain-English bullets on why the signal fired."""
    b: list[str] = []
    if not np.isnan(p.iv):
        iv_fmt = f"{p.iv:.2f}" if p.iv < 10 else f"{p.iv:.0f}"
        b.append(
            f"Trades at ${p.last:.2f} vs an estimated worth of ~${iv_fmt} "
            f"({p.mos_now:.0%} discount). His rule: only buy below "
            f"${p.max_buy:.2f} (50% of that estimate) - right now the "
            "answer is "
            + ("yes." if p.last <= p.max_buy else "no."))
        if p.n_est == 1:
            b.append("Heads up: only 1 of 3 value estimators produced a "
                     "number - negative earnings or free cash flow knocked "
                     "the others out, so treat that IV as very rough.")
    else:
        b.append("Intrinsic value can't be estimated from the data "
                 "(no reliable earnings or cash flow), so there is no real "
                 "margin-of-safety read.")

    debt = _num(p.fund.get("totalDebt"))
    cash = _num(p.fund.get("totalCash"))
    if not np.isnan(debt) and not np.isnan(cash):
        if cash > debt:
            b.append("Safety: more cash in the bank than total debt - "
                     "'tails, you don't lose much.'")
        elif debt < cash * 3:
            b.append("Safety: debt outweighs cash but is manageable - "
                     "the balance sheet can survive a bad year.")
        else:
            b.append("Safety: heavy debt vs cash - if the business stumbles, "
                     "the equity can get hurt badly.")

    gm = _num(p.fund.get("grossMargins"))
    roe = _num(p.fund.get("returnOnEquity"))
    if not np.isnan(gm):
        tail = " - real pricing power." if gm > 0.4 else \
               " - thin economics, weak moat."
        b.append(f"Moat: {gm:.0%} gross margins"
                 + (f" and {roe:.0%} return on equity" if not np.isnan(roe)
                    else "") + tail)

    ins = _num(p.fund.get("heldByInsiders"))
    if not np.isnan(ins):
        b.append(f"Skin in the game: insiders own ~{ins:.0%} of the company."
                 + (" High - they think like owners."
                    if ins > 0.08 else " Low - mostly hired hands."))

    verdict = {
        "DHANDHO": "Verdict: deep discount plus a survivable balance sheet - "
                   "the closest thing to 'heads I win, tails I don't lose "
                   "much.' Read the filings before betting big.",
        "BUY ZONE": "Verdict: below his 50%-of-IV buy line - a legitimate "
                    "candidate; run the card checklist before sizing.",
        "APPROACHING": "Verdict: interesting but not cheap enough yet - "
                       "set an alert near the max-buy price.",
        "WAIT": "Verdict: decent business, wrong price - patience; "
                "he would rather sit in cash.",
        "PASS": "Verdict: priced at or above estimate - he'd pass "
                "without a second thought.",
        "N/A": "Verdict: not scoreable this way - only consider it as a "
               "special situation, if at all.",
    }
    for k, v in verdict.items():
        if p.signal.startswith(k):
            b.append(v)
            break
    return b[:5]


def cannibal_check(prof: dict) -> tuple[float, str]:
    """Share-count CAGR from income-statement history (needs profile fetch).
    Pabrai's 'Uber cannibals' shrink shares fast; >2%/yr earns the flag."""
    sh = (prof or {}).get("shares_yr") or {}
    yrs = sorted(sh)
    if len(yrs) < 2 or sh[yrs[0]] <= 0:
        return np.nan, ""
    cagr = (sh[yrs[-1]] / sh[yrs[0]]) ** (1 / (yrs[-1] - yrs[0])) - 1
    if cagr <= -0.05:
        return cagr, "UBER CANNIBAL - share count shrinking fast"
    if cagr <= -0.02:
        return cagr, "cannibal - steady buybacks"
    if cagr >= 0.03:
        return cagr, "diluting - issuing shares"
    return cagr, "flat share count"


def position_sizing(portfolio_value: float, price: float) -> dict:
    """Pabrai concentrates: ~10% positions on highest conviction, and is
    happy holding cash when nothing qualifies."""
    return {
        "conviction_position_$": round(portfolio_value * 0.10, 2),
        "conviction_shares": int(portfolio_value * 0.10 // price) if price else 0,
        "note": "He holds few names and lots of patience; only size up "
                "when the odds are overwhelming.",
    }
