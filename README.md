# Pabrai Screener

A daily deep-value screen for US (S&P 500 + S&P 400) and Canadian (TSX
Composite) stocks, built around **Mohnish Pabrai's Dhandho framework**:
"heads I win, tails I don't lose much."

## Run it

```bash
/opt/homebrew/bin/python3.11 -m venv .venv     # once
.venv/bin/pip install -r requirements.txt      # once
.venv/bin/streamlit run app.py
```

Press **Run daily screen** in the sidebar. First run downloads ~1,100
tickers of price history (a few minutes); cached in `data/cache/` all day.

## What it does

| Tab | Pabrai concept | Output |
|---|---|---|
| Deep Value | 7-question checklist, quantified | Ranked list: IV estimate, max-buy price (IV×0.5), margin of safety, downside/moat/ownership sub-scores |
| Unloved & Cannibals | Distressed industries + Uber Cannibals | Names >25% below 52w high; share-count-shrink scan |
| Long Calls | Parity tab (he doesn't trade options) | Same chain scanner; size like a stock position if used |
| Business Cards | Circle of competence is a human call | Per-company infographic: business, funding mix, earnings/FCF/share-count history |
| Sidebar | Portfolio rules | ~10% conviction sizing, sell at ~90% IV, patience |

## Methodology

**Pabrai Score /100** = MOS /30 + Downside /25 + Moat /25 + Ownership /20.

- **Margin of safety /30**: discount of price to intrinsic value.
  IV = normalized FCF/share × (10 + 50×growth), growth capped at 15%,
  multiple capped ~17.5x — cross-checked against Graham's
  EPS × (8.5 + 2g); the *lower* estimate wins (his conservatism).
  **Max buy price = IV × 0.5** — his literal rule.
- **Downside /25**: net cash or low leverage, profitable, low beta,
  P/B asset cover, cash as % of market cap.
- **Moat /25**: gross margin (pricing power), operating margin, ROE, scale.
- **Ownership /20**: insider ownership %, disciplined payout, coverage.

**Flags**: `Unloved` = >25% below 52w high (distressed industries are his
hunting ground — see his coal and Turkey trades). `UBER CANNIBAL` = share
count shrinking >5%/yr (from income-statement share history in the card).

**Not automatable** (surfaced on every card): circle of competence,
management honesty, spawners, willingness to bet big.

## Caveats

- Yahoo Finance data (free): ~15min delayed, occasionally rate-limited;
  TSX names use `.TO` tickers in CAD.
- IV estimates are mechanical — Pabrai stresses computing IV *yourself*
  with confidence. Treat the estimate as a first pass, verify on the card.
- Screening aid, not financial advice.
