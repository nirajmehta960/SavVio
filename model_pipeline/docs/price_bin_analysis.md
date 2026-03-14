# Price Bin Analysis — Product Price Tiers for Training Data

This document tracks the evolution of product price bins used in `training_data_generator.py`
to create balanced training data for the SavVio ML model. Each configuration was tested against
real user financial profiles and product data to measure label distribution (GREEN / YELLOW / RED).

---

## Why price bins matter

The deterministic engine evaluates whether a user should buy a product. If most products in the
training data are cheap (under $100), almost every scenario gets labeled GREEN because the purchase
is trivially affordable. The model never learns to distinguish YELLOW or RED situations.

Price bins control how products are sampled during graduated session generation. Each user evaluates
one product per tier (budget → mid → premium), with cumulative spending carrying forward.

**Key user financial statistics (from real data):**

| Stat | Monthly Income | Discretionary Income | Liquid Savings |
|------|---------------:|---------------------:|---------------:|
| P10  | $1,454         | -$8,238              | $2,304         |
| P25  | $2,658         | -$2,009              | $5,660         |
| P50  | $3,998         | $721                 | $11,811        |
| P75  | $5,352         | $1,728               | $23,103        |
| P90  | $6,521         | $2,734               | $33,562        |

33.1% of users have **negative** discretionary income — they're already spending more than they earn
monthly and rely entirely on savings for purchases.

**Product price distribution:**

| Stat  | Price    |
|-------|----------|
| P50   | $26.99   |
| P75   | $33.62   |
| P90   | $87.95   |
| P95   | $169.17  |
| P99   | $805.73  |

91.5% of products are under $100. The median price is $26.99. This extreme left-skew means
naive random sampling produces almost entirely GREEN labels.

---

## Configurations tested

### Config 1: Original bins `[0, 50, 200, ∞]` — 3 tiers

| Tier    | Range        | Products |
|---------|-------------|----------|
| budget  | $0 – $50    | 88,412   |
| mid     | $50 – $200  | 4,915    |
| premium | $200+       | 1,000    |

**Result:** GREEN 91.7% / YELLOW 2.0% / RED 6.4%

**Problem:** Budget tier is dominated by cheap products. The $6.99 and $26.99 mass points
mean budget purchases are trivially GREEN for everyone. Model barely sees YELLOW.

---

### Config 2: 4-tier bins `[0, 100, 500, 1500, ∞]`

| Tier    | Range            | Products |
|---------|-----------------|----------|
| budget  | $0 – $100       | 86,292   |
| mid     | $100 – $500     | 6,700    |
| high    | $500 – $1,500   | 881      |
| premium | $1,500+         | 454      |

**Result:** GREEN 79.6% / YELLOW 7.5% / RED 12.9%

**Improvement:** Adding the $500+ tiers forced the engine to evaluate purchases that genuinely
stress users' finances. But the budget tier ($0–$100) still contributed only GREEN labels — it
was wasted signal.

---

### Config 3: `[100, 500, 1500, ∞]` — 3 tiers, no sub-$100 (current)

Eliminated products under $100 entirely. A $6.99 product is always GREEN — no learning signal.

| Tier    | Range            | Products | % of pool |
|---------|-----------------|----------|-----------|
| budget  | $100 – $500     | 6,700    | 83.4%     |
| mid     | $500 – $1,500   | 881      | 11.0%     |
| premium | $1,500+         | 454      | 5.6%     |

**Result:** GREEN 72.8% / YELLOW 10.4% / RED 16.9%

**Why this wins:**
- Best label balance of all tested configurations.
- Every product in the pool actually challenges users' finances.
- Bin boundaries align with engine PIR thresholds:
  - $100–$500 = 10–15% of a low-income user's paycheck → triggers YELLOW rules.
  - $500–$1,500 = 10–20% of median income → triggers YELLOW/RED rules.
  - $1,500+ = significant even for high earners → triggers RED rules.

---

### Other configurations tested

| Config | Bins | GREEN | YELLOW | RED |
|--------|------|------:|-------:|----:|
| `[100, 400, 1000, ∞]` — 3 tiers | $100/$400/$1K+ | 77.3% | 7.7% | 15.0% |
| `[100, 300, 800, 2000, ∞]` — 4 tiers | $100/$300/$800/$2K+ | 75.5% | 10.5% | 14.0% |
| `[0, 15, 50, ∞]` — 3 tiers (early) | $0/$15/$50+ | 15%/68%/17% (bins) | — | — |

Config `[100, 400, 1000+]` was considered but produced fewer YELLOW labels (7.7% vs 10.4%).
Config `[100, 300, 800, 2000+]` with 4 tiers had comparable balance but thinner product pools
in the upper tiers (314 premium products vs 454 in the chosen config).

---

## Final configuration

```python
PRICE_BINS  = [100, 500, 1_500, float("inf")]
PRICE_LABELS = ["budget", "mid", "premium"]

INCOME_BINS  = [0, 3_000, 5_000, float("inf")]
INCOME_LABELS = ["low", "mid", "high"]
```

Label distribution on 50,000 scenarios: **GREEN 72.8% / YELLOW 10.4% / RED 16.9%**

---

## Cross-dataset price analysis

For reference, we also analyzed price distributions across Amazon product categories to
understand which datasets produce the best bin balance:

| Dataset | Best 3-tier bins | Split |
|---------|-----------------|-------|
| Electronics (1.6M items) | `[0, 15, 50, ∞]` | 35% / 37% / 28% |
| Video Games (62K items) | `[0, 15, 40, ∞]` | 31% / 41% / 29% |
| General product_featured (94K items) | `[0, 15, 50, ∞]` | 15% / 68% / 17% |

The general product dataset has a 39% mass point at $26.99, making balanced low-range bins
impossible. Our solution: skip the low range entirely and use bins starting at $100, which
aligns with the financial significance threshold for our user base.
