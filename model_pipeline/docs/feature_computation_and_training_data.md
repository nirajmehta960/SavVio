# Feature Computation & Training Data Generation

This document covers two modules that work together to produce the training dataset:

1. **`financial_features.py`** — Computes 6 financial features for a single user-product pair (used at inference time).
2. **`training_data_generator.py`** — Generates 50K+ labeled scenarios by pairing real users with real products across graduated price tiers (used for ML training).

Source files:
- `model_pipeline/src/features/financial_features.py`
- `model_pipeline/src/features/training_data_generator.py`

---

## Part 1: Financial Features (`financial_features.py`)

### Purpose

Computes 6 derived financial features that capture how well a user can absorb a specific purchase.
These features are the inputs to the deterministic decision engine (Layer 1).

This module is used at **inference time** — when a real user queries whether they should buy a
specific product. For batch training data generation, `training_data_generator.py` reimplements
the same math in a vectorized form for performance.

### The 6 computed features

| # | Feature | Formula | What it measures |
|---|---------|---------|-----------------|
| 1 | `affordability_score` | `discretionary_income − product_price` | Can the user's monthly cash flow absorb this purchase? Negative = income alone can't cover it. |
| 2 | `price_to_income_ratio` (PIR) | `product_price / monthly_income` | How big is this purchase relative to income? PIR > 0.10 flags non-trivial purchases in RED/YELLOW rules. |
| 3 | `residual_utility_score` (RUS) | `(liquid_savings − product_price) / total_obligations` | After buying this from savings, how many months of financial runway remain? |
| 4 | `savings_to_price_ratio` (SPR) | `liquid_savings / product_price` | How many times over can savings cover this product? SPR < 1.5 means savings barely cover it. |
| 5 | `net_worth_indicator` (NWI) | `(liquid_savings − loan_amount) / monthly_income` | Is the user underwater? Negative NWI means outstanding debt exceeds savings. |
| 6 | `credit_risk_indicator` (CRI) | `(credit_score − 299) / 550` | Normalized credit score (0–1 range). A brand-new credit holder (score=300) gets CRI ≈ 0.002, not zero. |

### Input features (from database)

These are pre-computed by the data pipeline and stored in PostgreSQL:

| Feature | Source |
|---------|--------|
| `monthly_income` | User's monthly income |
| `discretionary_income` | Income minus all recurring expenses (rent, utilities, subscriptions, etc.) |
| `liquid_savings` | Accessible savings (checking + savings accounts, not retirement/locked funds) |
| `monthly_expenses` | Total monthly expenses |
| `monthly_emi` | Monthly loan EMI payments |
| `loan_amount` | Outstanding loan balance |
| `credit_score` | Credit score (300–850 range) |
| `debt_to_income_ratio` | Total debt payments / income |
| `saving_to_income_ratio` | Savings / income |
| `monthly_expense_burden_ratio` | (Expenses + EMI) / income |
| `emergency_fund_months` | Savings / (expenses + EMI) — months of runway without income |

### Cumulative spending — DI-first priority model

When a user evaluates multiple products in a session, prior purchases reduce their available
resources. The spending follows a **DI-first** priority:

1. **Discretionary income (DI) absorbs the purchase first.** DI represents the user's monthly
   cash flow after recurring expenses. If DI can cover the product price, savings are untouched.

2. **If DI is insufficient, liquid savings cover the shortfall.** Only the gap between the
   product price and available DI is deducted from savings.

**Example — user with DI=$1,200 and savings=$8,000 buying 3 products:**

| Purchase | Price | DI covers | Savings cover | DI after | Savings after |
|----------|------:|----------:|--------------:|---------:|--------------:|
| Budget   | $200  | $200      | $0            | $1,000   | $8,000        |
| Mid      | $800  | $800      | $0            | $200     | $8,000        |
| Premium  | $1,500| $200      | $1,300        | $0       | $6,700        |

DI absorbed the first two purchases entirely. On the third, DI only had $200 left, so savings
covered the remaining $1,300.

**Why not deduct from both equally?** Because that double-counts the spending. A user with
$1,200 DI buying a $200 product still has $1,000 DI and the same savings. Their savings
shouldn't decrease for a purchase their income can handle.

### API

```python
from features.financial_features import compute_affordability

result = compute_affordability(
    user_financial_profile={
        "monthly_income": 5000.0,
        "discretionary_income": 1200.0,
        "liquid_savings": 8000.0,
        "monthly_expenses": 2800.0,
        "monthly_emi": 1000.0,
        "loan_amount": 15000.0,
        "credit_score": 720,
    },
    product_price=799.99,
    cumulative_spend=500.0,  # user already bought $500 worth this session
)
# result.affordability_score → -99.99
# result.savings_to_price_ratio → 10.0
```

---

## Part 2: Training Data Generator (`training_data_generator.py`)

### Purpose

Generates the labeled training dataset by pairing real user financial profiles with real
products, computing features, and applying the deterministic engine to produce
GREEN/YELLOW/RED labels. The output CSV is used to train the downstream ML model.

### Sampling strategies

The generator supports three strategies (controlled by `graduated` and `stratified` parameters):

#### 1. Graduated (default, recommended)

Each user evaluates one product per price tier in ascending order: **budget → mid → premium**.
This simulates a realistic shopping session where a user considers increasingly expensive items.

**Key behaviors:**
- **Cumulative spending** — After each GREEN/YELLOW purchase, the product price is deducted
  from the user's available resources (DI first, then savings) before evaluating the next tier.
- **RED early-stop** — If a user gets RED on any tier, they stop. No further tiers are evaluated
  (the engine already determined this user can't afford more).
- **Session grouping** — Each user's graduated journey is assigned a `session_id` and rows are
  sorted so tiers appear consecutively: budget → mid → premium.

**Price tier bins (current):**

| Tier    | Price range      | Product pool |
|---------|-----------------|-------------|
| budget  | $100 – $500     | 6,700       |
| mid     | $500 – $1,500   | 881         |
| premium | $1,500+         | 454         |

Products under $100 are excluded — they're always GREEN and provide no learning signal.

#### 2. Stratified

Equal representation across income × price bracket combinations (3 income × 3 price = 9 cells).
Each cell gets `n_scenarios / 9` rows. Used when you want balanced coverage without the
session/cumulative-spending semantics.

#### 3. Random

Pure uniform random pairing. Legacy mode for quick experiments.

### Output schema

Each row in `training_scenarios.csv` represents one (user, product) evaluation:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `session_id` | Groups rows belonging to the same user's graduated journey (0, 1, 2, ...) |
| `price_tier` | Which tier this product was sampled from: `budget`, `mid`, or `premium` |
| `product_id` | Product identifier |
| `product_price` | Product price |
| `monthly_income` | User's monthly income (unchanged across tiers) |
| `discretionary_income` | User's DI **at the time of this evaluation** (decreases across tiers as prior purchases consume it) |
| `liquid_savings` | User's savings **at the time of this evaluation** (decreases only when DI is exhausted) |
| `affordability_score` | Computed feature (6 total) |
| `price_to_income_ratio` | Computed feature |
| `residual_utility_score` | Computed feature |
| `savings_to_price_ratio` | Computed feature |
| `net_worth_indicator` | Computed feature |
| `credit_risk_indicator` | Computed feature |
| `financial_label` | Final label: GREEN / YELLOW / RED (after both Layer 1 and Layer 2) |
| `downgraded` | `1` if Layer 2 (downgrade engine) changed the label, `0` otherwise |

### How a graduated session works — step by step

```
User U31070: DI = -$798.53, Savings = $16,354.36

1. Budget tier ($189.99):
   - DI is negative → savings cover $189.99
   - Savings: $16,354 → $16,164
   - Engine evaluates: GREEN
   - User proceeds to next tier.

2. Mid tier ($1,299.95):
   - DI still negative → savings cover $1,299.95
   - Savings: $16,164 → $14,864
   - Engine evaluates: GREEN
   - User proceeds to next tier.

3. Premium tier ($1,889.00):
   - DI still negative → savings cover $1,889
   - Savings: $14,864 → $12,975
   - Engine evaluates: RED (savings_to_price_ratio dropped below threshold)
   - Session stops. User gets 3 rows in the output.
```

### Layer 2 downgrade

After Layer 1 (financial engine) labels all scenarios, Layer 2 (downgrade engine) optionally
downgrades labels based on product/review quality. The `downgraded` column flags which rows
were affected. See `decision_engine.md` for Layer 2 rule details.

### Truncation and ordering

When the generator produces more rows than `n_scenarios`, it drops complete sessions from the
end (never splits a session mid-way). Session IDs are renumbered to be consecutive (0, 1, 2, ...),
and rows are sorted by `(session_id, tier_order)` so each user's journey is contiguous.

### API

```python
from features.training_data_generator import generate_scenarios

scenarios = generate_scenarios(
    financial_profiles=financial_df,   # DataFrame from DB
    products=products_df,              # DataFrame from DB
    reviews_df=reviews_df,             # Optional — enables Layer 2
    n_scenarios=50000,
    random_state=42,
    graduated=True,                    # default
)
scenarios.to_csv("training_scenarios.csv", index=False)
```

### Current label distribution (50K scenarios)

| Label  | Count  | %     |
|--------|-------:|------:|
| GREEN  | 36,391 | 72.8% |
| YELLOW | 5,176  | 10.4% |
| RED    | 8,432  | 16.9% |

224 rows were downgraded by Layer 2.
