## Product & Review Features (Layer 2 Inputs)

This document explains the **product** and **review** feature engineering used by the Layer 2 downgrade engine.

Source files:
- `model_pipeline/src/features/product_features.py`
- `model_pipeline/src/features/review_features.py`

These features are **not** used by the Layer 1 financial engine. They are used by:
- `model_pipeline/src/deterministic_engine/downgrade_engine.py` (Layer 2), which can **only downgrade** the financial label by one step when both product and review risk corroborate.

---

## Why these features exist

Layer 1 answers: **“Can the user afford it?”**

Layer 2 adds a separate question: **“Even if they can afford it, is this product/review pattern risky enough that we should add caution?”**

Downgrading requires:
- ≥ 1 product-rule trigger **and**
- ≥ 1 review-rule trigger

This “two independent sources” requirement reduces false positives from any single weak signal.

---

## Product features (`product_features.py`)

### Inputs required per product row

Batch computation requires these columns:
- `price`
- `average_rating`
- `rating_number`
- `rating_variance`

For category-normalized features, it also expects:
- `category` (if missing, the code falls back to treating each product as its own category)

### Output features (7)

The batch API `compute_product_features_batch(products_df)` returns a DataFrame with these columns added:

1. **`value_density` (VD)**  
   - **Formula**: `average_rating / log(price + 1)`  
   - **Intuition**: a “value per dollar” proxy. A high rating at a very high price is less impressive than the same rating at a low price.

2. **`review_confidence` (RC)**  
   - **Formula**: `log(rating_number + 1) / log(max_rating_number + 1)`  
   - **Range**: ~0 to 1  
   - **Intuition**: how much evidence backs the rating, normalized to the largest review count in the dataset.

3. **`rating_polarization` (RP)**  
   - **Formula**: `rating_variance / (average_rating * (5 - average_rating) + EPS)`  
   - **Intuition**: captures “love it or hate it” rating patterns. Normalizes variance by the maximum possible variance at the given mean rating.

4. **`quality_risk_score` (QRS)**  
   - **Formula**: `(5 - average_rating) * (1 - review_confidence)`  
   - **Intuition**: low ratings are more risky when evidence is weak. This is high when `average_rating` is low and `rating_number` is small.

5. **`cold_start_flag` (CSF)**  
   - **Formula**: `1 if rating_number < 10 else 0`  
   - **Intuition**: products with very few reviews are inherently less reliable.

6. **`price_category_rank` (PCR)**  
   - **Formula**: `(price - category_min_price) / (category_max_price - category_min_price + EPS)`  
   - **Range**: ~0 to 1  
   - **Intuition**: how expensive this product is **within its category**.

7. **`category_rating_deviation` (CRD)**  
   - **Formula**: `average_rating - category_mean_rating`  
   - **Intuition**: how far above/below category average this product is rated.

### Category stats precomputation

`compute_category_stats(products_df)` builds a mapping per `category`:
- `min_price`
- `max_price`
- `mean_rating`

Those stats are used for PCR and CRD.

### APIs

**Single product row**

```python
from features.product_features import compute_category_stats, compute_product_features

category_stats = compute_category_stats(products_df)
max_rating_number = float(products_df["rating_number"].max() or 0.0)
pf = compute_product_features(product_row, category_stats, max_rating_number)
```

**Batch**

```python
from features.product_features import compute_product_features_batch

products_with_features = compute_product_features_batch(products_df)
```

---

## Review features (`review_features.py`)

### Inputs expected per review row

`compute_review_features_batch(reviews_df)` groups by `product_id` and expects:
- `product_id` (required)

It also uses these if present (with safe defaults if missing):
- `verified_purchase` (defaults to False)
- `helpful_vote` (defaults to 0)
- `rating` (defaults to 0)
- `review_text` (defaults to empty string)
- `user_id` (defaults to None)

### Output features (6)

The batch API returns a DataFrame indexed by `product_id` with these columns:

1. **`verified_purchase_ratio` (VPR)**  
   - **Formula**: `(# verified_purchase == True) / n_reviews`  
   - **Intuition**: low VPR can indicate unreliable reviews.

2. **`helpful_concentration` (HC)**  
   - **Formula**: `max(helpful_vote) / (sum(helpful_vote) + EPS)`  
   - **Intuition**: when one review dominates helpfulness votes, the overall review set may be less representative.

3. **`sentiment_spread` (SS)**  
   - **Formula**: `(# ratings >= 4 - # ratings <= 2) / n_reviews`  
   - **Range**: -1 to 1  
   - **Intuition**: direction of sentiment (positive vs negative) without using NLP.

4. **`review_depth_score` (RDS)**  
   - **Formula**: `min(mean_word_count / 100, 1)`  
   - **Range**: 0 to 1  
   - **Intuition**: shallow reviews are less informative; detailed reviews score higher.

5. **`reviewer_diversity` (RD)**  
   - **Formula**: `n_unique_reviewers / n_reviews`  
   - **Range**: 0 to 1  
   - **Intuition**: a product whose reviews come from many unique users is more reliable than one dominated by a few accounts.

6. **`extreme_rating_ratio` (ERR)**  
   - **Formula**: `(# rating == 1 or rating == 5) / n_reviews`  
   - **Intuition**: unusually high extremes can indicate manipulation or polarization.

### Empty-review behavior

If a product has **no reviews**, the code returns **neutral/zeroed** features:
- VPR=0, HC=0, SS=0, RDS=0, RD=0, ERR=0

This ensures downstream code can still run and makes the downgrade engine conservative (it typically needs both product+review triggers).

### APIs

**Single product’s reviews**

```python
from features.review_features import compute_review_features

rf = compute_review_features(reviews_for_product_df)
```

**Batch**

```python
from features.review_features import compute_review_features_batch

review_features_by_product = compute_review_features_batch(reviews_df)
# index: product_id
```

---

## How these features connect to Layer 2 rules

The downgrade engine (`downgrade_engine.py`) consumes:
- Product features: `value_density`, `review_confidence`, `rating_polarization`, `quality_risk_score`, `cold_start_flag`, `price_category_rank`, `category_rating_deviation`
- Review features: `verified_purchase_ratio`, `helpful_concentration`, `sentiment_spread`, `review_depth_score`, `reviewer_diversity`, `extreme_rating_ratio`

And applies rule triggers such as:
- PR1: below-category-average rating AND weak review evidence (CRD + RC)
- RR1: fake-review pattern (VPR + ERR + RDS)

Only when **at least one PR and at least one RR** trigger will the label be downgraded by one step.

