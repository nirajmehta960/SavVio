# SavVio Data Pipeline Implementation Plan

## Overview

Pipeline for **3 datasets** using a **hybrid storage approach**:

| Dataset | Format | Storage Strategy |
|---------|--------|------------------|
| `financial_data.csv` (4MB) | CSV | Traditional columns |
| `product_data.jsonl` (285MB) | JSONL | **JSONB** + vector columns |
| `review_data.jsonl` (929MB) | JSONL | **JSONB** + vector columns |

---

<<<<<<< HEAD
=======
## Feature Analysis

### Financial Data (`financial_data.csv`)
**All Available Columns:**
- `user_id`, `age`, `gender`, `education_level`, `employment_status`, `job_title`
- `monthly_income_usd`, `monthly_expenses_usd`, `savings_usd`
- `has_loan`, `loan_type`, `loan_amount_usd`, `loan_term_months`, `monthly_emi_usd`, `loan_interest_rate_pct`
- `debt_to_income_ratio`, `credit_score`, `savings_to_income_ratio`, `region`, `record_date`

**Selected Features:**
- **Raw columns**: `monthly_income_usd`, `monthly_expenses_usd`, `savings_usd`, `debt_to_income_ratio`, `credit_score`
- **Engineered**: `discretionary_income`, `expense_burden_ratio`, `emergency_fund_months`

### Product Data (`product_data.jsonl`)
**All Available Keys:**
- `main_category`, `title`, `average_rating`, `rating_number`
- `features`, `description`, `price`, `images`, `videos`
- `store`, `categories`, `details`, `parent_asin`

**Selected for Embeddings:**
- `features`, `description`, `details`

### Review Data (`review_data.jsonl`)
**All Available Keys:**
- `rating`, `title`, `text`, `images`, `asin`, `parent_asin`
- `user_id`, `timestamp`, `helpful_vote`, `verified_purchase`

**Selected for Embeddings:**
- `title`, `text`

---

>>>>>>> 96a9ee6 (feat: update implementation plan with AI price imputation and null handling strategy)
## Execution Order

### Phase 1: Data Loading
| Step | Dataset | Action |
|------|---------|--------|
| 1.1 | `financial_data.csv` | Parse CSV with pandas |
| 1.2 | `product_data.jsonl` | Stream JSONL |
| 1.3 | `review_data.jsonl` | Stream JSONL |

### Phase 2: Preprocessing
| Dataset | Actions |
|---------|---------|
| **Financial** | Handle missing values, type conversion, validate ranges |
| **Product** | Validate JSON, check `parent_asin`, handle nulls for embeddings |
| **Review** | Validate JSON, check `parent_asin`, handle nulls for embeddings |

**Missing Value Handling (Product/Review):**
| Field | Issue | Action |
|-------|-------|--------|
| `price` | 43% nulls | **AI-based imputation** (LLM analyzes dataset price patterns) |
| `bought_together` | 100% nulls | **Drop field** (no data available) |
| `description` | Empty `[]` | → empty string for embedding |
| `features` | Missing | Default `[]` → empty string |
| `details` | Missing keys | Extract available keys only |
| `title`/`text` | Empty | → empty string for embedding |

### Phase 3: Schema & Validation
| Dataset | Validation Rules |
|---------|-----------------|
| Financial | `monthly_income_usd >= 0`, `credit_score` in [300, 850] |
| Product | Valid JSON, `parent_asin` exists, `average_rating` in [1.0, 5.0], `price > 0` |
| Review | Valid JSON, `parent_asin` exists, `rating` in [1.0, 5.0] |

### Phase 4: Feature Engineering

**Financial Features:**
| Feature | Formula |
|---------|---------|
| `discretionary_income` | `monthly_income_usd - monthly_expenses_usd - monthly_emi_usd` |
| `expense_burden_ratio` | `monthly_expenses_usd / monthly_income_usd` |
| `emergency_fund_months` | `savings_usd / monthly_expenses_usd` |

**Text Extraction for Embeddings:**
| Dataset | Extract From JSONB |
|---------|-------------------|
| Product | `data->'features'`, `data->'description'`, `data->'details'` |
| Review | `data->>'title'`, `data->>'text'` |

### Phase 5: Vector Embeddings
| Column | Dataset | Dimensions |
|--------|---------|------------|
| `description_embedding` | Product | 384 |
| `features_embedding` | Product | 384 |
| `details_embedding` | Product | 384 |
| `title_embedding` | Review | 384 |
| `text_embedding` | Review | 384 |

---

## PostgreSQL Schema (Hybrid)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Financial: Traditional Columns
CREATE TABLE financial_data (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(20) UNIQUE NOT NULL,
    age INTEGER, gender VARCHAR(20), education_level VARCHAR(50),
    employment_status VARCHAR(30), job_title VARCHAR(100),
    monthly_income_usd DECIMAL(12,2), monthly_expenses_usd DECIMAL(12,2),
    savings_usd DECIMAL(15,2), has_loan BOOLEAN, loan_type VARCHAR(50),
    loan_amount_usd DECIMAL(15,2), loan_term_months INTEGER,
    monthly_emi_usd DECIMAL(12,2), loan_interest_rate_pct DECIMAL(5,2),
    debt_to_income_ratio DECIMAL(5,2), credit_score INTEGER,
    savings_to_income_ratio DECIMAL(8,2), region VARCHAR(50), record_date DATE,
    -- Engineered features
    discretionary_income DECIMAL(12,2),
    expense_burden_ratio DECIMAL(5,4),
    emergency_fund_months DECIMAL(8,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Product: JSONB + Vectors
CREATE TABLE product_data (
    id SERIAL PRIMARY KEY,
    parent_asin VARCHAR(20) UNIQUE NOT NULL,
    data JSONB NOT NULL,
    description_embedding vector(384),
    features_embedding vector(384),
    details_embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Review: JSONB + Vectors
CREATE TABLE review_data (
    id SERIAL PRIMARY KEY,
    parent_asin VARCHAR(20) NOT NULL,
    data JSONB NOT NULL,
    title_embedding vector(384),
    text_embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (parent_asin) REFERENCES product_data(parent_asin)
);

-- Indexes
CREATE INDEX idx_product_gin ON product_data USING GIN (data);
CREATE INDEX idx_review_gin ON review_data USING GIN (data);
CREATE INDEX idx_product_desc_vec ON product_data 
    USING ivfflat (description_embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX idx_review_text_vec ON review_data 
    USING ivfflat (text_embedding vector_cosine_ops) WITH (lists=100);
```

---

## JSONB Query Examples

```sql
-- Query product by category
SELECT data->>'title', data->>'price' 
FROM product_data WHERE data->>'main_category' = 'Electronics';

-- Vector similarity search
SELECT data->>'title', description_embedding <=> $query_vector AS distance
FROM product_data ORDER BY distance LIMIT 10;
```

---

## Why Hybrid Approach?

| Benefit | Description |
|---------|-------------|
| **Faster MVP** | No flattening/preprocessing for JSONL |
| **Flexible schema** | JSONB handles varying structures |
| **Fast queries** | GIN indexes on JSONB, IVFFlat on vectors |

---

