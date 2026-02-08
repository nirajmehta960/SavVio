import os

from preprocess.financial import preprocess_financial
from preprocess.product import preprocess_product
from preprocess.review import preprocess_review


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    validated_dir = os.path.join(base_dir, "data", "validated")
    processed_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(processed_dir, exist_ok=True)

    def resolve_input_path(filename):
        validated_path = os.path.join(validated_dir, filename)
        raw_path = os.path.join(raw_dir, filename)
        if os.path.exists(validated_path):
            return validated_path
        if os.path.exists(raw_path):
            return raw_path
        raise FileNotFoundError(f"Missing input file: {filename}")

    financial_input = resolve_input_path("financial_data.csv")
    product_input = resolve_input_path("product_data.csv")
    review_input = resolve_input_path("review_data.csv")

    financial_output = os.path.join(processed_dir, "financial_processed.csv")
    product_output = os.path.join(processed_dir, "products_processed.csv")
    review_output = os.path.join(processed_dir, "review_processed.csv")

    preprocess_financial(financial_input, financial_output)
    preprocess_product(product_input, product_output)
    preprocess_review(review_input, review_output)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
