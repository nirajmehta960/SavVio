import csv
import json
import os
import random


def _flatten_list_value(value, separator=" | "):
    if isinstance(value, list):
        return separator.join(str(item) for item in value)
    if value is None:
        return ""
    return value


def _coerce_price(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_fieldnames(records):
    fieldnames = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _build_price_ranges(records):
    category_ranges = {}
    global_prices = []

    for record in records:
        price = _coerce_price(record.get("price"))
        if price is None:
            continue
        category = record.get("main_category")
        if category not in category_ranges:
            category_ranges[category] = {"min": price, "max": price}
        else:
            category_ranges[category]["min"] = min(category_ranges[category]["min"], price)
            category_ranges[category]["max"] = max(category_ranges[category]["max"], price)
        global_prices.append(price)

    global_range = None
    if global_prices:
        global_range = {"min": min(global_prices), "max": max(global_prices)}

    return category_ranges, global_range


def _fill_missing_prices(records, seed=42):
    rng = random.Random(seed)
    category_ranges, global_range = _build_price_ranges(records)

    missing_before = 0
    missing_after = 0

    for record in records:
        price = _coerce_price(record.get("price"))
        if price is None:
            missing_before += 1
            category = record.get("main_category")
            range_info = category_ranges.get(category, global_range)
            if not range_info:
                missing_after += 1
                record["price"] = 0.0
                continue

            low = range_info["min"]
            high = range_info["max"]
            if low == high:
                record["price"] = round(low, 2)
            else:
                record["price"] = round(rng.uniform(low, high), 2)
        else:
            record["price"] = round(price, 2)

    return missing_before, missing_after


def convert_product_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    for record in records:
        for column in ["features", "description", "categories"]:
            if column in record:
                record[column] = _flatten_list_value(record[column])

    missing_before, missing_after = _fill_missing_prices(records)
    fieldnames = _collect_fieldnames(records)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(
        f"Saved products CSV to {output_path}. "
        f"Missing prices filled: {missing_before - missing_after}."
    )


def convert_review_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    fieldnames = _collect_fieldnames(records)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved reviews CSV to {output_path}.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    raw_dir = os.path.join(base_dir, "data", "raw")

    os.makedirs(raw_dir, exist_ok=True)

    product_json = os.path.join(raw_dir, "product_data.json")
    review_json = os.path.join(raw_dir, "review_data.json")

    product_csv = os.path.join(raw_dir, "product_data.csv")
    review_csv = os.path.join(raw_dir, "review_data.csv")

    convert_product_json(product_json, product_csv)
    convert_review_json(review_json, review_csv)


if __name__ == "__main__":
    main()

