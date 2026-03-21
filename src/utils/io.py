import csv
import json
import os
from typing import Any


def prepend_key_to_dict(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def save_to_csv(payload: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", "value"])
        for key, value in payload.items():
            writer.writerow([key, value])


def save_records_to_csv(records: list[dict[str, Any]], output_path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_ready_record: dict[str, Any] = {}
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    csv_ready_record[key] = json.dumps(value, sort_keys=True)
                else:
                    csv_ready_record[key] = value
            writer.writerow(csv_ready_record)


def save_to_json(payload: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
