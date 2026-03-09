import csv
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
