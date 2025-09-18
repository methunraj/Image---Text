from __future__ import annotations

from scripts.export_records import ensure_records, all_columns


def test_ensure_records_explodes_primary_list():
    payload = {
        "invoice_number": "INV-1",
        "entries": [
            {"description": "Item A", "amount": 10},
            {"description": "Item B", "amount": 20},
        ],
    }

    rows = ensure_records(payload)

    assert len(rows) == 2
    assert rows[0]["invoice_number"] == "INV-1"
    assert rows[0]["entries.description"] == "Item A"
    assert rows[1]["entries.amount"] == 20


def test_ensure_records_flattens_nested_dicts():
    payload = {
        "vendor": {"name": "ACME", "address": {"city": "NYC"}},
        "total": 99.5,
    }

    rows = ensure_records(payload)

    assert rows == [{"vendor.name": "ACME", "vendor.address.city": "NYC", "total": 99.5}]


def test_all_columns_preserves_order():
    rows = [
        {"a": 1, "b": 2},
        {"b": 3, "c": 4},
    ]

    assert all_columns(rows) == ["a", "b", "c"]
