"""Formatting helpers for Streamlit display and exports."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def matrix_to_dataframe(matrix, row_names, column_names) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=row_names, columns=column_names)


def format_number(value) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def shipment_details(allocation, source_names, destination_names, show_dummy=False):
    details = []
    for i, source in enumerate(source_names):
        for j, destination in enumerate(destination_names):
            amount = allocation[i][j]
            if amount <= 0:
                continue
            is_dummy = source.startswith("Dummy") or destination.startswith("Dummy")
            if is_dummy and not show_dummy:
                continue
            details.append(f"{source} -> {destination} = {format_number(amount)} unit")
    return details


def build_excel_report(result_bundle) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_bundle["cost_df"].to_excel(writer, sheet_name="Input Cost Matrix")
        result_bundle["supply_df"].to_excel(writer, sheet_name="Supply", index=False)
        result_bundle["demand_df"].to_excel(writer, sheet_name="Demand", index=False)
        result_bundle["vam_allocation_df"].to_excel(writer, sheet_name="VAM Allocation")
        result_bundle["optimal_allocation_df"].to_excel(
            writer, sheet_name="MODI Optimal Allocation"
        )
        pd.DataFrame(
            [
                ["Total Supply", result_bundle["total_supply"]],
                ["Total Demand", result_bundle["total_demand"]],
                ["Status", result_bundle["status"]],
                ["Dummy", result_bundle["dummy_info"]],
                ["Total Biaya Awal", result_bundle["initial_cost"]],
                ["Total Biaya Optimal", result_bundle["optimal_cost"]],
                ["Penghematan", result_bundle["initial_cost"] - result_bundle["optimal_cost"]],
            ],
            columns=["Item", "Nilai"],
        ).to_excel(writer, sheet_name="Summary", index=False)
    return output.getvalue()
