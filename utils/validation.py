"""Input validation and balancing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_inputs(supply_df: pd.DataFrame, demand_df: pd.DataFrame, cost_df: pd.DataFrame):
    errors = []

    if supply_df.empty or demand_df.empty:
        errors.append("Tabel supply dan demand tidak boleh kosong.")
        return errors

    if supply_df["Nama Sumber"].astype(str).str.strip().eq("").any():
        errors.append("Nama sumber tidak boleh kosong.")
    if demand_df["Nama Tujuan"].astype(str).str.strip().eq("").any():
        errors.append("Nama tujuan tidak boleh kosong.")

    for label, frame, column in [
        ("Supply", supply_df, "Supply"),
        ("Demand", demand_df, "Demand"),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            errors.append(f"{label} harus berupa angka.")
        elif (values < 0).any():
            errors.append(f"{label} harus bernilai >= 0.")

    numeric_costs = cost_df.apply(pd.to_numeric, errors="coerce")
    if numeric_costs.isna().any().any():
        errors.append("Semua biaya harus berupa angka.")
    elif (numeric_costs < 0).any().any():
        errors.append("Semua biaya harus bernilai >= 0.")

    expected_shape = (len(supply_df), len(demand_df))
    if cost_df.shape != expected_shape:
        errors.append("Ukuran matriks biaya belum sesuai dengan jumlah sumber dan tujuan.")

    return errors


def prepare_balanced_data(supply_df: pd.DataFrame, demand_df: pd.DataFrame, cost_df: pd.DataFrame):
    source_names = supply_df["Nama Sumber"].astype(str).str.strip().tolist()
    destination_names = demand_df["Nama Tujuan"].astype(str).str.strip().tolist()
    supply = pd.to_numeric(supply_df["Supply"], errors="coerce").to_numpy(dtype=float)
    demand = pd.to_numeric(demand_df["Demand"], errors="coerce").to_numpy(dtype=float)
    costs = cost_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    total_supply = float(np.sum(supply))
    total_demand = float(np.sum(demand))
    status = "Balanced"
    dummy_info = ""

    if total_supply > total_demand:
        difference = total_supply - total_demand
        destination_names.append("Dummy Tujuan")
        demand = np.append(demand, difference)
        costs = np.column_stack([costs, np.zeros(len(source_names))])
        status = "Unbalanced"
        dummy_info = f"Dummy Tujuan ditambahkan dengan demand {difference:g}."
    elif total_demand > total_supply:
        difference = total_demand - total_supply
        source_names.append("Dummy Sumber")
        supply = np.append(supply, difference)
        costs = np.vstack([costs, np.zeros(len(destination_names))])
        status = "Unbalanced"
        dummy_info = f"Dummy Sumber ditambahkan dengan supply {difference:g}."

    return {
        "cost_matrix": costs,
        "supply": supply,
        "demand": demand,
        "source_names": source_names,
        "destination_names": destination_names,
        "total_supply": total_supply,
        "total_demand": total_demand,
        "status": status,
        "dummy_info": dummy_info,
    }
