"""Vogel Approximation Method implementation."""

from __future__ import annotations

import numpy as np


def _penalty(values: np.ndarray) -> float:
    if values.size == 0:
        return -np.inf
    ordered = np.sort(values.astype(float))
    if ordered.size == 1:
        return float(ordered[0])
    return float(ordered[1] - ordered[0])


def _penalty_detail(values: np.ndarray) -> str:
    if values.size == 0:
        return ""
    ordered = np.sort(values.astype(float))
    if ordered.size == 1:
        return f"({ordered[0]:g})"
    return f"({ordered[1]:g}-{ordered[0]:g})"


def vogel_approximation_method(
    cost_matrix,
    supply,
    demand,
    source_names,
    destination_names,
):
    """Build an initial transportation allocation using VAM."""
    costs = np.array(cost_matrix, dtype=float)
    remaining_supply = np.array(supply, dtype=float).copy()
    remaining_demand = np.array(demand, dtype=float).copy()
    m, n = costs.shape
    allocation = np.zeros((m, n), dtype=float)
    active_rows = set(range(m))
    active_cols = set(range(n))
    steps = []
    iteration = 1

    while active_rows and active_cols:
        row_penalties = {}
        col_penalties = {}
        row_penalty_details = {}
        col_penalty_details = {}
        supply_before = remaining_supply.copy()
        demand_before = remaining_demand.copy()

        for i in active_rows:
            row_costs = costs[i, list(active_cols)]
            row_penalties[i] = _penalty(row_costs)
            row_penalty_details[i] = _penalty_detail(row_costs)

        for j in active_cols:
            col_costs = costs[list(active_rows), j]
            col_penalties[j] = _penalty(col_costs)
            col_penalty_details[j] = _penalty_detail(col_costs)

        candidates = []
        for i, penalty in row_penalties.items():
            min_cost = float(np.min(costs[i, list(active_cols)]))
            candidates.append(("row", i, penalty, min_cost))
        for j, penalty in col_penalties.items():
            min_cost = float(np.min(costs[list(active_rows), j]))
            candidates.append(("col", j, penalty, min_cost))

        selected_type, selected_index, _, _ = max(
            candidates,
            key=lambda item: (item[2], -item[3]),
        )

        if selected_type == "row":
            i = selected_index
            min_cost = min(costs[i, j] for j in active_cols)
            possible_cols = [j for j in active_cols if costs[i, j] == min_cost]
            j = max(possible_cols, key=lambda col: remaining_demand[col])
        else:
            j = selected_index
            min_cost = min(costs[i, j] for i in active_rows)
            possible_rows = [i for i in active_rows if costs[i, j] == min_cost]
            i = max(possible_rows, key=lambda row: remaining_supply[row])

        amount = min(remaining_supply[i], remaining_demand[j])
        allocation[i, j] += amount
        remaining_supply[i] -= amount
        remaining_demand[j] -= amount

        steps.append(
            {
                "iteration": iteration,
                "row_penalties": {
                    source_names[row]: row_penalties[row] for row in row_penalties
                },
                "row_penalty_details": {
                    source_names[row]: row_penalty_details[row]
                    for row in row_penalty_details
                },
                "column_penalties": {
                    destination_names[col]: col_penalties[col] for col in col_penalties
                },
                "column_penalty_details": {
                    destination_names[col]: col_penalty_details[col]
                    for col in col_penalty_details
                },
                "selected": f"{'Baris' if selected_type == 'row' else 'Kolom'} "
                f"{source_names[selected_index] if selected_type == 'row' else destination_names[selected_index]}",
                "selected_cell": (source_names[i], destination_names[j]),
                "selected_cell_indices": (i, j),
                "selected_axis": selected_type,
                "selected_axis_index": selected_index,
                "allocated": float(amount),
                "supply_before": {
                    source_names[row]: float(supply_before[row]) for row in range(m)
                },
                "demand_before": {
                    destination_names[col]: float(demand_before[col]) for col in range(n)
                },
                "remaining_supply": {
                    source_names[row]: float(remaining_supply[row]) for row in range(m)
                },
                "remaining_demand": {
                    destination_names[col]: float(remaining_demand[col]) for col in range(n)
                },
                "allocation_after": allocation.copy(),
            }
        )

        if np.isclose(remaining_supply[i], 0):
            active_rows.discard(i)
        if np.isclose(remaining_demand[j], 0):
            active_cols.discard(j)

        iteration += 1

    total_cost = float(np.sum(allocation * costs))
    return {"allocation": allocation, "total_cost": total_cost, "steps": steps}
