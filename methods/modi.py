"""Modified Distribution Method implementation."""

from __future__ import annotations

from collections import deque

import numpy as np


EPSILON = 1e-9


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        self.parent[root_right] = root_left
        return True


def _total_cost(costs: np.ndarray, allocation: np.ndarray) -> float:
    return float(np.sum(costs * allocation))


def _basis_from_allocation(allocation: np.ndarray) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i in range(allocation.shape[0])
        for j in range(allocation.shape[1])
        if allocation[i, j] > EPSILON
    }


def _complete_basis(basis: set[tuple[int, int]], m: int, n: int) -> set[tuple[int, int]]:
    """Add zero-allocation basis cells until the basis has m+n-1 safe cells."""
    completed = set(basis)
    dsu = _DisjointSet(m + n)
    for i, j in completed:
        dsu.union(i, m + j)

    for i in range(m):
        for j in range(n):
            if len(completed) >= m + n - 1:
                return completed
            if (i, j) in completed:
                continue
            if dsu.union(i, m + j):
                completed.add((i, j))

    return completed


def _potentials(costs: np.ndarray, basis: set[tuple[int, int]]):
    m, n = costs.shape
    u = [None] * m
    v = [None] * n
    u[0] = 0.0

    changed = True
    while changed:
        changed = False
        for i, j in basis:
            if u[i] is not None and v[j] is None:
                v[j] = float(costs[i, j] - u[i])
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = float(costs[i, j] - v[j])
                changed = True

    # Degenerate disconnected fragments are anchored at zero as a fallback.
    for i, value in enumerate(u):
        if value is None:
            u[i] = 0.0
            changed = True
            while changed:
                changed = False
                for row, col in basis:
                    if u[row] is not None and v[col] is None:
                        v[col] = float(costs[row, col] - u[row])
                        changed = True
                    elif v[col] is not None and u[row] is None:
                        u[row] = float(costs[row, col] - v[col])
                        changed = True
    v = [0.0 if value is None else value for value in v]
    return np.array(u, dtype=float), np.array(v, dtype=float)


def _opportunity_costs(costs: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return costs - u[:, None] - v[None, :]


def find_closed_loop(
    entering_cell: tuple[int, int],
    basis: set[tuple[int, int]],
    m: int,
    n: int,
) -> list[tuple[int, int]]:
    """Find the improvement loop created by adding entering_cell to the basis."""
    start_row, start_col = entering_cell
    start_node = ("c", start_col)
    target_node = ("r", start_row)
    graph = {}

    for i, j in basis:
        row_node = ("r", i)
        col_node = ("c", j)
        graph.setdefault(row_node, []).append((col_node, (i, j)))
        graph.setdefault(col_node, []).append((row_node, (i, j)))

    queue = deque([(start_node, [])])
    visited = {start_node}

    while queue:
        node, path_edges = queue.popleft()
        if node == target_node:
            return [entering_cell] + path_edges
        for next_node, edge in graph.get(node, []):
            if next_node in visited:
                continue
            visited.add(next_node)
            queue.append((next_node, path_edges + [edge]))

    return []


def modi_method(cost_matrix, initial_allocation, source_names, destination_names):
    """Optimize a transportation allocation using the MODI method."""
    costs = np.array(cost_matrix, dtype=float)
    allocation = np.array(initial_allocation, dtype=float).copy()
    m, n = costs.shape
    basis = _complete_basis(_basis_from_allocation(allocation), m, n)
    iterations = []
    max_iterations = m * n * 10

    for iteration in range(1, max_iterations + 1):
        allocation_before = allocation.copy()
        u, v = _potentials(costs, basis)
        opportunity = _opportunity_costs(costs, u, v)

        for i, j in basis:
            opportunity[i, j] = 0.0

        non_basis = [
            (i, j, opportunity[i, j])
            for i in range(m)
            for j in range(n)
            if (i, j) not in basis
        ]
        entering = min(non_basis, key=lambda item: item[2], default=None)

        iteration_data = {
            "iteration": iteration,
            "allocation": allocation_before.copy(),
            "u": u.copy(),
            "v": v.copy(),
            "opportunity_cost": opportunity.copy(),
            "entering_cell": None,
            "loop": [],
            "theta": 0.0,
            "allocation_after": allocation.copy(),
            "total_cost": _total_cost(costs, allocation),
        }

        if entering is None or entering[2] >= -EPSILON:
            iterations.append(iteration_data)
            return {
                "allocation": allocation,
                "total_cost": _total_cost(costs, allocation),
                "iterations": iterations,
                "is_optimal": True,
            }

        entering_cell = (entering[0], entering[1])
        loop = find_closed_loop(entering_cell, basis, m, n)
        if not loop:
            iterations.append(iteration_data)
            return {
                "allocation": allocation,
                "total_cost": _total_cost(costs, allocation),
                "iterations": iterations,
                "is_optimal": False,
            }

        minus_cells = loop[1::2]
        theta = min(allocation[i, j] for i, j in minus_cells)

        for index, (i, j) in enumerate(loop):
            if index % 2 == 0:
                allocation[i, j] += theta
            else:
                allocation[i, j] -= theta
                if abs(allocation[i, j]) < EPSILON:
                    allocation[i, j] = 0.0

        basis.add(entering_cell)
        leaving_candidates = [cell for cell in minus_cells if allocation[cell] <= EPSILON]
        if leaving_candidates:
            basis.discard(leaving_candidates[0])
        basis = _complete_basis(basis, m, n)

        iteration_data.update(
            {
                "entering_cell": entering_cell,
                "loop": loop,
                "theta": float(theta),
                "allocation_after": allocation.copy(),
                "total_cost": _total_cost(costs, allocation),
            }
        )
        iterations.append(iteration_data)

    return {
        "allocation": allocation,
        "total_cost": _total_cost(costs, allocation),
        "iterations": iterations,
        "is_optimal": False,
    }
