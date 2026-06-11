from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from methods.modi import modi_method
from methods.vam import vogel_approximation_method
from utils.formatter import (
    build_excel_report,
    format_number,
    matrix_to_dataframe,
    shipment_details,
)
from utils.validation import prepare_balanced_data, validate_inputs


st.set_page_config(page_title="Transportation Problem Solver", layout="wide")


EXAMPLE_SUPPLY = pd.DataFrame(
    {
        "Nama Sumber": [
            "SPPG SAYANG JATINANGOR",
            "SPPG CIBEUSI 2",
            "SPPG CIKERUH",
            "SPPG CIPACING 2",
            "SPPG Yayasan KH Abu Syaeri Cikalama",
        ],
        "Supply": [2440, 1182, 2966, 1722, 3000],
    }
)
EXAMPLE_DEMAND = pd.DataFrame(
    {
        "Nama Tujuan": [
            "MI ASY SYAFIIYAH",
            "MIS BAITURRAHMAN",
            "SD NEGERI CIBEUSI",
            "MIS CIBEUSI",
            "SMKS PASUNDAN JATINANGOR",
            "SD NEGERI CIPACING I",
            "SDIT IMAM BUKHARI",
            "SMPN 3 JATINANGOR",
            "SMAN JATINANGOR",
            "MTS MAARIF CIKERUH",
        ],
        "Demand": [776, 604, 676, 646, 1310, 803, 1025, 850, 1855, 1102],
    }
)
EXAMPLE_COSTS = pd.DataFrame(
    [
        [0.168, 2.362, 2.590, 2.667, 0.267, 0.914, 0.571, 5.562, 2.057, 1.524, 0],
        [1.752, 2.895, 2.590, 2.590, 2.514, 0.610, 2.819, 6.781, 4.190, 3.810, 0],
        [1.143, 3.276, 1.600, 1.600, 0.838, 1.676, 1.448, 4.876, 1.524, 1.143, 0],
        [1.829, 1.295, 1.981, 1.981, 1.981, 0.457, 2.286, 6.095, 0.457, 3.124, 0],
        [3.810, 6.095, 3.886, 3.581, 3.505, 3.886, 3.657, 5.790, 2.133, 2.286, 0],
    ]
)


def make_blank_supply(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nama Sumber": [f"Sumber {index + 1}" for index in range(count)],
            "Supply": [0] * count,
        }
    )


def make_blank_demand(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nama Tujuan": [f"Tujuan {index + 1}" for index in range(count)],
            "Demand": [0] * count,
        }
    )


def resize_supply_df(existing: pd.DataFrame, count: int) -> pd.DataFrame:
    resized = existing.copy().reset_index(drop=True)
    for index in range(len(resized), count):
        resized.loc[index] = [f"Sumber {index + 1}", 0]
    return resized.iloc[:count].reset_index(drop=True)


def resize_demand_df(existing: pd.DataFrame, count: int) -> pd.DataFrame:
    resized = existing.copy().reset_index(drop=True)
    for index in range(len(resized), count):
        resized.loc[index] = [f"Tujuan {index + 1}", 0]
    return resized.iloc[:count].reset_index(drop=True)


def apply_data_editor_changes(frame: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    """Merge Streamlit data_editor's pending edits back into our dataframe."""
    editor_state = st.session_state.get(editor_key)
    if not isinstance(editor_state, dict):
        return frame

    updated = frame.copy()
    for row_index, changes in editor_state.get("edited_rows", {}).items():
        row_index = int(row_index)
        if row_index >= len(updated):
            continue
        for column, value in changes.items():
            if column in updated.columns:
                updated.at[row_index, column] = value
    return updated


def apply_cost_editor_changes(
    frame: pd.DataFrame, editor_key: str, demand_df: pd.DataFrame
) -> pd.DataFrame:
    editor_state = st.session_state.get(editor_key)
    if not isinstance(editor_state, dict):
        return frame

    updated = frame.copy()
    destinations = demand_df["Nama Tujuan"].astype(str).str.strip().tolist()
    for row_index, changes in editor_state.get("edited_rows", {}).items():
        row_index = int(row_index)
        if row_index >= len(updated):
            continue
        for column, value in changes.items():
            if column == "Nama Sumber" or column not in destinations:
                continue
            col_index = destinations.index(column)
            if col_index < updated.shape[1]:
                updated.iat[row_index, col_index] = value
    return updated


def apply_pending_editor_changes():
    if "supply_df" in st.session_state:
        st.session_state.supply_df = apply_data_editor_changes(
            st.session_state.supply_df, "supply_editor"
        )
    if "demand_df" in st.session_state:
        st.session_state.demand_df = apply_data_editor_changes(
            st.session_state.demand_df, "demand_editor"
        )
    if "cost_df" in st.session_state:
        st.session_state.cost_df = apply_cost_editor_changes(
            st.session_state.cost_df,
            "cost_editor",
            st.session_state.get("demand_df", pd.DataFrame({"Nama Tujuan": []})),
        )


def sync_cost_matrix(supply_df: pd.DataFrame, demand_df: pd.DataFrame, existing=None):
    row_count = len(supply_df)
    col_count = len(demand_df)
    current = existing if isinstance(existing, pd.DataFrame) else pd.DataFrame()
    synced = pd.DataFrame(0.0, index=range(row_count), columns=range(col_count))

    for row_index in range(row_count):
        for col_index in range(col_count):
            if row_index < current.shape[0] and col_index < current.shape[1]:
                synced.iloc[row_index, col_index] = current.iloc[row_index, col_index]
    return synced


def build_cost_editor_df(
    cost_df: pd.DataFrame, supply_df: pd.DataFrame, demand_df: pd.DataFrame
) -> pd.DataFrame:
    sources = supply_df["Nama Sumber"].astype(str).str.strip().tolist()
    destinations = demand_df["Nama Tujuan"].astype(str).str.strip().tolist()
    display_df = cost_df.copy()
    display_df.columns = destinations
    display_df.insert(0, "Nama Sumber", sources)
    return display_df


def extract_cost_matrix_from_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    cost_only = editor_df.drop(columns=["Nama Sumber"], errors="ignore")
    cost_only = cost_only.apply(pd.to_numeric, errors="coerce")
    cost_only.index = range(len(cost_only))
    cost_only.columns = range(cost_only.shape[1])
    return cost_only


def reset_state():
    for key in [
        "supply_df",
        "demand_df",
        "cost_df",
        "result",
        "last_use_example",
        "last_source_count",
        "last_destination_count",
        "supply_editor",
        "demand_editor",
        "cost_editor",
    ]:
        st.session_state.pop(key, None)


def initialize_input_state(use_example: bool, source_count: int, destination_count: int):
    changed_mode = st.session_state.get("last_use_example") != use_example
    changed_size = (
        st.session_state.get("last_source_count") != source_count
        or st.session_state.get("last_destination_count") != destination_count
    )

    if use_example and (changed_mode or "supply_df" not in st.session_state):
        st.session_state.supply_df = EXAMPLE_SUPPLY.copy()
        st.session_state.demand_df = EXAMPLE_DEMAND.copy()
        st.session_state.cost_df = EXAMPLE_COSTS.copy()
    elif not use_example and (changed_mode or "supply_df" not in st.session_state):
        st.session_state.supply_df = make_blank_supply(source_count)
        st.session_state.demand_df = make_blank_demand(destination_count)
        st.session_state.cost_df = sync_cost_matrix(
            st.session_state.supply_df, st.session_state.demand_df
        )
    elif not use_example and changed_size:
        st.session_state.supply_df = resize_supply_df(
            st.session_state.supply_df, source_count
        )
        st.session_state.demand_df = resize_demand_df(
            st.session_state.demand_df, destination_count
        )
        st.session_state.cost_df = sync_cost_matrix(
            st.session_state.supply_df,
            st.session_state.demand_df,
            st.session_state.get("cost_df"),
        )

    st.session_state.last_use_example = use_example
    st.session_state.last_source_count = source_count
    st.session_state.last_destination_count = destination_count


def run_solver(supply_df: pd.DataFrame, demand_df: pd.DataFrame, cost_df: pd.DataFrame):
    errors = validate_inputs(supply_df, demand_df, cost_df)
    if errors:
        for error in errors:
            st.error(error)
        return

    balanced = prepare_balanced_data(supply_df, demand_df, cost_df)
    vam_result = vogel_approximation_method(
        balanced["cost_matrix"],
        balanced["supply"],
        balanced["demand"],
        balanced["source_names"],
        balanced["destination_names"],
    )
    modi_result = modi_method(
        balanced["cost_matrix"],
        vam_result["allocation"],
        balanced["source_names"],
        balanced["destination_names"],
    )

    vam_allocation_df = matrix_to_dataframe(
        vam_result["allocation"], balanced["source_names"], balanced["destination_names"]
    )
    optimal_allocation_df = matrix_to_dataframe(
        modi_result["allocation"], balanced["source_names"], balanced["destination_names"]
    )
    balanced_cost_df = matrix_to_dataframe(
        balanced["cost_matrix"], balanced["source_names"], balanced["destination_names"]
    )

    st.session_state.result = {
        **balanced,
        "cost_df": balanced_cost_df,
        "supply_df": pd.DataFrame(
            {"Nama Sumber": balanced["source_names"], "Supply": balanced["supply"]}
        ),
        "demand_df": pd.DataFrame(
            {"Nama Tujuan": balanced["destination_names"], "Demand": balanced["demand"]}
        ),
        "vam": vam_result,
        "modi": modi_result,
        "vam_allocation_df": vam_allocation_df,
        "optimal_allocation_df": optimal_allocation_df,
        "initial_cost": vam_result["total_cost"],
        "optimal_cost": modi_result["total_cost"],
    }
    st.success("Perhitungan selesai.")


def dict_to_table(data: dict, name_column: str, value_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{name_column: key, value_column: value} for key, value in data.items()]
    )


def vam_summary_table(step: dict) -> pd.DataFrame:
    source, destination = step["selected_cell"]
    return pd.DataFrame(
        [
            ["Iterasi", step["iteration"]],
            ["Baris/Kolom Dipilih", step["selected"]],
            ["Cell Dipilih", f"{source} -> {destination}"],
            ["Jumlah Alokasi", step["allocated"]],
        ],
        columns=["Keterangan", "Nilai"],
    )


def potentials_table(item: dict, source_names: list[str], destination_names: list[str]):
    u_table = pd.DataFrame({"Sumber": source_names, "u": item["u"]})
    v_table = pd.DataFrame({"Tujuan": destination_names, "v": item["v"]})
    return u_table, v_table


def modi_summary_table(item: dict, source_names: list[str], destination_names: list[str]):
    entering = item["entering_cell"]
    if entering:
        entering_label = f"{source_names[entering[0]]} -> {destination_names[entering[1]]}"
    else:
        entering_label = "Tidak ada"

    return pd.DataFrame(
        [
            ["Iterasi", item["iteration"]],
            ["Entering Cell", entering_label],
            ["Theta", item["theta"]],
            ["Total Cost", item["total_cost"]],
        ],
        columns=["Keterangan", "Nilai"],
    )


def loop_table(loop: list[tuple[int, int]], source_names: list[str], destination_names: list[str]):
    rows = []
    for index, (source_index, destination_index) in enumerate(loop):
        rows.append(
            {
                "Urutan": index + 1,
                "Tanda": "+" if index % 2 == 0 else "-",
                "Sumber": source_names[source_index],
                "Tujuan": destination_names[destination_index],
            }
        )
    return pd.DataFrame(rows)


def penalty_label(value, detail="") -> str:
    if value is None:
        return ""
    detail_html = f"<small>{escape(str(detail))}</small>" if detail else ""
    return f"<strong>{format_number(value)}</strong>{detail_html}"


def vam_tableau_html(step: dict, result: dict) -> str:
    costs = result["cost_matrix"]
    allocation = step.get("allocation_after", result["vam"]["allocation"])
    source_names = result["source_names"]
    destination_names = result["destination_names"]
    selected_i, selected_j = step.get("selected_cell_indices", (-1, -1))
    selected_axis = step.get("selected_axis")
    selected_axis_index = step.get("selected_axis_index")
    row_penalties = step.get("row_penalties", {})
    row_details = step.get("row_penalty_details", {})
    col_penalties = step.get("column_penalties", {})
    col_details = step.get("column_penalty_details", {})
    supply_values = step.get("supply_before", step.get("remaining_supply", {}))
    demand_values = step.get("demand_before", step.get("remaining_demand", {}))
    total_supply = sum(float(value) for value in supply_values.values())

    html = [
        """
        <style>
        .vam-tableau-wrap { overflow-x: auto; padding: 0.25rem 0 0.75rem; }
        table.vam-tableau {
            border-collapse: collapse;
            width: 100%;
            min-width: 820px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            table-layout: fixed;
            font-family: Arial, sans-serif;
            background: var(--background-color);
        }
        .vam-tableau th,
        .vam-tableau td {
            border: 1px solid rgba(49, 51, 63, 0.2);
            min-width: 96px;
            height: 76px;
            text-align: center;
            vertical-align: middle;
            position: relative;
            color: var(--text-color);
            background: var(--background-color);
        }
        .vam-tableau th {
            height: 46px;
            background: var(--secondary-background-color);
            font-size: 1.05rem;
            font-weight: 800;
        }
        .vam-tableau .row-header {
            background: var(--secondary-background-color);
            font-weight: 800;
            color: var(--text-color);
            width: 120px;
        }
        .vam-tableau .cost-badge {
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 1px 7px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 6px;
            background: var(--secondary-background-color);
            color: var(--text-color);
            font-weight: 800;
            box-shadow: none;
        }
        .vam-tableau .allocation {
            font-size: 1.45rem;
            font-weight: 900;
            color: var(--text-color);
        }
        .vam-tableau .selected-cell {
            background: color-mix(in srgb, var(--secondary-background-color) 72%, var(--text-color) 8%);
            outline: 2px solid rgba(49, 51, 63, 0.35);
            outline-offset: -2px;
        }
        .vam-tableau .selected-row-col {
            background: var(--secondary-background-color);
        }
        .vam-tableau .supply-header,
        .vam-tableau .supply-cell {
            background: var(--secondary-background-color);
        }
        .vam-tableau .supply-header {
            background: var(--secondary-background-color);
        }
        .vam-tableau .supply-cell strong {
            color: var(--text-color);
            font-size: 1.35rem;
        }
        .vam-tableau .demand-header,
        .vam-tableau .demand-cell {
            background: var(--secondary-background-color);
            color: var(--text-color);
            font-weight: 900;
        }
        .vam-tableau .demand-cell {
            font-size: 1.15rem;
        }
        .vam-tableau .total-cell {
            background: var(--secondary-background-color);
            color: var(--text-color);
            font-size: 1.25rem;
            font-weight: 900;
        }
        .vam-tableau .penalty-header,
        .vam-tableau .penalty-cell,
        .vam-tableau .penalty-label {
            background: var(--secondary-background-color);
            color: var(--text-color);
        }
        .vam-tableau .penalty-cell strong {
            display: block;
            color: var(--text-color);
            font-size: 1.2rem;
            margin-bottom: 0.2rem;
        }
        .vam-tableau .penalty-cell small {
            display: block;
            color: color-mix(in srgb, var(--text-color) 70%, transparent);
        }
        .vam-tableau .selected-penalty {
            background: color-mix(in srgb, var(--secondary-background-color) 72%, var(--text-color) 8%);
            outline: 2px solid rgba(49, 51, 63, 0.35);
            outline-offset: -2px;
        }
        </style>
        <div class="vam-tableau-wrap">
        <table class="vam-tableau">
        """
    ]

    html.append("<tr><th> </th>")
    for destination in destination_names:
        html.append(f"<th>{escape(destination)}</th>")
    html.append("<th class='supply-header'>Supply</th><th class='penalty-header'>P Baris</th></tr>")

    for i, source in enumerate(source_names):
        html.append(f"<tr><td class='row-header'>{escape(source)}</td>")
        for j, _ in enumerate(destination_names):
            classes = []
            if i == selected_i and j == selected_j:
                classes.append("selected-cell")
            elif (
                selected_axis == "row"
                and i == selected_axis_index
                or selected_axis == "col"
                and j == selected_axis_index
            ):
                classes.append("selected-row-col")
            amount = float(allocation[i][j])
            amount_html = (
                f"<div class='allocation'>{format_number(amount)}</div>" if amount > 0 else ""
            )
            html.append(
                f"<td class='{' '.join(classes)}'>"
                f"<span class='cost-badge'>{format_number(costs[i][j])}</span>"
                f"{amount_html}</td>"
            )
        supply = supply_values.get(source, "")
        html.append(
            f"<td class='supply-cell'><strong>{format_number(supply)}</strong></td>"
        )
        row_penalty = row_penalties.get(source)
        row_detail = row_details.get(source, "")
        row_penalty_class = (
            "penalty-cell selected-penalty"
            if selected_axis == "row" and i == selected_axis_index
            else "penalty-cell"
        )
        html.append(
            f"<td class='{row_penalty_class}'>{penalty_label(row_penalty, row_detail)}</td></tr>"
        )

    html.append("<tr><td class='demand-header'>Demand</td>")
    for destination in destination_names:
        html.append(
            f"<td class='demand-cell'>{format_number(demand_values.get(destination, 0))}</td>"
        )
    html.append(f"<td class='total-cell'>{format_number(total_supply)}</td><td class='total-cell'></td></tr>")

    html.append("<tr><td class='penalty-label'><strong>P Kolom</strong></td>")
    for j, destination in enumerate(destination_names):
        col_penalty = col_penalties.get(destination)
        col_detail = col_details.get(destination, "")
        col_penalty_class = (
            "penalty-cell selected-penalty"
            if selected_axis == "col" and j == selected_axis_index
            else "penalty-cell"
        )
        html.append(
            f"<td class='{col_penalty_class}'>{penalty_label(col_penalty, col_detail)}</td>"
        )
    html.append("<td colspan='2'></td></tr></table></div>")
    return "".join(html)


with st.sidebar:
    st.header("VAM & MODI Solver")
    source_count = st.number_input("Jumlah sumber", min_value=1, value=3, step=1)
    destination_count = st.number_input("Jumlah tujuan", min_value=1, value=4, step=1)
    use_example = st.checkbox("Gunakan Data Contoh")
    if st.button("Reset Data", use_container_width=True):
        reset_state()
        st.rerun()

initialize_input_state(use_example, int(source_count), int(destination_count))
apply_pending_editor_changes()

st.title("Transportation Problem Solver")
st.write(
    "Aplikasi untuk menyelesaikan masalah transportasi menggunakan VAM sebagai solusi "
    "awal dan MODI sebagai metode optimasi."
)

tabs = st.tabs(
    [
        "Input Data",
        "Solusi Awal VAM",
        "Optimasi MODI",
        "Solusi Optimal",
        "Export",
    ]
)

with tabs[0]:
    st.subheader("Input Data")
    left, right = st.columns(2)

    with left:
        st.markdown("**Tabel Supply**")
        supply_df = st.data_editor(
            st.session_state.supply_df,
            num_rows="fixed",
            use_container_width=True,
            key="supply_editor",
        )

    with right:
        st.markdown("**Tabel Demand**")
        demand_df = st.data_editor(
            st.session_state.demand_df,
            num_rows="fixed",
            use_container_width=True,
            key="demand_editor",
        )

    st.session_state.supply_df = supply_df
    st.session_state.demand_df = demand_df
    st.session_state.cost_df = sync_cost_matrix(
        supply_df, demand_df, st.session_state.get("cost_df")
    )

    st.markdown("**Matriks Biaya**")
    cost_df = st.data_editor(
        build_cost_editor_df(st.session_state.cost_df, supply_df, demand_df),
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        disabled=["Nama Sumber"],
        key="cost_editor",
    )
    st.session_state.cost_df = extract_cost_matrix_from_editor(cost_df)

    total_supply = pd.to_numeric(supply_df["Supply"], errors="coerce").sum()
    total_demand = pd.to_numeric(demand_df["Demand"], errors="coerce").sum()
    status = "Balanced" if total_supply == total_demand else "Unbalanced"

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Total Supply", format_number(total_supply))
    metric_b.metric("Total Demand", format_number(total_demand))
    metric_c.metric("Status", status)
    if status == "Balanced":
        st.success("Total supply sama dengan total demand.")
    elif total_supply > total_demand:
        st.warning(
            f"Unbalanced. Dummy Tujuan akan ditambahkan sebesar "
            f"{format_number(total_supply - total_demand)}."
        )
    else:
        st.warning(
            f"Unbalanced. Dummy Sumber akan ditambahkan sebesar "
            f"{format_number(total_demand - total_supply)}."
        )

    if st.button("Hitung Solusi", type="primary"):
        run_solver(supply_df, demand_df, st.session_state.cost_df)

result = st.session_state.get("result")

with tabs[1]:
    if not result:
        st.info("Klik tombol Hitung Solusi pada tab Input Data terlebih dahulu.")
    else:
        st.metric("Total Biaya Awal", format_number(result["initial_cost"]))
        st.dataframe(result["vam_allocation_df"], use_container_width=True)
        for step in result["vam"]["steps"]:
            with st.expander(f"Iterasi VAM {step['iteration']}"):
                st.markdown(
                    vam_tableau_html(step, result),
                    unsafe_allow_html=True,
                )
with tabs[2]:
    if not result:
        st.info("Klik tombol Hitung Solusi pada tab Input Data terlebih dahulu.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Biaya Awal", format_number(result["initial_cost"]))
        col_b.metric("Total Biaya Optimal", format_number(result["optimal_cost"]))
        col_c.metric("Iterasi MODI", len(result["modi"]["iterations"]))
        col_d.metric("Status", "Optimal" if result["modi"]["is_optimal"] else "Belum optimal")

        if result["modi"]["is_optimal"]:
            st.success("Solusi optimal ditemukan.")
        else:
            st.warning("MODI berhenti sebelum status optimal. Periksa detail iterasi.")

        for item in result["modi"]["iterations"]:
            with st.expander(f"Iterasi MODI {item['iteration']}"):
                st.markdown("**Ringkasan Iterasi**")
                st.dataframe(
                    modi_summary_table(
                        item, result["source_names"], result["destination_names"]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("**Allocation Saat Ini**")
                st.dataframe(
                    matrix_to_dataframe(
                        item["allocation"],
                        result["source_names"],
                        result["destination_names"],
                    ),
                    use_container_width=True,
                )

                u_table, v_table = potentials_table(
                    item, result["source_names"], result["destination_names"]
                )
                potential_left, potential_right = st.columns(2)
                with potential_left:
                    st.markdown("**Nilai u**")
                    st.dataframe(u_table, hide_index=True, use_container_width=True)
                with potential_right:
                    st.markdown("**Nilai v**")
                    st.dataframe(v_table, hide_index=True, use_container_width=True)

                st.markdown("**Opportunity Cost**")
                st.dataframe(
                    matrix_to_dataframe(
                        item["opportunity_cost"],
                        result["source_names"],
                        result["destination_names"],
                    ),
                    use_container_width=True,
                )

                st.markdown("**Loop Perbaikan**")
                if item["loop"]:
                    st.dataframe(
                        loop_table(
                            item["loop"],
                            result["source_names"],
                            result["destination_names"],
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.dataframe(
                        pd.DataFrame(
                            [{"Keterangan": "Tidak ada loop karena solusi sudah optimal."}]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

                st.markdown("**Allocation Setelah Perbaikan**")
                st.dataframe(
                    matrix_to_dataframe(
                        item["allocation_after"],
                        result["source_names"],
                        result["destination_names"],
                    ),
                    use_container_width=True,
                )

with tabs[3]:
    if not result:
        st.info("Klik tombol Hitung Solusi pada tab Input Data terlebih dahulu.")
    else:
        col_a, col_b, col_c = st.columns(3)
        savings = result["initial_cost"] - result["optimal_cost"]
        col_a.metric("Total Biaya Awal", format_number(result["initial_cost"]))
        col_b.metric("Total Biaya Optimal", format_number(result["optimal_cost"]))
        col_c.metric("Penghematan", format_number(savings))

        st.dataframe(result["optimal_allocation_df"], use_container_width=True)
        show_dummy = st.checkbox("Tampilkan dummy", value=False)
        details = shipment_details(
            result["modi"]["allocation"],
            result["source_names"],
            result["destination_names"],
            show_dummy=show_dummy,
        )
        st.markdown("**Rincian Pengiriman**")
        if details:
            for detail in details:
                st.write(detail)
        else:
            st.write("Tidak ada alokasi yang ditampilkan.")

        st.success(
            "Berdasarkan perhitungan VAM dan MODI, diperoleh total biaya distribusi "
            f"minimum sebesar {format_number(result['optimal_cost'])}. Solusi dinyatakan "
            "optimal karena seluruh nilai opportunity cost sudah lebih besar atau sama dengan 0."
        )

with tabs[4]:
    if not result:
        st.info("Klik tombol Hitung Solusi pada tab Input Data terlebih dahulu.")
    else:
        csv_data = result["optimal_allocation_df"].to_csv().encode("utf-8")
        st.download_button(
            "Download Allocation Optimal CSV",
            data=csv_data,
            file_name="optimal_allocation.csv",
            mime="text/csv",
        )
        try:
            excel_data = build_excel_report(result)
        except ModuleNotFoundError as error:
            if error.name == "openpyxl":
                st.error(
                    "Export Excel membutuhkan package openpyxl. Jalankan "
                    "`pip install openpyxl` atau `pip install -r requirements.txt`, "
                    "lalu restart Streamlit."
                )
            else:
                raise
        else:
            st.download_button(
                "Download Laporan Excel",
                data=excel_data,
                file_name="transportation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
