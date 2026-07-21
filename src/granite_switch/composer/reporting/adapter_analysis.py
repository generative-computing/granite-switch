# SPDX-License-Identifier: Apache-2.0
"""Adapter analysis and diagnostics — printing utilities."""


def print_source_adapter_analysis(analysis: dict):
    """Print the source adapter analysis table."""
    adapter_names = analysis["adapter_names"]
    module_types = analysis["module_types"]
    status = analysis["status"]
    adapter_ranks = analysis["adapter_ranks"]
    file_info = analysis["file_info"]
    adapter_targets = analysis["adapter_targets"]

    # Print file info
    print("\nAdapter Files:")
    for info in file_info:
        if info["file"]:
            print(
                f"  {info['adapter']}: {info['file']} "
                f"({info['format']}, {info['size_mb']:.2f} MB)"
            )
        else:
            print(f"  {info['adapter']}: WARNING — No weight file found")

    # Print target modules for each adapter
    print("\nDeclared Target Modules (from adapter_config.json):")
    for idx, adapter_name in enumerate(adapter_names):
        targets = (
            sorted(list(adapter_targets[idx])) if idx < len(adapter_targets) else []
        )
        print(f"  {adapter_name}: {targets}")

    # Print header with adapter info
    print("\nSource Adapter Content (target_modules vs. actual file content):")
    print(f"  Ranks: {adapter_ranks}")
    print()

    # Calculate column widths
    module_col_width = max(len(mt) for mt in module_types) + 2
    adapter_col_width = max(15, max(len(name) for name in adapter_names) + 2)

    # Print header row
    header = f"{'Module Type':<{module_col_width}}"
    for adapter_name in adapter_names:
        header += f" | {adapter_name:<{adapter_col_width}}"
    print(header)
    print("-" * len(header))

    # Print data rows
    for module_type in module_types:
        row = f"{module_type:<{module_col_width}}"
        for adapter_name in adapter_names:
            cell_status = status[module_type].get(adapter_name, "unknown")
            row += f" | {cell_status:<{adapter_col_width}}"
        print(row)

    # Print legend
    print("\nLegend:")
    print("  populated    : Module in target_modules AND in file with non-zero values")
    print("  not-targeted : Module NOT in target_modules, correctly absent from file")
    print("  missing*     : Module in target_modules BUT not found in file")
    print("  zero*        : Module in target_modules AND in file BUT all zeros")
    print("  unexpected   : Module NOT in target_modules BUT present in file")
    print("  no-file      : Adapter weight file not found")

    # Summary statistics
    print("\nSummary:")
    total_cells = len(module_types) * len(adapter_names)
    populated_count = sum(
        1
        for mt in module_types
        for an in adapter_names
        if status[mt].get(an) == "populated"
    )
    not_targeted_count = sum(
        1
        for mt in module_types
        for an in adapter_names
        if status[mt].get(an) == "not-targeted"
    )
    missing_count = sum(
        1
        for mt in module_types
        for an in adapter_names
        if status[mt].get(an) == "missing*"
    )
    zero_count = sum(
        1
        for mt in module_types
        for an in adapter_names
        if status[mt].get(an) == "zero*"
    )
    unexpected_count = sum(
        1
        for mt in module_types
        for an in adapter_names
        if status[mt].get(an) == "unexpected"
    )

    correct_count = populated_count + not_targeted_count
    problem_count = missing_count + zero_count + unexpected_count

    print(f"  Total cells: {total_cells}")
    print(f"  Correct: {correct_count} ({100 * correct_count / total_cells:.1f}%)")
    print(f"    - populated: {populated_count}")
    print(f"    - not-targeted: {not_targeted_count}")
    if problem_count > 0:
        print(f"  Problems: {problem_count} ({100 * problem_count / total_cells:.1f}%)")
        if missing_count > 0:
            print(f"    - missing*: {missing_count} (declared but not in file)")
        if zero_count > 0:
            print(f"    - zero*: {zero_count} (in file but all zeros)")
        if unexpected_count > 0:
            print(f"    - unexpected: {unexpected_count} (in file but not declared)")

    print("\n" + "=" * 80)
