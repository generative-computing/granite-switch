# SPDX-License-Identifier: Apache-2.0
"""Post-build parameter validation.

Checks that all model parameters are properly initialized after weight
transfer, using the architecture descriptor to parameterize module group
knowledge.
"""

from collections import defaultdict

import torch

from .arch import ArchDescriptor


def validate_all_parameters(
    model,
    arch: ArchDescriptor,
    adapter_paths: list[str] | None = None,
    adapter_names: list[str] | None = None,
    target_module_sets: list[set] | None = None,
):
    """Validate that all model parameters are properly initialized.

    Args:
        model: The model to validate.
        arch: Architecture descriptor.
        adapter_paths: Optional list of adapter paths (for detailed LoRA validation).
        adapter_names: Display names for each adapter.  When ``None``,
            derived from the directory structure.
        target_module_sets: Pre-loaded target module sets per adapter.
            When ``None`` and *adapter_paths* is given, loaded from disk.
    """
    print("\nValidating model parameters...")

    uninit_params = []
    expected_zero_lora = []
    unexpected_zero_lora = []

    # Build adapter module map if paths provided
    adapter_has_module: dict[int, set] = {}
    if adapter_paths:
        if target_module_sets is None:
            from .adapter_loader import load_adapter_target_modules

            target_module_sets = load_adapter_target_modules(adapter_paths)
        adapter_has_module = dict(enumerate(target_module_sets))

    switch_to_peft = arch.switch_to_peft

    for name, param in model.named_parameters():
        # Skip config buffers
        if any(kw in name for kw in arch.buffer_keywords):
            continue

        is_all_zero = torch.all(param == 0)
        has_nan = torch.any(torch.isnan(param))

        if has_nan:
            uninit_params.append((name, "NaN", None))
        elif is_all_zero:
            is_lora = any(kw in name for kw in arch.lora_keywords)

            if is_lora and adapter_paths:
                module_key = arch.extract_module_key(name)

                if module_key:
                    peft_modules = switch_to_peft.get(module_key, [])

                    should_be_populated = False
                    missing_from_adapters = []

                    for adapter_idx, has_modules in adapter_has_module.items():
                        if peft_modules and any(
                            pm in has_modules for pm in peft_modules
                        ):
                            should_be_populated = True
                        else:
                            if adapter_names is not None:
                                label = adapter_names[adapter_idx]
                            else:
                                from pathlib import Path as _Path

                                label = _Path(
                                    adapter_paths[adapter_idx]
                                ).parent.parent.name
                            missing_from_adapters.append(f"{label}({adapter_idx})")

                    if should_be_populated:
                        if len(missing_from_adapters) == len(adapter_paths):
                            unexpected_zero_lora.append(
                                (
                                    name,
                                    module_key,
                                    "all_adapters_missing",
                                    missing_from_adapters,
                                )
                            )
                        else:
                            expected_zero_lora.append(
                                (
                                    name,
                                    module_key,
                                    "zero_padding_or_partial",
                                    missing_from_adapters,
                                )
                            )
                    else:
                        expected_zero_lora.append(
                            (
                                name,
                                module_key,
                                "no_adapter_targets",
                                missing_from_adapters,
                            )
                        )
                else:
                    expected_zero_lora.append((name, "unknown", "unknown_module", []))
            elif is_lora:
                expected_zero_lora.append((name, None, "no_adapter_info", []))
            else:
                uninit_params.append((name, "all_zero", None))

    # ---- Report ----

    if uninit_params:
        print(
            f"\nWARNING: {len(uninit_params)} base model parameters "
            f"appear uninitialized:"
        )
        for name, reason, _ in uninit_params[:10]:
            print(f"  - {name} ({reason})")
        if len(uninit_params) > 10:
            print(f"  ... and {len(uninit_params) - 10} more")
        print("\nThis is unexpected and may indicate a problem with weight transfer")

    if unexpected_zero_lora:
        print(
            f"\nWARNING: {len(unexpected_zero_lora)} LoRA parameters "
            f"are unexpectedly zero:"
        )
        for name, module, reason, adapters in unexpected_zero_lora[:10]:
            print(f"  - {name}")
            print(f"      Module: {module}, Reason: {reason}")
            if adapters:
                print(
                    f"      Missing from: "
                    f"{', '.join(adapters[:3])}"
                    f"{'...' if len(adapters) > 3 else ''}"
                )
        if len(unexpected_zero_lora) > 10:
            print(f"  ... and {len(unexpected_zero_lora) - 10} more")
        print("\nThese should have been populated by adapters")

    if expected_zero_lora:
        print(
            f"\nINFO: {len(expected_zero_lora)} LoRA parameters are zero (as expected):"
        )

        by_reason = defaultdict(list)
        for name, module, reason, adapters in expected_zero_lora:
            by_reason[reason].append((name, module, adapters))

        for reason, items in by_reason.items():
            print(f"\n  {reason}: {len(items)} parameters")
            if reason == "no_adapter_targets":
                print("    -> No adapter targets these modules")
            elif reason == "zero_padding_or_partial":
                print("    -> Zero-padding or some adapters don't target these modules")

            for name, module, adapters in items[:3]:
                print(f"      - {name}")
                if adapters:
                    print(
                        f"          Missing from: "
                        f"{', '.join(adapters[:2])}"
                        f"{'...' if len(adapters) > 2 else ''}"
                    )
            if len(items) > 3:
                print(f"      ... and {len(items) - 3} more")

        print("\n  These zeros are normal and expected")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\nParameter summary:")
    print(f"  Total: {total_params:,}")
    print(
        f"  Trainable: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)"
    )
    print(f"  Frozen: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
