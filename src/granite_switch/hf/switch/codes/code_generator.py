#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kerdock and Delsarte-Goethals code generation for memory-based switching.

Implements on-the-fly generation of Kerdock and Delsarte-Goethals codes
using Z4-linear codes and the Gray map (Hammons et al. 1994).

Reference: theory/z4_kerdock_dg_construction.pdf
Implementation: theory/z4_kerdock_dg.py
"""

from typing import Literal

import numpy as np
import torch

from .z4_kerdock_dg import GaloisRing, gray_map


class KerdockDGCodeGenerator:
    """Generate Kerdock/DG code vectors on-the-fly for memory addressing.

    Four configurations available (antipodal-free, a ∈ {0,1} only):
    - Kerdock K(6): N=64, capacity=2K, μ=1/8
    - DG(6,1): N=64, capacity=65K, μ=1/4
    - Kerdock K(8): N=256, capacity=32K, μ=1/16
    - DG(8,1): N=256, capacity=4.2M, μ=1/8

    Implementation: On-the-fly generation using Galois ring arithmetic.
    Memory efficient: Only generate codes as needed, no precomputed storage.

    Args:
        m: Parameter m (6 or 8), determines dimension N=2^m
        code_type: "kerdock" or "dg1"
        verbose: Print initialization info
    """

    def __init__(
        self,
        m: int,
        code_type: Literal["kerdock", "dg1"] = "kerdock",
        verbose: bool = True,
    ):
        if m not in [6, 8]:
            raise ValueError(f"m must be 6 or 8, got {m}")
        if code_type not in ["kerdock", "dg1"]:
            raise ValueError(f"code_type must be 'kerdock' or 'dg1', got {code_type}")

        self.m = m
        self.code_type = code_type
        self.N = 2**m  # Binary dimension (attention head dim)
        self.verbose = verbose

        # Initialize Galois ring GR(4, m-1) with precomputed trace tables
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Initializing {code_type.upper()} Code Generator (m={m})")
            print(f"{'=' * 60}")

        self.gr = GaloisRing(m)

        # Compute capacity (antipodal-free: a ∈ {0,1} only, not {0,1,2,3})
        # Codewords c_{a,b} and c_{a+2,b} are exact antipodals after Gray map,
        # so we restrict a to {0,1} to exclude antipodal pairs.
        if code_type == "kerdock":
            self.capacity = 2 ** (2 * m - 1)  # 2 * 4^(m-1)
            self.coherence = 1.0 / np.sqrt(self.N)
        else:  # dg1
            self.capacity = 2 ** (3 * m - 2)  # 2 * 4^(m-1) * 2^(m-1)
            self.coherence = 2.0 / np.sqrt(self.N)

        if verbose:
            print(f"\n✓ {code_type.upper()} generator initialized")
            print(f"  Dimension N: {self.N}")
            print(f"  Capacity: {self.capacity:,} addressable codes")
            print(f"  Theoretical coherence μ: {self.coherence:.6f}")
            print(f"{'=' * 60}\n")

    def generate_code_vector(self, address: int) -> torch.Tensor:
        """Generate single code vector on-the-fly.

        Process:
        1. Decode address into (a, b, [γ₁]) parameters
        2. Compute Z4 codeword: ca,b,[γ₁](ξ) for all ξ ∈ Teichmüller set
        3. Apply Gray map: Z4 → F2²
        4. Convert to ±1: 0→+1, 1→-1
        5. Normalize to unit norm

        Args:
            address: Integer address in [0, capacity-1]

        Returns:
            Unit-norm code vector [N] as torch.Tensor (float32)
        """
        if address >= self.capacity:
            raise ValueError(f"Address {address} exceeds capacity {self.capacity}")

        # Decode address
        if self.code_type == "kerdock":
            a, b_coeffs = self._decode_kerdock_address(address)
            z4_codeword = self._compute_kerdock_codeword(a, b_coeffs)
        else:  # dg1
            a, b_coeffs, gamma1_idx = self._decode_dg1_address(address)
            z4_codeword = self._compute_dg1_codeword(a, b_coeffs, gamma1_idx)

        # Apply Gray map
        binary_codeword = self._apply_gray_map(z4_codeword)

        # Convert to ±1 (0→+1, 1→-1)
        signed_vector = 1.0 - 2.0 * binary_codeword.astype(np.float32)

        # Normalize to unit norm
        norm = np.linalg.norm(signed_vector)
        if norm > 0:
            signed_vector = signed_vector / norm

        # Convert to torch tensor
        return torch.from_numpy(signed_vector.astype(np.float32))

    def precompute_codebook(self, dtype=torch.float32) -> torch.Tensor:
        """Pre-compute all code vectors as a [capacity, N] tensor.

        For use as a registered buffer in MultiSwitch. Enables torch.compile-
        compatible index lookup instead of on-the-fly generation.

        Raises ValueError if the codebook would exceed 128 MB (e.g. DG(8,1)
        with 4.2M vectors of dim 256). If that configuration is ever needed,
        implement an on-the-fly fallback.
        """
        size_bytes = self.capacity * self.N * 4  # float32 = 4 bytes
        max_bytes = 128 * 1024 * 1024  # 128 MB
        if size_bytes > max_bytes:
            raise ValueError(
                f"Codebook too large to precompute: {self.capacity} x {self.N} "
                f"= {size_bytes / 1024**2:.0f} MB (limit {max_bytes // 1024**2} MB). "
                f"Implement on-the-fly fallback for this configuration."
            )
        addresses = torch.arange(self.capacity, dtype=torch.long)
        return self.generate_code_vectors_batch(addresses, dtype=dtype)

    def generate_code_vectors_batch(
        self, addresses: torch.Tensor, device=None, dtype=None
    ) -> torch.Tensor:
        """Generate multiple code vectors efficiently (torch.compile friendly).

        Vectorized implementation that processes multiple addresses in parallel
        using pure tensor operations. Compatible with torch.compile.

        Args:
            addresses: [B] Tensor of integer addresses in [0, capacity-1]
            device: Target device (defaults to addresses.device)
            dtype: Target dtype (defaults to torch.float32)

        Returns:
            Unit-norm code vectors [B, N] as torch.Tensor
        """
        if device is None:
            device = addresses.device
        if dtype is None:
            dtype = torch.float32

        B = addresses.shape[0]
        T_size = len(self.gr.teichmuller_set)

        # Convert trace tables to torch tensors (cached after first call)
        if not hasattr(self, "_trace_table_z4_torch"):
            self._trace_table_z4_torch = torch.from_numpy(self.gr.trace_table_z4).to(
                device=device, dtype=torch.long
            )
            if self.code_type == "dg1":
                self._trace_table_f2_torch = torch.from_numpy(
                    self.gr.trace_table_f2
                ).to(device=device, dtype=torch.long)

        # Decode addresses in parallel (antipodal-free: a ∈ {0,1} only)
        if self.code_type == "kerdock":
            four_to_deg = 4**self.gr.deg
            a = addresses // four_to_deg  # [B], a ∈ {0,1}
            b_indices = addresses % four_to_deg  # [B]

            # Convert b_indices to coefficient matrix [B, deg]
            b_coeffs = torch.zeros((B, self.gr.deg), device=device, dtype=torch.long)
            for i in range(self.gr.deg):
                b_coeffs[:, i] = b_indices % 4
                b_indices = b_indices // 4

            # Compute traces vectorized: [B, deg] @ [deg, T_size] = [B, T_size]
            traces = torch.matmul(
                b_coeffs.float(), self._trace_table_z4_torch.float()
            )  # [B, T_size]
            traces = traces.round().long() % 4

            # Compute Z4 codewords: [B, T_size]
            z4_codewords = (a.unsqueeze(1) + traces) % 4  # [B, T_size]

        else:  # dg1
            kerdock_size = 2 * (4**self.gr.deg)  # a ∈ {0,1} only
            gamma1_indices = addresses // kerdock_size  # [B]
            kerdock_addrs = addresses % kerdock_size  # [B]

            # Decode Kerdock part
            four_to_deg = 4**self.gr.deg
            a = kerdock_addrs // four_to_deg  # a ∈ {0,1}
            b_indices = kerdock_addrs % four_to_deg

            b_coeffs = torch.zeros((B, self.gr.deg), device=device, dtype=torch.long)
            for i in range(self.gr.deg):
                b_coeffs[:, i] = b_indices % 4
                b_indices = b_indices // 4

            traces = torch.matmul(b_coeffs.float(), self._trace_table_z4_torch.float())
            traces = traces.round().long() % 4
            z4_codewords = (a.unsqueeze(1) + traces) % 4

            # Add DG correction: 2*Tr(γ̄₁·ξ̄³)
            # trace_table_f2[3, gamma1_idx, xi_idx] for all xi
            corrections = (
                2 * self._trace_table_f2_torch[3, gamma1_indices, :]
            )  # [B, T_size]
            z4_codewords = (z4_codewords + corrections) % 4

        # Apply Gray map vectorized: Z4 → F2^2
        # Gray map: 0→(0,0), 1→(1,0), 2→(1,1), 3→(0,1)
        gray_map_table = torch.tensor(
            [[0, 0], [0, 1], [1, 1], [1, 0]], device=device, dtype=torch.long
        )
        binary_codewords = gray_map_table[z4_codewords]  # [B, T_size, 2]
        binary_codewords = binary_codewords.reshape(B, 2 * T_size)  # [B, N]

        # Convert to ±1: 0→+1, 1→-1
        signed_vectors = 1.0 - 2.0 * binary_codewords.float()  # [B, N]

        # Normalize to unit norm
        norms = torch.linalg.norm(signed_vectors, dim=1, keepdim=True)  # [B, 1]
        code_vectors = signed_vectors / norms  # [B, N]

        return code_vectors.to(dtype=dtype)

    def _decode_kerdock_address(self, address: int) -> tuple[int, tuple]:
        """Decode address to (a, b) where a ∈ {0,1}, b ∈ GR(4, m-1).

        Restricts a to {0,1} to exclude antipodal pairs. Codewords with
        a and a+2 are exact negatives after Gray map, so we only use a ∈ {0,1}.
        """
        # address = a * 4^(m-1) + b_index, with a ∈ {0,1}
        four_to_deg = 4**self.gr.deg
        a = address // four_to_deg
        b_index = address % four_to_deg

        # Convert b_index to coefficient tuple
        b_coeffs = []
        for _ in range(self.gr.deg):
            b_coeffs.append(b_index % 4)
            b_index //= 4
        return a, tuple(b_coeffs)

    def _decode_dg1_address(self, address: int) -> tuple[int, tuple, int]:
        """Decode address to (a, b, γ₁_idx).

        Uses antipodal-free Kerdock base: a ∈ {0,1}, so kerdock_size = 2 * 4^(m-1).
        """
        kerdock_size = 2 * (4**self.gr.deg)  # a ∈ {0,1} only

        gamma1_idx = address // kerdock_size
        kerdock_addr = address % kerdock_size

        a, b_coeffs = self._decode_kerdock_address(kerdock_addr)
        return a, b_coeffs, gamma1_idx

    def _compute_kerdock_codeword(self, a: int, b_coeffs: tuple) -> np.ndarray:
        """Compute Kerdock Z4 codeword: c(ξ) = a + tr(b·ξ)."""
        T_size = len(self.gr.teichmuller_set)
        z4_codeword = np.zeros(T_size, dtype=np.uint8)

        for xi_idx in range(T_size):
            # Fast trace using precomputed table
            tr_val = 0
            for i in range(self.gr.deg):
                tr_val += b_coeffs[i] * self.gr.trace_table_z4[i, xi_idx]
            tr_val %= 4

            z4_codeword[xi_idx] = (a + tr_val) % 4

        return z4_codeword

    def _compute_dg1_codeword(
        self, a: int, b_coeffs: tuple, gamma1_idx: int
    ) -> np.ndarray:
        """Compute DG(m,1) Z4 codeword with correction term."""
        # Start with Kerdock part
        z4_codeword = self._compute_kerdock_codeword(a, b_coeffs)

        # Add correction: 2*Tr(γ̄₁·ξ̄³)
        T_size = len(self.gr.teichmuller_set)
        for xi_idx in range(T_size):
            # Correction using precomputed F2 trace table (power 3 for DG(m,1))
            tr_f2 = self.gr.trace_table_f2[3, gamma1_idx, xi_idx]
            correction = 2 * tr_f2
            z4_codeword[xi_idx] = (z4_codeword[xi_idx] + correction) % 4

        return z4_codeword

    def _apply_gray_map(self, z4_codeword: np.ndarray) -> np.ndarray:
        """Apply Gray map to Z4 codeword to get binary vector."""
        n = len(z4_codeword)
        binary = np.zeros(2 * n, dtype=np.uint8)
        for i, val in enumerate(z4_codeword):
            b0, b1 = gray_map(int(val))
            binary[2 * i] = b0
            binary[2 * i + 1] = b1
        return binary


def verify_coherence(codes: torch.Tensor) -> tuple[float, float]:
    """Verify mutual coherence of code vectors (for testing).

    Args:
        codes: [N, d] unit-norm code vectors

    Returns:
        (max_coherence, mean_coherence)
    """
    gram = torch.matmul(codes, codes.t())
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    coherences = torch.abs(gram[mask])
    return coherences.max().item(), coherences.mean().item()


def main():
    """Test code generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Z4 code generation")
    parser.add_argument(
        "--m", type=int, default=6, choices=[6, 8], help="Parameter m (6→N=64, 8→N=256)"
    )
    parser.add_argument(
        "--code-type",
        type=str,
        default="kerdock",
        choices=["kerdock", "dg1"],
        help="Code type",
    )
    parser.add_argument(
        "--num-codes",
        type=int,
        default=100,
        help="Number of codes to generate for testing",
    )
    args = parser.parse_args()

    # Create generator
    gen = KerdockDGCodeGenerator(m=args.m, code_type=args.code_type, verbose=True)

    # Generate some codes
    print(f"Generating {args.num_codes} code vectors...")
    codes = []
    for i in range(args.num_codes):
        code = gen.generate_code_vector(i)
        codes.append(code)
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{args.num_codes}...")

    codes = torch.stack(codes)
    print(f"\n✓ Generated {len(codes)} codes")
    print(f"  Shape: {codes.shape}")

    # Verify coherence
    print("\nComputing coherence...")
    max_coh, mean_coh = verify_coherence(codes)
    print(f"  Maximum coherence: {max_coh:.6f}")
    print(f"  Mean coherence: {mean_coh:.6f}")
    print(f"  Theoretical: {gen.coherence:.6f}")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
