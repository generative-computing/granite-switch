#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Kerdock and Delsarte-Goethals codes via Z_4-linear codes
and the Gray map, following the construction in z4_kerdock_dg_construction.tex

This implementation generates:
- Kerdock code K(m) = DG(m,0)
- Delsarte-Goethals code DG(m,1)

For even m >= 4, using Galois ring arithmetic over GR(4, m-1).

Author: Implementation based on Hammons et al. 1994
"""

import argparse
from pathlib import Path

import numpy as np


class GaloisRing:
    """
    Galois Ring GR(4, m-1) = Z_4[X] / (h(X))
    where h(X) is a Hensel lift of a primitive polynomial over F_2.
    """

    def __init__(self, m: int):
        """
        Initialize GR(4, m-1) for even m >= 4.

        Args:
            m: Even integer >= 4. Binary code length will be 2^m.
        """
        if m % 2 != 0 or m < 4:
            raise ValueError("m must be even and >= 4")

        self.m = m
        self.deg = m - 1  # Degree of the Galois ring
        self.n = (1 << self.deg) - 1  # 2^(m-1) - 1
        self.N = 1 << m  # Binary code length

        # Get primitive polynomial over F_2
        self.h2_coeffs = self._get_primitive_polynomial_f2(self.deg)

        # Hensel lift to Z_4
        self.h_coeffs = self._hensel_lift(self.h2_coeffs)

        # Build Teichmüller set
        self.teichmuller_set = self._build_teichmuller_set()

        # Precompute trace table for fast codeword generation
        self.trace_table_z4 = self._build_trace_table_z4()
        self.trace_table_f2 = self._build_trace_table_f2()

    def _get_primitive_polynomial_f2(self, deg: int) -> np.ndarray:
        """
        Get a primitive polynomial of given degree over F_2.
        Returns coefficients [a_0, a_1, ..., a_deg] where a_deg = 1.
        """
        # Table of primitive polynomials (binary representation)
        # Polynomial sum_{i} a_i X^i is encoded as sum_{i} a_i * 2^i
        primitive_polys = {
            3: 0b1011,  # X^3 + X + 1
            5: 0b100101,  # X^5 + X^2 + 1
            7: 0b10001001,  # X^7 + X^3 + 1
            9: 0b1000010001,  # X^9 + X^4 + 1
        }

        if deg not in primitive_polys:
            raise NotImplementedError(
                f"Primitive polynomial for degree {deg} not implemented"
            )

        poly_int = primitive_polys[deg]
        coeffs = np.zeros(deg + 1, dtype=np.uint8)
        for i in range(deg + 1):
            coeffs[i] = (poly_int >> i) & 1
        return coeffs

    def _poly_str(self, coeffs: np.ndarray, modulus: int) -> str:
        """Pretty print polynomial."""
        terms = []
        for i, c in enumerate(coeffs):
            if c != 0:
                if i == 0:
                    terms.append(f"{c % modulus}")
                elif i == 1:
                    if c == 1:
                        terms.append("X")
                    else:
                        terms.append(f"{c % modulus}*X")
                else:
                    if c == 1:
                        terms.append(f"X^{i}")
                    else:
                        terms.append(f"{c % modulus}*X^{i}")
        return " + ".join(terms) if terms else "0"

    def _poly_add_f2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add polynomials over F_2 (XOR)."""
        max_len = max(len(a), len(b))
        result = np.zeros(max_len, dtype=np.uint8)
        result[: len(a)] = a
        result[: len(b)] ^= b
        return result

    def _poly_mult_f2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply polynomials over F_2."""
        result = np.zeros(len(a) + len(b) - 1, dtype=np.uint8)
        for i in range(len(a)):
            if a[i]:
                for j in range(len(b)):
                    result[i + j] ^= (a[i] * b[j]) & 1
        return result

    def _poly_mod_f2(self, a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Polynomial modulo over F_2."""
        a = a.copy()
        m_deg = len(m) - 1
        while np.count_nonzero(a) > 0 and len(a) > m_deg:
            if a[-1] == 0:
                a = a[:-1]
                continue
            # Subtract (XOR) m * X^(len(a) - len(m))
            shift = len(a) - len(m)
            for i in range(len(m)):
                a[shift + i] ^= m[i]
            a = a[:-1]
        return a if len(a) > 0 else np.array([0], dtype=np.uint8)

    def _poly_add_z4(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add polynomials over Z_4."""
        max_len = max(len(a), len(b))
        result = np.zeros(max_len, dtype=np.uint8)
        result[: len(a)] = a
        result[: len(b)] = (result[: len(b)] + b) % 4
        return result

    def _poly_mult_z4(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply polynomials over Z_4."""
        result = np.zeros(len(a) + len(b) - 1, dtype=np.uint8)
        for i in range(len(a)):
            if a[i]:
                for j in range(len(b)):
                    result[i + j] = (result[i + j] + a[i] * b[j]) % 4
        return result

    def _poly_mod_z4(self, a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Polynomial modulo over Z_4."""
        a = a.copy()
        m_deg = len(m) - 1

        # Trim leading zeros
        while len(a) > 1 and a[-1] == 0:
            a = a[:-1]

        while len(a) > m_deg:
            # Get leading coefficient
            lead = a[-1]
            if lead == 0:
                a = a[:-1]
                continue

            # Need to divide by leading coefficient of m (which is 1)
            # lead * X^shift needs to be eliminated
            shift = len(a) - len(m)

            # Subtract lead * m * X^shift
            for i in range(len(m)):
                # Cast to int to avoid uint8 overflow warnings
                a[shift + i] = (int(a[shift + i]) - int(lead * m[i])) % 4

            # Trim the leading term
            a = a[:-1]

            # Trim any new leading zeros
            while len(a) > 1 and a[-1] == 0:
                a = a[:-1]

        return a if len(a) > 0 else np.array([0], dtype=np.uint8)

    def _hensel_lift(self, h2: np.ndarray) -> np.ndarray:
        """
        Hensel lift primitive polynomial from F_2 to Z_4.
        Returns h in Z_4[X] such that:
        - h ≡ h2 (mod 2)
        - h | X^n - 1 in Z_4[X]

        For known small cases, use pre-computed lifts.
        """
        # For small degrees, use known Hensel lifts
        # These can be computed using computer algebra systems like SageMath

        known_lifts = {
            # Degree 3: X^3 + X + 1 over F_2 -> 3 + X + 2X^2 + X^3 over Z_4
            3: {
                (1, 1, 0, 1): np.array([3, 1, 2, 1], dtype=np.uint8),
            },
            # Degree 5: X^5 + X^2 + 1 over F_2 -> 3 + 2X + 3X^2 + X^5 over Z_4
            5: {
                (1, 0, 1, 0, 0, 1): np.array([3, 2, 3, 0, 0, 1], dtype=np.uint8),
            },
            # Degree 7: X^7 + X^3 + 1 over F_2 -> 3 + X^3 + 2X^5 + X^7 over Z_4
            7: {
                (1, 0, 0, 1, 0, 0, 0, 1): np.array(
                    [3, 0, 0, 1, 0, 2, 0, 1], dtype=np.uint8
                ),
            },
        }

        h2_tuple = tuple(h2)
        deg = len(h2) - 1

        if deg in known_lifts and h2_tuple in known_lifts[deg]:
            return known_lifts[deg][h2_tuple]

        # Otherwise, try the natural lift and verify
        h_tilde = h2.copy().astype(np.uint8)

        # Verify that h_tilde divides X^n - 1
        x = np.array([0, 1], dtype=np.uint8)  # X
        x_power = np.array([1], dtype=np.uint8)  # X^0 = 1

        # Compute X^n mod h_tilde using binary exponentiation
        n = self.n
        base = x.copy()
        while n > 0:
            if n & 1:
                x_power = self._poly_mult_z4(x_power, base)
                x_power = self._poly_mod_z4(x_power, h_tilde)
            base = self._poly_mult_z4(base, base)
            base = self._poly_mod_z4(base, h_tilde)
            n >>= 1

        # Check if X^n ≡ 1 (mod h_tilde)
        is_one = len(x_power) == 1 and x_power[0] == 1
        if not is_one:
            import warnings

            warnings.warn(
                f"Natural lift does not satisfy X^{self.n} ≡ 1 (mod h); "
                f"got X^{self.n} ≡ {x_power} (mod h). "
                f"Code may have suboptimal properties."
            )
        return h_tilde

    def _build_teichmuller_set(self) -> list[np.ndarray]:
        """
        Build Teichmüller set T = {0, 1, ζ, ζ^2, ..., ζ^(n-1)}
        where ζ = X mod h(X).
        """
        T = []

        # Add 0
        T.append(np.array([0], dtype=np.uint8))

        # Add 1
        T.append(np.array([1], dtype=np.uint8))

        # Add powers of ζ = X
        zeta = np.array([0, 1], dtype=np.uint8)  # X
        current = zeta.copy()

        for i in range(2, self.n + 1):
            T.append(current.copy())
            if i < self.n:
                # Multiply by zeta
                current = self._poly_mult_z4(current, zeta)
                current = self._poly_mod_z4(current, self.h_coeffs)

        return T

    def frobenius(self, a: np.ndarray) -> np.ndarray:
        """
        Apply Frobenius automorphism σ: a -> a^2 in the Galois ring.
        For Teichmüller elements, σ(t) = t^2.
        """
        # Square the element
        a_squared = self._poly_mult_z4(a, a)
        return self._poly_mod_z4(a_squared, self.h_coeffs)

    def trace_z4(self, a: np.ndarray) -> int:
        """
        Galois ring trace: tr(a) = sum_{i=0}^{m-2} σ^i(a) in Z_4.
        """
        result = np.zeros(max(1, len(a)), dtype=np.uint8)
        current = a.copy()

        for i in range(self.deg):
            # Add current to result
            result = self._poly_add_z4(result, current)
            # Apply Frobenius
            current = self.frobenius(current)

        # The trace maps to Z_4, so we return the constant term
        return int(result[0] % 4)

    def _build_trace_table_z4(self) -> np.ndarray:
        """
        Precompute trace table for Z_4 trace.
        trace_table[i][j] = tr(X^i · T[j]) where T is Teichmüller set.

        This allows fast computation: tr(b·ξ) = sum_i b[i] · trace_table[i][ξ_idx] mod 4
        where b = b_0 + b_1·X + ... + b_{deg-1}·X^{deg-1}
        """
        T_size = len(self.teichmuller_set)
        trace_table = np.zeros((self.deg, T_size), dtype=np.uint8)

        # Precompute powers of X: X^0, X^1, ..., X^{deg-1}
        x_powers = []
        for i in range(self.deg):
            x_power = np.zeros(i + 1, dtype=np.uint8)
            x_power[i] = 1  # X^i
            x_powers.append(x_power)

        for i in range(self.deg):
            for j, xi in enumerate(self.teichmuller_set):
                # Compute X^i · xi
                product = self._poly_mult_z4(x_powers[i], xi)
                product = self._poly_mod_z4(product, self.h_coeffs)

                # Compute trace
                trace_table[i, j] = self.trace_z4(product)

        return trace_table

    def _build_trace_table_f2(self) -> np.ndarray:
        """
        Precompute trace table for F_2 field trace.
        trace_table_f2[power][gamma_idx][xi_idx] = Tr(γ · ξ^power)
        where γ, ξ are Teichmüller elements reduced mod 2.

        Used for DG(m,r) correction terms.
        """
        T_size = len(self.teichmuller_set)

        # For DG codes, we need powers 3, 5, 7, ... up to m-1
        # For now, precompute for powers up to 2*m (covers all DG levels)
        max_power = 2 * self.m
        trace_table_f2 = np.zeros((max_power, T_size, T_size), dtype=np.uint8)

        for power in range(max_power):
            for gamma_idx, gamma in enumerate(self.teichmuller_set):
                gamma_bar = self.reduce_mod_2(gamma)
                for xi_idx, xi in enumerate(self.teichmuller_set):
                    xi_bar = self.reduce_mod_2(xi)

                    # Compute xi_bar^power
                    xi_bar_power = self.poly_power_f2(xi_bar, power)

                    # Compute gamma_bar * xi_bar^power
                    product = self._poly_mult_f2(gamma_bar, xi_bar_power)
                    product = self._poly_mod_f2(product, self.h2_coeffs)

                    # Field trace
                    trace_table_f2[power, gamma_idx, xi_idx] = self.trace_f2(product)

        return trace_table_f2

    def reduce_mod_2(self, a: np.ndarray) -> np.ndarray:
        """Reduce polynomial coefficients mod 2 (project to F_{2^(m-1)})."""
        return a % 2

    def trace_f2(self, a_bar: np.ndarray) -> int:
        """
        Field trace: Tr: F_{2^(m-1)} -> F_2.
        Tr(a) = sum_{i=0}^{m-2} a^{2^i} over F_2.
        """
        result = np.zeros(max(1, len(a_bar)), dtype=np.uint8)
        current = a_bar.copy()

        for i in range(self.deg):
            # Add current to result (XOR for F_2)
            result = self._poly_add_f2(result, current)
            # Apply Frobenius (square and reduce mod h2)
            current = self._poly_mult_f2(current, current)
            current = self._poly_mod_f2(current, self.h2_coeffs)

        # Return constant term
        return int(result[0] % 2)

    def poly_power_f2(self, a: np.ndarray, exp: int) -> np.ndarray:
        """
        Compute a^exp in F_2[X]/(h2(X)) using binary exponentiation.
        """
        if exp == 0:
            return np.array([1], dtype=np.uint8)

        result = np.array([1], dtype=np.uint8)
        base = a.copy()

        while exp > 0:
            if exp & 1:
                result = self._poly_mult_f2(result, base)
                result = self._poly_mod_f2(result, self.h2_coeffs)
            base = self._poly_mult_f2(base, base)
            base = self._poly_mod_f2(base, self.h2_coeffs)
            exp >>= 1

        return result


def gray_map(z4_value: int) -> tuple[int, int]:
    """
    Gray map: Z_4 -> F_2^2
    0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
    Returns (bit0, bit1) in F_2.
    """
    z4_value = z4_value % 4
    if z4_value == 0:
        return (0, 0)
    elif z4_value == 1:
        return (0, 1)
    elif z4_value == 2:
        return (1, 1)
    else:  # z4_value == 3
        return (1, 0)


def gray_map_vector(z4_vector: np.ndarray) -> np.ndarray:
    """
    Apply Gray map to a vector over Z_4.
    Input: length-n vector over Z_4
    Output: length-2n binary vector over F_2
    """
    n = len(z4_vector)
    binary = np.zeros(2 * n, dtype=np.uint8)
    for i, val in enumerate(z4_vector):
        b0, b1 = gray_map(int(val))
        binary[2 * i] = b0
        binary[2 * i + 1] = b1
    return binary


def generate_kerdock(gr: GaloisRing) -> tuple[np.ndarray, int]:
    """
    Generate all Kerdock codewords K(m) = DG(m,0).

    Returns:
        z4_codewords: Array of shape (num_codewords, 2^(m-1)) over Z_4
        num_codewords: Total number of codewords = 2^(2m)
    """
    print(f"\nGenerating Kerdock code K({gr.m})...")

    T_size = len(gr.teichmuller_set)  # 2^(m-1)
    num_codewords = 4 * (4**gr.deg)  # 4 * 4^(m-1) = 2^(2m)

    z4_codewords = np.zeros((num_codewords, T_size), dtype=np.uint8)

    idx = 0
    # Enumerate all a in Z_4
    for a in range(4):
        # Enumerate all b in GR(4, m-1)
        # We enumerate by coefficient vectors: b = sum b_i * X^i
        for b_coeffs in np.ndindex(*([4] * gr.deg)):
            # Compute codeword: c(ξ) = a + tr(b·ξ) for each ξ in T
            # Use precomputed trace table: tr(b·ξ) = sum_i b[i] · tr(X^i · ξ)
            for j in range(T_size):
                # Fast trace computation using precomputed table
                tr_val = 0
                for i in range(gr.deg):
                    tr_val += b_coeffs[i] * gr.trace_table_z4[i, j]
                tr_val %= 4

                # Codeword value
                z4_codewords[idx, j] = (a + tr_val) % 4

            idx += 1

            if idx % 10000 == 0:
                print(f"  Generated {idx}/{num_codewords} codewords...")

    print(f"✓ Generated {num_codewords} Kerdock codewords")
    print(f"  Z_4 length: {T_size}")
    print(f"  Binary length (after Gray map): {2 * T_size} = {gr.N}")

    return z4_codewords, num_codewords


def generate_dg1(gr: GaloisRing) -> tuple[np.ndarray, int]:
    """
    Generate all DG(m,1) codewords.

    Adds half-strength correction: 2*Tr(γ₁ · ξ^3) for γ₁ in T.

    Returns:
        z4_codewords: Array of shape (num_codewords, 2^(m-1)) over Z_4
        num_codewords: Total number of codewords = 2^(3m-1)
    """
    print(f"\nGenerating DG({gr.m},1) code...")

    T_size = len(gr.teichmuller_set)  # 2^(m-1)
    num_codewords = 4 * (4**gr.deg) * T_size  # 2^(3m-1)

    # For large m, this becomes huge. We'll generate in batches.
    print(f"  Total codewords: {num_codewords:,}")
    print(f"  Z_4 length: {T_size}")

    z4_codewords = []
    codeword_count = 0

    # Enumerate all a in Z_4
    for a in range(4):
        print(f"  Processing a={a}/3...")

        # Enumerate all b in GR(4, m-1)
        for b_coeffs in np.ndindex(*([4] * gr.deg)):
            # For each γ₁ in T
            for gamma1_idx in range(T_size):
                # Compute codeword: c(ξ) = a + tr(b·ξ) + 2*Tr(γ₁_bar · ξ_bar^3)
                codeword = np.zeros(T_size, dtype=np.uint8)

                for xi_idx in range(T_size):
                    # Kerdock part: a + tr(b·ξ) using precomputed table
                    tr_val = 0
                    for i in range(gr.deg):
                        tr_val += b_coeffs[i] * gr.trace_table_z4[i, xi_idx]
                    tr_val %= 4
                    base_val = (a + tr_val) % 4

                    # Correction part: 2*Tr(γ₁_bar · ξ_bar^3) using precomputed table
                    # Power 3 for DG(m,1)
                    tr_f2 = gr.trace_table_f2[3, gamma1_idx, xi_idx]

                    # Add correction (multiply by 2 to get value in {0, 2})
                    correction = 2 * tr_f2

                    codeword[xi_idx] = (base_val + correction) % 4

                z4_codewords.append(codeword)
                codeword_count += 1

                if codeword_count % 50000 == 0:
                    print(
                        f"    Generated {codeword_count:,}/{num_codewords:,} codewords..."
                    )

    z4_codewords = np.array(z4_codewords, dtype=np.uint8)

    print(f"✓ Generated {codeword_count:,} DG(m,1) codewords")
    print(f"  Binary length (after Gray map): {2 * T_size} = {gr.N}")

    return z4_codewords, codeword_count


def pack_binary_to_uint8(binary_vectors: np.ndarray) -> np.ndarray:
    """
    Pack binary vectors into uint8 arrays for storage.
    Bit 0 -> +1, Bit 1 -> -1 (for correlation computation).

    Args:
        binary_vectors: Array of shape (n_vectors, n_bits) with values in {0, 1}

    Returns:
        Packed array of shape (n_vectors, n_bytes) where n_bytes = ceil(n_bits / 8)
    """
    n_vectors, n_bits = binary_vectors.shape
    n_bytes = (n_bits + 7) // 8

    packed = np.zeros((n_vectors, n_bytes), dtype=np.uint8)

    for i in range(n_vectors):
        for j in range(n_bits):
            byte_idx = j // 8
            bit_idx = j % 8
            if binary_vectors[i, j]:
                packed[i, byte_idx] |= 1 << bit_idx

    return packed


def save_vectors(vectors: np.ndarray, output_path: Path, code_type: str, m: int):
    """
    Save vectors in the format expected by max_correlation_numba.

    Args:
        vectors: Binary vectors of shape (n_vectors, n_bits)
        output_path: Directory to save files
        code_type: "kerdock" or "dg1"
        m: Parameter m
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Pack vectors
    packed = pack_binary_to_uint8(vectors)

    # Save as single file (or multiple shards for large datasets)
    n_vectors = packed.shape[0]

    if code_type == "kerdock":
        filename = f"kerdock_m{m}_vectors.npy"
    else:  # dg1
        filename = f"dg1_m{m}_vectors.npy"

    filepath = output_path / filename
    np.save(filepath, packed)

    print(f"\n✓ Saved {n_vectors:,} vectors to {filepath}")
    print(f"  Array shape: {packed.shape}")
    print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kerdock and DG(m,1) codes via Z_4 construction"
    )
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Parameter m (must be even, >= 4). Binary length = 2^m",
    )
    parser.add_argument(
        "--code",
        type=str,
        choices=["kerdock", "dg1"],
        required=True,
        help="Code type: 'kerdock' or 'dg1'",
    )
    parser.add_argument(
        "--output", type=str, default="./z4_codes", help="Output directory for vectors"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save vectors to disk (for large codes, this requires significant storage)",
    )

    args = parser.parse_args()

    # Initialize Galois ring
    gr = GaloisRing(args.m)

    # Generate code
    if args.code == "kerdock":
        z4_codewords, num_codewords = generate_kerdock(gr)
    else:  # dg1
        z4_codewords, num_codewords = generate_dg1(gr)

    # Apply Gray map to get binary vectors
    print("\nApplying Gray map...")
    n_codewords = z4_codewords.shape[0]
    binary_vectors = np.zeros((n_codewords, gr.N), dtype=np.uint8)

    for i in range(n_codewords):
        binary_vectors[i] = gray_map_vector(z4_codewords[i])
        if (i + 1) % 10000 == 0:
            print(f"  Mapped {i + 1}/{n_codewords} codewords...")

    print("✓ Applied Gray map to all codewords")
    print(f"  Binary vector shape: {binary_vectors.shape}")

    # Expected coherence
    if args.code == "kerdock":
        expected_coh = 1.0 / np.sqrt(gr.N)
    else:  # dg1
        expected_coh = 2.0 / np.sqrt(gr.N)

    print(f"\nExpected maximum coherence: {expected_coh:.6f}")

    # Save if requested
    if args.save:
        output_path = Path(args.output)
        save_vectors(binary_vectors, output_path, args.code, args.m)

        print("\n" + "=" * 60)
        print("To test coherence, run:")
        print("  python max_correlation_numba.py \\")
        print(f"    --data_dir {output_path} \\")
        print(f"    --m {args.m} \\")
        print(f"    --vector_type {args.code}")
        print("=" * 60)

    return binary_vectors


if __name__ == "__main__":
    main()
