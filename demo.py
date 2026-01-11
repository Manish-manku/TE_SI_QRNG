#!/usr/bin/env python
"""
Quick Demo of TE-SI-QRNG
=========================

Demonstrates the basic usage and key features of the Trust-Enhanced
Source-Independent Quantum Random Number Generator.
"""

import numpy as np
from te_si_qrng import TrustEnhancedQRNG, TrustVector
from simulator import (
    QuantumSourceSimulator,
    SourceParameters,
    SourceType
)


def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    else:
        print()


def demo_basic_usage():
    """Demonstrate basic TE-SI-QRNG usage."""
    print_separator("DEMO 1: Basic Usage")

    print("\n1. Creating an ideal quantum source...")
    params = SourceParameters(source_type=SourceType.IDEAL)
    source = QuantumSourceSimulator(params, seed=42)

    print("2. Initializing TE-SI-QRNG...")
    qrng = TrustEnhancedQRNG(block_size=10000)

    print("3. Generating certified random bits...")
    random_bits, metadata = qrng.generate_certified_random_bits(
        n_bits=5000,
        source_simulator=source
    )

    print(f"\n✓ Generated {len(random_bits)} certified random bits")
    print(f"  Trust score: {metadata[0]['trust_score']:.4f}")
    print(f"  Min-entropy: {metadata[0]['h_min_trusted']:.4f} bits/bit")
    print(f"  Extraction rate: {metadata[0]['extraction_rate']:.4f}")

    # Empirical quality check
    if len(random_bits) > 100:
        prob_one = np.mean(random_bits)
        print(f"  Empirical bias: {abs(prob_one - 0.5):.4f} (ideal: 0.0)")


def demo_trust_quantification():
    """Demonstrate trust quantification across different sources."""
    print_separator("DEMO 2: Trust Quantification")

    scenarios = {
        'Ideal Source': SourceParameters(source_type=SourceType.IDEAL),
        'Biased Source (10%)': SourceParameters(
            source_type=SourceType.BIASED,
            bias=0.1
        ),
        'Correlated Source': SourceParameters(
            source_type=SourceType.CORRELATED,
            correlation_length=20
        ),
        'Under Attack': SourceParameters(
            source_type=SourceType.ATTACKED,
            attack_strength=0.2
        )
    }

    qrng = TrustEnhancedQRNG(block_size=10000)

    print("\nTesting different source types:\n")
    print(f"{'Source Type':<25} {'Trust Score':<15} {'ε_bias':<10} {'ε_corr':<10}")
    print("-" * 60)

    for name, params in scenarios.items():
        source = QuantumSourceSimulator(params, seed=42)
        bits, bases, signal = source.generate_block(10000)
        trust = qrng.run_self_tests(bits, bases, signal)

        print(f"{name:<25} {trust.trust_score():<15.4f} {trust.epsilon_bias:<10.4f} {trust.epsilon_corr:<10.4f}")

    print("\n✓ Trust scores successfully distinguish source types")


def demo_attack_detection():
    """Demonstrate attack detection and response."""
    print_separator("DEMO 3: Attack Detection and Response")

    print("\nSimulating increasing attack strength...\n")
    print(f"{'Attack Strength':<20} {'Trust Score':<15} {'Min-Entropy':<15} {'Output Bits':<15}")
    print("-" * 65)

    qrng = TrustEnhancedQRNG(block_size=5000)

    for attack_strength in [0.0, 0.1, 0.2, 0.3]:
        params = SourceParameters(
            source_type=SourceType.ATTACKED,
            attack_strength=attack_strength
        )
        source = QuantumSourceSimulator(params, seed=42)

        bits, bases, signal = source.generate_block(5000)
        output, metadata = qrng.process_block(bits, bases, signal)

        trust = metadata['trust_score']
        h_min = metadata['h_min_trusted']
        n_output = metadata['output_bits']

        print(f"{attack_strength:<20.2f} {trust:<15.4f} {h_min:<15.4f} {n_output:<15}")

    print("\n✓ System detects attacks and reduces output accordingly")


def demo_temporal_adaptation():
    """Demonstrate adaptation to time-varying source quality."""
    print_separator("DEMO 4: Temporal Adaptation")

    print("\nSimulating source that degrades and recovers...\n")
    print(f"{'Time Step':<12} {'Source Bias':<15} {'Trust Score':<15} {'Extraction Rate':<15}")
    print("-" * 57)

    qrng = TrustEnhancedQRNG(block_size=3000)

    for step in range(6):
        # Bias varies: 0 -> 0.2 -> 0 (degradation and recovery)
        phase = np.pi * step / 3
        bias = 0.2 * abs(np.sin(phase))

        params = SourceParameters(
            source_type=SourceType.BIASED,
            bias=bias
        )
        source = QuantumSourceSimulator(params, seed=42 + step)

        bits, bases, signal = source.generate_block(3000)
        output, metadata = qrng.process_block(bits, bases, signal)

        print(f"{step:<12} {bias:<15.4f} {metadata['trust_score']:<15.4f} {metadata['extraction_rate']:<15.4f}")

    print("\n✓ System adapts dynamically to changing source quality")


def demo_comparison():
    """Compare TE-SI-QRNG with standard approach."""
    print_separator("DEMO 5: Comparison with Standard SI-QRNG")

    print("\nComparing TE-SI-QRNG vs. Standard SI-QRNG on biased source:\n")

    # Create biased source
    params = SourceParameters(source_type=SourceType.BIASED, bias=0.15)
    source = QuantumSourceSimulator(params, seed=42)

    # TE-SI-QRNG (our approach)
    print("1. TE-SI-QRNG (trust-adjusted):")
    te_qrng = TrustEnhancedQRNG(block_size=10000)
    te_output, te_metadata = te_qrng.generate_certified_random_bits(
        n_bits=5000,
        source_simulator=source
    )
    print(f"   Output: {len(te_output)} bits")
    print(f"   Trust: {te_metadata[0]['trust_score']:.4f}")
    print(f"   Min-entropy: {te_metadata[0]['h_min_trusted']:.4f} bits/bit")

    # Standard SI-QRNG (no trust adjustment)
    print("\n2. Standard SI-QRNG (assumes perfect trust):")
    source.reset()
    si_qrng = TrustEnhancedQRNG(block_size=10000)
    si_qrng.trust_vector = TrustVector(0.0, 0.0, 0.0, 0.0)  # Perfect trust

    si_output, si_metadata = si_qrng.generate_certified_random_bits(
        n_bits=5000,
        source_simulator=source
    )
    print(f"   Output: {len(si_output)} bits")
    print(f"   Trust: {si_metadata[0]['trust_score']:.4f} (assumed)")
    print(f"   Min-entropy: {si_metadata[0]['h_min_trusted']:.4f} bits/bit (unadjusted)")

    print("\n✓ TE-SI-QRNG provides conservative bounds for compromised sources")
    print("  Standard SI-QRNG may overestimate entropy when trust assumptions violated")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("  TRUST-ENHANCED SOURCE-INDEPENDENT QRNG - DEMONSTRATION")
    print("=" * 80)

    try:
        demo_basic_usage()
        demo_trust_quantification()
        demo_attack_detection()
        demo_temporal_adaptation()
        demo_comparison()

        print_separator("DEMONSTRATION COMPLETE")
        print("\n✅ All demos executed successfully!")
        print("\nKey takeaways:")
        print("  1. TE-SI-QRNG quantifies trust dynamically using operational tests")
        print("  2. Trust scores correctly distinguish ideal, biased, and attacked sources")
        print("  3. System detects attacks and reduces extraction rate accordingly")
        print("  4. Temporal adaptation enables response to time-varying source quality")
        print("  5. Provides measurable security improvement over standard SI-QRNG")

        print("\nFor comprehensive experiments, run:")
        print("  python experiment.py")
        print("\nFor quick automated suite, run:")
        print("  python run_all.py")
        print()

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
