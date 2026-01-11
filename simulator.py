"""
Quantum Source Simulator for TE-SI-QRNG Testing
================================================

Simulates various quantum random number generation sources with
realistic imperfections, drift, and attack scenarios.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    """Types of quantum sources to simulate."""
    IDEAL = "ideal"
    BIASED = "biased"
    DRIFTING = "drifting"
    CORRELATED = "correlated"
    ATTACKED = "attacked"
    PHOTON_COUNTING = "photon_counting"
    PHASE_NOISE = "phase_noise"


@dataclass
class SourceParameters:
    """Parameters for quantum source simulation."""
    source_type: SourceType = SourceType.IDEAL
    bias: float = 0.0  # Bias toward 1 [0, 0.5]
    drift_rate: float = 0.0  # Temporal drift rate
    correlation_length: int = 0  # Memory length
    noise_level: float = 0.0  # Physical noise level
    attack_strength: float = 0.0  # Adversarial manipulation strength
    detector_efficiency: float = 0.95  # Detector efficiency
    dark_count_rate: float = 0.001  # Dark count probability


class QuantumSourceSimulator:
    """
    Simulates a quantum random number source with configurable imperfections.

    Supports multiple source types:
    - Ideal: Perfect quantum randomness
    - Biased: Static bias in output distribution
    - Drifting: Time-varying bias (temperature, aging)
    - Correlated: Memory effects between bits
    - Attacked: Adversarial manipulation
    - Photon counting: Realistic photon detection
    - Phase noise: Homodyne/heterodyne detection
    """

    def __init__(self, params: SourceParameters, seed: Optional[int] = None):
        """
        Args:
            params: Source parameters
            seed: Random seed for reproducibility
        """
        self.params = params
        self.rng = np.random.RandomState(seed)

        # State variables
        self.time_step = 0
        self.current_bias = params.bias
        self.memory_buffer = []

    def generate_block(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a block of quantum random bits with metadata.

        Returns:
            (bits, bases, raw_signal): Quantum bits, measurement bases, raw physical signal
        """
        if self.params.source_type == SourceType.IDEAL:
            return self._generate_ideal(n_bits)
        elif self.params.source_type == SourceType.BIASED:
            return self._generate_biased(n_bits)
        elif self.params.source_type == SourceType.DRIFTING:
            return self._generate_drifting(n_bits)
        elif self.params.source_type == SourceType.CORRELATED:
            return self._generate_correlated(n_bits)
        elif self.params.source_type == SourceType.ATTACKED:
            return self._generate_attacked(n_bits)
        elif self.params.source_type == SourceType.PHOTON_COUNTING:
            return self._generate_photon_counting(n_bits)
        elif self.params.source_type == SourceType.PHASE_NOISE:
            return self._generate_phase_noise(n_bits)
        else:
            return self._generate_ideal(n_bits)

    def _generate_ideal(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ideal quantum random bits."""
        bits = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits)  # Gaussian noise

        self.time_step += n_bits
        return bits, bases, raw_signal

    def _generate_biased(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate biased quantum bits (static bias)."""
        prob_one = 0.5 + self.params.bias

        bits = (self.rng.rand(n_bits) < prob_one).astype(np.uint8)
        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits) + self.params.bias * 2

        self.time_step += n_bits
        return bits, bases, raw_signal

    def _generate_drifting(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate bits with temporal drift in bias."""
        bits = np.zeros(n_bits, dtype=np.uint8)
        raw_signal = np.zeros(n_bits)

        for i in range(n_bits):
            # Update drift (sinusoidal + linear)
            drift = self.params.drift_rate * (
                np.sin(2 * np.pi * self.time_step / 5000) +
                self.time_step / 100000
            )
            current_prob = 0.5 + drift

            # Clip to valid range
            current_prob = np.clip(current_prob, 0.01, 0.99)

            bits[i] = 1 if self.rng.rand() < current_prob else 0
            raw_signal[i] = self.rng.randn() + drift * 2

            self.time_step += 1

        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        return bits, bases, raw_signal

    def _generate_correlated(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate bits with memory/correlation."""
        bits = np.zeros(n_bits, dtype=np.uint8)
        raw_signal = np.zeros(n_bits)

        correlation_strength = min(self.params.correlation_length / 100, 0.9)

        for i in range(n_bits):
            if len(self.memory_buffer) == 0 or self.rng.rand() > correlation_strength:
                # Generate new random bit
                bit = self.rng.randint(0, 2)
            else:
                # Correlated with recent history
                bit = self.memory_buffer[-1]

            bits[i] = bit
            raw_signal[i] = self.rng.randn()

            # Update memory buffer
            self.memory_buffer.append(bit)
            if len(self.memory_buffer) > self.params.correlation_length:
                self.memory_buffer.pop(0)

            self.time_step += 1

        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        return bits, bases, raw_signal

    def _generate_attacked(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate bits under adversarial attack.

        Attack model: Adversary tries to introduce patterns and bias.
        """
        # Start with quantum random bits
        bits = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        # Adversary introduces bias
        attack_mask = self.rng.rand(n_bits) < self.params.attack_strength
        bits[attack_mask] = 1  # Force to 1

        # Adversary introduces patterns (every 100 bits)
        pattern_strength = int(self.params.attack_strength * 10)
        for i in range(0, n_bits, 100):
            if i + pattern_strength < n_bits:
                bits[i:i+pattern_strength] = 1  # Inject pattern

        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)
        raw_signal = self.rng.randn(n_bits) + self.params.attack_strength

        self.time_step += n_bits
        return bits, bases, raw_signal

    def _generate_photon_counting(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate photon counting QRNG.

        Single photons in superposition states measured by detectors.
        Realistic model includes:
        - Detector efficiency
        - Dark counts
        - Dead time effects (simplified)
        """
        bits = np.zeros(n_bits, dtype=np.uint8)
        raw_signal = np.zeros(n_bits)  # Photon counts
        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        for i in range(n_bits):
            # Quantum photon in superposition |0> + |1> -> 50/50 detection
            true_bit = self.rng.randint(0, 2)

            # Detector efficiency
            detected = self.rng.rand() < self.params.detector_efficiency

            if detected:
                bits[i] = true_bit
                raw_signal[i] = 1.0  # Photon detected
            else:
                # Dark count
                if self.rng.rand() < self.params.dark_count_rate:
                    bits[i] = self.rng.randint(0, 2)
                    raw_signal[i] = 0.5  # Dark count (distinguishable)
                else:
                    # No detection - assign random
                    bits[i] = self.rng.randint(0, 2)
                    raw_signal[i] = 0.0

        self.time_step += n_bits
        return bits, bases, raw_signal

    def _generate_phase_noise(self, n_bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate phase noise QRNG (homodyne/heterodyne detection).

        Measures quadratures of vacuum state -> Gaussian distribution.
        Bits extracted from sign or other processing.
        """
        # Generate Gaussian random variables (vacuum quadratures)
        x_quadrature = self.rng.randn(n_bits)
        p_quadrature = self.rng.randn(n_bits)

        # Add technical noise
        x_quadrature += self.rng.randn(n_bits) * self.params.noise_level
        p_quadrature += self.rng.randn(n_bits) * self.params.noise_level

        # Measurement basis choice
        bases = self.rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        # Extract bits from quadrature measurements
        bits = np.zeros(n_bits, dtype=np.uint8)
        raw_signal = np.zeros(n_bits)

        for i in range(n_bits):
            if bases[i] == 0:
                # Measure X quadrature
                measured_value = x_quadrature[i]
            else:
                # Measure P quadrature
                measured_value = p_quadrature[i]

            # Extract bit from sign
            bits[i] = 1 if measured_value > 0 else 0
            raw_signal[i] = measured_value

        self.time_step += n_bits
        return bits, bases, raw_signal

    def inject_temporary_fault(self, fault_type: str, duration: int):
        """
        Inject a temporary fault for testing fault detection.

        Args:
            fault_type: Type of fault ('bias', 'drift', 'correlation')
            duration: Duration in number of bits
        """
        # This would modify internal state temporarily
        # Useful for testing self-testing response
        pass

    def reset(self):
        """Reset simulator state."""
        self.time_step = 0
        self.current_bias = self.params.bias
        self.memory_buffer = []


class AttackScenarioSimulator:
    """
    Simulates specific attack scenarios on QRNG.

    Attack types:
    - Source tampering
    - Side-channel injection
    - Basis choice manipulation
    - Post-processing backdoor
    """

    def __init__(self, base_source: QuantumSourceSimulator):
        """
        Args:
            base_source: Underlying quantum source
        """
        self.base_source = base_source

    def source_tampering_attack(self, n_bits: int, tamper_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate adversary tampering with quantum source.

        Adversary replaces fraction of quantum bits with predictable values.
        """
        bits, bases, raw_signal = self.base_source.generate_block(n_bits)

        # Tamper with some bits
        tamper_mask = self.base_source.rng.rand(n_bits) < tamper_rate

        # Adversary sets pattern
        tamper_pattern = (np.arange(n_bits) % 2).astype(np.uint8)
        bits[tamper_mask] = tamper_pattern[tamper_mask]

        return bits, bases, raw_signal

    def side_channel_injection_attack(self, n_bits: int, injection_strength: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate side-channel signal injection.

        Adversary injects electromagnetic signals to bias detector.
        """
        bits, bases, raw_signal = self.base_source.generate_block(n_bits)

        # Inject periodic bias via side channel
        injection_signal = injection_strength * np.sin(2 * np.pi * np.arange(n_bits) / 50)

        # Bias toward 1 when injection is positive
        bias_mask = (injection_signal > 0) & (self.base_source.rng.rand(n_bits) < abs(injection_signal))
        bits[bias_mask] = 1

        # Visible in raw signal
        raw_signal += injection_signal

        return bits, bases, raw_signal


def create_test_scenarios() -> dict:
    """
    Create standard test scenarios for TE-SI-QRNG evaluation.

    Returns:
        Dictionary of scenario name -> SourceParameters
    """
    scenarios = {
        'ideal': SourceParameters(
            source_type=SourceType.IDEAL
        ),
        'small_bias': SourceParameters(
            source_type=SourceType.BIASED,
            bias=0.05
        ),
        'large_bias': SourceParameters(
            source_type=SourceType.BIASED,
            bias=0.2
        ),
        'temporal_drift': SourceParameters(
            source_type=SourceType.DRIFTING,
            drift_rate=0.1
        ),
        'short_correlation': SourceParameters(
            source_type=SourceType.CORRELATED,
            correlation_length=10
        ),
        'long_correlation': SourceParameters(
            source_type=SourceType.CORRELATED,
            correlation_length=100
        ),
        'weak_attack': SourceParameters(
            source_type=SourceType.ATTACKED,
            attack_strength=0.1
        ),
        'strong_attack': SourceParameters(
            source_type=SourceType.ATTACKED,
            attack_strength=0.3
        ),
        'realistic_photon': SourceParameters(
            source_type=SourceType.PHOTON_COUNTING,
            detector_efficiency=0.9,
            dark_count_rate=0.001
        ),
        'degraded_photon': SourceParameters(
            source_type=SourceType.PHOTON_COUNTING,
            detector_efficiency=0.7,
            dark_count_rate=0.01
        ),
        'phase_noise_clean': SourceParameters(
            source_type=SourceType.PHASE_NOISE,
            noise_level=0.1
        ),
        'phase_noise_noisy': SourceParameters(
            source_type=SourceType.PHASE_NOISE,
            noise_level=0.5
        ),
    }

    return scenarios


if __name__ == "__main__":
    print("Quantum Source Simulator for TE-SI-QRNG")
    print("=" * 80)

    # Test scenarios
    scenarios = create_test_scenarios()

    print(f"\nAvailable test scenarios: {len(scenarios)}")
    for name, params in scenarios.items():
        print(f"  - {name}: {params.source_type.value}")

    # Example: Generate from ideal source
    print("\nExample: Ideal source")
    ideal_source = QuantumSourceSimulator(scenarios['ideal'], seed=42)
    bits, bases, signal = ideal_source.generate_block(1000)
    print(f"  Generated {len(bits)} bits")
    print(f"  Mean: {np.mean(bits):.4f} (expected ~0.5)")
    print(f"  Std: {np.std(bits.astype(float)):.4f} (expected ~0.5)")

    # Example: Biased source
    print("\nExample: Biased source (bias=0.2)")
    biased_source = QuantumSourceSimulator(scenarios['large_bias'], seed=42)
    bits, bases, signal = biased_source.generate_block(10000)
    print(f"  Generated {len(bits)} bits")
    print(f"  Mean: {np.mean(bits):.4f} (expected ~0.7)")
    print(f"  Bias detected: {abs(np.mean(bits) - 0.5):.4f}")
