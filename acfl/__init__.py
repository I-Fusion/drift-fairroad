"""
ACFL: Adversarially-Conditioned Federated Learning

Poisoning-resilient FL defense framework for avionics and drone systems.
Integrates with drift-fairroad as the evaluation testbed.
"""

from acfl.config import ACFLConfig
from acfl.defense import ACFLDefense
from acfl.layer3_detection import SpoofingDetector

__version__ = "0.1.0"
__all__ = ["ACFLConfig", "ACFLDefense", "SpoofingDetector"]
