"""kokoro-sv — Swedish voices for Kokoro-82M, one line to speak.

    from kokoro_sv import SwedishKokoro
    SwedishKokoro().speak("Hej!", voice="Stina", out="hej.wav")
"""
from .core import SwedishKokoro

__version__ = "0.1.1"
__all__ = ["SwedishKokoro"]
