"""Compatibility entry point for the YIFE training pipeline.

The canonical implementation is ``src/train/trainer.py``. Keeping this wrapper
prevents the older script from silently using a different preprocessing or
training protocol.
"""

from src.train.trainer import main


if __name__ == "__main__":
    main()
