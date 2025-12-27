import os
import random
from typing import Optional

import numpy as np

def set_deterministic(seed: int = 0) -> None:
    """Best-effort determinism across python/random/numpy (and torch if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # harmless on ROCm builds
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not support full determinism; keep best effort.
            pass
    except Exception:
        pass

