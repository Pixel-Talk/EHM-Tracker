from dataclasses import dataclass
from .base_config import PrintableConfig

@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    dsize: int = 512  # crop size
    scale: float = 1.5  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    direction: str = 'large-small'  # If detect multiple faces, select strategy, e.g., left-right, right-left, top-bottom, bottom-top, small-large, large-small, distance-from-retarget-face, default: large-small