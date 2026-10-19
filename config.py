"""集中管理数据、输出和固定候选配置。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PUBLIC_MODEL_ROOT = Path("/data0/public/guanhuancong/cell_best_model")
DATA_ROOT = PUBLIC_MODEL_ROOT / "data"

DEFAULT_TABLE_INPUT = DATA_ROOT / "final_table_input.csv"
DEFAULT_PHASE_INPUT = DATA_ROOT / "final_stage_input.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/default_run"

RANDOM_SEED = 20260822
COUNT_MODEL_C_VALUES = (0.05, 0.2, 1.0)

# 已有视频排序分数只允许用以下小权重阶段分数修正。
VIDEO_CANDIDATES = {
    "current": {},
    "global05": {"phase_global_all": 0.05},
    "global10": {"phase_global_all": 0.10},
    "early05": {"phase_global_early": 0.05},
    "early075": {"phase_global_early": 0.075},
    "dynneg025": {"within_dynamics_all": -0.025},
    "dynneg05": {"within_dynamics_all": -0.05},
    "spatial025": {"spatial_early": 0.025},
    "spatial05": {"spatial_early": 0.05},
    "spatial075": {"spatial_early": 0.075},
    "spatial05_dynneg025": {
        "spatial_early": 0.05,
        "within_dynamics_all": -0.025,
    },
    "global05_dynneg025": {
        "phase_global_all": 0.05,
        "within_dynamics_all": -0.025,
    },
}

