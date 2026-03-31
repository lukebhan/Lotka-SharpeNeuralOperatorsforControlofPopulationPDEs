from pathlib import Path


ROOT = Path(__file__).resolve().parent

SEED = 1234
N_GRID = 501
N_FAMILIES = 1000
TARGET_R0_MIN = 1.2

NPZ_PATH = ROOT / "datasets" / "simple_parametric_dataset.npz"
CSV_PATH = ROOT / "datasets" / "simple_parametric_summary.csv"
PLOT_PREFIX = ROOT / "figures" / "simple_param"
SHOW_PLOTS = True

# -------------------------------------------------------------------
# Fertility family
# K(a) = K_base + K_amp * exp(-(a-K_center)^2 / (2 K_sigma^2))
# -------------------------------------------------------------------
K_BASE_MIN = 0.40
K_BASE_MAX = 0.80

K_AMP_MIN = 2.00
K_AMP_MAX = 3.00

K_CENTER_MIN = 0.11
K_CENTER_MAX = 0.35

K_SIGMA_MIN = 0.05
K_SIGMA_MAX = 0.23

# -------------------------------------------------------------------
# Mortality family
# mu(a) = MU_min + MU_juv * exp(-MU_r_juv a) + MU_sen * a^(MU_p_sen)
# -------------------------------------------------------------------
MU_MIN_MIN = 0.03
MU_MIN_MAX = 0.10

MU_JUV_MIN = 0.05
MU_JUV_MAX = 0.19

MU_R_JUV_MIN = 3.5
MU_R_JUV_MAX = 5.5

MU_SEN_MIN = 0.03
MU_SEN_MAX = 0.17

MU_P_SEN_MIN = 1.7
MU_P_SEN_MAX = 2.9

# -------------------------------------------------------------------
# Interaction family
# g(a) = G_OFFSET + G_AMP * exp(-(a-G_center)^2 / (2 G_sigma^2))
# -------------------------------------------------------------------
G_OFFSET_MIN = 0.05
G_OFFSET_MAX = 0.13

G_AMP_MIN = 0.20
G_AMP_MAX = 0.50

G_CENTER_MIN = 0.37
G_CENTER_MAX = 0.63

G_SIGMA_MIN = 0.05
G_SIGMA_MAX = 0.31
