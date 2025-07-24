from enum import Enum

class BOEnum(str, Enum):
    """Abstract base for any BO-related enumerations."""
    pass

class CriterionBO(BOEnum):
    EI = "EI"
    LCB = "LCB"
    LW_LCB = "LW-LCB"


class TrainingDataSource(BOEnum):
    COMPUTE = "compute_points"
    LOAD = "load"


class OptimizationMode(BOEnum):
    RUN_OPT = "run_opt"
    LOAD_OPT = "load_opt_points"


class DeviceType(BOEnum):
    CPU = "cpu"
    GPU = "gpu"



class Prior(BOEnum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"


class BOECGParameter(BOEnum):
    INIT_LENGTH = "init_length"
    LENGTH = "length"
    W = "w"
    L_SEGMENT = "l_segment"
    FASCICLES_LENGTH = "fascicles_length"
    FASCICLES_ANGLES = "fascicles_angles"
    BRANCH_ANGLE = "branch_angle"
    ROOT_TIME = "root_time"
    CV = "cv"