from pathlib import Path
import sys

PWD = Path(__file__).parent.absolute()
ROOT_PATH = PWD.parent.parent
sys.path.append(ROOT_PATH.as_posix())

TRAIN_CONF_PATH = PWD.parent.joinpath("training_conf.yml")
TRAINING_DATASET_PATH = ROOT_PATH.joinpath("data/train_val_test_sets")
EMULATION_DATASET_PATH = ROOT_PATH.joinpath("data/emulation.dataset")
PATHS_DATASET_PATH = ROOT_PATH.joinpath("data/path.dataset")