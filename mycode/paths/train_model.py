"""Draft script to train API sequence classification"""
# Standard library imports
from pathlib import Path
from collections import Counter
import sys
import logging

# Non-standard imports
import numpy as np
import torch
from torch import LongTensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

# Repository imports
PWD = Path(__file__).parent.absolute()
sys.path.append(PWD.parent.as_posix())
from common.context import PATHS_DATASET_PATH
from preprocessing.text import normalize_path
from preprocessing.array import pad_array, byte_filter, remap
from models import Filepath


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("paths_training")

BENIGN_PATH = PATHS_DATASET_PATH.joinpath("dataset_benign_win10.txt")
MALICIOUS_PATH = PATHS_DATASET_PATH.joinpath("dataset_malicious_augmented.txt")
# Configuration setup
PADDING_LENGTH = 150
N_KEEP_BYTES = 150
EMBEDDING_DIM = 64
N_EPOCHS = 200


# Loading paths data
logger.info("Loading the dataset ...")
paths_data = []
with (open(BENIGN_PATH) as benign_txt, open(MALICIOUS_PATH) as malicious_txt):
    # Add benign file paths (coupled with value 0)
    paths_data.extend([(path, 0) for path in benign_txt.readlines()])
    # Add malicious file paths (coupled with value 1)
    paths_data.extend([(path, 1) for path in malicious_txt.readlines()])

raw_paths, binary_labels = zip(*paths_data) # Splitting paths from labels

# Normalising and UTF-8-encoding the paths
logger.info("Started preprocessing ...")
paths_vectors = []
for path in raw_paths:
    encoded_path = normalize_path(path).encode("utf-8", "ignore")
    utf_vector = np.array(list(encoded_path), dtype=int)
    padded_vector = pad_array(utf_vector, PADDING_LENGTH)
    paths_vectors.append(padded_vector)

paths_vectors = np.vstack(paths_vectors)

# Byte filtering
logger.info("Filtering and remapping paths bytes ...")
byte_counter = Counter(paths_vectors.flatten())
# Select the first column of the byte counts — i.e. the byte value
keep_bytes = np.array(byte_counter.most_common(N_KEEP_BYTES+1))[:, 0]
filtered_paths = byte_filter(paths_vectors, keep_bytes)

# Byte remapping
bytes_set = set([0, 1] + list(keep_bytes))
bytes_map = {byte_value: index
            for index, byte_value
            in enumerate(bytes_set)}
remapped_paths = remap(filtered_paths, bytes_map)

# Prepare training data
X = remapped_paths.copy()
y = np.array(binary_labels, dtype=np.int)


# Start training
logger.warning(f"Model training started for {N_EPOCHS} epochs ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(TensorDataset(LongTensor(X), LongTensor(y)),
                          batch_size = 1024, shuffle=True)
paths_model = Filepath(keep_bytes, device, embedding_dim=EMBEDDING_DIM)
optimizer = Adam(paths_model.model.parameters(), lr=.001, weight_decay=0)
loss_function = CrossEntropyLoss()
paths_model.fit(N_EPOCHS, optimizer, loss_function, train_loader)
