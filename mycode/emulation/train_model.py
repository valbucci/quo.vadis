"""Draft script to train API sequence classification"""
# Standard library imports
from pathlib import Path
from collections import Counter
import sys
import logging

# Non-standard imports
import pandas as pd
import numpy as np
import torch
from torch import LongTensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

# Repository imports
PWD = Path(__file__).parent.absolute()
sys.path.append(PWD.parent.as_posix()) # Include utility modules in the syspath
from common.context import EMULATION_DATASET_PATH
from common.utils import get_api_sequences
from preprocessing.array import rawseq2array
from utils.functions import flatten
from models import Emulation


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("emulation_training")

# Configuration setup
VOCABULARY_SIZE = 600
PADDING_LENGTH = 150
EMBEDDING_DIM = 96
N_EPOCHS = 200

# Load API sequences data
logger.info("Loading the dataset ...")
preloaded_apis_path = PWD.joinpath("api_sequences.pickle")
reports_df = None

if preloaded_apis_path.exists():
    reports_df = pd.read_pickle(preloaded_apis_path)
else:
    reports_df = get_api_sequences(EMULATION_DATASET_PATH)
    reports_df.to_pickle(preloaded_apis_path)
    
# Identify clean/malware classes
reports_df["is_malware"] = reports_df["family"] != "clean"
# Use only relevant data
reports_df = reports_df[["pe_hash", "api_sequence", "is_malware"]]

# Account for hash duplicates (i.e. multiple classes)
reports_df.drop_duplicates("pe_hash", inplace=True)
reports_df.reset_index(drop=True, inplace=True)

# Preprocessing
logger.info("Started preprocessing ...")
api_sequences = reports_df["api_sequence"].values

# Retain only the V most occurring APIs, where V is the vocabulary size  
api_counter = Counter(flatten(api_sequences))
apis_preserved = [x[0] for x in api_counter.most_common(VOCABULARY_SIZE)]
# Encode each API with numeric value
api_map = dict(zip(apis_preserved, range(2, VOCABULARY_SIZE+2)))

# Now encode the dataset
encoded_data = []
for _, (pe_hash, api_sequence, is_malware) in reports_df.iterrows():
    sequence_array = rawseq2array(api_sequence, api_map, PADDING_LENGTH)
    encoded_data.append((pe_hash, sequence_array, int(is_malware)))
    
_, sequence_arrays, binary_labels = zip(*encoded_data)
X = np.array(sequence_arrays)
y = np.array(binary_labels)
del sequence_arrays
del binary_labels

# Start training
logger.warning(f"Model training started for {N_EPOCHS} epochs ...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(TensorDataset(LongTensor(X), LongTensor(y)),
                          batch_size = 1024, shuffle=True)
emulation_model = Emulation(api_map, device, embedding_dim=EMBEDDING_DIM)
optimizer = Adam(emulation_model.model.parameters(), lr=.001, weight_decay=0)
loss_function = CrossEntropyLoss()
emulation_model.fit(N_EPOCHS, optimizer, loss_function, train_loader)
