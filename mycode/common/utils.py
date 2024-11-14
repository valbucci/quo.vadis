"""utils.py 
contains a set of custom utility functions for data import and preprocessing.
"""
# Standard imports
from pathlib import Path
from collections import Counter
import re
import sys
import logging
import pickle
import pandas as pd
import numpy as np


# Repository imports
PWD = Path(__file__).parent.absolute()
sys.path.append(PWD.as_posix())
from context import TRAINING_DATASET_PATH
from preprocessing.reports import report_to_apiseq
from preprocessing.array import rawseq2array
from utils.functions import flatten


SHA256_PATTERN = r'[a-f0-9]{64}'


def assert_paths_exist(paths:list[Path]) -> None:
    for path in paths:
        try:
            assert path.exists()
        except AssertionError:
            raise AssertionError(f"Could not find path '{path.abs()}'.")


def get_name_stem(file_path) -> str:
    """
    Get the base name stem from a file path.
        e.g. "path/to/file/stem.ext" -> "stem"

    Args:
        file_path (str|Path): The path, or the file name.

    Returns:
        str: The base name stem, which is the file name excluding the path
            and the extension.
    """
    base_name = Path(file_path).name
    return base_name.split(".")[0]


def get_api_sequences(emulation_dataset_path: Path, 
                      skip_clean=False) -> pd.DataFrame:
    """
    Extract the sequence of API calls and associated family class, given the 
    base path to the emulation dataset folder containing the JSON-formatted 
    reports. Each family class should be named `report_<FAMILY_CLASS>`, where
    <FAMILY_CLASS> could be e.g. "ransomware" or "clean". Furthermore, each
    emulation report should be named as `<SHA256>.json`, where <SHA256> is the
    digest of the corresponding PE binary.

    Args:
        emulation_dataset_path (Path): base path to the emulation dataset. 
        skip_clean (bool, optional): flag for skipping the extraction of API
            calls by benignware. Defaults to False.

    Returns:
        pd.DataFrame: Table containing the extracted information structured as:
            - pe_hash (str): the sha256 digest of the PE binary;
            - family (str): the family class of the executable;
            - api_sequence (list[str]): the sequence of API calls. 
    """
    logger = logging.getLogger("get_api_sequences")

    assert emulation_dataset_path.exists()
    api_sequences = []
    
    # Iterate over every malware family folder
    # (includes benignware in "report_clean")
    for family_reports_dir in emulation_dataset_path.glob("report_*"):
        family_name = family_reports_dir.name.split("_")[-1]
        if skip_clean and family_name == "clean":
            # Skip benignware reports
            logger.info(f"Skipping benignware ...")
            continue
        
        logger.info(f"Getting family '{family_name}' ...")
        
        # Iterate over all JSON files in the family folder
        # (files with extension `.err` are empty)
        for report_path in family_reports_dir.glob("*.json"):
            malware_hash = get_name_stem(report_path)
            if re.match(SHA256_PATTERN, malware_hash) is None:
                # If the filename is not SHA256 it's not an emulation report
                logger.warning(f"Skipping non-SHA256 '{report_path}' ...")
                continue
            
            report_features = report_to_apiseq(report_path)
            api_sequence = tuple(report_features["api.seq"])
            assert report_features["api.seq.len"] == len(api_sequence)

            api_sequences.append((malware_hash, family_name, api_sequence))
        
    return pd.DataFrame(api_sequences, columns=["pe_hash", "family", 
                                                "api_sequence"])
    
    
def preprocess_api_sequences(api_sequences: np.ndarray,
                             vocabulary_size: int=600,
                             padding_length: int=150) -> np.ndarray:
    """
    Encodes sequences of API calls to integer values based on a globally-defined 
    vocabulary size (V) and padding length.
    
    0      -> Reserved for padding.
    1      -> For APIs rarer than V most frequently occurring APIs.
    2..V+2 -> APIs ranked based on their frequency of occurrence, with the most
                  frequent APIs being encoded with lower values.

    Args:
        api_sequences (np.ndarray): Dataset as bi-dimensional array where each 
            row contains the sequence of API calls made by one sample.
        vocabulary_size (int, optional): Amount of APIs to encode with different
            values, starting from the most common ones. Defaults to 600.
        padding_length (int, optional): Size of the input vector, such that each
            API sequence is padded with zeroes or truncated. Defaults to 150.

    Returns:
        np.ndarray: The API sequences encoded as integers within [0, V+2].
    """
    # Retain only the V most occurring APIs, where V is the vocabulary size  
    api_counter = Counter(flatten(api_sequences))
    apis_preserved = [
        x[0] for x in api_counter.most_common(vocabulary_size)]
    
    # Encode each API with numeric value
    api_map = dict(zip(apis_preserved, range(2, vocabulary_size+2)))
    
    return np.vstack([rawseq2array(x, api_map, padding_length)
                      for x in api_sequences])
    
    
def get_train_val_test_datasets(dataset_path: Path=TRAINING_DATASET_PATH)\
    -> tuple[tuple, tuple]:
    """
    Loads training, validation, and test datasets from specified file paths 
    within the given dataset directory. Retrieves datasets split into 
    features (X) and labels (y). 
    
    It expects specific file formats for each dataset:
    - Feature data (X): Stored in `.pickle.set` files
    - Label data (y): Stored in `.arr` files
    
    The function loads feature files using pickle and label files using numpy.

    Args:
        dataset_path (Path, optional):
            The root directory where the dataset files are located. Defaults 
                to `TRAINING_DATASET_PATH`.
    
    Returns:
        tuple[tuple, tuple]:
            A tuple containing two tuples:
                - The first tuple contains the loaded feature datasets.
                - The second tuple contains the loaded label datasets.

            The structure is: `((X_train, X_test, X_val), 
                                (y_train, y_test, y_val))`

    Example Usage:
    ```python
    dataset_path = Path("path/to/dataset")
    (X_data, y_data) = get_train_val_test_datasets(dataset_path)
    X_train, X_test, X_val = X_data
    y_train, y_test, y_val = y_data
    ```
    """
    
    def pickle_load_paths(paths:list[Path]) -> tuple[any]:
        assert_paths_exist(paths)
        loaded_list = []
        for path in paths:
            with open(path, "rb") as file:
                unpickled = pickle.load(file)
                loaded_list.append(unpickled)
                
        return tuple(loaded_list)

    def numpy_load_paths(paths:list[Path]) -> tuple[any]:
        assert_paths_exist(paths)
        loaded_list = []
        for path in paths:
            loaded = np.load(path)
            loaded_list.append(loaded)
                
        return tuple(loaded_list)    
    
    X_train_path = dataset_path.joinpath("X_train.pickle.set")
    X_test_path = dataset_path.joinpath("X_test.pickle.set")
    X_val_path = dataset_path.joinpath("X_val.pickle.set")

    y_train_path = dataset_path.joinpath("y_train.arr")
    y_test_path = dataset_path.joinpath("y_test.arr")
    y_val_path = dataset_path.joinpath("y_val.arr")
    
    return (pickle_load_paths([X_train_path, X_test_path, X_val_path]),
            numpy_load_paths([y_train_path, y_test_path, y_val_path]))