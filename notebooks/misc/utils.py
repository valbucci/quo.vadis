"""utils.py 
contains a set of custom utility functions for data import and preprocessing.
"""
# Standard imports
from pathlib import Path
import re
import sys
import logging
import pandas as pd

# Repository imports
REPO_ROOT = Path("../../")
sys.path.append(REPO_ROOT.as_posix())

from preprocessing.reports import report_to_apiseq


SHA256_PATTERN = r'[a-f0-9]{64}'

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