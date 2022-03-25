import os
import pandas as pd

def get_filepath_db(PE_DB_PATH="/data/quo.vadis/data/pe.dataset/PeX86Exe"):
    hashpath_DB = pd.DataFrame()
    for root, dirs, _ in os.walk(PE_DB_PATH):
        for name in dirs:
            fullpath = os.path.join(root, name)
            csvdb = os.path.join(fullpath, [x for x in os.listdir(fullpath) if x.endswith(".csv")][0])
            df = pd.read_csv(csvdb, header=None, names=["hash", "path"])
            hashpath_DB = pd.concat([hashpath_DB,df])
    return hashpath_DB


def get_filepath_from_hash(h, db):
    return db[db["hash"] == h].path.values[0]


def get_report_db(REPORT_PATH="/data/quo.vadis/data/emulation.dataset"):
    db = {}
    for root, dirs, _ in os.walk(REPORT_PATH):
        dirs = [x for x in dirs if "report_" in x]
        # if no any "report_" subfolders, take JSON files from REPORT_PATH
        if not any(dirs): 
            dirs = [""]
        for name in dirs:
            fullpath = os.path.join(root, name)
            reportlist = [x for x in os.listdir(fullpath) if x.endswith(".json")]
            for h in reportlist:
                db[h.rstrip(".json")] = os.path.join(fullpath, h)
    return db


def get_rawpe_db(PE_DB_PATH="/data/quo.vadis/data/pe.dataset/PeX86Exe"):
    db = {}
    for root, dirs, _ in os.walk(PE_DB_PATH):
        for name in dirs+[root]:
            if name != root:
                fullpath = os.path.join(root, name)
                files = os.listdir(fullpath)
            else:
                fullpath = root
                files = set(os.listdir(root)) - set(dirs)
            for pehash in files:
                db[pehash] = os.path.join(fullpath, pehash)
    return db