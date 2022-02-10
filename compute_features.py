import ssgetpy
from scipy.io import mmread
import pandas as pd
import numpy as np
import json
import more_itertools as mit
from collections import defaultdict

DATA_PATH = './data/'
MATRICES_PATH = DATA_PATH + 'matrices.csv'
DATASET_PATH = DATA_PATH + 'dataset.csv'
META_PATH = DATA_PATH + 'matrices_meta.json'


def get_meta_dict():
    meta_dict = dict()
    df = pd.read_csv(MATRICES_PATH)
    for i in range(len(df)):
        matrix_id = int(df.ProblemId[i])
        matrix = ssgetpy.search(matrix_id)[0]
        file_path = matrix.download(extract=True)
        meta_dict[matrix_id] = {
            "path": file_path[0] + "/" + matrix.name + ".mtx",
            "rows": matrix.rows,
            "cols": matrix.cols,
            "nonzeros": matrix.nnz,
            "posdef": matrix.isspd,
            "nsym": matrix.nsym,
            "psym": matrix.psym,
        }

    with open(META_PATH, 'w') as outfile:
        json.dump(meta_dict, outfile, indent=4, sort_keys=True)
        print("Saved collected metas into " + META_PATH)

    return meta_dict


def nnz_per_row(mtx):
    return np.unique(mtx.nonzero()[0], return_counts=True)[1]


def chunks_per_row(mtx):
    chunk_dict = defaultdict(list)
    for x, y in zip(mtx.row, mtx.col):
        chunk_dict[x].append(y)
    chunks = []
    chunksizes = []
    for row in list(chunk_dict.values()):
        groups = [list(group) for group in mit.consecutive_groups(row)]
        chunksizes.extend([len(group) for group in groups])
        chunks.append(len(groups))
    return np.array(chunks), np.array(chunksizes)


def get_feature_df():
    feature_dict = get_meta_dict()
    for key, meta in feature_dict.items():
        if meta["nonzeros"] > 10000000:
            continue
        print(f'reading matrix {meta["path"]}')
        mtx = mmread(meta["path"])
        feature_dict[key]["id"] = key
        density = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
        feature_dict[key]["density"] = density
        feature_dict[key]["avg_nnz"] = mtx.getnnz() / mtx.shape[0]

        feature_dict[key]["max_nnz"] = int(nnz_per_row(mtx).max())
        feature_dict[key]["std_nnz"] = np.std(nnz_per_row(mtx))
        chunks, chunk_sizes = chunks_per_row(mtx)
        feature_dict[key]["avg_row_block_count"] = np.mean(chunks)
        feature_dict[key]["std_row_block_count"] = np.std(chunks)
        feature_dict[key]["min_row_block_count"] = np.min(chunks)
        feature_dict[key]["max_row_block_count"] = np.max(chunks)
        feature_dict[key]["avg_row_block_size"] = np.mean(chunk_sizes)
        feature_dict[key]["std_row_block_size"] = np.std(chunk_sizes)
        feature_dict[key]["min_row_block_size"] = np.min(chunk_sizes)
        feature_dict[key]["max_row_block_size"] = np.max(chunk_sizes)
        feature_dict[key]["block_count"] = np.sum(chunks)

    return pd.DataFrame(data=feature_dict).T
