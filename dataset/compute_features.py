import ssgetpy
from scipy.io import mmread
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import more_itertools as mit
from collections import defaultdict
import data_files
import os

class ComputeFeatures:

    def get_meta_dict(df):
        exist_df = os.path.exists(data_files.DATASET_PATH)
        if exist_df:
            existed_df = pd.read_csv(data_files.DATASET_PATH)
        meta_dict = dict()
        for i in range(len(df)):
            problem_id = df.ProblemId[i]
            if (problem_id != problem_id.split('-')[0]):
                matrix_id, size, split = problem_id.split('-')
                matrix_id = int(matrix_id)
            else:
                matrix_id = int(problem_id)
                size = split = None
            if (df.conv0[i] == 0 and df.conv1[i] == 0):
                continue
            if (exist_df and matrix_id in existed_df.id.unique() and problem_id in existed_df.ProblemId.unique()):
                continue
            matrix = ssgetpy.search(matrix_id)[0]
            file_path = matrix.download(extract=True)
            meta_dict[problem_id] = {
                "problem_id": problem_id,
                "id": matrix_id,
                "size": size,
                "split": split,
                "path": file_path[0] + "/" + matrix.name + ".mtx",
                "rows": matrix.rows,
                "cols": matrix.cols,
                "nonzeros": matrix.nnz,
                "posdef": matrix.isspd,
                "nsym": matrix.nsym,
                "psym": matrix.psym,
            }

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

    def read_matrix(path, size, split):
        mtx = mmread(path)
        if (size != None and split != None):
            start = int(split) - 1
            end = start + int(size)
            mtx = coo_matrix(mtx.A[start:end, start:end])
        return mtx


    def p_n_symmetry(mtx):
        size = mtx.shape
        mtx.setdiag(0)
        nzoffdiag = mtx.count_nonzero()
        a_csr = mtx.tocsr()
        a_t_csr = a_csr.transpose()
        p_sym = a_csr.multiply(a_t_csr).count_nonzero() / nzoffdiag

        n_sym = size[0] * size[1] - (a_csr - a_t_csr).count_nonzero()
        double_zero = size[0] * size[1] - (a_csr.multiply(a_csr) + a_t_csr.multiply(a_t_csr)).count_nonzero()
        n_sym = (n_sym - double_zero) / nzoffdiag
        if (n_sym == 1):
            posdef = 1
        else:
            posdef = 0

        return p_sym, n_sym, posdef


    def get_feature_df(self, label_df):
        feature_dict = ComputeFeatures.get_meta_dict(label_df)
        for key, meta in feature_dict.items():
            print(f'reading matrix {meta["path"]}')
            mtx = ComputeFeatures.read_matrix(meta["path"], meta["size"], meta["split"])
            if (meta["size"] != None and meta["split"] != None):
                feature_dict[key]["rows"] = mtx.shape[0]
                feature_dict[key]["cols"] = mtx.shape[1]
                feature_dict[key]["nonzeros"] = mtx.getnnz()
                if meta["nsym"] != 1:
                    psym, feature_dict[key]["nsym"] = ComputeFeatures.p_n_symmetry(mtx)
                    feature_dict[key]["psym"] = max(meta["psym"], psym)

            feature_dict[key].update(ComputeFeatures.compute_common_features(mtx))

        return pd.DataFrame(data=feature_dict).T

    def compute_common_features(mtx):
        features = dict()
        features["density"] = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
        features["avg_nnz"] = mtx.getnnz() / mtx.shape[0]
        features["max_nnz"] = int(ComputeFeatures.nnz_per_row(mtx).max())
        features["std_nnz"] = np.std(ComputeFeatures.nnz_per_row(mtx))
        chunks,chunk_sizes = ComputeFeatures.chunks_per_row(mtx)
        features["avg_row_block_count"] = np.mean(chunks)
        features["std_row_block_count"] = np.std(chunks)
        features["min_row_block_count"] = np.min(chunks)
        features["max_row_block_count"] = np.max(chunks)
        features["avg_row_block_size"] = np.mean(chunk_sizes)
        features["std_row_block_size"] = np.std(chunk_sizes)
        features["min_row_block_size"] = np.min(chunk_sizes)
        features["max_row_block_size"] = np.max(chunk_sizes)
        features["block_count"] = np.sum(chunks)

        return features

    def merge(self, label_df, feature_df):
        dataset = label_df.merge(feature_df, left_on="ProblemId", right_on="problem_id")
        if os.path.exists(data_files.DATASET_PATH):
            existed_df = pd.read_csv(data_files.DATASET_PATH)
            dataset = pd.concat([dataset, existed_df])
        return dataset