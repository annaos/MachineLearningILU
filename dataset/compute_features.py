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

    def __init__(self):
        self.exist_df = os.path.exists(data_files.DATASET_PATH)
        if self.exist_df:
            self.existed_df = pd.read_csv(data_files.DATASET_PATH)


    def skip(self, item, matrix_id):
        conv = item.conv0 == 0 and item.conv1 == 0
        existed = self.exist_df \
               and matrix_id in self.existed_df.id.unique() \
               and item.ProblemId in self.existed_df.ProblemId.unique()
        return conv or existed


    def get_matrix_info(self, problem_id):
        if (problem_id != problem_id.split('-')[0]):
            matrix_id, size, split = problem_id.split('-')
            matrix_id = int(matrix_id)
        else:
            matrix_id = int(problem_id)
            size = split = None
        return matrix_id, size, split


    def nnz_per_row(self, mtx):
        return np.unique(mtx.nonzero()[0], return_counts=True)[1]


    def chunks_per_row(self, mtx):
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


    def read_matrix(self, path, size, split):
        mtx = mmread(path)
        if (size != None and split != None):
            start = int(split) - 1
            end = start + int(size)
            mtx = coo_matrix(mtx.A[start:end, start:end])
        return mtx


    def p_n_symmetry(self, input_matrix):
        mtx = input_matrix.copy()
        size = mtx.shape
        mtx.setdiag(0)
        nzoffdiag = mtx.count_nonzero()
        if nzoffdiag == 0:
            return 1, 1
        a_csr = mtx.tocsr()
        a_t_csr = a_csr.transpose()
        p_sym = a_csr.multiply(a_t_csr).count_nonzero() / nzoffdiag

        n_sym = size[0] * size[1] - (a_csr - a_t_csr).count_nonzero()
        double_zero = size[0] * size[1] - (a_csr.multiply(a_csr) + a_t_csr.multiply(a_t_csr)).count_nonzero()
        n_sym = (n_sym - double_zero) / nzoffdiag

        return p_sym, n_sym


    def get_features(self, problem_id):
        matrix_id, size, split = self.get_matrix_info(problem_id)
        matrix = ssgetpy.search(matrix_id)[0]
        file_path = matrix.download(extract=True)
        path = file_path[0] + "/" + matrix.name + ".mtx"

        features = {
            "problem_id": problem_id,
            "id": matrix_id,
            "size": size,
            "split": split,
            "path": path,
            "rows": matrix.rows,
            "cols": matrix.cols,
            "nonzeros": matrix.nnz,
            "posdef": matrix.isspd,
            "nsym": matrix.nsym,
            "psym": matrix.psym,
        }

        mtx = self.read_matrix(path, size, split)
        if size != None and split != None:
            features["rows"] = mtx.shape[0]
            features["cols"] = mtx.shape[1]
            features["nonzeros"] = mtx.count_nonzero()
            if matrix.nsym != 1:
                psym, features["nsym"] = self.p_n_symmetry(mtx)
                features["psym"] = psym
        features.update(self.compute_common_features(mtx))
        return features


    def get_feature_df(self, label_df):
        feature_dict = dict()

        for i, item in label_df.iterrows():
            problem_id = item.ProblemId
            if (i + 1) % 100 == 0:
                print(f'processing matrix #{i + 1} {problem_id}')

            matrix_id, size, split = self.get_matrix_info(problem_id)
            if self.skip(item, matrix_id):
                continue
            feature_dict[problem_id] = self.get_features(problem_id)

        return pd.DataFrame(data=feature_dict).T


    def compute_common_features(self, mtx):
        features = dict()
        features["density"] = mtx.count_nonzero() / (mtx.shape[0] * mtx.shape[1])
        features["avg_nnz"] = mtx.count_nonzero() / mtx.shape[0]
        features["max_nnz"] = int(self.nnz_per_row(mtx).max())
        features["std_nnz"] = np.std(self.nnz_per_row(mtx))
        chunks,chunk_sizes = self.chunks_per_row(mtx)
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
        if self.exist_df:
            dataset = pd.concat([dataset, self.existed_df])
        return dataset