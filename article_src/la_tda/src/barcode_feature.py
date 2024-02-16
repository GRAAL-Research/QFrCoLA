import concurrent.futures
import itertools
import json
from collections import defaultdict

import numpy as np
import ripserplusplus as rpp
from tqdm import tqdm


###################################
# RIPSER FEATURE CALCULATION FORMAT
###################################
# Format: "h{dim}\_{type}\_{args}"

# Dimension: 0, 1, etc.; homology dimension

# Types:

#     1. s: sum of lengths; example: "h1_s".
#     2. m: mean of lengths; example: "h1_m"
#     3. v: variance of lengths; example "h1_v"
#     4. e: entropy of persistence diagram.
#     2. n: number of barcodes with time of birth/death more/less then threshold.
#         2.1. b/d: birth or death
#         2.2. m/l: more or less than threshold
#         2.2. t: threshold value
#        example: "h0_n_d_m_t0.5", "h1_n_b_l_t0.75"
#     3. t: time of birth/death of the longest barcode (not incl. inf).
#         3.1. b/d: birth of death
#             example: "h0_t_d", "h1_t_b"

####################################


def barcode_pop_inf(barcode):
    """Delete all infinite barcodes"""
    for dim in barcode:
        if len(barcode[dim]):
            barcode[dim] = barcode[dim][barcode[dim]["death"] != np.inf]
    return barcode


def barcode_number(barcode, dim=0, bd="death", ml="m", t=0.5):
    """Calculate number of barcodes in h{dim} with time of birth/death more/less then threshold"""
    if len(barcode[dim]):
        if ml == "m":
            return np.sum(barcode[dim][bd] >= t)
        elif ml == "l":
            return np.sum(barcode[dim][bd] <= t)
        else:
            raise Exception("Wrong more/less type in barcode_number calculation")
    else:
        return 0.0


def barcode_time(barcode, dim=0, bd="birth"):
    """Calculate time of birth/death in h{dim} of longest barcode"""
    if len(barcode[dim]):
        max_len_idx = np.argmax(barcode[dim]["death"] - barcode[dim]["birth"])
        return barcode[dim][bd][max_len_idx]
    else:
        return 0.0


def barcode_number_of_barcodes(barcode, dim=0):
    return len(barcode[dim])


def barcode_entropy(barcode, dim=0):
    if len(barcode[dim]):
        lengths = barcode[dim]["death"] - barcode[dim]["birth"]
        lengths /= np.sum(lengths)
        return -np.sum(lengths * np.log(lengths))
    else:
        return 0.0


# def barcode_lengths(barcode, dim=0):
#     return barcode[dim]['death'] - barcode[dim]['birth']


def barcode_sum(barcode, dim=0):
    """Calculate sum of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.sum(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def barcode_mean(barcode, dim=0):
    """Calculate mean of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.mean(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def barcode_std(barcode, dim=0):
    """Calculate std of lengths of barcodes in h{dim}"""
    if len(barcode[dim]):
        return np.std(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def count_ripser_features(barcodes, feature_list=["h0_m"]):
    """Calculate all provided ripser features"""
    # first pop all infs from barcodes
    barcodes = [barcode_pop_inf(barcode) for barcode in barcodes]
    # calculate features
    features = []
    for feature in feature_list:
        feature = feature.split("_")
        # dimension, feature type and args
        dim, ftype, fargs = int(feature[0][1:]), feature[1], feature[2:]
        if ftype == "s":
            feat = [barcode_sum(barcode, dim) for barcode in barcodes]
        elif ftype == "m":
            feat = [barcode_mean(barcode, dim) for barcode in barcodes]
        elif ftype == "v":
            feat = [barcode_std(barcode, dim) for barcode in barcodes]
        elif ftype == "n":
            bd, ml, t = fargs[0], fargs[1], float(fargs[2][1:])
            if bd == "b":
                bd = "birth"
            elif bd == "d":
                bd = "death"
            feat = [barcode_number(barcode, dim, bd, ml, t) for barcode in barcodes]
        elif ftype == "t":
            bd = fargs[0]
            if bd == "b":
                bd = "birth"
            elif bd == "d":
                bd = "death"
            feat = [barcode_time(barcode, dim, bd) for barcode in barcodes]
        elif ftype == "nb":
            feat = [barcode_number_of_barcodes(barcode, dim) for barcode in barcodes]
        elif ftype == "e":
            feat = [barcode_entropy(barcode, dim) for barcode in barcodes]
        features.append(feat)
    return np.swapaxes(np.array(features), 0, 1)  # samples X n_features


def matrix_to_ripser(matrix, ntokens, lower_bound=0.0):
    """Convert matrix to appropriate ripser++ format"""
    matrix = cutoff_matrix(matrix, ntokens)
    matrix = (matrix > lower_bound).astype(int) * matrix
    matrix = 1.0 - matrix
    matrix -= np.diag(np.diag(matrix))  # 0 on diagonal
    matrix = np.minimum(
        matrix.T, matrix
    )  # symmetrical, edge emerges if at least one direction is working
    return matrix


def run_ripser_on_matrix(matrix, dim):
    barcode = rpp.run(f"--format distance --dim {dim}", data=matrix)
    return barcode


def get_barcodes(matricies, ntokens_array, dim=1, lower_bound=0.0):
    """Get barcodes from matrix"""
    barcodes = []

    for i, matrix in enumerate(matricies):
        matrix = matrix_to_ripser(matrix, ntokens_array[i], lower_bound)
        barcode = run_ripser_on_matrix(matrix, dim)
        barcodes.append(barcode)
    return barcodes


def cutoff_matrix(matrix, ntokens):
    """Return normalized submatrix of first n_tokens"""
    matrix = matrix[:ntokens, :ntokens]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound, verbose=False):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    iter = itertools.product(layers, heads)
    iter = list(iter)
    if verbose:
        iter = tqdm(iter, "Layer, Head", leave=False, total=len(iter))
    for layer, head in iter:
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = get_barcodes(
            matricies, ntokens_array, dim, lower_bound
        )
    return barcodes


def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]


def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, "w"))


def split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits):
    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits)
    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
    return splitted


def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b, d in barcode[dim]],
                dtype=[("birth", "<f4"), ("death", "<f4")],
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes
