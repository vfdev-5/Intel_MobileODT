
import numpy as np


def repeat(id_type_list, output_size):
    n = int(np.ceil(output_size * 1.0 / len(id_type_list)))
    out = np.tile(id_type_list, [n, 1])
    return out[:output_size]


def generate_trainval_kfolds(id_type_list, n_folds, seed):
    types = (('Type_1', 'AType_1'), ('Type_2', 'AType_2'), ('Type_3', 'AType_3'))
    out = [None, None, None]
    for i, ts in enumerate(types):
        o = id_type_list[(id_type_list[:, 1] == ts[0]) | (id_type_list[:, 1] == ts[1])]
        out[i] = o

    ll = max([len(o) for o in out])
    out = np.array([repeat(o, ll) for o in out])
    out = out.reshape((3 * ll, 2))
    np.random.seed(seed)
    np.random.shuffle(out)

    for val_fold_index in range(n_folds):
        ll = len(out)
        size = int(ll * 1.0 / n_folds + 1.0)
        overlap = (size * n_folds - ll) * 1.0 / (n_folds - 1.0)
        val_start = int(round(val_fold_index * (size - overlap)))
        val_end = val_start + size

        val_id_type_list = out[val_start:val_end]
        train_id_type_list = np.array([
                [i[0], i[1]] for i in out if np.sum(np.prod(i == val_id_type_list, axis=1)) == 0
        ])
        yield train_id_type_list, val_id_type_list
