import os
import pickle as pkl
import numpy as np
import scipy.io
from scipy import sparse
import math
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import normalize
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)


def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    # np.asarray(num_label)[i][0]
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred


def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_label(file, variable_name="lables"):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, file)
    data = scipy.io.loadmat(my_file)
    logger.info("loading mat file %s", file)
    sA = sparse.csr_matrix(data[variable_name])
    label = sA.todense().astype(np.int)
    label = np.array(label)
    print(label.shape, type(label), label.min(), label.max())
    return label


def predict_cv_fixed(embedding, label, train_index, test_index, C=1.):
    # this is the function that use the libliner to classify the objects based on the fixed train_indexs and test_indexes

    micro, macro = [], []
    for i in range(0, 10):
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = label[train_index], label[test_index]
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                solver="liblinear",
                multi_class="ovr"),
            n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        # logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    logger.info("%d fold validation, training ratio %f",
                len(micro), len(train_index)/len(train_index))
    logger.info("Average micro %.2f, Average macro %.2f",
                np.mean(micro) * 100,
                np.mean(macro) * 100)
    print(micro)

    return np.mean(micro), np.mean(macro), len(train_index)/len(test_index)*100.0


def predict_cv(embedding, label, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    # this is the function that use the libliner to classify the objects based on the train_ratio
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
                           random_state=random_state)
    for train_index, test_index in shuffle.split(embedding):
        print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == embedding.shape[0]
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = label[train_index], label[test_index]
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                solver="liblinear",
                multi_class="ovr"),
            n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    logger.info("%d fold validation, training ratio %f",
                len(micro), train_ratio)
    logger.info("Average micro %.2f, Average macro %.2f",
                np.mean(micro) * 100,
                np.mean(macro) * 100)
    print(micro)
    return np.mean(micro), np.mean(macro), len(train_index)/len(test_index)*100.0


def load_indices(file, variable_name="train"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]

def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora',
                        help='dataset to load.')
    parser.add_argument("--label", type=str, required=True,
                        help="input file path for labels (.mat)")
    parser.add_argument("--embedding", type=str, required=True,
                        help="input file path for embedding (.npy)")
    parser.add_argument("--matfile-variable-name", type=str, default='group',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--seed", type=int, required=True,
                        help="seed used for random number generator when randomly split data into training/test set.")
    parser.add_argument("--start-train-ratio", type=int, default=10,
                        help="the start value of the train ratio (inclusive).")
    parser.add_argument("--stop-train-ratio", type=int, default=90,
                        help="the end value of the train ratio (inclusive).")
    parser.add_argument("--num-train-ratio", type=int, default=9,
                        help="the number of train ratio choosed from [train-ratio-start, train-ratio-end].")
    parser.add_argument("--C", type=float, default=1.0,
                        help="inverse of regularization strength used in logistic regression.")
    parser.add_argument("--num-split", type=int, default=10,
                        help="The number of re-shuffling & splitting for each train ratio.")
    parser.add_argument("--train-test-indices", type=str, default=None,
                        help='variable name of indices used for training/testing inside a .mat file.')
    parser.add_argument('--dimension', type=int, default=6,
                        help='Embedding Dimension')
    return parser


if __name__ == "__main__":
    # Parsing the parametres of the input
    # The format of the input is the file in the format of mat. In this file we have four sets of data
    # 1- The Embedding matrix with the name of 'network'
    # 2- The labels matrix with the name of 'labels'
    # 3- The list of Train indexes with the name of 'train_indexes'
    # 4- The list of Test indexes with the name of 'test_indexes'

    ###############################################
    # Reading the Parameters from input
    parser = read_parameters()
    args = parser.parse_args()


    logging.basicConfig(
        #filename="%s.log" % args.embedding, filemode="w", # uncomment this to log to file
        level=logging.INFO,
        format='%(asctime)s %(message)s')  # include timestamp

    ###############################################
    #Loading the labels matrix

    logger.info("Loading label from %s...", args.label+str(args.dimension)+'_dim.mat')

    label = load_label(
        file= args.label+str(args.dimension)+'_dim.mat', variable_name=args.matfile_variable_name)
    logger.info("Label loaded!")

    ###############################################
    #Loading the embedding matrix

    logger.info("Loading network embedding from %s...", args.label+str(args.dimension)+'_dim.mat')
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, args.label + str(args.dimension) + '_dim.mat')
    data = scipy.io.loadmat(my_file)
    logger.info("loading mat file %s", args.label + str(args.dimension) + '_dim.mat')
    embedding = data['network']
    embedding_artificial = embedding.copy()
    # embedding = random.uniform(0, 1)
    # embedding = np.random(random.uniform(0,1), size=(19717,6))
    random_matrix_array = np.random.rand(19717, 6)

    normed_embedding = normalize(embedding, axis=1, norm='l2')


    logger.info("Network embedding loaded!")

    ###################################################
    # loading the train indexes and test indexes
    train_indices = data['train_indexes']
    test_indices = data['test_indexes']

    ###################################################
    # calling the liblinear with the fixed training and testing labels

    avg_micro, avg_macro, train_ratio = predict_cv_fixed(normed_embedding, label, train_indices[0], test_indices[0])
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'ami_normalized_' + args.dataset + '_fixed_' + str(args.dimension) + '.mat')
    scipy.io.savemat(my_file, {'ami': avg_micro, 'ama': avg_macro, 'tr': train_ratio})


    # train_ratios = np.linspace(args.start_train_ratio, args.stop_train_ratio,
    #                           args.num_train_ratio)
    
    # manual ask for certain training ratio.
    train_ratios = np.linspace(1, 9.5,
                               10)
    # train_ratios = np.linspace(10,98,10)

    avg_micros = []
    avg_macros = []
    trs = []
    ###################################################
    # calling the liblinear based on the amount of train_ratios
    for tr in train_ratios:

        avg_micro, avg_macro, train_ratio = predict_cv(normed_embedding, label, train_ratio=tr/100., n_splits=args.num_split, C=args.C, random_state=args.seed)
        avg_micros.append(avg_micro)
        avg_macros.append(avg_macro)
        trs.append(train_ratio)

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'ami_normalized_'+args.dataset+'_NotFixed_'+str(args.dimension)+'.mat')
    scipy.io.savemat(my_file, {'ami': avg_micros, 'ama': avg_macros, 'tr': trs})  # instnt of average micro

