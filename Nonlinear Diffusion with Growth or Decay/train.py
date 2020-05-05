import argparse
import numpy as np
import random
import scipy.io as sio
# import matlab.engine

from utils import read_parameters, load_data, calculate_normalized_laplacian, calculate_quality_measures
from nonlinear_diffusion import calculate_nonlinear_diffusion, calculate_two_nonlinear_diffusions, self_learning, CalculateDiffusionGraph, CalculateSSPTree, find_Min_rank, Process_Result, calculate_nonlinear_diffusion_embeddings
from predict import predict_cv, predict_cv_fixed

from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

parser = read_parameters()
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
np.random.RandomState(args.seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, graph = load_data(args.dataset,
                                                                                                  args.normalize)
print("Done loading data")
idx_train = [i for i, x in enumerate(train_mask) if x]
idx_val = [i for i, x in enumerate(val_mask) if x]
idx_test = [i for i, x in enumerate(test_mask) if x]
n = adj.shape[0]

if args.generate_knn == 1:
    dense_features = features
    Tree = KDTree(dense_features)
    dists, idxs = Tree.query(dense_features, k=args.knighbors_number)
    dists = np.exp(-(dists ** 2 / (2 * args.rad ** 2)))
    G = np.zeros((n, n))
    G[np.tile(range(0, n), args.knighbors_number), np.reshape(idxs, n * args.knighbors_number, 1)] = np.reshape(dists,
                                                                                                                n * args.knighbors_number,
                                                                                                                1)
    # Zero diagonal elements.
    for i in range(0, n):
        G[i, i] = 0.0
    # Make G symmtric!
    Gtranspose = np.transpose(G)
    G = G + Gtranspose - np.sqrt(G * Gtranspose)
    print("Done building knn graph")
    dsqrt = np.diag(np.sqrt(G.sum(axis=0)))
    x, y = np.nonzero(dsqrt)
    nnz = x.shape[0]
    for i in range(0, nnz):
        if dsqrt[x[i], y[i]] != 0:
            dsqrt[x[i], y[i]] = 1.0 / dsqrt[x[i], y[i]];
    G = np.dot(np.dot(dsqrt, G), dsqrt);
    np.save('G_' + args.dataset + '_k_' + str(args.knighbors_number) + '_rad_' + str(args.rad) + '.npy', G);
else:
    # print("Done loading knn graph")
    G = np.load('G_' + args.dataset + '_k_' + str(args.knighbors_number) + '_rad_' + str(args.rad) + '.npy');
    print("Done loading knn graph")


[n, nclass] = labels.shape
if args.use_two_diffusions == 0:
    G = args.w * G + (1.0 - args.w) * adj.A
    GG = np.dot(G, G)
    G = G + GG
    # sio.savemat('raw' + args.dataset + '.mat', {'network': G})
    B = adj.A
    B = B.astype(float)
    load_data = True
    y = np.argmax(labels, 1)
    if load_data == True:
        L = np.load('L_' + args.dataset + 'Normalized laplacian' + '.npy')
        print("Done loading Normalized laplacian")
        pinvD = np.load('pinVD_' + args.dataset + 'PinVD' + '.npy')
        print("Done loading PinVD")
        preds = np.load('Preds_' + args.dataset + 'Preds' + '.npy')
        print("Done loading Preds")
        labelsMatrix = labels.copy()
        labelsMatrix[len(idx_train):labelsMatrix.shape[0], :] = 0
        embedd_index,preds_embedding = calculate_nonlinear_diffusion_embeddings(idx_train, y, args.t, args.h, args.p1, L, n, nclass, args.function, args.number_of_samples_per_label, args.dataset)
        sio.savemat('all_labels'+'_'+args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim'+ '.mat', {'network': preds_embedding, 'lables': labels, 'train_indexes': embedd_index, 'test_indexes': idx_test})
    else:
        L, pinvD = calculate_normalized_laplacian(G)
        preds = calculate_nonlinear_diffusion(idx_train, y, args.t, args.h, args.p1, L, pinvD, n, nclass, args.function)
        labelsMatrix = labels.copy()
        labelsMatrix[len(idx_train):labelsMatrix.shape[0], :] = 0
        embedd_index,preds_embedding = calculate_nonlinear_diffusion_embeddings(idx_train, y, args.t, args.h, args.p1, L, n, nclass, args.function, args.number_of_samples_per_label, args.dataset)
        sio.savemat('all_labels'+'_'+args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim'+ '.mat', {'network': preds_embedding, 'lables': labels, 'train_indexes': embedd_index, 'test_indexes': idx_test})
        np.save('L_' + args.dataset + 'Normalized laplacian' + '.npy', L)
        np.save('pinVD_' + args.dataset + 'PinVD' + '.npy', pinvD)
        np.save('Preds_' + args.dataset + 'Preds' + '.npy', preds)

    # C = CalculateDiffusionGraph(B, nclass,preds)
    # SSSP_Rank,SSSP_Weight = CalculateSSPTree(idx_train, y, nclass,C)
    # DicOfRanks = find_Min_rank(SSSP_Rank,SSSP_Weight)
    # result = Process_Result(DicOfRanks)

else:
    LF, pinvDF = calculate_normalized_laplacian(G)
    Ladj, pinvDadj = calculate_normalized_laplacian(1.0 * adj.A)
    y = np.argmax(labels, 1)
    if args.self_learning == 0:
        preds = calculate_two_nonlinear_diffusions(idx_train, y, args.t, args.h, args.p1, args.p2, LF, Ladj, args.sigma,
                                                   pinvDF, pinvDadj, args.w, n, nclass, args.function)
    else:
        preds = self_learning(idx_train, y, args.t, args.h, args.p1, args.p2, LF, Ladj, args.sigma, pinvDF, pinvDadj,
                              args.w, n, nclass, y, idx_test, args.rank_based, args.function)
    C = CalculateDiffusionGraph((adj.A).astype(float), nclass,preds)
    SSSP_Rank,SSSP_Weight = CalculateSSPTree(idx_train, y, nclass, C)
    DicOfRanks = find_Min_rank(SSSP_Rank,SSSP_Weight)
    result = Process_Result(DicOfRanks)
print("Done calculating the diffusions")

if args.rank_based == 0:
    # Get predicted labels for each data point.
    pred_labels = np.argmax(preds, 0)
    # Report accuracy on test set.
    y = np.argmax(labels[idx_val, :], 1)
    pred_labels_test = pred_labels[idx_test]
    y = np.argmax(labels[idx_test, :], 1)
    calculate_quality_measures(y, pred_labels_test)
else:
    sorted_pred = np.argsort(-preds)
    sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
    for i in range(0, sorted_pred.shape[0]):
        for j in range(0, sorted_pred.shape[1]):
            sorted_pred_idx[i, sorted_pred[i][j]] = j
    pred_labels = np.argmin(sorted_pred_idx, 0)
    # Report accuracy on test set.
    y = np.argmax(labels[idx_val, :], 1)
    pred_labels_test = pred_labels[idx_test]
    y = np.argmax(labels[idx_test, :], 1)
    calculate_quality_measures(y, pred_labels_test)


    labelSorted = np.load('labelsSortedlabelsSorted' + '.npy')
    newMethod_labels_test = labelSorted[idx_test]
    rankedSorted = np.load('rankSorted_rankSorted' + '.npy')
    newMethod_ranked_test = rankedSorted[idx_test]
    calculate_quality_measures(y, newMethod_labels_test)

    X=np.arange(0.0, len(newMethod_ranked_test), 1)


    # plt.plot(X, newMethod_ranked_test)  # Plotting the line plot
    # plt.plot(X, pred_labels_test)  # Plotting the line plot
    plt.scatter(X[0:100], newMethod_ranked_test[0:100], c='b', marker='o')
    plt.scatter(X[0:100], pred_labels_test[0:100], c='r', marker='o')
    # Labeling the X-axis
    plt.xlabel('X-axis')
    # Labeling the Y-axis
    plt.ylabel('Y-axis')
    # Give a title to the graph
    plt.title('Simple Line Plot')
    plt.show()
