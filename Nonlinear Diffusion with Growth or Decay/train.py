import argparse
import numpy as np
import os
import random
import scipy.io as sio
from sklearn.preprocessing import normalize
import networkx as nx
# import matlab.engine

from utils import read_parameters, load_data, calculate_normalized_laplacian, calculate_quality_measures
from nonlinear_diffusion import calculate_nonlinear_diffusion, calculate_two_nonlinear_diffusions, self_learning, CalculateDiffusionGraph, CalculateSSPTree, find_Min_rank, Process_Result, calculate_nonlinear_diffusion_embeddings, Calculate_Embedding_Based_Graph, calculate_SSPTree_embedding, Calculate_Directed_Diffusion_Graph
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
print("Done loading data of" + " "+args.dataset)
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
    # edges_num_1 = (nx.from_numpy_matrix(G)).number_of_edges()
    print("Done loading knn graph")


# edges_num = (nx.from_numpy_matrix(G)).number_of_edges()
[n, nclass] = labels.shape
if args.use_two_diffusions == 0:
    G = args.w * G + (1.0 - args.w) * adj.A
    countedges = np.transpose(np.nonzero(G))
    print("# Edges in G:"+str(len(countedges)))

    # GG = np.dot(G, G)
    # G = G + GG
    # countedges_GG = np.transpose(np.nonzero(G))
    # print("# Edges in G:" + str(len(countedges_GG)))
    #
    # B = adj.A
    # B = B.astype(float)

    y = np.argmax(labels, 1)
    if bool(args.load_data) == True:
        L = np.load('L_' + args.dataset + '_Normalized_laplacian_G' + '.npy')
        print("Done loading Normalized laplacian")
        pinvD = np.load('pinVD_' + args.dataset + '_PinVD_G' + '.npy')
        print("Done loading PinVD")
        preds = np.load('Preds_' + args.dataset + '_Preds_G' + '.npy')
        print("Done loading Preds")
        # predstttt = np.load('rank_Result_GG_embediingcora.npy', allow_pickle=True)

        # G, CDirected = Calculate_Directed_Diffusion_Graph(G, nclass,preds)
        # # CUnDirected = CalculateDiffusionGraph(G, nclass, preds)
        # np.save('diffusion_graph_directed_G_all_lables_' + args.dataset + '.npy', CDirected)
        # # C = np.load('diffusion_Graph_GG_CORA_all_lables.npy')
        # SSSP_Rank,SSSP_Weight = CalculateSSPTree(idx_train, y, nclass,CDirected,1)
        # np.save('Directed_rank_Result_G_'+ args.dataset+ '.npy', SSSP_Rank)
        # np.save('Directed_weight_Result_G_' +args.dataset+ '.npy',SSSP_Weight)
        # DicOfRanks = find_Min_rank(SSSP_Rank,SSSP_Weight)
        # rank_sorted, labels_sorted = Process_Result(DicOfRanks)
        # np.save('Directed_rank_sorted_with_G_' + args.dataset + '.npy', rank_sorted)
        # np.save('Directed_labels_sorted_with_G_' + args.dataset + '.npy', labels_sorted)

        # THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        # my_file = os.path.join(THIS_FOLDER, 'all_labels_' + args.dataset + '_140' + '_dim.mat')
        # data = sio.loadmat(my_file)
        # embedding = data['network']
        # normed_embedding = normalize(embedding, axis=1, norm='l2')
        # labelsMatrix = labels.copy()
        # labelsMatrix[len(idx_train):labelsMatrix.shape[0], :] = 0
        # embedd_index,preds_embedding = calculate_nonlinear_diffusion_embeddings(idx_train, y, args.t, args.h, args.p1, L, n, nclass, args.function, args.number_of_samples_per_label, args.dataset)
        # sio.savemat('all_labels'+'_'+args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim'+ '.mat', {'network': preds_embedding, 'lables': labels, 'train_indexes': embedd_index, 'test_indexes': idx_test})
    else:
        L, pinvD = calculate_normalized_laplacian(G)
        preds = calculate_nonlinear_diffusion(idx_train, y, args.t, args.h, args.p1, L, pinvD, n, nclass, args.function)
        print("Done calculating  laplacian and Preds")
        # labelsMatrix = labels.copy()
        # labelsMatrix[len(idx_train):labelsMatrix.shape[0], :] = 0
        # embedd_index,preds_embedding = calculate_nonlinear_diffusion_embeddings(idx_train, y, args.t, args.h, args.p1, L, n, nclass, args.function, args.number_of_samples_per_label, args.dataset)
        # normed_embedding = normalize(preds_embedding, axis=1, norm='l2')
        # sio.savemat('all_labels'+'_'+args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim'+ '.mat', {'network': preds_embedding, 'lables': labels, 'train_indexes': embedd_index, 'test_indexes': idx_test})
        np.save('L_' + args.dataset + '_Normalized_laplacian_G' + '.npy', L)
        np.save('pinVD_' + args.dataset + '_PinVD_G' + '.npy', pinvD)
        np.save('Preds_' + args.dataset + '_Preds_G' + '.npy', preds)

    G ,C_undirected = CalculateDiffusionGraph(G, nclass,preds)
    np.save('diffusion_graph_simple_G_all_lables_' + args.dataset + '.npy', C_undirected)
    print("Done calculating  diffusion_graph_simple_G_all_lables")
    SSSP_Rank_undirected,SSSP_Weight_undirected = CalculateSSPTree(idx_train, y, nclass,C_undirected, 0)
    np.save('rank_Result_G_simple_'+ args.dataset+ '.npy', SSSP_Rank_undirected)
    np.save('weight_Result_G_simple_' +args.dataset+ '.npy',SSSP_Weight_undirected)
    print("Done calculating  SSSP_Rank_undirected_graph_simple_G_all_lables")
    DicOfRanks_undirected = find_Min_rank(SSSP_Rank_undirected,SSSP_Weight_undirected)
    rank_sorted_undirected, labels_sorted_undirected = Process_Result(DicOfRanks_undirected)
    np.save('rank_sorted_G_simple_' + args.dataset + '.npy', rank_sorted_undirected)
    np.save('labels_sorted_G_simple_' + args.dataset + '.npy', labels_sorted_undirected)
    print("Done calculating the undirected")

    G, C_directed = Calculate_Directed_Diffusion_Graph(G, nclass, preds)
    np.save('diffusion_graph_directed_G_all_lables_' + args.dataset + '.npy', C_directed)
    print("Done calculating  diffusion_graph_directed_G_all_lables")
    SSSP_Rank_directed, SSSP_Weight_directed = CalculateSSPTree(idx_train, y, nclass, C_directed, 1)
    np.save('rank_Result_G_directed_' + args.dataset + '.npy', SSSP_Rank_directed)
    np.save('weight_Result_G_directed_' + args.dataset + '.npy', SSSP_Weight_directed)
    print("Done calculating  SSSP_Rank_directed_graph_directed_G_all_lables")
    DicOfRanks_directed = find_Min_rank(SSSP_Rank_directed, SSSP_Weight_directed)
    rank_sorted_directed, labels_sorted_directed = Process_Result(DicOfRanks_directed)
    np.save('rank_sorted_G_directed_' + args.dataset + '.npy', rank_sorted_directed)
    np.save('labels_sorted_G_directed_' + args.dataset + '.npy', labels_sorted_directed)
    print("Done calculating the directed")

    embedd_index,preds_embedding = calculate_nonlinear_diffusion_embeddings(idx_train, y, args.t, args.h, args.p1, L, n, nclass, args.function, args.number_of_samples_per_label, args.dataset)
    sio.savemat('all_labels_G'+'_'+args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim'+ '.mat', {'network': preds_embedding, 'lables': labels, 'train_indexes': embedd_index, 'test_indexes': idx_test})
    print("Done calculating the embedding_" +args.dataset+'_'+str(args.number_of_samples_per_label*nclass) +'_'+'dim')
    normed_embedding = normalize(preds_embedding, axis=1, norm='l2')
    G ,C_embedding = Calculate_Embedding_Based_Graph(G, nclass, normed_embedding)
    np.save('diffusion_graph_embedding_G_all_lables_' + args.dataset + '.npy', C_embedding)
    print("Done calculating  diffusion_graph_embedding_G_all_lables")
    SSSP_Rank_embedding, SSSP_Weight_embedding = calculate_SSPTree_embedding(idx_train, y, nclass, C_embedding)
    np.save('rank_Result_G_embedding_'+ args.dataset+ '.npy', SSSP_Rank_embedding)
    np.save('weight_Result_G_embedding_' +args.dataset+ '.npy',SSSP_Weight_embedding)
    print("Done calculating  SSSP_Rank_embedding_graph_embedding_G_all_lables")
    DicOfRanks_embedding = find_Min_rank(SSSP_Rank_embedding, SSSP_Weight_embedding)
    rank_sorted_embedding, labels_sorted_embedding = Process_Result(DicOfRanks_embedding)
    np.save('rank_sorted_G_embedding_' + args.dataset + '.npy', rank_sorted_embedding)
    np.save('labels_sorted_G_embedding_' + args.dataset + '.npy', labels_sorted_embedding)
    print("Done calculating the embedding ")

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

# Get predicted labels for each data point.
pred_labels_not_rank_based = np.argmax(preds, 0)
# Report accuracy on test set.
y = np.argmax(labels[idx_val, :], 1)
pred_labels_test_not_rank_base = pred_labels_not_rank_based[idx_test]
y = np.argmax(labels[idx_test, :], 1)
print("accuracy for not rank based:")
calculate_quality_measures(y, pred_labels_test_not_rank_base)

sorted_pred = np.argsort(-preds)
sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
for i in range(0, sorted_pred.shape[0]):
    for j in range(0, sorted_pred.shape[1]):
        sorted_pred_idx[i, sorted_pred[i][j]] = j
pred_labels_rank_based = np.argmin(sorted_pred_idx, 0)
# Report accuracy on test set.
y = np.argmax(labels[idx_val, :], 1)
pred_labels_test_rank_based = pred_labels_rank_based[idx_test]
y = np.argmax(labels[idx_test, :], 1)
print("accuracy for rank based:")
calculate_quality_measures(y, pred_labels_test_rank_based)

labels_sorted_G_simple = np.load('labels_sorted_G_simple_' + args.dataset + '.npy')
labels_sorted_G_simple_test = labels_sorted_G_simple[idx_test]
rank_sorted_G_simple = np.load('rank_sorted_G_simple_' + args.dataset + '.npy')
rank_sorted_G_simple_test = rank_sorted_G_simple[idx_test]
print("accuracy for rank based undirected graph:")
calculate_quality_measures(y, labels_sorted_G_simple_test)

labels_sorted_G_directed = np.load('labels_sorted_G_directed_' + args.dataset + '.npy')
labels_sorted_G_directed_test = labels_sorted_G_directed[idx_test]
rank_sorted_G_directed = np.load('rank_sorted_G_directed_' + args.dataset + '.npy')
rank_sorted_G_directed_test = rank_sorted_G_directed[idx_test]
print("accuracy for rank based directed graph:")
calculate_quality_measures(y, labels_sorted_G_directed_test)

labels_sorted_G_embedding = np.load('labels_sorted_G_embedding_' + args.dataset + '.npy')
labels_sorted_G_embedding_test = labels_sorted_G_embedding[idx_test]
rank_sorted_G_embedding = np.load('rank_sorted_G_embedding_' + args.dataset + '.npy')
rank_sorted_G_embedding_test = rank_sorted_G_embedding[idx_test]
print("accuracy for rank based embedding graph:")
calculate_quality_measures(y, labels_sorted_G_embedding_test)

# if args.rank_based == 0:
#     # Get predicted labels for each data point.
#     pred_labels = np.argmax(preds, 0)
#     # Report accuracy on test set.
#     y = np.argmax(labels[idx_val, :], 1)
#     pred_labels_test = pred_labels[idx_test]
#     y = np.argmax(labels[idx_test, :], 1)
#     calculate_quality_measures(y, pred_labels_test)
#     labelSorted = np.load('labels_sorted' + args.dataset + '.npy')
#     newMethod_labels_test = labelSorted[idx_test]
#     rankedSorted = np.load('rank_sorted' + args.dataset + '.npy')
#     newMethod_ranked_test = rankedSorted[idx_test]
#     calculate_quality_measures(y, newMethod_labels_test)
# else:
#     sorted_pred = np.argsort(-preds)
#     sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
#     for i in range(0, sorted_pred.shape[0]):
#         for j in range(0, sorted_pred.shape[1]):
#             sorted_pred_idx[i, sorted_pred[i][j]] = j
#     pred_labels = np.argmin(sorted_pred_idx, 0)
#     # Report accuracy on test set.
#     y = np.argmax(labels[idx_val, :], 1)
#     pred_labels_test = pred_labels[idx_test]
#     y = np.argmax(labels[idx_test, :], 1)
#     calculate_quality_measures(y, pred_labels_test)
#
#     labelSorted = np.load('Directed_labels_sorted_with_G_' + args.dataset + '.npy')
#     newMethod_labels_test = labelSorted[idx_test]
#     rankedSorted = np.load('Directed_rank_sorted_with_G_' + args.dataset + '.npy')
#     newMethod_ranked_test = rankedSorted[idx_test]
#     calculate_quality_measures(y, newMethod_labels_test)
    #
    # X=np.arange(0.0, len(newMethod_ranked_test), 1)
    #
    #
    # # plt.plot(X, newMethod_ranked_test)  # Plotting the line plot
    # # plt.plot(X, pred_labels_test)  # Plotting the line plot
    # plt.scatter(X[0:100], newMethod_ranked_test[0:100], c='b', marker='o')
    # plt.scatter(X[0:100], pred_labels_test[0:100], c='r', marker='o')
    # # Labeling the X-axis
    # plt.xlabel('X-axis')
    # # Labeling the Y-axis
    # plt.ylabel('Y-axis')
    # # Give a title to the graph
    # plt.title('Simple Line Plot')
    # plt.show()
