import numpy as np
import networkx as nx
import random
from tqdm import tqdm



def calculate_nonlinear_diffusion(idx_train, labels, t, h, p, L, pinvD, n, nclass, nnlinear_function):
    preds = np.zeros((nclass, n))
    # Start diffusion
    labels_train = labels[idx_train]
    for j in range(0, nclass):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        u = np.zeros(n)
        u[indexes] = 1.0 / (len(indexes))
        for tt in range(0, t):
            if nnlinear_function == "power":
                u = u - h * np.dot(L, np.power(u, p))
                u[u < 0] = 0.0
                u[u > 1] = 1.0
            if nnlinear_function == "tanh":
                u = u - h * np.dot(L, np.tanh(u))
        train_j_class = j
        preds[train_j_class, :] = np.maximum(preds[train_j_class, :], u)
    return preds


def calculate_nonlinear_diffusion_embeddings(idx_train, labels, t, h, p, L, n, nclass, nnlinear_function,
                                             numSamplesPerLabels, dataset):
    # create the matrix for saving the result the number of rows is the number of the samples we pick times the the number of labels
    # so we can say that each row of this class is a diffusion vector for each of the seeds, finally we return the transpose of this matrix which is the embedding of the matrix
    preds = np.zeros(( numSamplesPerLabels * nclass, n)).astype(float)
    # Start diffusion
    labels_train = labels[idx_train]  #Fetch the labels for the train indexes
    embedd_train_index = []
    # we pick number of Samples Per Label with variable (numSamplesPerLabels) in this loop and also save the indexes of the samples that we picked
    for j in tqdm(range(0, nclass)):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        sampled_list = random.sample(indexes, numSamplesPerLabels)
        embedd_train_index.extend(sampled_list)
        # for each of the seeds that we pick by random from this specific label we start the diffusion and at  the end of the loop we save the diffusion vector for each of the seeds in the (preds) matrix
        for k in range(0, len(sampled_list)):
            u = np.zeros(n)
            u[sampled_list[k]] = 1.0
            for tt in range(0, t):
                if nnlinear_function == "power":
                    u = u - h * np.dot(L, np.power(u, p))
                    u[u < 0] = 0.0
                    u[u > 1] = 1.0
                if nnlinear_function == "tanh":
                    u = u - h * np.dot(L, np.tanh(u))
            preds[j * numSamplesPerLabels + k, :] = np.maximum(preds[j * numSamplesPerLabels + k, :], u)

    return embedd_train_index, np.transpose(preds) # finally we return the transpose of the preds which is our embedding and the indicies for the seeds

def CalculateDiffusionGraph(B, nclass, diffusions):
    result = [None] * nclass
    adjMatrix = B.copy()
    x = np.transpose(np.nonzero(B))
    for i in range(0, nclass):
        diffusion_Vector = diffusions[i, :]
        for j in tqdm(range(len(x))):
            tempValue = np.abs(diffusion_Vector[x[j][0]] - diffusion_Vector[x[j][1]])
            if tempValue == 0:
                B[x[j][0]][x[j][1]] = 0.00000000000000000000000000000001
            else:
                B[x[j][0]][x[j][1]] = tempValue
        result[i] = B
        B = adjMatrix.copy()
    return B, result

def Calculate_Embedding_Based_Graph(B, nclass, embedding):
    adjMatrix = B.copy()
    x = np.transpose(np.nonzero(B))
    for j in tqdm(range(len(x))):
        dist = np.linalg.norm(embedding[x[j][0]] - embedding[x[j][1]])
        B[x[j][0]][x[j][1]] = dist

    return adjMatrix, B

def Calculate_Directed_Diffusion_Graph(adj, nclass, diffusions):
    result = [None] * nclass
    adjMatrix = adj.copy()
    x = np.transpose(np.nonzero(adj))
    for i in range(0, nclass):
        diffusion_Vector = diffusions[i, :]
        for j in tqdm(range(len(x))):
            if diffusion_Vector[x[j][0]] >= diffusion_Vector[x[j][1]]:
               tempValue = np.abs(diffusion_Vector[x[j][0]] - diffusion_Vector[x[j][1]])
               if tempValue == 0:
                   adj[x[j][0]][x[j][1]] = 0.00000000000000000000000000000001
               else:
                   adj[x[j][0]][x[j][1]] = tempValue
            else:
                adj[x[j][0]][x[j][1]] = 0
        result[i] = adj.copy()
        adj = adjMatrix.copy()
    adj = adjMatrix.copy()
    return adj, result

def CalculateSSPTree(idx_train, labels, nclass, B,isdirected):
    labels_train = labels[idx_train]
    rank_result = [None] * nclass
    weight_result = [None] * nclass
    for j in range(0, nclass):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        dict_l = {}
        dict_w = {}
        if bool(isdirected) == True:
            G = nx.from_numpy_matrix(np.matrix(B[j]), create_using=nx.DiGraph)
        else:
            G = nx.from_numpy_matrix(B[j])

        for z in tqdm(range(len(indexes))):
            length, path = nx.single_source_dijkstra(G, indexes[z], target=None, cutoff=None, weight='weight')
            level = {k: len(v) - 1 for k, v in path.items()}
            if len(dict_l) == 0:
                dict_l = level
                dict_w = length
            else:
                for k, v in level.items():
                    if k in dict_l:
                        if dict_l[k] > v:
                            dict_l[k] = v
                            dict_w[k] = length[k]

                        if dict_l[k] == v:
                            if dict_w[k] > length[k]:
                                dict_w[k] = length[k]
                    else:
                        dict_l[k] = v
                        dict_w[k] = length[k]

                # dict_l = {k: min(v, dict_l[k]) for k, v in level.items() if k in dict_l}
        rank_result[j] = dict_l
        weight_result[j] = dict_w
    return rank_result, weight_result

def calculate_SSPTree_embedding(idx_train, labels, nclass, B):
    labels_train = labels[idx_train]
    rank_result = [None] * nclass
    weight_result = [None] * nclass
    G = nx.from_numpy_matrix(B)
    for j in range(0, nclass):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        dict_l = {}
        dict_w = {}
        for z in tqdm(range(len(indexes))):
            length, path = nx.single_source_dijkstra(G, indexes[z], target=None, cutoff=None, weight='weight')
            level = {k: len(v) - 1 for k, v in path.items()}
            if len(dict_l) == 0:
                dict_l = level
                dict_w = length
            else:
                for k, v in level.items():
                    if k in dict_l:
                        if dict_l[k] > v:
                            dict_l[k] = v
                            dict_w[k] = length[k]

                        if dict_l[k] == v:
                            if dict_w[k] > length[k]:
                                dict_w[k] = length[k]
                    else:
                        dict_l[k] = v
                        dict_w[k] = length[k]
        rank_result[j] = dict_l
        weight_result[j] = dict_w
    return rank_result, weight_result

def find_Min_rank(rank_dicts, weight_dicts):
    # G = nx.from_numpy_matrix(G)
    # nodes = list(G.nodes)
    result = {}
    weight = {}

    for i in range(0, len(rank_dicts)):
        for k, v in rank_dicts[i].items():
            if k in result:
                if result[k][0] > v:
                    result[k] = (v, i)
                    weight[k] = weight_dicts[i][k]
                if result[k][0] == v:
                    if weight_dicts[i][k] < weight[k]:
                        result[k] = (v, i)
                        weight[k] = weight_dicts[i][k]
            else:
                result[k] = (v, i)
                weight[k] = weight_dicts[i][k]

    return result

def Process_Result(result):
    # sorted(results.items(), key = lambda kv: (kv[1], kv[0]))
    rank_sorted = []
    labels_sorted = []
    for i in sorted(result):
        rank_sorted.append(result[i][0])
        labels_sorted.append(result[i][1])
    return rank_sorted, labels_sorted

def calculate_nonlinear_diffusion_treeBased(idx_train, labels, t, h, p, L, pinvD, n, nclass, G, nnlinear_function):
    preds = np.zeros((nclass, n))
    # Start diffusion
    labels_train = labels[idx_train]
    for j in range(0, nclass):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        u = np.zeros(n)
        u[indexes] = 1.0 / (len(indexes))
        for tt in range(0, t):
            if nnlinear_function == "power":
                u = u - h * np.dot(L, np.power(u, p))
                u[u < 0] = 0.0
                u[u > 1] = 1.0
            if nnlinear_function == "tanh":
                u = u - h * np.dot(L, np.tanh(u))
        train_j_class = j
        preds[train_j_class, :] = np.maximum(preds[train_j_class, :], u)
    return preds

def calculate_two_nonlinear_diffusions(idx_train, labels, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass,
                                       nnlinear_function):
    preds = np.zeros((nclass, n))
    labels_train = labels[idx_train]
    # Start diffusion
    for j in range(0, nclass):
        indexes = [idx_train[i] for i, x in enumerate(labels_train) if x == j]
        uF = np.zeros(n)
        uF[indexes] = 1.0 / (len(indexes))
        uadj = np.zeros(n)
        uadj[indexes] = 1.0 / (len(indexes))
        print("Running nonlinear diffusion for class = %d" % (j))
        for tt in range(0, t):
            if nnlinear_function == "power":
                uadj = uadj - h * np.dot(Ladj, np.power(uadj, p1)) - sigma * (uadj - uF)
                uF = uF - h * np.dot(LF, np.power(uF, p2)) - sigma * (uF - uadj)
                uF[uF < 0] = 0.0
                uF[uF > 1] = 1.0
                uadj[uadj < 0] = 0.0
                uadj[uadj > 1] = 1.0
            if nnlinear_function == "tanh":
                uadj = uadj - h * np.dot(Ladj, uadj) - sigma * (uadj - uF)
                uadj = np.tanh(uadj)
                uF = uF - h * np.dot(LF, uF) - sigma * (uF - uadj)
                uF = np.tanh(uF)
        train_j_class = j
        f = w * np.dot(pinvDF, uF) + (1 - w) * np.dot(pinvDadj, uadj)
        preds[train_j_class, :] = np.maximum(preds[train_j_class, :], f)
    return preds

def choose_samples(preds, samples_no, remove_colums):
    # Choose samples such that we preserve class distribution.
    [nclass, n] = preds.shape
    preds[:, remove_colums] = -1
    sorted_index = np.argsort(preds)
    selected_samples = np.array([])
    for i in range(0, nclass):
        samples_per_class = int(samples_no / nclass);
        selected_samples_i = np.squeeze(sorted_index[i, n - samples_per_class:n])
        selected_samples = np.unique(np.append(selected_samples, selected_samples_i)).astype(int)
    return selected_samples


def self_learning(idx_train, y, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass, labels, idx_test,
                  rank_based, nnlinear_function):
    preds = calculate_two_nonlinear_diffusions(idx_train, y, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n,
                                               nclass, nnlinear_function)
    if rank_based == 0:
        pred_labels = np.argmax(preds, 0)
    else:
        sorted_pred = np.argsort(-preds)
        sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
        for i in range(0, sorted_pred.shape[0]):
            for j in range(0, sorted_pred.shape[1]):
                sorted_pred_idx[i, sorted_pred[i][j]] = j
        pred_labels = np.argmin(sorted_pred_idx, 0)
    new_samples = idx_train;
    for iter in range(5):
        selected_samples = choose_samples(preds, 10, new_samples)
        pred_labels[idx_train] = labels[idx_train]
        new_samples = np.unique(np.concatenate((selected_samples, new_samples)));
        preds = calculate_two_nonlinear_diffusions(new_samples, pred_labels, t, h, p1, p2, LF, Ladj, sigma, pinvDF,
                                                   pinvDadj, w, n, nclass, nnlinear_function)
        if rank_based == 0:
            pred_labels = np.argmax(preds, 0)
        else:
            sorted_pred = np.argsort(-preds)
            sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
            for i in range(0, sorted_pred.shape[0]):
                for j in range(0, sorted_pred.shape[1]):
                    sorted_pred_idx[i, sorted_pred[i][j]] = j
            pred_labels = np.argmin(sorted_pred_idx, 0)
    return preds
