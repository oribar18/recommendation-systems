import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy import sparse, spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


def create_matrix(data):
    data = data.pivot(index='userID', columns='artistID', values='weight')
    data = data.fillna(0)
    data = sparse.csr_matrix(data.values)
    return data


def find_r_avg(R):
    return R.sum() / R.count_nonzero()


def index_dict(data):
    users_to_index = {}
    artists_to_index = {}
    index_to_users = {}
    index_to_artists = {}
    users_list = pd.unique(data['userID'])
    artists_list = pd.unique(data['artistID'])
    for i, user in enumerate(users_list):
        users_to_index[user] = i
        index_to_users[i] = user
    for i, artist in enumerate(artists_list):
        artists_to_index[artist] = i
        index_to_artists[i] = artist
    return users_to_index, index_to_users, artists_to_index, index_to_artists


def build_artist_info_dicts(data):
    plays_by_artist = {}
    for i, row in data.iterrows():
        if row['artistID'] not in plays_by_artist.keys():
            plays_by_artist[row['artistID']] = {row['userID']: row['weight']}
        else:
            plays_by_artist[row['artistID']][row['userID']] = row['weight']
    return plays_by_artist


def find_biases(data, plays, lamda, r_avg):
    rows = len(data['weight'])
    artists = list(pd.unique(data['artistID']))
    users = list(pd.unique(data['userID']))
    cols = users + artists
    A = sparse.dok_matrix((rows, len(cols)))
    index = 0
    c = []
    for artist in plays.keys():
        for user in plays[artist].keys():
            A[index, users_to_index[user]] = 1
            A[index, len(users) + artists_to_index[artist]] = 1
            c.append(plays[artist][user] - r_avg)
            index += 1
    c = np.array(c)
    zeros = np.zeros(A.shape[1])
    lamda_matrix = np.eye(A.shape[1]) * math.sqrt(lamda)
    A = sparse.vstack([A, lamda_matrix])
    c = np.concatenate((c, zeros), axis=None)
    return scipy.sparse.linalg.lsqr(A, c)[0]


def build_bias_dict(bias_vec, index_to):
    bias_dict = {}
    for i, bias in enumerate(bias_vec):
        bias_dict[index_to[i]] = bias
    return bias_dict


def build_prediction_matrix(R, users_bias, artists_bias, r_avg, index_to_users, index_to_artists):
    prediction_matrix = R.copy()
    rows, cols = prediction_matrix.nonzero()
    for row, col in zip(rows, cols):
        prediction_matrix[row, col] = r_avg + users_bias[index_to_users[row]] + artists_bias[index_to_artists[col]]
    return prediction_matrix


# def build_dict_of_artists_vectors(data):
#     artists = pd.unique(data['artistID'])
#     artists_dict = {}
#     for artist in artists:
#         artists_dict[artist] =


# def build_similarity_matrix(R, data, plays):
#     D = sparse.dok_matrix((R.shape[1], R.shape[1]))
#     artists_dict = index_dict(data)[1]
#     # values_per_artist = artists_vectors_dict
#     for i, artist1 in enumerate(plays.keys()):
#         for j, artist2 in enumerate(plays.keys()):
#             if artist1 == artist2 or j >= i:
#                 continue
#             print(np.array(R[:, i]))
#             print(np.array(R[:, j]))
#             print(cosine_similarity(R[:, i], R[:, j], dense_output=False))
#     print(D)


def neighbourhood_predictor(col, L, errors_matrix, similarity_matrix, user):
    non_zeros_index = similarity_matrix[col].nonzero()[1]
    if len(non_zeros_index) <= 1 or L == 0:
        return 0
    non_zeros = {}
    for index in non_zeros_index:
        non_zeros[index] = similarity_matrix[col, index]
    non_zeros_abs_sorted = sorted(non_zeros, key=lambda dict_key: abs(non_zeros[dict_key]))
    numerator = 0
    denominator = 0
    if len(non_zeros_abs_sorted) < L:
        for i in range(len(non_zeros_abs_sorted)):
            numerator += non_zeros[non_zeros_abs_sorted[i]] * errors_matrix[user, non_zeros_abs_sorted[i]]
            denominator += abs(non_zeros[non_zeros_abs_sorted[i]])
        return numerator / denominator
    for i in range(L):
        numerator += non_zeros[non_zeros_abs_sorted[i]] * errors_matrix[user, non_zeros_abs_sorted[i]]
        denominator += abs(non_zeros[non_zeros_abs_sorted[i]])
    return numerator/denominator


def build_final_prediction_matrix(prediction_matrix, L, errors_matrix, similarity_matrix):
    final_prediction_matrix = prediction_matrix.copy()
    rows, cols = final_prediction_matrix.nonzero()
    for row, col in zip(rows, cols):
        final_prediction_matrix[row, col] = final_prediction_matrix[row, col] + neighbourhood_predictor(col, L,
                                                                                errors_matrix, similarity_matrix, row)
        if final_prediction_matrix[row, col] < 0:
            final_prediction_matrix[row, col] = 1
    return final_prediction_matrix


def compute_loss(final_prediction_matrix, R):
    loss = 0
    rows, cols = R.nonzero()
    for row, col in zip(rows, cols):
        if final_prediction_matrix[row, col] < 0:
            final_prediction_matrix[row, col] = 1
        loss += pow(R[row, col] - final_prediction_matrix[row, col], 2)
    return loss


def compute_MSE(final_prediction_matrix, R):
    sum = 0
    count = 0
    rows, cols = R.nonzero()
    for row, col in zip(rows, cols):
        sum += math.pow(R[row, col] - final_prediction_matrix[row, col], 2)
        count += 1
    return sum/count


def build_test_predictions(test, r_avg, L):
    weight = []
    for i, row in test.iterrows():
        if row['userID'] not in users_bias_dict.keys() and row['artistID'] in artists_bias_dict.keys():
            weight.append(r_avg + artists_bias_dict[row['artistID']])
        elif row['userID'] in users_bias_dict.keys() and row['artistID'] not in artists_bias_dict.keys():
            weight.append(r_avg + users_bias_dict[row['userID']])
        elif row['userID'] not in users_bias_dict.keys() and row['artistID'] not in artists_bias_dict.keys():
            weight.append(r_avg)
        else:
            weight.append(r_avg + users_bias_dict[row['userID']] + artists_bias_dict[row['artistID']]
                          + neighbourhood_predictor(artists_to_index[row['artistID']], L, errors_matrix,
                                                    similarity_matrix, users_to_index[row['userID']]))
    weight = [10**x for x in weight]
    test['weight'] = weight
    return test.to_csv('316137371_314968595_task1.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv("user_artist.csv")
    test = pd.read_csv("test.csv")
    # results = {}
    # for i in range(5):
    #     print("sample num " + str(i) + ":")
        # data_train, data_test = train_test_split(data, test_size=0.25, random_state=i)
    data['weight'] = np.log10(data['weight'])
    # data_test['weight'] = np.log10(data_test['weight'])
    R = create_matrix(data)
    users_to_index, index_to_users, artists_to_index, index_to_artists = index_dict(data)
    plays = build_artist_info_dicts(data)
    r_avg = find_r_avg(R)
    # lamdas = [5, 10, 20, 30, 40]
    # for lamda in lamdas:
    # print("lamda is: " + str(lamda))
    chosen_lamda = 30
    bias = find_biases(data, plays, chosen_lamda, r_avg)
    users_bias = bias[:R.shape[0]]
    users_bias_dict = build_bias_dict(users_bias, index_to_users)
    artists_bias = bias[R.shape[0]:]
    artists_bias_dict = build_bias_dict(artists_bias, index_to_artists)
    # R_test = create_matrix(test)
    prediction_matrix = build_prediction_matrix(R, users_bias_dict, artists_bias_dict, r_avg, index_to_users
                                                , index_to_artists)
        # result = compute_loss(prediction_matrix, R)
        # if lamda not in results.keys():
        #     results[lamda] = result
        # else:
        #     results[lamda] += result
        # print(compute_loss(prediction_matrix, R))
    # for key in results.keys():
    #     results[key] /= 5
    # plt.plot(results.keys(), results.values())
    # plt.xlabel('lamda')
    # plt.ylabel('loss')
    # plt.scatter(results.keys(), results.values(), color='r', label='Lamda points')
    # plt.legend()
    # plt.show()
    errors_matrix = R - prediction_matrix
    similarity_matrix = cosine_similarity(errors_matrix.T, dense_output=False)
    chosen_L = 750
    test_prediction = build_test_predictions(test, r_avg, chosen_L)
        # for L in [730, 740, 750, 760]:
        #     print("neighbourhood size is " + str(L) + ":")
        #     final_prediction_matrix = build_final_prediction_matrix(prediction_matrix, L, errors_matrix,
        #                                                             similarity_matrix)
        #     result = compute_loss(final_prediction_matrix, R_test)
        #     print("loss is: " + str(result))
    #         if L not in results.keys():
    #             results[L] = result
    #         else:
    #             results[L] += result
    # for key in results.keys():
    #     results[key] /= 5
    # plt.plot(results.keys(), results.values())
    # plt.xlabel('L')
    # plt.ylabel('loss')
    # plt.scatter(results.keys(), results.values(), color='r', label='L points')
    # plt.legend()
    # plt.show()
