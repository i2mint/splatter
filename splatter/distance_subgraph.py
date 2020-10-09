from operator import itemgetter

import networkx as nx
from numpy import random, sqrt, triu_indices
import numpy as np
from scipy.spatial.distance import pdist

from sss.misc.generalized_star_clustering import (
    default_priority_func,
    degree_and_weight_priority_func,
    layered_clustering,
)

MIN_NUM_LEVELS_IN_STAR_CLUS = 1
MAX_NUM_LEVELS_IN_STAR_CLUS = 20


# from oto.pj.stag.util.graph import json_links_from_graph
def json_links_from_graph(graph, link_distance_attr='weight'):
    return [{'source': i, 'target': j, 'linkDistance': d[link_distance_attr]}
            for i, j, d in graph.edges_iter(data=True)]


def euclidean_distance_square(A, B):
    '''A and B are two numpy arrays and the function computes the square of the distance'''
    D = A - B
    return sum(x ** 2 for x in D)


def farthest_from_point(point, point_set):
    """
    find the farthest point in point_set from point and return its coordinate and its distance squared to point_set
    """
    record = []
    for i in point_set:
        distance = euclidean_distance_square(point, i)
        record.append([i, distance])
    # create a list of [i,distance] where i is a point in point_set and distance is the distance from i to point

    sorted_record = sorted(record, key=itemgetter(1))
    # sort the record list according to the second item of each entry, i.e, the distance from point to point_set

    return sorted_record[len(point_set) - 1]
    # return a list with the coordinate of the point first in an array and its distance to point_set in second


def largest_distance(point_set):
    """find the largest distance squared in point_set"""

    max_distance = 0
    # initialize max_distance

    for i in range(len(point_set) - 1):
        distance = farthest_from_point(point_set[i + 1], point_set)[1]
        if distance > max_distance:
            max_distance = distance
            # go through each point of point_set and record its maximum distance.
            # goes through each pair twice so could be improved by removing pairs already considered

    return max_distance
    # return the largest distance squared in point_set


def array_to_dist_graph(point_set, point_weight=1, metric='sqeuclidean'):
    """take a set of points P and create the complete graph with vertex set P and weights equal to the distance between pairs"""
    link_idx = triu_indices(len(point_set), k=1)
    g = nx.Graph(data=list(zip(link_idx[0],
                               link_idx[1],
                               [{'weight': w} for w in pdist(point_set, metric=metric)])))

    if point_weight is not None:
        if isinstance(point_weight, (int, float)):
            for vertex in g.nodes():
                g.node[vertex]["vertex_weight"] = point_weight
        else:
            assert g.number_of_nodes() == len(point_weight), \
                "point_weight must be a single number or a list/array of length the number of nodes"
            for vertex, weight in zip(g.nodes(), point_weight):
                g.node[vertex]["vertex_weight"] = weight

    return g


def dist_prob(norm_distance):
    '''takes normalized distances as input and assign a corresponding probability'''

    zero = 0.95
    half = 0
    one = 0.3
    # set the value of the quadratic function at zero, 0.5 and one.

    a = 2 * one - 4 * half + 2 * zero
    b = -one + 4 * half - 3 * zero
    c = zero
    # computes the coefficients of the quadratic function

    prob = a * norm_distance ** 2 + b * norm_distance + c
    # compute the probability corresponding to a given distance
    # note: the distance will be normalized when used in array_to_subgraph

    return prob


def ATG_Kruskal(point_set, n_small=2, n_large=0, n_matching=0, power=2):
    """finds the distance graph and compute its minimum spanning tree using Kruskal algorithm, then remove that tree from the graph
    and iterate the procedure n_small times. After that, finds the maximum spanning tree out of the remaining edges, removes its
    edges and iterate n_large times. All the edges selected that way form the graph used to construct the dict.
    The parameter power will change the power of the distance used on the weights of the graphs.
    Choose 1 for squared distance, 1/2 for distance and 2 for 4th power of the distance
    The parameter n_matching had n maximum matching but it is very slow and not very useful anyway. Should be replace by maximal matching
     at the very least."""

    graph = array_to_dist_graph(point_set)
    selected_1 = nx.Graph()
    """this will be the graph selected by the minimum spanning tree phase."""
    selected_2 = nx.Graph()
    """this will be the graph selected by the maximum spanning tree phase. """
    link_list = list()

    for i in range(n_small):
        tree = list(nx.minimum_spanning_edges(graph, weight='weight', data=True))
        selected_1.add_edges_from(tree)
        graph.remove_edges_from(tree)

    for i in range(len(selected_1.edges())):
        d = dict()
        d['source'] = selected_1.edges()[i][0]
        d['target'] = selected_1.edges()[i][1]
        d['linkDistance'] = (selected_1[selected_1.edges()[i][0]][selected_1.edges()[i][1]]['weight']) ** power
        link_list.append(d)

    graph_matching = nx.Graph()
    for i in range(n_matching):
        dict_matching = nx.max_weight_matching(graph)
        for j in dict_matching:
            w = graph[i][dict_matching[j]]['weight']
            graph_matching.add_edge(i, dict_matching[j], weight=w)
        graph.remove_edges_from(graph_matching.edges())

    for i in range(len(graph_matching.edges())):
        d = dict()
        d['source'] = graph_matching.edges()[i][0]
        d['target'] = graph_matching.edges()[i][1]
        d['linkDistance'] = \
            (graph_matching[graph_matching.edges()[i][0]][graph_matching.edges()[i][1]]['weight']) ** power
        link_list.append(d)

    for u, v, d in graph.edges(data=True):
        d['weight'] = -d['weight']

    for i in range(n_large):
        tree = list(nx.minimum_spanning_edges(graph, weight='weight', data=True))
        selected_2.add_edges_from(tree)
        graph.remove_edges_from(tree)

    for i in range(len(selected_2.edges())):
        d = dict()
        d['source'] = selected_2.edges()[i][0]
        d['target'] = selected_2.edges()[i][1]
        d['linkDistance'] = (selected_2[selected_2.edges()[i][0]][selected_2.edges()[i][1]]['weight']) ** power
        link_list.append(d)

    return link_list


def function_inverse(x):
    return 1 / (1 + 10 * x)


def function_prob_exp_pow(x):
    return (2 ** (-x)) ** 20


def function_prob_exp_npow(x):
    return (2 ** (x)) ** 10


def max_function_pow(x):
    return max(function_prob_exp_pow(x), function_prob_exp_npow(x))


def function_prob_quad1(x):
    return dist_prob(x, 0.5, 0.01)


def fast_decrease(x):
    return 1 / float(1 + 100 * x ** 0.1)


def make_more_sense(x):
    return 1 - float(x ** 0.2)


def linear_test(x):
    return 1 - float(x)


def random_chooser(point_set, function=linear_test, sparsing_constant=1):
    """create a graph with vertices being the points in the array point_set and with edges weighed by
    the square of the distance between the corresponding two points.
    The sparsing constant set the maximum probability of an edge to be chosen to that particular value
    """
    M = largest_distance(point_set)
    graph = nx.Graph()
    size = len(point_set)
    link_list = list()

    for i in range(size):
        graph.add_node(i)
        for j in range(0, i):
            random_1 = random.uniform(0, 1)
            # print "D=" ,D, "function=", function(D),"prob =", prob, prob - adjust_constant
            if random_1 < sparsing_constant:

                # for each unordered pair
                D = float(euclidean_distance_square(point_set[i], point_set[j]))
                # find the normalized distance of from point #i to point #j when
                prob = function(D / float(M))

                # compute the probability that an edge with D value is kept
                randome = random.uniform(0, 1)

                if randome < prob:
                    graph.add_edge(i, j, weight=D)

    for i in range(len(graph.edges())):
        d = dict()
        d['source'] = graph.edges()[i][0]
        d['target'] = graph.edges()[i][1]
        d['linkDistance'] = graph[graph.edges()[i][0]][graph.edges()[i][1]]['weight']
        link_list.append(d)

    return link_list


def mk_link_list(graph, distance_exageration_power):
    link_list = list()

    for i in range(len(graph.edges())):
        d = dict()
        d['source'] = graph.edges()[i][0]
        d['target'] = graph.edges()[i][1]
        d['linkDistance'] = (graph[graph.edges()[i][0]][graph.edges()[i][1]]['weight']) ** distance_exageration_power
        link_list.append(d)

    return link_list


def kruskal_subgraph(point_set, max_n_edges=None):
    n = len(point_set)
    if max_n_edges is None:
        max_n_edges = min(1000, int(n * (n - 1) / 2))

    graph = array_to_dist_graph(point_set)
    subgraph = nx.Graph()

    while True:
        tree = list(nx.minimum_spanning_edges(graph, weight='weight', data=True))
        if subgraph.number_of_edges() + len(tree) <= max_n_edges:
            subgraph.add_edges_from(tree)
            graph.remove_edges_from(tree)
        else:
            remaining_edges = max_n_edges - subgraph.number_of_edges()
            tree = tree[:remaining_edges]
            subgraph.add_edges_from(tree)
            graph.remove_edges_from(tree)  # only needed if you are going to do further work
            break

    return subgraph


def kruskal_links(point_set, max_n_edges=None, distance_exageration_power=2):
    return mk_link_list(kruskal_subgraph(point_set, max_n_edges=max_n_edges),
                        distance_exageration_power=distance_exageration_power)


def generalized_star_clustering_link(X, level_thresholds=None, function=None, threshold_method='percentile'):
    """
    generates a json (dict) specification for a graph, meant to be uploaded by a force directed graph interface

    Inputs:
        X: Data points (rows are indexed by points, columns by dimensions)
        level_thresholds:
            an int: Number of levels you want (will choose threshold values to be percentiles of the distances of X
            a list of numbers: The actual threshold values you want
        function: function(vertex_degree, vertex_weight, sum_adj_vert_weight) that will return a priority score
    """

    complete_graph = array_to_dist_graph(X)

    if function is None:
        function = degree_and_weight_priority_func
    if level_thresholds is None:
        level_thresholds = int(min(MAX_NUM_LEVELS_IN_STAR_CLUS,
                                   max(MIN_NUM_LEVELS_IN_STAR_CLUS, sqrt(len(X)))))
    if isinstance(level_thresholds, int):  # then consider level_thresholds to be the spec of how many levels we want

        level_thresholds = _choose_level_thresholds_from_complete_graph(complete_graph,
                                                                        level_thresholds,
                                                                        method=threshold_method)

    graph = layered_clustering(complete_graph, level_thresholds, function)

    # connecting the graph with a minimum spanning tree if the graph is disconnected
    if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        tree = nx.minimum_spanning_tree(complete_graph, weight='weight')
        for edge in tree.edges(data=True):
            if not graph.has_edge(edge[0], edge[1]):
                graph.add_edge(edge[0], edge[1], edge[2])

    # return the links json for this graph
    return json_links_from_graph(graph, link_distance_attr='weight')


def gstar_with_kruskal_init(X, level_thresholds=None, function=None, threshold_method='percentile'):
    """Same algorithm as generalized_star_clustering_link but first find 1 kruskal tree and work on what is left.
       Will output the union of the Krukal tree and the subgraph given by the generalized_star_clustering_link
       in the form of a json, meant to be uploaded to the force directed graph interface """

    graph = array_to_dist_graph(X)

    tree = nx.minimum_spanning_edges(graph, weight='weight', data=True)
    graph.remove_edges_from(tree)

    if function is None:
        function = default_priority_func
    if level_thresholds is None:
        level_thresholds = int(min(MAX_NUM_LEVELS_IN_STAR_CLUS,
                                   max(MIN_NUM_LEVELS_IN_STAR_CLUS, sqrt(len(X)))))
    if isinstance(level_thresholds, int):  # then consider level_thresholds to be the spec of how many levels we want

        level_thresholds = _choose_level_thresholds_from_complete_graph(graph,
                                                                        level_thresholds,
                                                                        method=threshold_method)

    subgraph_2 = layered_clustering(graph, level_thresholds, function)
    subgraph_2.add_edges_from(tree)

    return json_links_from_graph(subgraph_2, link_distance_attr='weight')


def _choose_level_thresholds_from_complete_graph(complete_graph, level_thresholds, method='percentile'):
    if method == 'percentile':
        # get the list of all distances between two nodes (i.e. the edge weights of complete graph
        distance_list = [edge[2].get('weight', np.nan) for edge in complete_graph.edges(data=True)]
        if len(distance_list) > 0:
            level_thresholds = list(np.percentile(distance_list,
                                                  np.linspace(start=0, stop=100, num=level_thresholds + 2)[1:-1]))
        else:
            level_thresholds = []
    else:
        raise ValueError("Unknown method: {}".format(method))
    return level_thresholds


def test():
    from sklearn.datasets import make_blobs
    from scipy.spatial.distance import cdist

    n = 100
    # X = np.random.rand(n, 2)
    X, _ = make_blobs(n_samples=n, n_features=2, centers=4)
    print((X.shape))

    d = generalized_star_clustering_link(X, level_thresholds=None, function=None)

    print(("Test 1 got {} links over {} nodes".format(len(d), len(X))))
    print(d[:4])

    distmat = cdist(X, X)
    thresh = np.percentile(distmat, q=5)
    d = generalized_star_clustering_link(X, level_thresholds=[thresh], function=None)

    print(("Test 2 got {} links over {} nodes".format(len(d), len(X))))
    print(d[:4])

    # d = gstar_with_kruskal_init(X, level_thresholds=None, function=None)

    # make sure that singleton data returns empty link list
    n = 1
    X = np.random.rand(n, 2)
    d = generalized_star_clustering_link(X, level_thresholds=None, function=None)
    assert len(d) == 0, "if X.shape[0]==1, you should get an empty list"

    # print(d['nodes'][:2])
    # print(d['links'][:2])
