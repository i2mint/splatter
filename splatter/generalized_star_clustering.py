
import networkx as nx
from numpy import inf, argmax

__doc__ = """

SIMPLE STAR GRAPH

 We initiate the queue to be all the vertices of the graph G and the set of centers to be empty.
 A vertex $V$ of highest priority (to be define/chosen later on but one can think of vertex degree as in the classical
 star algorithm) in the queue is processed in each iteration.
 The algorithm creates a new star with center $V$ (i.e. adds $V$ to the center set)
 if $V$ has no adjacent centers or if all its adjacent centers have lower priority.

 The latter case destroys the stars adjacent to $V$ (and removes their center from the center set),
 and their satellites are placed in the queue.
 The vertex $V$ is then removed from the queue and the cycle is repeated until the queue is empty.

 One can easily prove that the algorithm terminates by noting that the vertices added to the queue have by construction
 lower priority than the vertex $V$:

 If a destroyed star center $c$ is adjacent to a satellite $s$ of greater priority,
 then $s$ must be adjacent to another center whose priority is equal to or greater than its own.
 Thus, breaking the star associated with $c$ cannot lead to the promotion of $s$ to the status of center,
 so $s$ need not be enqueued.

 In particular, since the priority of satellites placed in the queue are less than the degree of the processed vertex,
 the queue must eventually be empty and the algorithm ends.

 The priority can be any function depending on:

1) the vertex degree

2) the vertex weight

3) the weight of the adjacent vertices

By choosing this priority_func to return the vertex degree, we obtain the regular star algorithm.
By choosing the priority_func to return the weight of the adjacent vertices, we obtain the weighted star algorithm.
The main reason to introduce vertex weight is to be able to generate a hierarchical "star clustering graph"
taking into account the "weight" of star center
(which for example could be how many original vertices this star center is representing).
"""


def vertex_degree_priority_func(vertex, graph):
    return graph.degree(vertex)


def weight_star_priority_func(vertex, graph):
    sum_adj_vert_weight = 0
    for neighbor in graph[vertex]:
        sum_adj_vert_weight += graph[vertex][neighbor]['weight']
    return sum_adj_vert_weight


def degree_and_weight_priority_func(vertex, graph):
    return graph.degree(vertex) * graph.node[vertex]['vertex_weight']


default_priority_func = vertex_degree_priority_func


def vertex_priority(vertex, graph, priority_func=default_priority_func):
    """defines a measure of the importance of a vertex, more important will be picked first out of the queue. The
    basic star algorithm defines the importance as the degree of the vertex"""

    degree = graph.degree(vertex)
    vertex_weight = graph.node[vertex]['vertex_weight']
    total_edge_weight = 0
    for neighbor in graph[vertex]:
        total_edge_weight = total_edge_weight + graph[vertex][neighbor]['weight']

    return priority_func(degree, vertex_weight, total_edge_weight)

#
# def vertex_priority_set(vertex_set, graph, priority_func=default_priority_func):
#     idx_of_max = argmax(list(imap(lambda vertex: priority_func(vertex, graph), vertex_set)))
#     return vertex_set[idx_of_max], priority_func(vertex_set[idx_of_max], graph)


def vertex_priority_set(vertex_set, graph, priority_func=default_priority_func):
    max_priority_vertex = None
    max_priority_value = -inf
    for vertex in vertex_set:
        if priority_func(vertex, graph) > max_priority_value:
            max_priority_vertex = vertex
            max_priority_value = priority_func(vertex, graph)
    return max_priority_vertex, max_priority_value


PRIORITY_VERTEX_IDX = 0
PRIORITY_VAL_IDX = 1


def update_single_generalized(graph, queue, center_set, priority_func=default_priority_func):
    """ takes a graph, set of queued vertices and the list of centers and satellites and returns the graph,
        queue, lists of centers after running the algorithm one iteration over the vertex of largest
        vertex priority in queue
    """

    # find the vertex of largest priority in the queue
    max_queue_vertex = vertex_priority_set(queue, graph, priority_func)[PRIORITY_VERTEX_IDX]

    # find the stars which are the neighbors of max_queue_vertex
    max_queue_star_neigbhors = list(set(graph.neighbors(max_queue_vertex)) & set(center_set))

    # if max_queue_vertex has no star in its neighborhood add max_queue_vertex to center_set
    if max_queue_star_neigbhors == []:
        center_set.append(max_queue_vertex)

    elif priority_func(max_queue_vertex, graph) \
            > vertex_priority_set(max_queue_star_neigbhors, graph, priority_func)[PRIORITY_VAL_IDX]:
        center_set.append(max_queue_vertex)
        center_set = [item for item in center_set if item not in graph.neighbors(max_queue_vertex)]

        # if the degree of max_queue_vertex is larger than than all the vertices in max_queue_star_neigbhors, then add
        # max_queue_vertex to center_set and remove max_queue_star_neigbhors from center_set
        for center in max_queue_star_neigbhors:
            for center_neighbor in graph.neighbors(center):
                # consider all star centers which are neighbors from max_queue_vertex and add their own neighbors
                # to the queue in the condition that their degree is less or equal than the center
                if priority_func(center, graph) > priority_func(center_neighbor, graph) - 1:
                    queue.append(center_neighbor)

    queue.remove(max_queue_vertex)
    # remove max_qeue from queue

    queue = list(set(queue))
    center_set = list(set(center_set))
    # eliminate duplicates (could start off with a set)

    return graph, queue, center_set


def update_iterate_generalized(graph, queue=None, center_set=None, priority_func=default_priority_func):
    if queue is None:
        queue = graph.nodes()
    if center_set is None:
        center_set = []

    while len(queue) > 0:
        graph, queue, center_set = update_single_generalized(graph, queue, center_set, priority_func)
        # iterate the algorithm until the queue is empty

    center_set_dict = dict()

    for vertex in center_set:
        center_set_dict[vertex] = sum(graph.node[x]['vertex_weight'] for x in graph[vertex]) + graph.node[vertex][
            'vertex_weight']

    return center_set, center_set_dict


def interval_subgraph(graph, delta_1, delta_2):
    """create of subgraph of edges with weight greater or equal to delta_1 and less than delta_2. If delta_2 is not a float
    then there is no upper bound."""

    edge_set = list()
    interval_graph = nx.Graph()
    interval_graph.add_nodes_from(graph.nodes(data=True))

    # below is my incompetent way to have the algorithm deal with the desired possibiliy of an infinite delta_2.
    # Thor gave me a nicer method to do that, using np.inf() but I don't want to change anything now, it is late already!

    if type(delta_2) == str:
        for edge in graph.edges(data=True):
            if graph[edge[0]][edge[1]]['weight'] >= delta_1:
                edge_set.append(edge)


    else:
        for edge in graph.edges(data=True):
            if graph[edge[0]][edge[1]]['weight'] < delta_2 and graph[edge[0]][edge[1]]['weight'] >= delta_1:
                edge_set.append(edge)

    interval_graph.add_edges_from(edge_set)
    return interval_graph


def layered_clustering(graph, delta_list, priority_func=default_priority_func):
    delta_list = delta_list + [inf, 0]
    delta_list.sort()
    delta_number = len(delta_list)
    list_edges = []

    # The goal is to make a list of the successive induced graphs on the set of center at each iteration
    # The vertices carry a weight corresponding to the number of vertices they represent

    graph_copy = graph.copy()
    output_graph = nx.Graph()
    output_graph.add_nodes_from(graph.nodes())

    for i in range(delta_number - 1):

        small_delta = delta_list[i]
        large_delta = delta_list[i + 1]
        range_graph = interval_subgraph(graph_copy, small_delta, large_delta).copy()

        update_output = update_iterate_generalized(range_graph, priority_func=priority_func)
        list_edges = list_edges + range_graph.edges(update_output[0], data=True)

        # making the new subgraph with updated vertex weights
        graph_copy = graph_copy.subgraph(update_output[0]).copy()

        for vertex in graph_copy.nodes():
            graph_copy.node[vertex]['vertex_weight'] = update_output[1][vertex]

            # print graph_copy.nodes(data=True)
            # clustering_list.append(graph_copy.nodes(data=True))

    output_graph.add_edges_from(list_edges)

    # return list({clustering_list[x][i][0] for x in range(delta_number-1) for i in range(len(clustering_list[x]))})
    return output_graph
