from attack import *


def graph_generation(g, solution, device):

    inject_graph = copy.deepcopy(g)
    for node_id, node_data in solution.items():

        for i in range(len(node_data['node_feat'])):
            feature = node_data['node_feat'][i].to(device)
            inject_graph = inject_node(inject_graph, feature)
            for edge in node_data['edges'][i]:
                inject_graph  = wire_edge(inject_graph , edge)
    print(inject_graph)
    return inject_graph