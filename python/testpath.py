import copy as copy
def find_all_paths(graph, start,end, path=[], at_risk = []):
    path = path + [start]
    
    at_risk1 = copy.copy(at_risk)
    at_risk1.extend(graph[start])
    at_risk1 = list(set(at_risk1))
    if start==end:
        return [path]
    paths = []
    for node in at_risk1:
        if node not in path:
            newpaths = find_all_paths(graph, node, end,
                                      path=path
                                      , at_risk = at_risk1)
            for i  in range(len(newpaths)):
                paths.append(newpaths[i])
    return paths

g= {'A': ['B', 'C'],
    'B': [],
    'C': [],
    'D': []}
all = []

for nd in g.keys():
    all.extend(find_all_paths(g, 'A', nd))
