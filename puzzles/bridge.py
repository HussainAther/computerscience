"""
The Seven Bridges of Königsberg (Konisberg konisberg seven bridges) involves devising a plan
to cross each bridge in a city once and only once.
"""

# Map each node to its connecting nodes using bridges
map = { "a" : ["c", "d"],
        "b" : ["c", "d"],
        "c" : ["a", "b", "d"],
        "d" : ["a", "b", "c"]
       }

walk_len = 2 # to cover all nodes
start_node = "a" # which node to begin
walk = [[start_node]] # where we have walked
for _ in range(walk_len + 1):
    next_nodes = list()
    for node in walk[-1]:
        next_nodes.extend(map[node])
    walk.append(next_nodes)
walk = walk[1:]
