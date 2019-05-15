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
