"""
Small Parsimony Problem. Find the most parsimonious labeling of the internal nodes of a rooted tree. 
"""

def parsimony(n, edges, labels):
    """
    For an input rooted binary tree n with each leaf labeled by a string of length m,
    output labeling of all other nodes of the tree by strings of length m 
    that minimizes the parsimony score of the tree.
    """
    # m, total node count
    m = 2*n-1    
    
    # Compute tree and reverse-tree from edges
    tree = {}
    parent = {}
    for edge in edges:
        node = edge[0]
        child = edge[1]
        tree.setdefault(node,[]).append(child)
        parent[child] = node

    # Get the root from parent list
    root = parent[parent.keys()[0]]
    while root in parent.keys():
        root = parent[root]

    # l, size of string labels
    l = len(labels[0])
    
    # Alphabet of characters of labels
    alphabet = sorted(list(set(''.join(labels))))
    
    # k, number of characters in string labels
    k = len(alphabet)
    
    # d, dictionnary of every character position in the alphabet
    d = dict(zip(alphabet,range(k)))

    # string labels 
    s = [[" "]*l for i in range(m)]
    
    sk = np.ndarray(shape=(m,k,l), dtype=int)
    
    # maximum parsimony value is (m-1)*l
    sk.fill((m-1)*l)
    
    # fill leaf sk according to labels
    for i,label in enumerate(labels):
        s[i] = list(label)
        for j,c in enumerate(label):
            sk[i,d[c],j] = 0
    
    # Depth-first search for each string element to fill sk values
    for i in range(l):
        def dfs_sk(node):
            if node < n:
                # leaf is at tree botton, simply return
                return
            lnode = tree[node][0]
            rnode = tree[node][1]
            dfs_sk(lnode)
            dfs_sk(rnode)
            for j in range(k):
                mask = np.ones(k)
                mask[j] = 0
                sk[node,j,i] = min(sk[lnode,:,i]+mask) + min(sk[rnode,:,i]+mask)
            return
        dfs_sk(root)
        
    parsimony = sum(sk[root].min(axis=0))
    
    # Depth-first search to back propagate the internal node string s values 
    for i in range(l):        
        def dfs_s(node):
            
            if node < n:
                # leaf is at tree botton, simply return
                return
            c = sk[node,:,i]
            if node == root:
                # when root simply choose the min score ever                
                s[node][i] = alphabet[c.argmin()]
            else:
                pnode = parent[node]
                j = d[s[pnode][i]]
                mask = np.ones(k)
                mask[j] = 0
                c += mask
                s[node][i] = alphabet[c.argmin()]
            
            lnode = tree[node][0]
            rnode = tree[node][1]
            dfs_s(lnode)
            dfs_s(rnode)
        dfs_s(root)
    
    ret = []
    for node,(lnode,rnode) in tree.iteritems():
        ps = "".join(s[node])
        ls = "".join(s[lnode])
        rs = "".join(s[rnode])
        ret.append((ps,ls))
        ret.append((ps,rs))
    # left and right branches
    lbranch = ("".join(s[root]), "".join(s[tree[node][0]]))
    rbranch = ("".join(s[root]), "".join(s[tree[node][1]]))
    return (parsimony,ret[:], rbranch, lbranch)

