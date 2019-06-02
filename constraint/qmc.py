"""
QMC Quine-McCluskey (quine mccluskey) algoritmh or method of prime implicants minimizes
Boolean functions.
"""

def comp(string1, string2):
    """
    Compare two strings.
    """
    l1 = list(string1); l2 = list(string2)
    count = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            count += 1
	    l1[i] = "_"
    if count > 1:
        return -1
    else:
        return("".join(l1))
