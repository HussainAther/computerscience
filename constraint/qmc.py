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

def check(binary):
    """
    Check the binary string for prime implicants.
    """
    pi = []
    while 1:
        check1 = ["$"]*len(binary)
        temp = []
        for i in range(len(binary)):
	    for j in range(i+1, len(binary)):
	        k = compare_string(binary[i], binary[j])
		if k != -1:
		    check1[i] = "*"
		    check1[j] = "*"
		    temp.append(k)
        for  i in range(len(binary)):
            if check1[i] == "$":
                pi.append(binary[i])
        if len(temp) == 0:
            return pi
        binary = list(set(temp))
