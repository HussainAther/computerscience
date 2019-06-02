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
	        k = comp(binary[i], binary[j])
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

def decimal_to_binary(no_of_variable, minterms):
    """
    Convert from decimal to binary with number of variables and minimum
    terms.
    """
    temp = []
    s = ""
    for m in minterms:
        for i in range(no_of_variable):
            s = str(m%2) + s
            m //= 2
        temp.append(s)
	s = ""
    return temp

def is_for_table(string1, string2, count):
    """
    Are there count mismatches between the two strings?
    """
    l1 = list(string1);l2=list(string2)
    count_n = 0 # Number of mismatches
    for i in range(len(l1)):
        if l1[i] != l2[i]:
	    count_n += 1
    if count_n == count:
        return True
    else:
        return False 

def selection(chart, prime_implicants):
    """
    Select implicants from the chart. 
    """
    temp = []
    select = [0]*len(chart)
    for i in range(len(chart[0])):
        count = 0
	rem = -1
	for j in range(len(chart)):
	    if chart[j][i] == 1:
	        count += 1
		rem = j
	    if count == 1:
		select[rem] = 1
	for i in range(len(select)):
            if select[i] == 1:
	        for j in range(len(chart[0])):
                    if chart[i][j] == 1:
		        for k in range(len(chart)):
			    chart[k][j] = 0 
		temp.append(prime_implicants[i])
    while 1:
        max_n = 0; rem = -1; count_n = 0
	for i in range(len(chart)):
            count_n = chart[i].count(1)
	    if count_n > max_n:
	        max_n = count_n
	        rem = i
        if max_n == 0:
            return temp
	temp.append(prime_implicants[rem])
        for i in range(len(chart[0])):
            if chart[rem][i] == 1:
	        for j in range(len(chart)):
		    chart[j][i] = 0
