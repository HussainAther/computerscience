"""
The   fusc   integer sequence is defined as:
+ fusc(0) = 0
+ fusc(1) = 1
+ for n>1, the nth term is defined as:
+ + if n is even;
+ + + fusc(n) = fusc(n/2)
+ + if n is odd;     
+ + + fusc(n) = fusc((n-1)/2) + fusc((n+1)/2)
"""
