from classify import *
import math
import numpy as np

"""
Methods for solving general constraint problems (csp CSP).
"""

from csp import BinaryConstraint, CSP, CSPState, Variable,\
    basic_constraint_checker, solve_csp_problem

def forward_checking(state, verbose=False):
    # Before running Forward checking we must ensure
    # that constraints are okay for this state.
    basic = basic_constraint_checker(state, verbose)
    if not basic:
        return False

    # Add your forward checking logic here.
    
    raise NotImplementedError


def forward_checking_prop_singleton(state, verbose=False):
    # Run forward checking first.
    fc_checker = forward_checking(state, verbose)
    if not fc_checker:
        return False

    # Add your propagate singleton logic here.
    raise NotImplementedError

from moose import moose_csp_problem
from mapcoloring import map_coloring_csp_problem

def csp_solver_tree(problem, checker):
    problem_func = globals()[problem]
    checker_func = globals()[checker]
    answer, search_tree = problem_func().solve(checker_func)
    return search_tree.tree_to_string(search_tree)


senate_people = read_congress_data("data/S110.ord")
senate_votes = read_vote_data("data/S110desc.csv")

house_people = read_congress_data("data/H110.ord")
house_votes = read_vote_data("data/H110desc.csv")

last_senate_people = read_congress_data("data/S109.ord")
last_senate_votes = read_vote_data("data/S109desc.csv")


### Nearest Neighbors
## An example of evaluating a nearest-neighbors classifier.
senate_group1, senate_group2 = crosscheck_groups(senate_people)
# evaluate(nearest_neighbors(hamming_distance, 1), senate_group1, senate_group2, verbose=1)

def euclidean_distance(list1, list2):
    return np.linalg.norm(a-b)(list1, list2)

# Once you have implemented euclidean_distance, you can check the results:
# evaluate(nearest_neighbors(euclidean_distance, 1), senate_group1, senate_group2)

## a classifier that makes at most 3 errors on the Senate.

my_classifier = nearest_neighbors(hamming_distance, 1)
#evaluate(my_classifier, senate_group1, senate_group2, verbose=1)

### ID Trees
# print(CongressIDTree(senate_people, senate_votes, homogeneous_disorder))

## information_disorder function to replace homogeneous_disorder. Thisshould lead to simpler trees.

def information_disorder(yes, no):
    return homogeneous_disorder(yes, no)

# print(CongressIDTree(senate_people, senate_votes, information_disorder))
# evaluate(idtree_maker(senate_votes, homogeneous_disorder), senate_group1, senate_group2)

## Now try it on the House of Representatives. However, do it over a data set
## that only includes the most recent n votes, to show that it is possible to
## classify politicians without ludicrous amounts of information.

def limited_house_classifier(house_people, house_votes, n, verbose = False):
    house_limited, house_limited_votes = limit_votes(house_people,
    house_votes, n)
    house_limited_group1, house_limited_group2 = crosscheck_groups(house_limited)

    if verbose:
        print("ID tree for first group:")
        print(CongressIDTree(house_limited_group1, house_limited_votes, information_disorder))
        print("ID tree for second group:")
        print(CongressIDTree(house_limited_group2, house_limited_votes, information_disorder))
        
    return evaluate(idtree_maker(house_limited_votes, information_disorder),
                    house_limited_group1, house_limited_group2)

                                   
## Find n that classifies at least 430 representatives correctly.
N_1 = 10
rep_classified = limited_house_classifier(house_people, house_votes, N_1)

## Find n that classifies at least 90 senators correctly.
N_2 = 10
senator_classified = limited_house_classifier(senate_people, senate_votes, N_2)

## Find n that classifies at least 95 of last year's senators correctly.
N_3 = 10
old_senator_classified = limited_house_classifier(last_senate_people, last_senate_votes, N_3)
