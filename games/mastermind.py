import random

"""
Python implementation of Mastermind, a puzzle game
in which the user must guess a 4-character pattern
based on feedback from a computer.
"""

def encode(c, g):
    """
    Return the correct c based on guess g of the user.
    """
    out = [""] * len(c) # output array
    for i, (cc, gc) in enumerate(zip(c, g)): # for each correct character
        if gc in c:                          # and guess character
            if gc == cc:
                out[i] = "X"
            else:
                out[i] = "-"
        else:
            out[i] = "O"   
    return "".join(out)

def sii(p, 
        minv, 
        maxv):
    """
    Safe integer input:
    For a given prompt p, minimun value minv, and maximum value
    maxv, return the input when asked how long the code to be guessed 
    should be.
    """
    while True:
        uinput = input(prompt) # userinput
        try:
            uinput = int(uinput)
        except ValueError:
            continue
        if minv <= uinput <= maxv:
            return uinput

print("Welcome to Mastermind.")
print("You will need to guess a random code.")
print("For each guess, you will receive a hint.")
print("In this hint, X denotes a correct letter, and O a letter in the original string but in a different position.")
print()

# number of letters
nol = sii("Select a number of possible letters for the code (2-20): ", 2, 20)

# code length
clength = sii("Select a length for the code (4-10): ", 4, 10)

letters = "ABCDEFGHIJKLMNOPQRST"[:number_of_letters]
code = "".join(random.choices(letters, k=code_length))
guesses = []

while True:
    print()
    guess = input(f"Enter a guess of length {code_length} ({letters}): ").upper().strip()
    if len(guess) != code_length or any([char not in letters for char in guess]):
        continue
    elif guess == code:
        print(f"\nYour guess {guess} was correct!")
        break
    else:
        guesses.append(f"{len(guesses)+1}: {" ".join(guess)} => {" ".join(encode(code, guess))}")
    for i_guess in guesses:
        print("------------------------------------")
        print(i_guess)
    print("------------------------------------")
