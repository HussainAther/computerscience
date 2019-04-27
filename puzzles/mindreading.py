"""
Solving this without import.

You are a Magician and a Mind Reader Extraordinaire. The Assistant goes into the
audience with an authentic deck of 52 cards while you are outside the room and can’t
possibly see anything. Five audience members each select one card from the deck. The
Assistant then gathers up the five cards. The Assistant shows the entire audience four
cards, one at a time. For each of these four cards, the Assistant asks the audience to
mentally focus on the card, while you look away and try to read their collective minds.
Then, after a few seconds you are shown the card. This helps you calibrate your mind
reading to the particular audience.

After you see these four cards, you claim that you are well calibrated to this audience and
leave the room. The Assistant shows the fifth card to the audience and puts it away.
Again, the audience mentally focuses on the fifth card. You return to the room,
concentrate for a short time and correctly name the secret, fifth card!

You are in cahoots with your Assistant and have planned and practiced this trick.
However, everyone is watching closely and the only information that the Assistant can
give you is the four cards.

How does this trick work?

It turns out that the order in which the Assistant reveals the cards tells the Magician what
the fifth card is! The Assistant needs to be able to decide which of the cards is going to be
hidden – he or she cannot allow the audience to pick the hidden card out of the five cards
that the audience picks. Here’s one way that the Assistant and the Magician can work
together.
"""

deck = ["A_C", "A_D", "A_H", "A_S", "2_C", "2_D", "2_H", "2_S", "3_C", "3_D", "3_H", "3_S",
        "4_C", "4_D", "4_H", "4_S", "5_C", "5_D", "5_H", "5_S", "6_C", "6_D", "6_H", "6_S",
        "7_C", "7_D", "7_H", "7_S", "8_C", "8_D", "8_H", "8_S", "9_C", "9_D", "9_H", "9_S",
        "10_C", "10_D", "10_H", "10_S", "J_C", "J_D", "J_H", "J_S",
        "Q_C", "Q_D", "Q_H", "Q_S", "K_C", "K_D", "K_H", "K_S"]

def outputFirstCard(numbers, oneTwo, cards):
    """
    This procedure figures out which card should be hidden based on the distance
    between the two cards that have the same suit.
    It returns the hidden card, the first exposed card, and the distance
    """
    encode = (numbers[oneTwo[0]] - numbers[oneTwo[1]]) % 13
    if encode > 0 and encode <= 6:
        hidden = oneTwo[0]
        other = oneTwo[1]
    else:
        hidden = oneTwo[1]
        other = oneTwo[0]
        encode = (numbers[oneTwo[1]] - numbers[oneTwo[0]]) % 13
    print("First card is:", cards[other])
    return hidden, other, encode

def outputNext3Cards(code, ind):
    """
    Simple if-else in accordance with problem.
    """
    if code == 1:
        second, third, fourth = ind[0], ind[1], ind[2]
    elif code == 2:
        second, third, fourth = ind[0], ind[2], ind[1]
    elif code == 3:
        second, third, fourth = ind[1], ind[0], ind[2]
    elif code == 4:
        second, third, fourth = ind[1], ind[2], ind[0]
    elif code == 5:
        second, third, fourth = ind[2], ind[0], ind[1]
    else:
        second, third, fourth = ind[2], ind[1], ind[0]
    print ("Second card is:", deck[second])
    print ("Third card is:", deck[third])
    print ("Fourth card is:", deck[fourth])

def sortList(tlist):
    """
    Implement sorting list elements in ascending order.
    """
    for index in range(0, len(tlist)-1):
        ismall = index
        for i in range(index, len(tlist)):
            if tlist[ismall] > tlist[i]:
                ismall = i
        tlist[index], tlist[ismall] = tlist[ismall], tlist[index]

    return

def ComputerAssistant():
    """
    Randomly generate five cards.
    """
    print ("Cards are character strings as shown below.")
    print ("Ordering is:", deck)
    cards, cind, cardsuits, cnumbers = [], [], [], []
    numsuits = [0, 0, 0, 0]
    number = 0
    while number < 99999:
        number = int(input("Please give random number of at least 6 digits:"))
    """
    Generate five "random" numbers from the input number
    """
    clist = []
    i = 0
    while len(clist) < 5:
        number = number * (i + 1) // (i + 2)
        n = number % 52
        i += 1
        if not n in clist:
            clist.append(n)
   for i in range(5):
        n = clist[i]
        cards.append(deck[n])
        cind.append(n)
        cardsuits.append(n % 4)
        cnumbers.append(n // 4)
        numsuits[n % 4] += 1
        if numsuits[n % 4] > 1:
            pairsuit = n % 4
    cardh = []
    for i in range(5):
        if cardsuits[i] == pairsuit:
            cardh.append(i)
    hidden, other, encode = outputFirstCard(cnumbers, cardh, cards)
    remindices = []
    for i in range(5):
        if i != hidden and i != other:
            remindices.append(cind[i])
    sortList(remindices)
    outputNext3Cards(encode, remindices)
    guess = input("What is the hidden card?")
    if guess == cards[hidden]:
        print("You are a Mind Reader Extraordinaire!")
    else:
        print("Sorry, not impressed!")
    return
