import tkinter
import random

"""
In this game player has to enter color of the word that appears on the screen and hence the score increases 
by one, the total time to play this game is 30 seconds. Colors used in this game are Red, Blue, Green, Pink, 
Black, Yellow, Orange, White, Purple and Brown. Interface will display name of different colors in different 
colors. Player has to identify the color and enter the correct color name to win the game.
"""

colors = ["Red","Blue","Green","Pink","Black", 
          "Yellow","Orange","White","Purple","Brown"] 
score = 0

timeleft = 30

def startGame(event):
    """
    Start the game with a given event.
    """
   if timeleft == 30: 
        # start the countdown timer. 
        countdown() 
          
    # run the function to 
    # choose the next color. 
    nextColor() 
  
def nextColor(): 
    """
    Display the next color.
    """
    global score 
    global timeleft 
  
    # if a game is currently in play 
    if timeleft > 0: 
  
        # make the text entry box active. 
        e.focus_set() 
  
        # if the color typed is equal 
        # to the color of the text 
        if e.get().lower() == colors[1].lower(): 
              
            score += 1
  
        # clear the text entry box. 
        e.delete(0, tkinter.END) 
          
        random.shuffle(colors) 
          
        # change the color to type, by changing the 
        # text _and_ the color to a random color value 
        label.config(fg = str(colors[1]), text = str(colors[0])) 
          
        # update the score. 
        scoreLabel.config(text = "Score: " + str(score)) 

def countdown(): 
    """
    It's the final countdown.
    """
    global timeleft 
  
    # if a game is in play 
    if timeleft > 0: 
  
        # decrement the timer. 
        timeleft -= 1
          
        # update the time left label 
        timeLabel.config(text = "Time left: "
                               + str(timeleft)) 
                                 
        # run the function again after 1 second. 
        timeLabel.after(1000, countdown) 

