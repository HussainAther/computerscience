"""
The Dining Philosopher Problem (dining philosopher) states that n philosophers seated around a 
circular table with one chopstick between each pair of philosophers. There is one chopstick between 
each philosopher. A philosopher may eat if he can pickup the two chopsticks adjacent to him. One 
chopstick may be picked up by any one of its adjacent followers but not both. When a philosopher 
cannot grab both forks it sits and waits. Eating takes random time, then the philosopher puts the 
forks down and leaves the dining room. After spending some random time thinking about the nature 
of the universe, he again becomes hungry, and the circle repeats itself.

Solving this to practice threading.
"""

class phil(threading.Thread):
    """
    Let's get philosophical up in here.
    """
    running = True
    def __init__(self, xname, forkOnLeft, forkOnRight):
        """
        Initialize variables for each person and forks on either side.
        """
        threading.Thread.__init__(self)
        self.name = xname
        self.forkOnLeft = forkOnLeft
        self.forkOnRight = forkOnRight
    def run(self):
        """
        The philosopher is sleeping or thinking. 
        """
        while(self.running):
            #  Philosopher is thinking (but really is sleeping).
            time.sleep(random.uniform(3,13))
            print("%s is hungry." % self.name)
            self.dine()
    def dine(self):
        """
        The philosopher is dining.
        """
        fork1, fork2 = self.forkOnLeft, self.forkOnRight
        while self.running:
            fork1.acquire(True)
            locked = fork2.acquire(False)
            if locked: break
            fork1.release()
            print("%s swaps forks" % self.name)
            fork1, fork2 = fork2, fork1
        else:
            return
        self.dining()
        fork2.release()
        fork1.release()
    def dining(self):		
        """
        After dining, begin sleeping or thinking.
        """	
        print("%s starts eating "% self.name)
        time.sleep(random.uniform(1,10))
        print("%s finishes eating and leaves to think." % self.name)

def DiningPhilosophers():
    forks = [threading.Lock() for n in range(5)]
    philosopherNames = ("Aristotle", "Kant", "Buddha", "Marx", "Russel")
    philosophers= [Philosopher(philosopherNames[i], forks[i%5], forks[(i+1)%5]) \
            for i in range(5)]
    random.seed(12345)
    Philosopher.running = True
    for p in philosophers: p.start()
    time.sleep(100)
    Philosopher.running = False
    print ("Now we're finishing.")
