import copy

"""
Markov Decision Process (MDP) for Grid world using Value iteration algorithm and Prioritized Sweeping.

GridWorld is a 2D rectangular grid of size (Nrows,Ncolumns) with an agent starting off at one grid cell, 
moving from cell to cell through the grid, and eventually exiting after collecting a reward. This grid 
environment is described as follows:

State space: GridWorld has N rows × N columns distinct states. We use s to denote the state. The agent
starts in the bottom-left cell (row 1, column 1, marked as a green cell). There exist one or more terminal
states (blue cells) that can be located anywhere in the grid (except the bottom-left cell). There may also
be walls (red cells) that the agent cannot be moved to.

Actions: ​At every non-terminal state, the agent can either walk or run in any of the four directions
(Up, Down, Left, and Right), which results in 8 possible actions: “Walk Up”, “Walk Down”, “Walk Left”,
“Walk Right”, “Run Up”, “Run Down”, “Run Left”, “Run Right”. At the terminal state, the only possible
action is “Exit”. We use A(s) to denote the set of all possible actions at state s.

Transition model​: GridWorld is stochastic because the actions can be unreliable. In this environment,
action “Walk ​X​” (​X can be Up, Down, Left, or Right) moves the agent one cell in the X direction with
probability pwalk , but with probabilities 0.5 (1 − pwalk ) and 0.5 (1 − pwalk ) moves the agent one
cell at angles of 90° and -90° to the direction ​X​, respectively.

If moving in a particular direction causes the agent to bump into a wall, the movement fails, and
the agent stays in the same cell "i, j" . We write P (s′|s, a) to denote the probability of reaching
state s′ if action a is done in state s . The following examples illustrate the environment dynamics:

Assume that the agent chooses action “Walk Up” at “4,4” as shown in figure below. This action moves
the agent to (5,4) with probability pwalk , but with probability 0.5 (1 − pwalk ) , it moves the
agent right to “4,5”, and with probability 0.5 (1 − pwalk ) , it moves the agent left to "4,3”.

Assume that the agent chooses action “Run Up” at “4,4” as shown in figure below. This action moves
the agent two cells up, but because it causes the agent to bump into a wall, the agent stays at “4,4”
with probability prun . With probability 0.5 (1 − prun) , the agent moves two cells right to “4,6”.
Finally, this action moves the agent two cells left with probability 0.5 (1 − prun) , but because
of the wall at “4,2”, the agent stays at “4,4” with probability 0.5 (1 − prun) . Hence, as a result
of this action, the agent moves to “4,6” with probability 0.5 (1 − prun) and stays at “4,4” with
probability prun + 0.5 (1 − prun) .

Rewards: ​When the agent takes action ​a ​in state s , it receives a reward, R(s, a) . For all non-terminal states, ​s​:
+ R​(​s,​Walk ​X​) ​= Rwalk (a constant).
+ R​(​s,​Run ​X​) ​=​ Rrun (a constant).
Furthermore, if there are K terminal states, the reward in the ​kt​ h terminal state, ​s is ​R​(​s,​Exit) = Rterminal (k) (a constant).

"""

class MDP(object):
    matrix=[];
    pwalk,prun=0.0,0.0;
    rrun,rwalk=0.0,0.0;
    terminal=[];
    m,n=0,0;
    rewards=[];
    wall=[];
    gamma=0.0;
    result=[];
    pq = []
    factor=0;
    policy=["Walk Up","Walk Down","Walk Left","Walk Right","Run Up","Run Down","Run Left","Run Right"];

    def prioritized_sweeping(self):
        while len(self.pq)>0:
            task = self.pq.pop(0);
            self.addNeigbhor(task);

    def addNeigbhor(self,task):
        i=task[0];
        j=task[1];

        if self.isValid(i-2,j) and [i-2,j] not in self.wall and [i-2,j] not in self.terminal:
            old_utility=self.matrix[i-2][j];
            self.matrix[i-2][j]=self.getUtility(i-2,j);
            sigma=abs(self.matrix[i-2][j]-old_utility);
            if  sigma > self.factor and [i-2,j] not in self.pq:
                self.pq.append([i-2,j]);

        if self.isValid(i+2,j) and [i+2,j] not in self.wall and [i+2,j] not in self.terminal:
            old_utility=self.matrix[i+2][j];
            self.matrix[i+2][j]=self.getUtility(i+2,j);
            sigma=abs(self.matrix[i+2][j]-old_utility);
            if sigma > self.factor and [i+2,j] not in self.pq:
                self.pq.append([i+2,j]);

        if self.isValid(i-1,j) and [i-1,j] not in self.wall and [i-1,j] not in self.terminal:
            old_utility=self.matrix[i-1][j];
            self.matrix[i-1][j]=self.getUtility(i-1,j);
            sigma=abs(self.matrix[i-1][j]-old_utility);
            if sigma > self.factor and [i-1,j] not in self.pq:
                self.pq.append([i-1,j]);

        if self.isValid(i+1,j) and [i+1,j] not in self.wall and [i+1,j] not in self.terminal:
            old_utility=self.matrix[i+1][j];
            self.matrix[i+1][j]=self.getUtility(i+1,j);
            sigma=abs(self.matrix[i+1][j]-old_utility);
            if sigma > self.factor and [i+1,j] not in self.pq:
                self.pq.append([i+1,j]);

        if self.isValid(i,j+2) and [i,j+2] not in self.wall and [i,j+2] not in self.terminal:
            old_utility=self.matrix[i][j+2];
            self.matrix[i][j+2]=self.getUtility(i,j+2);
            sigma=abs(self.matrix[i][j+2]-old_utility);
            if sigma > self.factor and [i,j+2] not in self.pq:
                self.pq.append([i,j+2]);

        if self.isValid(i,j-2) and [i,j-2] not in self.wall and [i,j-2] not in self.terminal:
            old_utility=self.matrix[i][j-2];
            self.matrix[i][j-2]=self.getUtility(i,j-2);
            sigma=abs(self.matrix[i][j-2]-old_utility);
            if sigma > self.factor and [i,j-2] not in self.pq:
                self.pq.append([i,j-2]);

        if self.isValid(i,j+1) and [i,j+1] not in self.wall and [i,j+1] not in self.terminal:
            old_utility=self.matrix[i][j+1];
            self.matrix[i][j+1]=self.getUtility(i,j+1);
            sigma=abs(self.matrix[i][j+1]-old_utility);
            if sigma > self.factor and [i,j+1] not in self.pq:
                self.pq.append([i,j+1]);

        if self.isValid(i,j-1) and [i,j-1] not in self.wall and [i,j-1] not in self.terminal:
            old_utility=self.matrix[i][j-1];
            self.matrix[i][j-1]=self.getUtility(i,j-1);
            sigma=abs(self.matrix[i][j-1]-old_utility);
            if sigma > self.factor and [i,j-1] not in self.pq:
                self.pq.append([i,j-1]);

    def isValid(self,i,j):
        if i<0 or j<0 or i>=self.m or j>=self.n:
            return False;

        return True;

    def getUtility(self,i,j):

        direction=[0 for x in range(8)];
        direction[0]=self.getWalkUp(i,j);
        direction[1]=self.getWalkDown(i,j);
        direction[2]=self.getWalkLeft(i,j);
        direction[3]=self.getWalkRight(i,j);
        direction[4]=self.getRunUp(i,j);
        direction[5]=self.getRunDown(i,j);
        direction[6]=self.getRunLeft(i,j);
        direction[7]=self.getRunRight(i,j);

        id=self.getMax(direction);
        self.result[i][j]=self.policy[id];
        return direction[id];


    def getMax(self,direction):
        id=0;
        for i in range(8):
            if direction[i]>direction[id]:
                id=i;
        return id;
        

    def getWalkUp(self,i,j):
        sum=0.0;

        if i-1<0 or [i-1,j] in self.wall:
            sum+=self.pwalk * (self.rwalk+ self.gamma * self.matrix[i][j]);
        else:
            sum+=self.pwalk * (self.rwalk+ self.gamma * self.matrix[i-1][j]);

        if j-1<0 or [i,j-1] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk+ self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk+ self.gamma * self.matrix[i][j-1]);

        if j+1>=self.n or [i,j+1] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk+ self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk+ self.gamma * self.matrix[i][j+1]);

        return sum;

    def getWalkDown(self,i,j):
        sum=0.0;

        if i+1>=self.m or [i+1,j] in self.wall:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i+1][j]);

        if j-1<0 or [i,j-1] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j-1]);

        if j+1>=self.n or [i,j+1] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j+1]);

        return sum;

    def getWalkLeft(self,i,j):
        sum=0.0;

        if j-1<0 or [i,j-1] in self.wall:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i][j-1]);

        if i-1<0 or [i-1,j] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i-1][j]);

        if i+1>=self.m or [i+1,j] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i+1][j]);

        return sum;

    def getWalkRight(self,i,j):
        sum=0.0;

        if j+1>=self.n or [i,j+1] in self.wall:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.pwalk * (self.rwalk + self.gamma * self.matrix[i][j+1]);

        if i-1<0 or [i-1,j] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i-1][j]);

        if i+1>=self.m or [i+1,j] in self.wall:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.pwalk) * (self.rwalk + self.gamma * self.matrix[i+1][j]);

        return sum;

    def getRunUp(self,i,j):
        sum=0.0;

        if i-2<0 or [i-2,j] in self.wall or [i-1,j] in self.wall:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i-2][j]);

        if j-2<0 or [i,j-2] in self.wall or [i,j-1] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j-2]);

        if j+2>=self.n or [i,j+2] in self.wall or [i,j+1] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j+2]);

        return sum;

    def getRunDown(self,i,j):
        sum=0.0;

        if i+2>=self.m or [i+2,j] in self.wall or [i+1,j] in self.wall:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i+2][j]);

        if j-2<0 or [i,j-2] in self.wall or [i,j-1] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j-2]);

        if j+2>=self.n or [i,j+2] in self.wall or [i,j+1] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j+2]);

        return sum;

    def getRunLeft(self,i,j):
        sum=0.0;

        if j-2<0 or [i,j-2] in self.wall or [i,j-1] in self.wall:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j-2]);

        if i-2<0 or [i-2,j] in self.wall or [i-1,j] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i-2][j]);

        if i+2>=self.m or [i+2,j] in self.wall or [i+1,j] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i+2][j]);

        return sum;

    def getRunRight(self,i,j):
        sum=0.0;

        if j+2>=self.n or [i,j+2] in self.wall or [i,j+1] in self.wall:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=self.prun * (self.rrun + self.gamma * self.matrix[i][j+2]);

        if i-2<0 or [i-2,j] in self.wall or [i-1,j] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i-2][j]);

        if i+2>=self.m or [i+2,j] in self.wall or [i+1,j] in self.wall:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i][j]);
        else:
            sum+=0.5 * (1-self.prun) * (self.rrun + self.gamma * self.matrix[i+2][j]);

        return sum;

    def main(self):
        inputFile = open('input.txt', 'r');
        #Read input File
        dimension=inputFile.readline().rstrip('\n').split(",");
        self.m=int(dimension[0]);
        self.n=int(dimension[1]);

        wallNo=int(inputFile.readline().rstrip('\n'));
        for i in range(wallNo):
            wallDim=inputFile.readline().rstrip('\n').split(",");
            self.wall.append([int(self.m-int(wallDim[0])),int(int(wallDim[1])-1)]);

        terminalNo=int(inputFile.readline().rstrip('\n'));
        for i in range(terminalNo):
            terminalDim=inputFile.readline().rstrip('\n').split(",");
            self.terminal.append([int(self.m-int(terminalDim[0])),int(int(terminalDim[1])-1)]);
            self.rewards.append(float(terminalDim[2]));
        
        prob=inputFile.readline().rstrip('\n').split(",");
        self.pwalk=float(prob[0]);
        self.prun=float(prob[1]);

        rew=inputFile.readline().rstrip('\n').split(",");
        self.rwalk=float(rew[0]);
        self.rrun=float(rew[1]);

        self.gamma=float(inputFile.readline().rstrip('\n'));

        self.matrix=[[0 for x in range(self.n)] for y in range(self.m)];
        self.result=[[0 for x in range(self.n)] for y in range(self.m)]

        for i in range(len(self.terminal)):
            temp=self.terminal[i];
            self.matrix[temp[0]][temp[1]]=self.rewards[i];
            self.pq.append(temp);
            self.result[temp[0]][temp[1]]="Exit"

        for w in self.wall:
            self.result[w[0]][w[1]]="None";

        self.prioritized_sweeping();

        output_file = open("output.txt", "w");
        for i in range(self.m):
            output_file.write(",".join(self.result[i]));
            if i<self.m-1:
                output_file.write("\n");
            
mdpObj=MDP();
mdpObj.main();
