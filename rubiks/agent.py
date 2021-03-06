import cube as c

"""
Agent performing the actions.
"""

class Agent:
   def __init__(self, cube):
      """
      Initialize the Agent using the proper parameters
      The reference to a Rubik's cube must be passed in
      """
      self.cube = cube
      # All actions are performed in a clockwise rotation of one of six sides
      self.actions = ["front", "back", "left", "right", "top", "bottom"]
      # The memory representaiton o 

   def get_actions(self):
      """
      Returns the action selection
      """
      return self.actions

   def perform_action(self, action):
      """
      Takes in one of the actions and performs the given move on the cube
      """
      
      if action == "front":
         self.cube.rotate_front()
      elif action == "back":
         self.cube.rotate_back()
      elif action == "left":
         self.cube.rotate_left()
      elif action == "right":
         self.cube.rotate_right()
      elif action == "top":
         self.cube.rotate_top()
      elif action == "bottom":
         self.cube.rotate_bottom()
