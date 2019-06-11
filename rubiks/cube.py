class Cube:
   """
   Initialize the cube using a give size.
   """
   def __init__(self):
      self.__size__ = 3
      self.__front__ = [[0 for x in range(self.__size__)] for x in range(self.__size__)]
      self.__left__ = [[1 for x in range(self.__size__)] for x in range(self.__size__)]
      self.__back__ = [[2 for x in range(self.__size__)] for x in range(self.__size__)]
      self.__top__ = [[3 for x in range(self.__size__)] for x in range(self.__size__)]
      self.__right__ = [[4 for x in range(self.__size__)] for x in range(self.__size__)]
      self.__bottom__ = [[5 for x in range(self.__size__)] for x in range(self.__size__)]
      
