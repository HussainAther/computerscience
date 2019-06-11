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
      
    def set_front(self, l):
        """
        Set the front of the cube.
        """
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__front__ = l
        else:
            raise Exception("Input lists incorrect length")

    def set_left(self, l):
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__left__ = l
        else:
            raise Exception("Input lists incorrect length")

    def set_back(self, l):
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__back__ = l
        else:
            raise Exception("Input lists incorrect length")

    def set_top(self, l):
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__top__ = l
        else:
            raise Exception("Input lists incorrect length")

    def set_right(self, l):
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__right__ = l
        else:
            raise Exception("Input lists incorrect length")

    def set_bottom(self, l):
        if len(l) == 3 & len(l[0]) == 3 & len(l[1]) == 3 & len(l[2]) == 3:
            self.__bottom__ = l
        else:
            raise Exception("Input lists incorrect length") 

    def get_front(self):
        """
        Get the front face of the cube
        """
        return self.__front__
   
    def get_back(self):
        """
        Get the back face of the cube
        """
        return self.__back__
   
    def get_left(self):
        """
        Get the back face of the cube
        """
        return self.__left__

    def get_right(self):
        """
        Get the back face of the cube
        """
        return self.__right__

    def get_top(self):
        """
        Get the back face of the cube
        """
        return self.__top__

    def get_bottom(self):
        """
        Get the back face of the cube
        """
        return self.__bottom__
