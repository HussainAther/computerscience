import sys
import imagematrix

"""
In “Seam Carving for Content-Aware Image Resizing”, Shai Avidan and Ariel Shamir describe a novel method
of resizing images. With a given iamge, we calculate the best vertical seam to remove.
A vertical seam is a connected path of pixels with one pixel in each row. The best vertical seam
minimizes the total "energy" of pixels in the seam.
"""

class ResizeableImage(imagematrix.ImageMatrix):
    def best_seam(self):
        """
        Return coordinates corresponding to the cheapest vertical seam
        to remove from an image.
        """
        # Calculate each energy once.
        energy = {}
        for i in range(self.width):
            for j in range(self.height):
              energy[i,j] = self.energy(i,j)

        dp = {}
        for i in range(self.width):
            dp[i,0] = energy[i,0]

        backpointer = {}
        for j in range(1, self.height):
            for i in range(self.width):
                # Down
                dp[i,j] = energy[i,j] + dp[i,j-1]
                backpointer[i,j] = 0
                # Down-left
                if i != 0:
                    if dp[i,j] > energy[i,j] + dp[i-1,j-1]:
                        dp[i,j] = energy[i,j] + dp[i-1,j-1]
                        backpointer[i,j] = -1
                # Down-right
                if i != self.width-1:
                    if dp[i,j] > energy[i,j] + dp[i+1,j-1]:
                        dp[i,j] = energy[i,j] + dp[i+1,j-1]
                        backpointer[i,j] = 1

        # Find best pixel in bottom row.
        best_value = sys.maxint
        index = None
        for i in range(self.width):
            if dp[i,self.height-1] < best_value:
                best_value = dp[i,self.height-1]
                index = i

        # Follow backpointers up.
        seam = []
        for j in range(self.height-1, 0, -1):
            seam.append((index, j))
            index = index + backpointer[index,j]
        seam.append((index, 0))

        return seam

    def remove_best_seam(self):
        self.remove_seam(self.best_seam())
