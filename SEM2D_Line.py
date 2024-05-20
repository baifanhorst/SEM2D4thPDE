# Line class
# Used for straight edges of an element


import numpy as np

# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
sys.path.append('../')

# Reloading the module
import importlib
import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

##########################################################

class Line():
    # Boundary line class
    
    def __init__(self, N, parameter_point_type, point_start, point_end):
        # N: the largest index of boundary points on the curve
        # The index range is 0,1,...,N
        self.N = N
        
        # Starting point and end point
        # Each point is a 1D numpy array containing the x and y coordinates
        self.point_start = point_start
        self.point_end = point_end
        
        # Other parameters created in class functions
        # self.parameter_points:
            # The parameter values corresponding to boundary points (the line is parametrized. In the textbook, the parameter is 's')
            # Created in set_parameter_points
            # These points range from -1 to 1
            # Usually, Legendre/Chebyshev Gauss Lobatto points are used
            # parameter_point_type: 'Legendre', 'Chebyshev'     
        self.set_parameter_points(parameter_point_type)
        
        # self.w_Bary:
            # Barycentric weights corresponding to self.parameter_points
        self.set_BarycentricWeights()
          
        # self.D:
            # Differentiation matrix wrt self.parameter_points
        self.set_DiffMatrix()
        
        # self.x_nodes, self.y_nodes:
            # The coordinates of nodes
        self.cal_coordinates_node()
       
        # self.x_deri_nodes, self.y_deri_nodes:
            # the derivative of x and y coordinates 
            # For a straight line, all derivatives are the same
        self.cal_derivatives_node()
        
        
    
    
    def set_parameter_points(self, point_type):
        # Set the parameter values at the boundary points
        # which are just Legendre-Gauss-Lobatto or Chebyshev-Gauss-Lobatto points
        
        if point_type=='Legendre':
            self.parameter_points = LegendreGaussLobattoNodes(self.N)
        
        if point_type=='Chebyshev':
            self.parameter_points = ChebyshevGaussLobattoNodes_Reversed(self.N)
        

        
    def set_BarycentricWeights(self):
        # Computing the Barycentric weights corresponding to self.parameter_points
        self.w_Bary = BarycentricWeights(self.parameter_points)
        
    def set_DiffMatrix(self):
        # Using the Barycentric weights to get the differentiation matrix wrt self.parameter_points
        self.D = PolynomialDiffMatrix(self.parameter_points, self.w_Bary)
    
    
    def cal_coordinates(self, s):
        # Calculating the coordinates of the point at s for a straight line
        return ((1-s) * self.point_start + (1+s) * self.point_end) / 2
        
    def cal_coordinates_node(self):
        # Calculating the coordinates of all the nodes
        self.x_nodes = ( (1 - self.parameter_points) * self.point_start[0] 
                        +(1 + self.parameter_points) * self.point_end[0] ) / 2
        self.y_nodes = ( (1 - self.parameter_points) * self.point_start[1] 
                        +(1 + self.parameter_points) * self.point_end[1] ) / 2
        
    def cal_derivatives(self, s=None, order=1):
        # Calculating the derivatives of the coordinates at any point on a straight line
        # s is a dummy input, just to make the definition of this function the same as that of the curve class
        # order: the order of the derivative, default is 1
        
        # Currently, only 1st-4th derivatives are allowed.
        if order<1 or order>4:
            print("The order of the derivative must be in [1,4], return None")
            return None
        
        if order == 1:
            deri = (self.point_end - self.point_start) / 2
        else:
            deri = np.zeros(self.point_start.shape)
        
        return deri
    
    def cal_derivatives_node(self, order=1):
        # Calculating the derivatives of the coordinates at all nodes
        # order: the order of the derivative, default is 1
        
        # Currently, only 1st-4th derivatives are allowed.
        if order<1 or order>4:
            print("The order of the derivative must be in [1,4], return None")
            return None
        
        if order == 1:
            x_deri, y_deri = self.cal_derivatives()
            self.x_deri_nodes = np.ones(self.parameter_points.shape) * x_deri
            self.y_deri_nodes = np.ones(self.parameter_points.shape) * y_deri
        else:
            self.x_deri_nodes = np.zeros(self.parameter_points.shape)
            self.y_deri_nodes = np.zeros(self.parameter_points.shape)
        
            
        
    