# Grid class for the grid on the computational (square) domain

import numpy as np

# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
sys.path.append('../')

# Reloading the module
import importlib
import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

class Grid():
    # A class that stores information of the nodes/collocation points
    # and related stuff
    def __init__(self, Nx, Ny, node_type_x, node_type_y):
        # The maximum index of nodes in the x direction
        # index = 0,1,...,N
        self.Nx = Nx
        # The maximum index of nodes in the y direction
        self.Ny = Ny
        
        # class variables created in other functions:
        # 1D Nodes in the x and y direction
        # self.nodes_x, self.nodes_y
        # Gauss Lobatto weights for the x and y nodes
        #self.w_GL_x, self.w_GL_y
        self.cal_nodes(node_type_x, node_type_y)
        
        # The Barycentric weights corresponding to self.nodes_x, self.nodes_y
        # self.w_bary_x, self.w_bary_y:
        self.cal_BarycentricWeights()
        
        # The differentiating matrices wrt x and y
        # self.Dx, self.Dy
        self.cal_DiffMatrix()
       
    
    def cal_nodes(self, node_type_x, node_type_y):
        # Generating the nodes along x and y directions
        # as well as the Gauss Lobatto weights
        # node_type_x, node_type_y: types of the nodes,
        # including: Legendre, Chebyshev
        
        if node_type_x=="Legendre":
            self.nodes_x = LegendreGaussLobattoNodes(self.Nx)
            self.w_GL_x = LegendreGaussLobattoWeights(self.nodes_x)
        elif node_type_x=="Chebyshev":
            self.nodes_x = ChebyshevGaussLobattoNodes_Reversed(self.Nx)
            self.w_GL_x = ChebyshevGaussLobattoWeights(self.nodes_x)
        else:
            print("Unknown node type")
            
        if node_type_y=="Legendre":
            self.nodes_y = LegendreGaussLobattoNodes(self.Ny)
            self.w_GL_y = LegendreGaussLobattoWeights(self.nodes_y)
        elif node_type_y=="Chebyshev":
            self.nodes_y = ChebyshevGaussLobattoNodes_Reversed(self.Ny)
            self.w_GL_y = ChebyshevGaussLobattoWeights(self.nodes_y)
        else:
            print("Unknown node type")
            
    def cal_BarycentricWeights(self):
        # Computing the Barycentric weights for the x and y nodes
        self.w_bary_x = BarycentricWeights(self.nodes_x)
        self.w_bary_y = BarycentricWeights(self.nodes_y)
            
            
    def cal_DiffMatrix(self):
        # Computing the differentiation matrix for the x and y directions
        self.Dx = PolynomialDiffMatrix(self.nodes_x, self.w_bary_x)
        self.Dy = PolynomialDiffMatrix(self.nodes_y, self.w_bary_y)
        
        # Computing the differentiation matrices up to 4th order
        
        # The 0th differentiation matrices are identical matrices
        Ix = np.identity(self.Nx + 1)
        Iy = np.identity(self.Ny + 1)
        self.Dx_list = [Ix, self.Dx]
        self.Dy_list = [Iy, self.Dy]
        
        
        for order in [2,3,4]:
            self.Dx_list.append(PolynomialDiffMatrix_HighOrder(order, self.nodes_x, self.w_bary_x, self.Dx))
            self.Dy_list.append(PolynomialDiffMatrix_HighOrder(order, self.nodes_y, self.w_bary_y, self.Dy))

            
    
        
        
        
        
        
        
        