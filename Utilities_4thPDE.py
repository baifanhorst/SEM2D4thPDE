import numpy as np
from numpy import pi, cos, sin, log, sqrt, exp


def set_Uth_S_RHS_BC_SingleElement_ver1(element_list):
    # Set up exact solution, source, RHS for a single element
    # The equation is the steady potential
    # The BCs are Dirichlet 
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = x**4 + 2 * y**4 + 3 * x**2 * y**2 + sin(x) + 2 * cos(x) + exp(x) 
        # Source term
        e.S = 18*x**2 + 30*y**2 + exp(x) - sin(x) - 2*cos(x)
        # RHS
        e.RHS = np.copy(e.U_th)
        e.RHS[1:-1, 1:-1] = e.S[1:-1, 1:-1] 
        

def set_Uth_S_RHS_BC_SingleElement_ver2(element_list):
    # Set up exact solution, source, RHS for a single element
    # The equation is the steady potential
    # The BCs are Dirichlet 
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = x + 4 * y + 3 * x**3 * y**2 +  y**4 + sin(x**2) + 2 * cos(x**3) + exp(-x**2) 
        # Source term
        e.S = -18*x**4*cos(x**3) + 6*x**3 - 4*x**2*sin(x**2) + 4*x**2*exp(-x**2) + 18*x*y**2 - 12*x*sin(x**3) + 12*y**2 + 2*cos(x**2) - 2*exp(-x**2)
        # RHS
        e.RHS = np.copy(e.U_th)
        e.RHS[1:-1, 1:-1] = e.S[1:-1, 1:-1] 
        
        

  


def set_BC_edge(element, location, coeffs):
    # Set BC for a single edge
    # element: the element that the target edge belongs to
    # location: location of the BC
        # location = (edge_idx, layer, gap_start, gap_end)
        # edge_idx: the index of the edge to which the BC is applied
        # layer: 0, 1,... whether the BC is applied to the outmost or the first inner layer and so on
        # gap_start: the number of grid points skipped in the beginning
        # gap_end: the number of grid points skipped at the end
        # e.g. location = (3, 1, 1, 1), the BC is applied to the first inner layer under edge 3
        # with the first and the last grid points on that layer skipped. Then the grid
        # index range is i = 1,...,N-1, j=M-1
    # coeffs: the coefficients in front of the derivative terms in the BC 
    # Assume that the highest derivative order is 3
        # coeffs = (A00, A10, A01, A20, A11, A02, A30, A21, A12, A03, rhs)
    
    imin, imax, jmin, jmax = element.set_LocationBCEquation(location)
        
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            print(i,j)



def set_Uth_S_RHS_ver1(element_list):
    # Set up exact solution, source, RHS
    # The exact solution is x^2+y^2
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = x**2 + y**2 
        # Source term
        e.S = 4 * np.ones(x.shape)
        # RHS
        e.RHS = e.J * e.S
     
        
    # Return two derivative functions for setting BC 
    def func_ux(x,y):
        return 2*x
    
    def func_uy(x,y):
        return 2*y
    
    return func_ux, func_uy
        

def set_Uth_S_RHS_ver2(element_list):
    # Set up exact solution, source, RHS
    # The exact solution is sin(x) + sin(y)
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = sin(x) + sin(y)
        # Source term
        e.S = - sin(x) - sin(y)
        # RHS
        e.RHS = e.J * e.S
        
    # Return two derivative functions for setting BC 
    def func_ux(x,y):
        return cos(x)
    
    def func_uy(x,y):
        return cos(y)
    
    return func_ux, func_uy
        
def set_Uth_S_RHS_ver3(element_list):
    # Set up exact solution, source, RHS
    # The exact solution is sin(x) + sin(y)
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = exp(x**2 + y**2)
        # Source term
        e.S = exp(x**2 + y**2) * (4 * x**2 + 4 * y**2 + 4)
        # RHS
        e.RHS = e.J * e.S
        
    # Return two derivative functions for setting BC 
    def func_ux(x,y):
        return exp(x**2 + y**2) * 2 * x
    
    def func_uy(x,y):
        return exp(x**2 + y**2) * 2 * y
    
    return func_ux, func_uy
        
def set_Uth_S_RHS_ver4(element_list):
    # Set up exact solution, source, RHS
    # The exact solution is sin(x) + sin(y)
    for e in element_list:
        x = e.nodes_phy_x
        y = e.nodes_phy_y
        # Exact solution
        e.U_th = x**2 + 2 * y**2 + sin(x) + exp(y)
        # Source term
        e.S = 6 - sin(x) + exp(y)
        # RHS
        e.RHS = np.copy(e.S)
        
    # Return two derivative functions for setting BC 
    def func_ux(x,y):
        return 2*x + cos(x)
    
    def func_uy(x,y):
        return 4*y + exp(y)
    
    return func_ux, func_uy
    

def set_BC_Dirichlet(element_list, alpha):
    # Dirichlet boundary conditions:
    for e in element_list:
        Nx = e.grid.Nx
        Ny = e.grid.Ny
        
        # Curve 1
        j = 0
        e.RHS[:,j] = e.U_th[:,j]
        # Curve 3
        j = Ny
        e.RHS[:,j] = e.U_th[:,j]
        
        # Curve 4
        i = 0
        e.RHS[i,:] = e.U_th[i,:] 
        # Curve 2
        i = Nx
        e.RHS[i,:] = e.U_th[i,:] 
        
        
def set_BC_BoundaryInnerNodes(e, func_ux, func_uy, alpha, beta):
    # Set boundary BC for element e
    # func_ux, func_uy: derivative functions
    # alpha, beta: boundary condition coefficients
    
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    
    # There are several ways to input alpha and beta:
    # (1) numbers, (2) lists, (3) 2D arrays of shape (Nx+1, Ny+1)
    
    if (not isinstance(alpha, np.ndarray)) and (not isinstance(alpha, list)):
        e.alpha = np.ones((Nx+1, Ny+1)) * alpha
    elif isinstance(alpha, np.ndarray):
        e.alpha = alpha
    elif isinstance(alpha, list):
        e.alpha = np.zeros((Nx+1, Ny+1))
        # Curve 1 
        e.alpha[1:Nx, 0] = alpha[0]
        # Curve 2 
        e.alpha[Nx, 1:Ny] = alpha[1]
        # Curve 3 
        e.alpha[1:Nx, Ny] = alpha[2]
        # Curve 4 
        e.alpha[0, 1:Ny] = alpha[3]
    
    if (not isinstance(beta, np.ndarray)) and (not isinstance(beta, list)):
        e.beta = np.ones((Nx+1, Ny+1)) * beta
    elif isinstance(beta, np.ndarray):
        e.beta = beta
    elif isinstance(beta, list):
        e.beta = np.zeros((Nx+1, Ny+1))
        # Curve 1 
        e.beta[1:Nx, 0] = beta[0]
        # Curve 2 
        e.beta[Nx, 1:Ny] = beta[1]
        # Curve 3 
        e.beta[1:Nx, Ny] = beta[2]
        # Curve 4 
        e.beta[0, 1:Ny] = beta[3]  
      
        
    for curve_idx, mark_curve in enumerate(e.mark_curves):
        if mark_curve=='boundary':
            if curve_idx + 1 == 1:
                j = 0
                for i in range(1, Nx):
                    x = e.nodes_phy_x[i,j]
                    y = e.nodes_phy_y[i,j]
                    nx, ny = e.norm_vect[i,j]
                    pu_pn = func_ux(x,y) * nx + func_uy(x,y) * ny
                    e.RHS[i,j] = e.alpha[i,j] * e.U_th[i,j] + e.beta[i,j] * pu_pn
                        
                    
            if curve_idx + 1 == 2:
                i = Nx
                for j in range(1, Ny):
                    x = e.nodes_phy_x[i,j]
                    y = e.nodes_phy_y[i,j]
                    nx, ny = e.norm_vect[i,j]
                    pu_pn = func_ux(x,y) * nx + func_uy(x,y) * ny
                    e.RHS[i,j] = e.alpha[i,j] * e.U_th[i,j] + e.beta[i,j] * pu_pn
                        
                        
            if curve_idx + 1 == 3:
                j = Ny
                for i in range(1, Nx):
                    x = e.nodes_phy_x[i,j]
                    y = e.nodes_phy_y[i,j]
                    nx, ny = e.norm_vect[i,j]
                    pu_pn = func_ux(x,y) * nx + func_uy(x,y) * ny
                    e.RHS[i,j] = e.alpha[i,j] * e.U_th[i,j] + e.beta[i,j] * pu_pn
                        
            if curve_idx + 1 == 4:
                i = 0
                for j in range(1, Ny):
                    x = e.nodes_phy_x[i,j]
                    y = e.nodes_phy_y[i,j]
                    nx, ny = e.norm_vect[i,j]
                    pu_pn = func_ux(x,y) * nx + func_uy(x,y) * ny
                    e.RHS[i,j] = e.alpha[i,j] * e.U_th[i,j] + e.beta[i,j] * pu_pn
                    
                    
def set_BC_CornerNodes(e):
    # Set the Dirichlet boundary condition for corner nodes
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    for i in [0, Nx]:
        for j in [0, Ny]:
            e.RHS[i,j] = e.U_th[i,j]
            
            
            
            

    
            
    
            
        
                    
        
        
        
        

