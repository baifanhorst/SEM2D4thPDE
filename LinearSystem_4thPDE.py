import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
sys.path.append('../symbolic')

# Reloading the module
import importlib

import Coefficients
importlib.reload(Coefficients)
from Coefficients import *



def map_idx(i, j, element, element_list):
    # Convert the grid index (i,j) of the input element 
    # to the index n of the whole linear system Cx=d 
    
    # i,j: the indices of grid of the input element
        # i=0,...,N, j=0,...,M, N,M are the maxium x and y grid labels of the element
    # element: input element    
    # element_list: list of all elements
        # The (0,0) node of each element corresponds to an index in C
        # which depends on the number of nodes of all previous elements
        # Thus, we need element_list
    
    
    # Create a list of numbers of nodes of all elements
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
    
    #idx_start: the starting index (the index of (0,0))
    # idx_start = the number of total grid nodes of all previous elements
    idx_start =  np.sum(num_nodes_list[0:element.idx], dtype=np.int64)
   
    # Within each element, the 2D nodal values are stretched into 1D as follows:
    # (u00, u01, ..., u0M,   u10, u11, ...,u1M, ..., uN0, uN1, ..., uNM)
    
    M = element.grid.Ny
    return i*(M+1) + j + idx_start


def map_idx_edgecommon_to_element(i, edge, element_idx):
    # Given a common edge node index i, find the node index pair
    # of the neighboring element labeled by element_idx (=0, 1)
    
    # Extract the element, the curve label, marker for alignment
    element, curve_idx, mark_align = edge.element_info[element_idx]
    
    Nx = element.grid.Nx
    Ny = element.grid.Ny

    
    if mark_align == 1:
        
        if curve_idx == 1:
            return (i, 0)
        elif curve_idx == 2:
            return (Nx, i)
        elif curve_idx ==3:
            return (i, Ny)
        elif curve_idx ==4:
            return (0, i)
        
    elif mark_align == -1:
        
        if curve_idx == 1:
            return (Nx-i, 0)
        elif curve_idx == 2:
            return (Nx, Ny-i)
        elif curve_idx ==3:
            return (Nx-i, Ny)
        elif curve_idx ==4:
            return (0, Ny-i)



    

def MatEqnConverter_Collocation_Init(element_list):
    # Initializing the linear system Cx = d
    # Create a list of number of nodes of all elements
    # The number of nodes of each element is its total number of grid points
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
        
    # Get the total number of nodes
    # Note that np.sum returns float as default
    # We must convert it to int in order to use it as an index
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64)
    
    # The linear system is Cx = d
    C = np.zeros((num_nodes_total, num_nodes_total))
    d = np.zeros(num_nodes_total)
    
    # Row index 
    ind_1st = 0
    
    return C, d, ind_1st


def MatEqnConverter_Collocation_SingleEquation(C, d, ind_1st, e, i, j, element_list, A_list, rhs, order_total):
    # Constructing the matrix and rhs for a single equation
    # The inner node equations and BC equations have the same form
    # This function gives a unified treatment
    # C, d, ind_1st: the linear system Cx=d and the starting index of the current equation
    # e: element
    # (i,j): grid point where the equation belongs to
    # element_list: list of all elements, used to find column index (ind_2nd)
    # A_list: list of coefficient functions
        # A_list = [[A00, A01, A02, A03, A04],[A10, A11, A12, A13],[A20, A21, A22],[A30, A31],[A40]]
        # or A_list = [[A00, A01, A02, A03],[A10, A11, A12],[A20, A21],[A30]]
    # rhs: rhs of the equation. For inner nodes, this is the source term. For BC, this is the function on the rhs of the BC equation
    # order_total: the total order of the partial derivatives. For inner nodes, this is 4. For BC, this is 3.
    
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    d[ind_1st] = rhs
    
    # Coordinates
    x = e.nodes_phy_x[i,j]
    y = e.nodes_phy_y[i,j]
    
    # Map derivatives
    map_deri = e.get_mapderi_singlenode(i, j)
    
    # Contribution of derivatives
    # 1<= a+b <= order_total
    for a_plus_b in range(1, order_total+1):
        # a = 0,..., a+b
        for a in range(0, a_plus_b+1):
            b = a_plus_b - a
            
            # Evaluate the coefficient at the current grid point
            A = A_list[a][b](x,y)
            
            # 1<= r+s <= a+b
            for r_plus_s in range(1, a_plus_b + 1):
                # r = 0,...,r+s
                for r in range(0, r_plus_s + 1):
                    s = r_plus_s - r
                    
                    # Evaluate the coefficient in front of the partial derivative wrt xi and eta
                    B = coeff_B(a, b, r, s, map_deri)
                    
                    # Add the contribution
                    for n in range(0, N+1):
                        for m in range(0, M+1):
                            ind_2nd = map_idx(n, m, e, element_list)
                            C[ind_1st, ind_2nd] += A * B * e.grid.Dx_list[r][i,n] * e.grid.Dy_list[s][j,m]
    
    
    # Contribution of the 0th order term
    ind_2nd = map_idx(i, j, e, element_list)
    C[ind_1st, ind_2nd] += A_list[0][0](x,y)

    
    
    # Update the equation index
    ind_1st += 1
    
    return C, d, ind_1st
    


def MatEqnConverter_Collocation_InnerNodes(C, d, ind_1st, element_list, locations, A_list):
    # Constructing the matrix and rhs for inner nodes
    
    # C, d: the initialized matrix and rhs of Cx = d
    # ind_1st: the index of the first equation to be created
    
    # locations = [(gap_i_start, gap_i_end, gap_j_start, gap_j_end), ... ]
        # This specifies the grid points to be skipped
        # i = gap_i_start, ..., N - gap_i_end. Similar for j
    
    # element_list: list of elements
    # A_list: coefficients list, which are functions
        # A_list = [[A00, A01, A02, A03, A04],[A10, A11, A12, A13],[A20, A21, A22],[A30, A31],[A40]]
        # Thus, A_list[i,j] = Aij
    
    # Constructing the inner node part of C and d
    for k, e in enumerate(element_list):

        imin, imax, jmin, jmax = e.set_LocationInnerNodeEquation(locations[k])        
        
        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                rhs = e.RHS[i,j]
                C, d, ind_1st = MatEqnConverter_Collocation_SingleEquation(C, d, ind_1st, 
                                                                           e, i, j, 
                                                                           element_list, A_list, rhs, order_total=4)
    return C, d, ind_1st



def MatEqnConverter_Collocation_Patching(edgecommon_list, element_list, C, d, ind_1st):
    # Constructing equations for nodes on common edges by using patching conditions
    # edgecommon_list: list of common edges
    # element_list :list of elements
    # C, d, ind_1st: result from MatEqnConverter_Collocation_InnerNodes
    
    
    for edge in edgecommon_list:
        # Patching the values
        for s in range(1, edge.N): # s = 1,...,N-1
            d[ind_1st] = 0
            
            for edge_element_idx in [0,1]: # edge_element_idx: the index of the neighboring elements of the common edge, only two values (0,1)
                element, curve_idx, mark_align = edge.element_info[edge_element_idx]
                # Finding the node index pair (i,j) in the neighboring element
                # corresponding to s
                i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                ind_2nd = map_idx(i, j, element, element_list)
                # The equation is element0nodevalue - element1nodevalue =0, thus each element's contribution to C has a different sign
                C[ind_1st, ind_2nd] = (-1)**edge_element_idx 
            
            ind_1st += 1
        
        # Patching the derivatives
        for s in range(1, edge.N): # s = 1,...,N-1
            d[ind_1st] = 0
            for edge_element_idx in [0,1]:
                element, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                
                MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, 1.0, ind_1st, C)

            ind_1st += 1

    return C, d, ind_1st



def MatEqnConverter_Collocation_BC(C, d, ind_1st, e, element_list, location, A_list):
    # Constructing the matrix and rhs for BSs
    
    # C, d: the initialized matrix and rhs of Cx = d
    # ind_1st: the index of the first equation to be created
    
    # e: the current element
    # element_list: list of elements
    
    # location = (edge_idx, layer, gap_start, gap_end)
        # See the element class' functon 'set_LocationBCEquation' for details
    
    # A_list: coefficients list, which are functions
        # A_list = [[A00, A01, A02, A03, A04],[A10, A11, A12, A13],[A20, A21, A22],[A30, A31],[A40]]
        # Thus, A_list[i,j] = Aij
    

    
    # Constructing the BC part of C and d
    
    # Find the index range for the BC
    imin, imax, jmin, jmax = e.set_LocationBCEquation(location)
    
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            rhs = e.RHS[i,j]
            C, d, ind_1st = MatEqnConverter_Collocation_SingleEquation(C, d, ind_1st, e, i, j, element_list, A_list, rhs, order_total=3)

    return C, d, ind_1st




def MatEqnSolver_NonsquareCollocation(C, d, element_list):
    # Solving the matrix Cu = d and reshape the solution
    u = np.linalg.solve(C, d)
    
    
    # Create a list of number of nodes of all elements
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
    
    # Reshape the solution
    for k, element in enumerate(element_list):
        idx_start = np.sum(num_nodes_list[0:k], dtype=np.int64)
        idx_end = np.sum(num_nodes_list[0:k+1], dtype=np.int64)
        Nx = element.grid.Nx
        Ny = element.grid.Ny
        element.U = u[idx_start : idx_end].reshape(Nx+1, Ny+1)
        
        
###########################################
# Visualization
###########################################    

    
def visualizing_all(elements):
    # Showing all elements' grid
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for k, element in enumerate(elements):
        ax.scatter(element.nodes_phy_x, element.nodes_phy_y, label=f"Element {k}")
    ax.legend()
    ax.set_aspect('equal')
        
   

def visualization_error(element_list):
    # Visualization of absolute error
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
    error_max_list = []
    for element in element_list:
        ax.plot_surface(element.nodes_phy_x, element.nodes_phy_y, np.abs(element.U - element.U_th), cmap=cm.coolwarm)
        error_max_list.append( np.max(np.abs(element.U - element.U_th)) )
    ax.set_title(f"Max error: {np.max(error_max_list)}")
       
    
    
def visualization_solution_2Dcontour(element_list):
    # Visualization
    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    # Min and max node values, used to adjust contour plot color scale
    data_mins = []
    data_maxs = []
    for element in element_list:
        data_mins.append(np.min(element.U))
        data_maxs.append(np.max(element.U))
    data_min = np.min(data_mins)
    data_max = np.max(data_maxs)
    
    # Location of the levels
    num_level = 10
    levels = np.linspace(data_min, data_max, num_level+1)
    
    
    for element in element_list:
        ct_numerical = ax[0].contourf(element.nodes_phy_x, element.nodes_phy_y, element.U, levels=levels, vmin=data_min, vmax=data_max)
        ct_th = ax[1].contourf(element.nodes_phy_x, element.nodes_phy_y, element.U_th, levels=levels, vmin=data_min, vmax=data_max)

    '''
    # Colorbars
    # Calculate ratio to adjust the colorbar size
    # ct_ratio = element.grid.Nx / element.grid.Ny
    ct_ratio = element.U.shape[0] / element.U.shape[1]
    cb_numerical = fig.colorbar(ct_numerical, ax=ax[0], fraction=0.05*ct_ratio, pad=0.04)
    cb_th = fig.colorbar(ct_th, ax=ax[1], fraction=0.05*ct_ratio, pad=0.04)
    
    # Change the tick label size
    cb_tick_label_size = 8
    cb_numerical.ax.tick_params(labelsize=cb_tick_label_size)
    cb_th.ax.tick_params(labelsize=cb_tick_label_size)
    '''
    
    
    
    # x,y labels
    label_size = 10
    for i in range(2):
        ax[i].set_xlabel(r'$x$', fontsize=label_size)
        ax[i].set_ylabel(r'$y$', rotation=0, fontsize=label_size)
    
    # Tick labels
    tick_label_size = 8
    for i in range(2):
        ax[i].tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax[i].tick_params(axis='both', which='minor', labelsize=tick_label_size)

    # Titles
    title_size = 10
    ax[0].set_title("Numerical", fontsize=title_size)
    ax[1].set_title("Theoretical", fontsize=title_size)


    # Shape adjustment
    # Aspect ratio
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    # Hozirontal spacing
    fig.subplots_adjust(wspace=0.5)  # Adjust the horizontal spacing

    # Save figure
    filename = "./figs/patching_.jpg"
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    
def visualizing_C(C):
    # Showing nonzero entries of C
    idx_nonzero_x, idx_nonzero_y = np.nonzero(C)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(idx_nonzero_x, idx_nonzero_y, s=0.25, color='black')
    
    
    







####################################################
# Old codes
####################################################
def MatEqnConverter_Collocation_InnerNodes_old(element_list):
    # Constructing the matrix and rhs for inner nodes
    
    # element_list: list of elements
    
    
    # Create a list of number of nodes of all elements
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
        
    # Get the total number of nodes
    # Note that np.sum returns float as default
    # We must convert it to int in order to use it as an index
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64)
    
    # The linear system is Cx = d
    C = np.zeros((num_nodes_total, num_nodes_total))
    d = np.zeros(num_nodes_total)
    
    # Row index 
    ind_1st = 0
    
    # Constructing the inner node part of C and d
    for k, e in enumerate(element_list):
        
        N = e.grid.Nx
        M = e.grid.Ny
        
        D_xi=e.grid.Dx
        D_eta=e.grid.Dy
    
        coeff_xi = (e.X_xi**2 + e.Y_xi**2) / e.J
        coeff_eta = (e.X_eta**2 + e.Y_eta**2) / e.J
        coeff_mixed = (e.X_xi * e.X_eta + e.Y_xi * e.Y_eta) / e.J
    
        RHS = e.RHS
        
        for i in range(1, N): # i=1,...,N-1
            for j in range(1,M): # j=1,...,M-1
                # Set d
                d[ind_1st] = RHS[i,j]
        
                # Set C
                # Find C[ind_1st,:]
                # The terms in 4 double sums are distributed to C
                for k in range(0,N+1): # k=0,...,N
                    for n in range(0,N+1): # n=0,...,N
                        # Find the 1D index n corresponding to (n,j)
                        ind_2nd = map_idx(n, j, e, element_list)
                        C[ind_1st, ind_2nd] += D_xi[i,k] * coeff_eta[k,j] * D_xi[k,n]
        
                for k in range(0,N+1): # k=0,...,N
                    for m in range(0,M+1): # m=0,...,M
                        # Find the 1D index n corresponding to (k,m)
                        ind_2nd = map_idx(k, m, e, element_list)
                        C[ind_1st, ind_2nd] -= D_xi[i,k] * coeff_mixed[k,j] * D_eta[j,m]
        
                for l in range(0,M+1): # l=0,...,M
                    for n in range(0,N+1): # n=0,...,N
                        # Find the 1D index n corresponding to (n,l)
                        ind_2nd = map_idx(n, l, e, element_list)
                        C[ind_1st, ind_2nd] -= D_eta[j,l] * coeff_mixed[i,l] * D_xi[i,n]
        
                for l in range(0,M+1): #l=0,...,M
                    for m in range(0, M+1): # m=0,...,M
                        # Find the 1D index n corresponding to (i,m)
                        ind_2nd = map_idx(i, m, e, element_list)
                        C[ind_1st, ind_2nd] += D_eta[j,l] * coeff_xi[i,l] * D_eta[l,m]
        
                # Update the equation index
                ind_1st += 1
    
            
    return C, d, ind_1st


def MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, coeff, ind_1st, C):
    # When setting up the patching of common edge normal derivatves
    # and Neunmann or Robin BC, we have equations where RHS includes normal derivatives
    # Given an element and its boundary node (i,j), this function set the contribution
    # of the normal derivative terms to C
    # For the patching condition, we simply have: normal derivative of element 1 + normal derivative of element 2 = 0
    # For Neumann and Robin BC, there is a coefficient in front of the normal derivative
    # This coefficient is 'coeff' here
    # ind_1st: the index of the equation to be contributed, which is also the row index of C
    
    Nx = element.grid.Nx
    Ny = element.grid.Ny

    D_xi = element.grid.Dx
    D_eta = element.grid.Dy

    J = element.J[i,j]

    X_xi = element.X_xi[i,j]
    X_eta = element.X_eta[i,j]
    Y_xi = element.Y_xi[i,j]
    Y_eta = element.Y_eta[i,j]

    nx = element.norm_vect[i,j,0]
    ny = element.norm_vect[i,j,1]
    
    coeff_u_xi = (nx * Y_eta - ny * X_eta) / J
    coeff_u_eta = (-nx * Y_xi + ny * X_xi) / J
    
    for n in range(0, Nx+1):
        ind_2nd = map_idx(n, j, element, element_list)
        C[ind_1st, ind_2nd] += coeff_u_xi * D_xi[i,n] * coeff

    for m in range(0, Ny+1):
        ind_2nd = map_idx(i, m, element, element_list)
        C[ind_1st, ind_2nd] += coeff_u_eta * D_eta[j,m] * coeff
        
        
def MatEqnConverter_Collocation_BC_old2(alpha, beta, element_list, C, d, ind_1st):
    # Constructing equations for boundary nodes
    # The BC equation is alpha * u + beta * pu_pn = gamma
    # alpha, beta: two coefficients in the equation
    # RHS: the boundary nodes of RHS include values of gamma
    # element_list: list of all elements
    # C, d: the linear system Cx=d
    # ind_1st: the starting row index of the equations
    
    for element in element_list:
        
        RHS = element.RHS
        
        Nx = element.grid.Nx
        Ny = element.grid.Ny
        
        for curve_idx, mark_curve in enumerate(element.mark_curves):
            if mark_curve=='boundary':
                if curve_idx + 1 == 1:
                    j = 0
                    for i in range(1, Nx):
                        
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += alpha
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, beta, ind_1st, C)

                        ind_1st += 1
                        
                    
                if curve_idx + 1 == 2:
                    i = Nx
                    for j in range(1, Ny):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += alpha
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, beta, ind_1st, C)

                        ind_1st += 1
                    
                if curve_idx + 1 == 3:
                    j = Ny
                    for i in range(1, Nx):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += alpha
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, beta, ind_1st, C)

                        ind_1st += 1
                    
                if curve_idx + 1 == 4:
                    i = 0
                    for j in range(1, Ny):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += alpha
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, beta, ind_1st, C)

                        ind_1st += 1
                    
    return C, d, ind_1st


def MatEqnConverter_Collocation_BC_old(element_list, C, d, ind_1st):
    # Constructing equations for boundary nodes
    # The BC equation is alpha * u + beta * pu_pn = gamma
    # RHS: the boundary nodes of RHS include values of gamma
    # element_list: list of all elements
    # C, d: the linear system Cx=d
    # ind_1st: the starting row index of the equations
    
    
        
    for element in element_list:
        
        RHS = element.RHS
        
        Nx = element.grid.Nx
        Ny = element.grid.Ny
        
        for curve_idx, mark_curve in enumerate(element.mark_curves):
            if mark_curve=='boundary':
                if curve_idx + 1 == 1:
                    j = 0
                    for i in range(1, Nx):
                        
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += element.alpha[i,j]
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, element.beta[i,j], ind_1st, C)

                        ind_1st += 1
                        
                    
                if curve_idx + 1 == 2:
                    i = Nx
                    for j in range(1, Ny):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += element.alpha[i,j]
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, element.beta[i,j], ind_1st, C)

                        ind_1st += 1
                    
                if curve_idx + 1 == 3:
                    j = Ny
                    for i in range(1, Nx):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += element.alpha[i,j]
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, element.beta[i,j], ind_1st, C)

                        ind_1st += 1
                    
                if curve_idx + 1 == 4:
                    i = 0
                    for j in range(1, Ny):
                        d[ind_1st] = RHS[i,j]
                        
                        # Contribution of alpha * u
                        ind_2nd = map_idx(i, j, element, element_list)
                        C[ind_1st, ind_2nd] += element.alpha[i,j]
                        # Contribution of beta * pu_pn
                        MatEqnConverter_Collocation_NormalDerivativeContribution(i, j, element, element_list, element.beta[i,j], ind_1st, C)

                        ind_1st += 1
                    
    return C, d, ind_1st


def MatEqnConverter_Corner(element_list, C, d, ind_1st, avg=None):
    # By default, applying Dirichlet BC to corner nodes
    # The BC values are stored at the corner nodes of RHS_list
    # element_list: list of all elements
    # C,d: the linear system Cx=d
    # ind_1st: the starting row index of C
    
    # If avg=='average', then a corner value is the average of neighboring values 
    
    if avg==None:
        for element in element_list:
            RHS = element.RHS
            Nx = element.grid.Nx
            Ny = element.grid.Ny
            for i in [0, Nx]:
                for j in [0, Ny]:
                    d[ind_1st] = RHS[i,j]
                    ind_2nd = map_idx(i, j, element, element_list)
                    C[ind_1st, ind_2nd] = 1
                    ind_1st += 1
                    
    if avg=='average':
        for element in element_list:
            Nx = element.grid.Nx
            Ny = element.grid.Ny
            
            # Corner 1
            d[ind_1st] = 0
            ind_2nd = map_idx(0, 0, element, element_list)
            C[ind_1st, ind_2nd] = 1
            #idx_neighbor = [(1,0), (0,1), (1,1)]
            idx_neighbor = [(1,0), (0,1)]
            for i,j in idx_neighbor:
                ind_2nd = map_idx(i, j, element, element_list)
                C[ind_1st, ind_2nd] = -1/len(idx_neighbor)
            ind_1st += 1
            
            # Corner 2
            d[ind_1st] = 0
            ind_2nd = map_idx(Nx, 0, element, element_list)
            C[ind_1st, ind_2nd] = 1
            #idx_neighbor = [(Nx-1,0), (Nx,1), (Nx-1,1)]
            idx_neighbor = [(Nx-1,0), (Nx,1)]
            for i,j in idx_neighbor:
                ind_2nd = map_idx(i, j, element, element_list)
                C[ind_1st, ind_2nd] = -1/len(idx_neighbor)
            ind_1st += 1
            
            # Corner 3
            d[ind_1st] = 0
            ind_2nd = map_idx(Nx, Ny, element, element_list)
            C[ind_1st, ind_2nd] = 1
            #idx_neighbor = [(Nx-1,Ny), (Nx,Ny-1), (Nx-1,Ny-1)]
            idx_neighbor = [(Nx-1,Ny), (Nx,Ny-1)]
            for i,j in idx_neighbor:
                ind_2nd = map_idx(i, j, element, element_list)
                C[ind_1st, ind_2nd] = -1/len(idx_neighbor)
            ind_1st += 1
            
            # Corner 4
            d[ind_1st] = 0
            ind_2nd = map_idx(0, Ny, element, element_list)
            C[ind_1st, ind_2nd] = 1
            #idx_neighbor = [(1,Ny), (0,Ny-1), (1,Ny-1)]
            idx_neighbor = [(1,Ny), (0,Ny-1)]
            for i,j in idx_neighbor:
                ind_2nd = map_idx(i, j, element, element_list)
                C[ind_1st, ind_2nd] = -1/len(idx_neighbor)
            ind_1st += 1
            
                
    return C, d, ind_1st