import numpy as np
import matplotlib.pyplot as plt


# Element class
class Element():
    def __init__(self, idx, shape, grid, curves, corners, mark_curves):
        # idx: element index, which is also the index of the element in the element list
        self.idx = idx
        
        # Shape of the element: 'quad', 'curved'
        # 'quad': quadrilateral with four straight edges
        # 'curved': curved quadrilateral, at least one edge is curved
        # We distinguish between the two shapes because the straight quadrilaterals, 
        # the physical nodes and derivatives are calculated differently
        self.shape = shape
        
        # grid: grid class
        self.grid = grid
        # Extract dimensions for further use
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        
        # Computing the total number of nodes
        # Used in constructing the whole linear system for all elements
        self.num_nodes = (Nx+1) * (Ny+1)
        
        
        # curves: list or tuple of curve objects
        # corners: list or tuple of corner coordinates
        # The corners are in counterclockwise direction
            # x1 -> x2 -> x3 -> x4
            # The edges are also in couterclockwise direction
            # e1 -> e2 -> e3 -> e4
            # The edges are directional. 
            # Starting and ending corners for each edge:
                # e1: x1, x2
                # e2: x2, x3
                # e3: x4, x3
                # e4: x1, x4
                
        self.curves = curves
        self.corners = corners
        
        # mark_curve = [mark1, mark2, mark3, mark4]
        # Markers of the four curves to indicate whether they are boundary or inner common edges
        # each marker takes two values: 'boundary', 'common'
        self.mark_curves = mark_curves
       
        # Computational nodes (xi, eta nodes)
        # indexing = 'ij': this guarantees that nodes_comp_x[i,j] = grid.nodes_x[i]
        self.nodes_comp_x, self.nodes_comp_y = np.meshgrid(self.grid.nodes_x,
                                                           self.grid.nodes_y,
                                                           indexing='ij')    
        
        
        # Physical nodes (x,y nodes)
        self.nodes_phy_x = np.zeros((Nx+1, Ny+1))
        self.nodes_phy_y = np.zeros((Nx+1, Ny+1))
        
        if self.shape == 'quad':
            self.cal_QuadMap_nodes()
        elif self.shape == 'curved':
            self.cal_Map_nodes()
            
            
        # Metrics (derivatives)
        # Old notations
            # self.X_xi: partial x partial xi
            # self.X_eta = partial x partial eta
            # self.Y_xi: partial y partial xi
            # self.Y_eta = partial y partial eta
        
        # Map derivatives of high orders
        # self.x_deri, self.y_deri
        # e.g. self.x[1][2]: \partial^3 x \partial xi \partial eta^2
   
        
        if self.shape == 'quad':
            self.cal_QuadMapDerivatives_nodes()
        elif self.shape == 'curved':
            self.cal_MapDerivatives_nodes()
            
        
        
        # Jacobian 
        self.J = np.zeros((Nx+1, Ny+1))
        self.cal_Jacobian()
        
        
        # The scaling factors and normal vectors on the four boundaries
        # self.scal_lower, self.norm_vect_lower
        # self.scal_upper, self.norm_vect_upper
        # self.scal_left, self.norm_vect_left
        # self.scal_right, self.norm_vect_right
        # self.norm_vect: This is a 3D array with boundary values be the normal derivatives
        # This is used when setting patching conditions or Neumann BC
        # The corner values takes one of the normal derivatives, but this does not matter since
        # we never use the normal derivatives at a corner node
        self.cal_normal_vector_nodes()
        
        
        # Arrays related to the collocation method
        # Source term, shape = (Nx+1, Ny+1), not created by class functions
        self.S = None
        # RHS, shape = (Nx+1, Ny+1), not created by class functions
        # The inner nodes store the rhs grid values of the governing equation
        # The boundary nodes store the BCs.
        self.RHS = None
        # Exact solution, shape = (Nx+1, Ny+1), not created by class functions
        self.U_th = None
        # Numerical solution, shape = (Nx+1, Ny+1), not created by class functions
        self.U = None
        # General boundary condition coefficients: alpha * u + beta * pu_pn = gamma
        # Alpha and beta are (Nx+1, Ny+1) 2D arrays
        # where only boundary nodes store alpha and beta values
        # gamma is stored at boundary nodes of RHS
        self.alpha = None
        self.beta = None
        
        
        #######################
        # Deprecated
        #######################
        
        # Locations for inner node equations
        # locations_InnerNodeEquation = [gap_i_start, gap_i_end, gap_j_start, gap_j_end]
            # This specifies the grid points to be skipped
            # i = gap_i_start, ..., N - gap_i_end. 
            # j = gap_j_start, ..., M - gap_j_end. 
        
        #self.location_InnerNodeEquation = None
        
        
        
        
    ########################################
    # Functions for quadrilateral maps
    ########################################
    
    def cal_QuadMap(self, xi, eta):
        # Finding the physical coordinnates for the quadrilateral map 
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        # Notation correspondence with David's book
        x1 = self.corners[0]
        x2 = self.corners[1]
        x3 = self.corners[2]
        x4 = self.corners[3]
        
        result = 1/4 * (x1 * (1-xi) * (1-eta)
                      + x2 * (1+xi) * (1-eta)
                      + x3 * (1+xi) * (1+eta)
                      + x4 * (1-xi) * (1+eta) )
        return result
    
    def cal_QuadMap_nodes(self):
        # Computing the physical nodes for the quadrilateral map 
        
        # Notation in the textbook
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.nodes_phy_x = 1/4 * ( x1 * (1-xi) * (1-eta)
                                 + x2 * (1+xi) * (1-eta)
                                 + x3 * (1+xi) * (1+eta)
                                 + x4 * (1-xi) * (1+eta) )
                                  
        
        
        self.nodes_phy_y = 1/4 * ( y1 * (1-xi) * (1-eta)
                                 + y2 * (1+xi) * (1-eta)
                                 + y3 * (1+xi) * (1+eta)
                                 + y4 * (1-xi) * (1+eta) )
    
        

    
    
    def cal_QuadMapDerivatives(self, coordinate, order, xi, eta):
        # Finding the partial derivatives for the map to a quadrilateral
        # at a single point (xi, eta)
        # coordinate: 'x', 'y', the variable to be differentiated
        # order: (i,j), the order wrt xi and eta
            # e.g: (2,0), \partial^2 (...) \partial xi^2
        
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        if coordinate == 'x':
            if order == (0, 0):
                return self.cal_QuadMap(xi, eta)[0]
            if order == (1, 0):
                return  0.25 * (1-eta) * (x2 - x1) + 0.25 * (1+eta) * (x3 - x4)
            elif order == (0, 1):
                return 0.25 * (1-xi) * (x4 - x1) + 0.25 * (1+xi) * (x3 - x2)
            elif order == (1,1):
                return 0.25 * (x1 + x3 - x2 - x4)
            else:
                return 0
            
        if coordinate == 'y':
            if order == (0, 0):
                return self.cal_QuadMap(xi, eta)[1]
            if order == (1, 0):
                return  0.25 * (1-eta) * (y2 - y1) + 0.25 * (1+eta) * (y3 - y4)
            elif order == (0, 1):
                return 0.25 * (1-xi) * (y4 - y1) + 0.25 * (1+xi) * (y3 - y2)
            elif order == (1,1):
                return 0.25 * (y1 + y3 - y2 - y4)
            else:
                return 0

      
    
    def cal_QuadMapDerivatives_nodes(self):
        # Finding the partial derivatives for the map to a quadrilateral
        # at all nodes
        
        # Initialize map derivatives
        # They are stored in a nested list
        # Note that x00 represents the map itself (no differentiation), whose value won't be used.
        # self.x_deri = [[x00, x01, x02, x03, x04],[x10, x11, x12, x13],[x20, x21, x22],[x30, x31],[x40]]
        # xij = self.x_deri[i][j]
        
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        
        # Zero map derivatives
        # For the quad map, many high order derivatives are zeros.
        # We can use the same zero matrix to save storage.
        deri_zeros = np.zeros((Nx+1, Ny+1))
        
        # Initialize the x derivatives
        self.x_deri = []
        for i in range(0, 4+1): # i=0,...,4
            deri = []
            for j in range(0, 4-i + 1): # j=0,...,4-i
                deri.append(deri_zeros)    
            self.x_deri.append(deri)
        
        # Initialize the y derivatives
        self.y_deri = []    
        for i in range(0, 4+1): # i=0,...,4
            deri = []
            for j in range(0, 4-i + 1): # j=0,...,4-i
                deri.append(deri_zeros)    
            self.y_deri.append(deri)

        
        # Old codes for 1st derivatives
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.X_xi = 0.25 * (1-eta) * (x2 - x1) + 0.25 * (1+eta) * (x3 - x4)
        self.Y_xi = 0.25 * (1-eta) * (y2 - y1) + 0.25 * (1+eta) * (y3 - y4)
                   
        self.X_eta = 0.25 * (1-xi) * (x4 - x1) + 0.25 * (1+xi) * (x3 - x2)
        self.Y_eta = 0.25 * (1-xi) * (y4 - y1) + 0.25 * (1+xi) * (y3 - y2) 
        
        # Computing all map derivatives
        # Only need to compute nonzero derivatives
        self.x_deri[1][0] = self.X_xi
        self.x_deri[0][1] = self.X_eta
        self.x_deri[1][1] = 0.25 * (x1 + x3 - x2 - x4) * np.ones(self.nodes_comp_x.shape)
        
        self.y_deri[1][0] = self.Y_xi
        self.y_deri[0][1] = self.Y_eta
        self.y_deri[1][1] = 0.25 * (y1 + y3 - y2 - y4) * np.ones(self.nodes_comp_x.shape)
        
        # x_deri[0][0] and y_deri[0][0] contain the physical nodes
        self.x_deri[0][0] = self.nodes_phy_x
        self.y_deri[0][0] = self.nodes_phy_y
        
    #########################################################
    # Functions for general maps
    #########################################################
    
    def cal_Map(self, xi, eta):
        # Finding the physical coordinnates for the map 
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        
        curve1 = self.curves[0]
        curve2 = self.curves[1]
        curve3 = self.curves[2]
        curve4 = self.curves[3]
        
        x1 = self.corners[0]
        x2 = self.corners[1]
        x3 = self.corners[2]
        x4 = self.corners[3]
        
        # Evaluate the curves at xi or eta
        
        # Evaluate the curves at xi or eta
        curve1_xi = curve1.cal_coordinates(xi)
        curve3_xi = curve3.cal_coordinates(xi)
        curve2_eta = curve2.cal_coordinates(eta)
        curve4_eta = curve4.cal_coordinates(eta)
        
        
        result = 1/2 * ((1-eta) * curve1_xi
                      + (1+eta) * curve3_xi
                      + (1+xi) * curve2_eta
                      + (1-xi) * curve4_eta) \
               - 1/4 * (1-xi) * ((1-eta) * x1 + (1+eta) * x4 ) \
               - 1/4 * (1+xi) * ((1-eta) * x2 + (1+eta) * x3 )
        return result
    
    
    def cal_MapDerivatives(self, xi, eta):
        # Finding the partial derivatives for a general map 
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
        
        curve1 = self.curves[0]
        curve2 = self.curves[1]
        curve3 = self.curves[2]
        curve4 = self.curves[3]
        
        x1 = self.corners[0]
        x2 = self.corners[1]
        x3 = self.corners[2]
        x4 = self.corners[3]
        
        # The partial derivatives of curve1 and 3 wrt xi
        p_curve1_p_xi = curve1.cal_derivatives(xi)
        p_curve3_p_xi = curve3.cal_derivatives(xi)
        # Evaluate curve2 and 4 at eta
        curve2_eta = curve2.cal_coordinates(eta)
        curve4_eta = curve4.cal_coordinates(eta)
        
        
        # Metrics wrt xi
        X_xi, Y_xi = 1/2 * (curve2_eta - curve4_eta 
                            + (1-eta)*p_curve1_p_xi 
                            + (1+eta)*p_curve3_p_xi) \
                    -1/4 * ((1-eta)*(x2 - x1) 
                          + (1+eta)*(x3 - x4))
        
        # The partial derivatives of curve2 and 4 wrt eta
        p_curve2_p_eta = curve2.cal_derivatives(eta)
        p_curve4_p_eta = curve4.cal_derivatives(eta)
        # Evaluate curve1 and 1 at xi
        curve1_xi = curve1.cal_coordinates(xi)
        curve3_xi = curve3.cal_coordinates(xi)
        # Metrics wrt eta
        X_eta, Y_eta = 1/2 * ((1-xi)*p_curve4_p_eta + (1+xi)*p_curve2_p_eta
                             - curve1_xi + curve3_xi) \
                     - 1/4 * ((1-xi)*(x4-x1)
                             +(1+xi)*(x3-x2))
        
        return X_xi, Y_xi, X_eta, Y_eta
    
    
    def cal_Map_nodes(self):
        # Computing the physical nodes for general map
        for i in range(self.grid.Nx + 1): #i=0,1,...,Nx
            for j in range(self.grid.Ny + 1): #j=0,1,...,Ny
                xi = self.nodes_comp_x[i,j]
                eta = self.nodes_comp_y[i,j] 
                self.nodes_phy_x[i,j], self.nodes_phy_y[i,j] = self.cal_Map(xi, eta)
                
                
    def cal_MapDerivatives_nodes(self):
        # Computing the derivatives for general map
        for i in range(self.grid.Nx + 1): #i=0,1,...,Nx
            for j in range(self.grid.Ny + 1): #j=0,1,...,Ny
                xi = self.nodes_comp_x[i,j]
                eta = self.nodes_comp_y[i,j]
                self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j] = \
                                    self.cal_MapDerivatives(xi, eta)
                                    
    def cal_Jacobian(self):
        # Computing the Jabobian at nodes
        self.J = self.X_xi * self.Y_eta - self.X_eta * self.Y_xi  
        
        
    def cal_normal_vector_nodes(self):
        # Computing the normal vectors on the boundary
        
        # Cartesian basis
        ex = np.array([1,0])
        ey = np.array([0,1])
        
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        
        
        # Create a norm vector 2D array to store normal vectors
        # This is used when setting patching conditions and Neumann BCs
        # Only boundary inner nodes of these arrays will be used
        self.norm_vect = np.zeros((Nx+1, Ny+1, 2))
        

        
        
        
        # 'Lower boundary'
        j = 0
        self.norm_vect_lower = np.zeros((Nx + 1, 2))
        self.scal_lower = np.zeros(Nx + 1)
        for i in range(Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_lower[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_lower[i] = -sign_J / self.scal_lower[i] * (X_xi * ey - Y_xi * ex)
            self.norm_vect[i,j] = self.norm_vect_lower[i]
            
        # 'Upper boundary'
        j = Ny
        self.norm_vect_upper = np.zeros((Nx + 1, 2))
        self.scal_upper = np.zeros(Nx + 1)
        for i in range(Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_upper[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector 
            self.norm_vect_upper[i] = sign_J / self.scal_upper[i] * (X_xi * ey - Y_xi * ex)
            self.norm_vect[i,j] = self.norm_vect_upper[i]
            
        # 'Left boundary'
        i = 0
        self.norm_vect_left = np.zeros((Ny + 1, 2))
        self.scal_left = np.zeros(Ny + 1)
        for j in range(Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_left[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_left[j] = -sign_J / self.scal_left[j] * (Y_eta * ex - X_eta * ey)
            self.norm_vect[i,j] = self.norm_vect_left[j]
            
            
            
        # 'Right boundary'
        i = Nx
        self.norm_vect_right = np.zeros((Ny + 1, 2))
        self.scal_right = np.zeros(Ny + 1)
        for j in range(Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_right[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector
            self.norm_vect_right[j] = sign_J / self.scal_right[j] * (Y_eta * ex - X_eta * ey)
            self.norm_vect[i,j] = self.norm_vect_right[j]
    
    
    ####################################        
    # Visualization           
    ####################################      
    
    def visualizing_curves(self):
        # Visualization of the curve nodes
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, c in enumerate(self.curves):
            ax.scatter(c.x_nodes, c.y_nodes, label='Curve {}'.format(i+1))
        ax.legend()
        ax.set_title('Boundary nodes')
        ax.set_aspect('equal')
    
        
        
    def visualizing_Grid_NormalVector(self):
        # Visualizing the nodes and the normal vectors
        # The normal vectors are from four 1D arrays
        fig, ax = plt.subplots(nrows=1, ncols=1)
        Nx = self.grid.Nx
        Ny = self.grid.Ny

        ax.scatter(self.nodes_phy_x, self.nodes_phy_y)

        j = 0
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect_lower[:,0], self.norm_vect_lower[:,1])
        
        j = Ny   
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect_upper[:,0], self.norm_vect_upper[:,1])    

        i = 0
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect_left[:,0], self.norm_vect_left[:,1])

        i = Nx
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect_right[:,0], self.norm_vect_right[:,1])   

        ax.set_title('Grid, Normal Vector')
        ax.set_aspect('equal')
    
    
    
        
        
    
    
    #########################################
    # Functions for constructing the linear system
    #########################################

    def set_LocationInnerNodeEquation(self, location):
        # Given the gaps stored in location, compute the grid point range 
        # for inner node equations
        # Used in the construction of the linear system
        # location = (gap_i_start, gap_i_end, gap_j_start, gap_j_end)
            # gap_i_start: the number of points skipped at the beginning of the x direction
            # gap_i_end: the number of points skipped at the end of the x direction
            # gap_j_start, gap_j_end are similar
        
        N = self.grid.Nx
        M = self.grid.Ny
        
        gap_i_start, gap_i_end, gap_j_start, gap_j_end = location
        # Starting i index
        imin = gap_i_start
        # End i index
        imax = N - gap_i_end
        # Starting j index
        jmin = gap_j_start
        # End j index
        jmax = M - gap_j_end
        
        return imin, imax, jmin, jmax
    
    def set_LocationBCEquation(self, location):
        # Given the information in location, compute the grid point range
        # for BC equations
        # Used in the construction of the linear system
        
        # location: location of the BC
            # location = (edge_idx, layer, gap_start, gap_end)
            # edge_idx: the index of the edge to which the BC is applied
            # layer: 0, 1,... whether the BC is applied to the outmost or the first inner layer and so on
            # gap_start: the number of grid points skipped in the beginning
            # gap_end: the number of grid points skipped at the end
            # e.g. location = (3, 1, 1, 1), the BC is applied to the first inner layer under edge 3
            # with the first and the last grid points on that layer skipped. Then the grid
            # index range is i = 1,...,N-1, j=M-1
        
        edge_idx, layer, gap_start, gap_end = location
        
        N = self.grid.Nx
        M = self.grid.Ny
        
        if edge_idx == 1:
            imin = gap_start
            imax = N - gap_end
            jmin = layer
            jmax = jmin
        elif edge_idx == 2:
            imin = N - layer
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
        elif edge_idx == 3:
            imin = gap_start
            imax = N - gap_end
            jmin = M - layer
            jmax = jmin
        elif edge_idx == 4:
            imin = layer
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
            
        return imin, imax, jmin, jmax
        
    def get_mapderi_singlenode(self, i, j):
        # Get all map derivatives at a single node (i,j)
        map_deri = []
        
        for a_plus_b in range(1, 4+1): #a+b = 1,2,3,4
            for b in range(0, a_plus_b+1): # b = 0,1,...,a+b
                a = a_plus_b - b
                map_deri.append(self.x_deri[a][b][i,j])
        
        for a_plus_b in range(1, 4+1): #a+b = 1,2,3,4
            for b in range(0, a_plus_b+1): # b = 0,1,...,a+b
                a = a_plus_b - b
                map_deri.append(self.y_deri[a][b][i,j])
            
        return map_deri
        
    
    
    
    
    
    
    ###########################
    # Old codes
    ##########################
    def cal_QuadMapDerivatives_old(self, xi, eta):
        # Finding 1st partial derivatives for the map to a quadrilateral
        # at a single point (xi, eta)
        # The mapping functions are X(xi, eta) and Y(xi, eta)
            
        # Notation correspondence with David's book
        x1 = self.corners[0]
        x2 = self.corners[1]
        x3 = self.corners[2]
        x4 = self.corners[3]
            
        X_xi, Y_xi = 1/4 * (1-eta) * (x2 - x1) \
                   + 1/4 * (1+eta) * (x3 - x4)
        X_eta, Y_eta = 1/4 * (1-xi) * (x4 - x1) \
                     + 1/4 * (1+xi) * (x3 - x2)
            
        return X_xi, Y_xi, X_eta, Y_eta
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #####################################
    # Depracated
    #####################################
    
    def visualizing_norm_vector_matrix(self):
        # Visualizing the nodes and the normal vectors
        # The normal vectors are from the normal vector matrix
        fig, ax = plt.subplots(nrows=1, ncols=1)
        Nx = self.grid.Nx
        Ny = self.grid.Ny

        ax.scatter(self.nodes_phy_x, self.nodes_phy_y)

        j = 0
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect[:,j,0], self.norm_vect[:,j,1])
        
        j = Ny   
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect[:,j,0], self.norm_vect[:,j,1])    

        i = 0
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect[i,:,0], self.norm_vect[i,:,1])

        i = Nx
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect[i,:,0], self.norm_vect[i,:,1])   

        ax.set_aspect('equal')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
    
            