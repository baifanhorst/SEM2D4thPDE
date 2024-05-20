import matplotlib.pyplot as plt

 
def visualizing_all(elements):
    # Showing all elements' grid
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for k, element in enumerate(elements):
        ax.scatter(element.nodes_phy_x, element.nodes_phy_y, label=f"Element {k}")
    ax.legend()
    ax.set_aspect('equal')
    














    
'''
# Functions for visualization of curve nodes, grid nodes, normal vectors
def visualizing_curves(curves):
    # Visualization of the curves
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, c in enumerate(curves):
        ax.scatter(c.x_nodes, c.y_nodes, label='Curve {}'.format(i+1))
    ax.legend()
    ax.set_title('Boundary nodes')
    ax.set_aspect('equal')
    
def visualizing_grid(element):
    # Visualizing the nodes and the normal vectors
    fig, ax = plt.subplots(nrows=1, ncols=1)
    Nx = element.grid.Nx
    Ny = element.grid.Ny

    ax.scatter(element.nodes_phy_x, element.nodes_phy_y)

    j = 0
    ax.quiver(element.nodes_phy_x[:,j], element.nodes_phy_y[:,j], 
              element.norm_vect_lower[:,0], element.norm_vect_lower[:,1])
    
    j = Ny   
    ax.quiver(element.nodes_phy_x[:,j], element.nodes_phy_y[:,j], 
              element.norm_vect_upper[:,0], element.norm_vect_upper[:,1])    

    i = 0
    ax.quiver(element.nodes_phy_x[i,:], element.nodes_phy_y[i,:], 
              element.norm_vect_left[:,0], element.norm_vect_left[:,1])

    i = Nx
    ax.quiver(element.nodes_phy_x[i,:], element.nodes_phy_y[i,:], 
              element.norm_vect_right[:,0], element.norm_vect_right[:,1])   

    ax.set_aspect('equal')
    
def visualizing_norm_vector_matrix(element):
    # Visualizing the nodes and the normal vectors
    # The normal vectors are from the normal vector matrix
    fig, ax = plt.subplots(nrows=1, ncols=1)
    Nx = element.grid.Nx
    Ny = element.grid.Ny

    ax.scatter(element.nodes_phy_x, element.nodes_phy_y)

    j = 0
    ax.quiver(element.nodes_phy_x[:,j], element.nodes_phy_y[:,j], 
              element.norm_vect[:,j,0], element.norm_vect[:,j,1])
    
    j = Ny   
    ax.quiver(element.nodes_phy_x[:,j], element.nodes_phy_y[:,j], 
              element.norm_vect[:,j,0], element.norm_vect[:,j,1])    

    i = 0
    ax.quiver(element.nodes_phy_x[i,:], element.nodes_phy_y[i,:], 
              element.norm_vect[i,:,0], element.norm_vect[i,:,1])

    i = Nx
    ax.quiver(element.nodes_phy_x[i,:], element.nodes_phy_y[i,:], 
              element.norm_vect[i,:,0], element.norm_vect[i,:,1])   

    ax.set_aspect('equal')
'''   
