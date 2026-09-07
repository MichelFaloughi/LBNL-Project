# This file is meant to define helper funcions and methods for the ipynb file
import numpy as np
import plotly.graph_objects as go
from gpcam import GPOptimizer


######################
## Global Variables ##
######################
my_seed = 656465
np.random.seed(my_seed)

socm_min = 30  # SOC_max min
socm_max = 100 # SOC_max max
omega_min = 2
omega_max = 8

##############
## Plotting ##
##############
def plot(x,y,z,data = None):
    fig = go.Figure()
    fig.add_trace(go.Surface(x = x, y = y, z=z))
    if data is not None:
        fig.add_trace(go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2],
                                   mode='markers',marker=dict(size=12,
                                                              color=data[:,2],         # Set color equal to a variable
                                                              #colorscale='Viridis',   # Choose a colorscale
                                                              opacity=0.8)
                                   )
        )

    fig.update_layout(title='Plot', autosize=True,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),scene=dict(xaxis_title='SOC_max', yaxis_title='Omega: Frequency', zaxis_title='t:Time'))


    fig.show()

###################
## Generate Data ##
###################

# Function that takes in a number of points to generate, and coordinate bounds
# return two lists of x and y coordinates, hence point_i = (x_data[i], y_data[i]) 
def generate_random_points(num_points, x_min, x_max, y_min, y_max, curr_seed=my_seed):
    np.random.seed(curr_seed)
    x_data = np.round(np.random.uniform(x_min, x_max, num_points), 3)
    y_data = np.round(np.random.uniform(y_min, y_max, num_points), 3)
        
    return x_data, y_data

# This one looks good check it out on Desmos 3D
def new_data_function(x,y):
    return -0.015*x**2 - 0.5*y**3 +500

def data_funtion(x, y):
    return -0.125*x**2 - 0.2*y**3 + 500

def maher_alternative_data_functtion(x,y):
    return -0.1*x**2 - 2*y**2 + 500

def get_data(num_points=20, x_min=socm_min, x_max=socm_max, y_min=omega_min, y_max=omega_max, mean_error=30, var_error=10, z_function=new_data_function, curr_seed=my_seed):

    soc_max, omega = generate_random_points(num_points, x_min, x_max, y_min, y_max, curr_seed=curr_seed)
    xy_data = np.column_stack((soc_max, omega))

    soc_max = xy_data[:,0]
    omega = xy_data[:,1]

    z_data = z_function(soc_max,omega)

    error = np.random.normal(loc=mean_error, scale=var_error, size=z_data.shape)
    z_data = z_data + error

    return xy_data, z_data

def get_seed():
    return my_seed

############
## Spaces ##
############

def get_spaces(n=100, x_min=socm_min, x_max=socm_max, y_min=omega_min, y_max=omega_max):

    # Create a design space
    x_space = np.linspace(x_min, x_max, n)
    y_space = np.linspace(y_min, y_max, n)
    x_space, y_space = np.meshgrid(x_space, y_space)

    # Reshape the arrays into a 2-column array with 10000 rows
    my_space = np.vstack((x_space.reshape(-1), y_space.reshape(-1))).T
    
    return x_space, y_space, my_space

#################
## GPOptimizer ##
#################

def create_and_train(x_data, y_data, my_trained_hps, hps_bounds, my_space, x_space, y_space,kernel_function=None, noise_function=None, mean_function=None):
    
    my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        gp_kernel_function= kernel_function,
                        gp_mean_function= mean_function,
                        gp_noise_function= noise_function
                        )

    my_gpo.train(hyperparameter_bounds=hps_bounds)

    f = my_gpo.posterior_mean(my_space)["f(x)"]
    v = my_gpo.posterior_covariance(my_space, add_noise=True)["v(x)"]

    f_re = f.reshape(100,100)

    print(f'Hyperparameters: {my_gpo.hyperparameters}')
    plot(x_space, y_space, f_re, data = np.column_stack([my_gpo.x_data, my_gpo.y_data]))
    

    return my_gpo

#############
## Kernels ##
#############

####################
## Mean Functions ##
####################

#####################
## Noise Functions ##
#####################

###################
## Other helpers ##
###################

def quick_demo():
    soc_max = np.arange(socm_min, socm_max, 0.1)
    o = np.arange(omega_min, omega_max, 0.1)
    SOC_MAX, O = np.meshgrid(soc_max, o)
    # y = data_funtion(SOC_MAX,O)
    y = new_data_function(SOC_MAX,O)
    plot(SOC_MAX, O, y)

