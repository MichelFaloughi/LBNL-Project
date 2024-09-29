# imports
import numpy as np
import matplotlib.pyplot as plt
from fvgp import GP
from fvgp.gp_kernels import * 

############################
## Fetch Data to train on ##
############################

x_data = np.random.rand(20).reshape(-1,1)     # Does this give us a normally distributed noise ?
x_data_test = np.random.rand(10).reshape(-1,1)
x_pred1D = np.linspace(0,1,100).reshape(-1,1)

def forreal(x):
    return ((6 * x - 2) ** 2 ) * np.sin(12 * x - 4 )

y_data = forreal(x_data[:,0]) + (np.random.rand(len(x_data)) - 0.5) * 2
y_data_test = forreal(x_data_test[:,0]) + (np.random.rand(len(x_data_test)) - 0.5) * 2

#####################
## Customize my GP ##
#####################

hps = np.array([
    0.1,  # length scale
    0.1,  # signal or output variance
    1,  # noise variance
    10   # mean function
])

def my_noise(x,hps=hps):
    return np.diag(np.ones((len(x))) * hps[2]) # noise variance
    

def rbf_kernel(x1,x2,hps=hps):
    sigma2 = hps[1]  # signal or output variance
    l = hps[0]       # length scale
    return sigma2 * np.exp((-1/2) * ((x1-x2)/l)**2)

# THIS HAS TO RETURN A MATRIX

def my_mean(x=0, hps=hps):
    return np.sin(hps[3] * x[:,0]) # mean function



my_gp1 = GP(x_data, y_data,
            init_hyperparameters=hps,
            noise_variances=None, #provding noise variances and a noise function will raise a warning 
            compute_device='cpu', 
            # gp_kernel_function=rbf_kernel, 
            gp_kernel_function_grad=None, 
            # gp_mean_function=my_mean, 
            gp_mean_function_grad=None,
            gp_noise_function=my_noise,
            gp2Scale = False,
            calc_inv=False, 
            ram_economy=False, 
            args=None,
            )

hps_bounds = np.array([[1e-5,1000],     # length scale for the kernel          # default kernel function has this as signal variance and vice versa
                       [0.01,5.],      # signal variance for the kernel
                       [0.001,100],     # noise
                       [0.01,60]      # mean
                      ])

my_gp1.train(hyperparameter_bounds=hps_bounds, # max_iter=100,
             init_hyperparameters=hps)


#let's make a prediction
x_pred = np.linspace(0,1,1000)

mean1 = my_gp1.posterior_mean(x_pred.reshape(-1,1))["f(x)"]
var1 =  my_gp1.posterior_covariance(x_pred.reshape(-1,1), variance_only=False, add_noise=True)["v(x)"]

plt.figure(figsize = (16,10))
plt.plot(x_pred,mean1, label = "posterior mean", linewidth = 4)
plt.plot(x_pred1D,forreal(x_pred1D), label = "latent function", linewidth = 4)
plt.fill_between(x_pred, mean1 - 3. * np.sqrt(var1), mean1 + 3. * np.sqrt(var1), alpha = 0.5, color = "grey", label = "var")
plt.scatter(x_data,y_data, color = 'black')


##looking at some validation metrics
print(my_gp1.rmse(x_pred1D, forreal(x_pred1D)))
print(my_gp1.crps(x_pred1D, forreal(x_pred1D)))