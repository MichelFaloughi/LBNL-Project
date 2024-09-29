# This is scratch paper for my project and where I keep old helper functions
import numpy as np
from gpcam import GPOptimizer



# Actually, we might not have to write if-elif-else cases and give the functions default None values
def create_and_train_2(x_data, y_data, my_trained_hps, hps_bounds, my_space, x_space, y_space,kernel_function=None, noise_function=None, mean_function=None):

    x_space_retrieved = my_space[:, 0].reshape(x_space.shape)
    y_space_retrieved = my_space[:, 1].reshape(y_space.shape)

    assert np.array_equal(x_space, x_space_retrieved), "x_space does not match!"
    assert np.array_equal(y_space, y_space_retrieved), "y_space does not match!"

    if not kernel_function and not noise_function and not mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        # gp_kernel_function= kernel_function,
                        # gp_mean_function= mean_function,
                        # gp_noise_function= noise_function
                        )
    elif not kernel_function and not noise_function and mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        # gp_kernel_function= kernel_function,
                        gp_mean_function= mean_function,
                        # gp_noise_function= noise_function
                        )
    elif not kernel_function and noise_function and not mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        # gp_kernel_function= kernel_function,
                        # gp_mean_function= mean_function,
                        gp_noise_function= noise_function
                        )
    elif kernel_function and not noise_function and not mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        gp_kernel_function= kernel_function,
                        # gp_mean_function= mean_function,
                        # gp_noise_function= noise_function
                        )
    elif kernel_function and  noise_function and not mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        gp_kernel_function= kernel_function,
                        # gp_mean_function= mean_function,
                        gp_noise_function= noise_function
                        )
    elif kernel_function and  not noise_function and not mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        gp_kernel_function= kernel_function,
                        gp_mean_function= mean_function,
                        # gp_noise_function= noise_function
                        )
    elif not kernel_function and noise_function and mean_function:
        my_gpo = GPOptimizer(x_data, y_data,
                        init_hyperparameters=my_trained_hps,
                        # hyperparameter_bounds=hps_bounds,
                        # gp_kernel_function= kernel_function,
                        gp_mean_function= mean_function,
                        gp_noise_function= noise_function
                        )
    else:
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

    plot(x_space, y_space, f_re, data = np.column_stack([my_gpo.x_data, my_gpo.y_data]))
    print(f'Hyperparameters: {my_gpo.hyperparameters}')

    return my_gpo




def get_data(num_points=20, lower_x=10, upper_x=40, lower_y=1, upper_y=10, mean_error=30, var_error=10):

    a, o = generate_random_points(num_points, lower_x, upper_x, lower_y, upper_y)
    xy_data = np.column_stack((a, o))

    a = xy_data[:,0]
    o = xy_data[:,1]

    z_data = -0.1*a**2 - 2*o**2 + 500

    error = np.random.normal(loc=mean_error, scale=var_error, size=z_data.shape)
    z_data = z_data + error

    return xy_data, z_data


hps_bounds = np.array([[0.001,10000]    # in kernel, signal variance
                   ,[0.001,10]   # in kernel, length scale
                   ,[0.001,10]   # in kernel, length scale 2
                   ,[0.1,10]      # s, slope
                   ,[0.1,10]      # s, offset
                   ,[0, 2500]     # mean, 
                   ,[0, 2500]     # mean
                   ])

hps = np.array([0.01      # in kernel
                , 0.01    # in kernel
                , 0.01    # in kernel
                , 1       # s
                , 1       # s
                , 800     # mean
                , 40      # mean
                ])







