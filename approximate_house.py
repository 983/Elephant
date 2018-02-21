# simple demo to show a way to approximate a house-like curve with fourier series
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import scipy.optimize

house = np.zeros(128)
house[20:50] = np.linspace(1, 2, 30)
house[50:80] = np.linspace(2, 1, 30)
house[65:75] = 1.9

# rule of thumb: stuff works better when data is normalized
house -= np.mean(house)
house /= np.std(house)

def make_curve(parameters):
    # make a curve, given some fourier series coefficients
    
    t = np.linspace(0, 2*np.pi, len(house), endpoint=False)

    result = np.zeros(len(t))

    for k in range(len(parameters)//2):
        a = parameters[k]
        b = parameters[k + len(parameters)//2]
        
        result += a*np.cos(k*t) + b*np.sin(k*t)
    
    return result

def loss(parameters):
    difference = house - make_curve(parameters)
    return np.mean(np.square(difference))

plt.title("fourier series of house")
plt.plot(house, label='house')

for n_params in range(10, 50+1, 20):
    # random guess is good enough
    guess = np.random.randn(n_params)

    # fit n_params many fourier coefficients to approximate house
    params = scipy.optimize.fmin_l_bfgs_b(loss, guess, grad(loss))[0]

    # find approximate house curve given the fitted fourier coefficients
    curve = make_curve(params)

    plt.plot(curve, label='%d parameters'%n_params)
plt.legend()
plt.show()
