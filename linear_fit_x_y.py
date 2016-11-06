import numpy as np
import matplotlib.pyplot
import random

x = np.array([10, 12, 14, 16, 18, 20])
y = np.array([8.2, 10.1, 11.8, 10.3, 12.7, 12.9])

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    y = []
    for i in range(hm):
        _ = val + random.randrange(-variance, variance)
        y.append(_)
        if correlation and correlation == 'pos':
            val += step
        if correlation and correlation == 'neg':
            val -= step
    x = [i for i in range(len(y))]
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)

x,y = create_dataset(40, 20, 2, correlation='pos')
matplotlib.pyplot.scatter(x, y)

def best_fit_line(x, y):

    x_mean = x.mean()
    y_mean = y.mean()

    # m = x_mean*y_mean - (x*y)_mean / x_mean^2 - (x^2)_mean
    m1 = (x_mean*y_mean - (x*y).mean())/(x_mean**2 - np.array(x**2).mean())
    m2 = sum((x - x.mean())*(y - y.mean())) / sum((x - x.mean())**2)
    b = y_mean - m1*x_mean
    
    return m1,b

def squared_error(y_orig, y_line):
    return sum((y_line - y_orig)**2)

def coefficient_of_determination(y_orig, y_line):
    y_mean_line = [y_orig.mean() for _ in y_orig]
    squared_error_regr = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean_line)
    return 1 - squared_error_regr/squared_error_mean


print(best_fit_line(x, y))

m, b = best_fit_line(x, y)
y_line = [m*_ + b for _ in x]
matplotlib.pyplot.plot(x, y_line)

predict_x = 10
predict_y = m*predict_x + b
matplotlib.pyplot.scatter(predict_x, predict_y, s = 100, color='r')

r_squared = coefficient_of_determination(y, y_line)
print(r_squared)

matplotlib.pyplot.show()

