#=============================================================
#-------------        Logistic regression        ------------- 
#=============================================================
# A kind of statistical regression model of variables 
# with Bernoulli distribution.
#=============================================================
# Value description
# x : Price
# n : Number of Customer
# y : Number of people responded to price
#=============================================================
# STEP 1 : Graph how many people responded to the price
#=============================================================
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(6, 8, 0.2)
n = np.array([70, 74, 68, 69, 74, 86, 43, 56, 78, 48])
y = np.array([7, 14, 19, 28, 38, 50, 30, 45, 70, 48])

n_y = n - y
p_ = y / n

print(p_)
plt.scatter(x, p_, color="c")
plt.show()

#=============================================================
# STEP 2 : Graph the regression curve of response probability
#=============================================================

def logistic(a):
    return 1 / (1 + np.exp(-a))

beta = np.array([0., 0.])

for i in range(100000):
    a = beta[0] + beta[1]*x
    p_k = (logistic(a) - 1)*y + logistic(a)*n_y
    grad = np.array([np.sum(p_k), np.sum(p_k*x)])
    beta[0] = beta[0] - 0.0001*grad[0]
    beta[1] = beta[1] - 0.0001*grad[1]
    
    if (i + 1)%20000 == 0:
        print("STEP_" + str(i + 1))
        print(beta)
        print(grad)
        print("--------------------------")
print(beta)

x_ = np.arange(5.5, 8.5, 0.05)
y_ = 1 / (1 + np.exp(-(beta[0] + beta[1]*x_)))

# The orange dot curve is the obtained value.
plt.scatter(x_, y_, color="coral")
# The blue dot curve is the original value.
plt.scatter(x, p_, color="c")
plt.show()

