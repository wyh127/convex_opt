import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt



def feature_normalization(train, test):
    tmp_min = np.min(train, axis = 0)
    tmp_max = np.max(train, axis = 0)
    train_normalized = (train - tmp_min)/(tmp_max - tmp_min)
    test_normalized = (test - tmp_min)/(tmp_max - tmp_min)
    return train_normalized, test_normalized

def compute_square_loss(X, y, theta):
    loss = 0 #initialize the square_loss
    m = np.shape(X)[0]
    tmp = np.dot(X, theta) - y
    loss = np.sum(np.square(tmp)) / (2*m)
    return loss

def compute_square_loss_gradient(X, y, theta):
    m = np.shape(X)[0]
    tmp1 = np.dot(X, theta) - y
    tmp2 = np.dot(np.transpose(X), tmp1)
    grad = 1 / m * tmp2
    return grad

def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    for i in range(num_features):
        tmp = np.zeros(num_features)
        tmp[i] = epsilon
        approx_grad[i] = compute_square_loss(X, y, theta+tmp) - compute_square_loss(X, y, theta-tmp)
    approx_grad = approx_grad / (2 * epsilon)
    dis = np.sqrt(np.sum(np.square(approx_grad - true_gradient)))
    return dis <= tolerance

def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    #TODO
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    for i in range(num_features):
        tmp = np.zeros(num_features)
        tmp[i] = epsilon
        approx_grad[i] = objective_func(X, y, theta+tmp) - objective_func(X, y, theta-tmp)

    approx_grad = approx_grad / (2 * epsilon)
    dis = np.sqrt(np.sum(np.square(approx_grad - true_gradient)))
    return dis <= tolerance

def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    theta = np.zeros(num_features)

    
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)

    for i in range(num_iter):
        if check_gradient:
            if not grad_checker(X, y, theta):
                sys.exit("wrong gradient")
        theta = theta - alpha * compute_square_loss_gradient(X, y, theta)
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)

    return theta_hist, loss_hist

def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    #TODO
    m = np.shape(X)[0]
    tmp1 = np.dot(X, theta) - y
    tmp2 = np.dot(np.transpose(X), tmp1)
    grad = 1 / m * tmp2 + 2 * lambda_reg * theta
    return grad

def regularized_grad_descent(X, y, alpha=0.01, lambda_reg=1, num_iter=1000):
    (num_instances, num_features) = X.shape
    #theta = np.ones(num_features) #Initialize theta
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        theta = theta - alpha * compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist

def stochastic_grad_descent(X, y, alpha=0.05, lambda_reg=0.01, num_iter=1000):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    #TODO
    step_size = 1

    for i in range(num_iter):
        shuffle = np.random.permutation(num_instances)
        for j in shuffle:
            if isinstance(alpha, str):
                if alpha == "1/sqrt(t)":
                    theta = theta - .1/np.sqrt(step_size) * compute_regularized_square_loss_gradient(X[j], y[j], theta, lambda_reg)
                elif alpha == "1/t":
                    theta = theta - .1/step_size * compute_regularized_square_loss_gradient(X[j], y[j], theta, lambda_reg)
            else:
                theta = theta - alpha * compute_regularized_square_loss_gradient(X[j], y[j], theta, lambda_reg)
            step_size = step_size+1

            theta_hist[i, j] = theta
            loss_hist[i, j] = compute_square_loss(X, y, theta) + lambda_reg*np.sum(np.square(theta))

    return theta_hist, loss_hist


df = pd.read_csv('hw1-data.csv', delimiter=',')
X = df.values[:,:-1]
y = df.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

X_train, X_test = feature_normalization(X_train, X_test)


X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term


'''
theta0 = np.zeros((49, ))
loss0 = compute_square_loss(X_train, y_train, theta0)
'''

'''
theta0 = np.zeros((49, ))
grad0 = compute_square_loss_gradient(X_train, y_train, theta0)
print(grad0)
'''

'''
theta0 = np.zeros((49, ))
print(grad_checker(X_train, y_train, theta0))
'''

'''
theta0 = np.zeros((49, ))
print(generic_gradient_checker(X_train, y_train, theta0, objective_func = compute_square_loss, gradient_func = compute_square_loss_gradient))
'''


# Plot
'''
fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111)
for a in [0.05, 0.01, 0.005, 0.001]:
    theta_hist1, loss_hist1 = batch_grad_descent(X_train, y_train, alpha=a)
    plt.plot(loss_hist1, label='step size = %r' %a)
plt.xlabel('Steps')
plt.ylabel('Average square loss')
#ax.set_xlabel('Steps')
#ax.set_ylabel('Average square loss')
plt.title('Average square loss as a function of steps in BGD')
plt.legend(loc=0)
plt.show()
'''

'''
theta0 = np.zeros((49, ))
print(compute_regularized_square_loss_gradient(X_train, y_train, theta0, 0.5))
'''

'''
for a in [0.05, 0.01, 0.005, 0.001]:
    theta_hist1, loss_hist1 = regularized_grad_descent(X_train, y_train, alpha=a)
    plt.plot(loss_hist1)

plt.show()
'''

'''
lambdas = np.linspace(-7, -1)
train_loss = np.zeros(len(lambdas))
test_loss = np.zeros(len(lambdas))

for i in range(len(lambdas)):
    theta_hist1, loss_hist1 = regularized_grad_descent(X_train, y_train, lambda_reg=10**lambdas[i], alpha = 0.05)
    train_loss[i] = compute_square_loss(X_train, y_train, theta_hist1[-1])
    test_loss[i] = compute_square_loss(X_test, y_test, theta_hist1[-1])

plt.plot(lambdas, train_loss)
plt.plot(lambdas, test_loss)
plt.show()
'''


'''
B = np.linspace(1, 4.6)
test_loss = np.zeros(len(B))

for b in range(len(B)):
    X_trainb = X_train
    X_testb = X_test
    X_trainb[-1] = B[b]*X_trainb[-1]  # Add bias term
    X_testb[-1] = B[b]*X_testb[-1] # Add bias term
        
    theta_hist1, loss_hist1 = regularized_grad_descent(X_trainb, y_train, alpha = 0.05, lambda_reg = 10**(-2))
    test_loss[b] = compute_square_loss(X_testb, y_test, theta_hist1[-1])

plt.plot(B, test_loss)
plt.show()
'''



theta, loss = stochastic_grad_descent(X_train, y_train, alpha = 0.01)
plt.plot(np.log(np.mean(loss, axis = 1)), label = "1")
theta, loss = stochastic_grad_descent(X_train, y_train, alpha = 0.005)
plt.plot(np.log(np.mean(loss, axis = 1)), label = "2")
theta, loss = stochastic_grad_descent(X_train, y_train, alpha = "1/t")
plt.plot(np.log(np.mean(loss, axis = 1)), label = "3")
theta, loss = stochastic_grad_descent(X_train, y_train, alpha = "1/sqrt(t)")
plt.plot(np.log(np.mean(loss, axis = 1)), label = "4")
#plt.ylim((-2.6, -2.53))
plt.legend()
plt.show()













