import numpy as np


# data generation
X = np.random.rand(150, 75)

theta = np.zeros((75, 1))
t = [-1, 1]
theta[0:10] = (np.random.choice(t, 10)).reshape((10, 1)) * 10

epsilon = np.random.randn(150, 1) * 0.1

y = np.matmul(X, theta) + epsilon

data = np.hstack((y, X))

train = data[0:80, :]
cv = data[80:100]
test = data[100:150, :]


# shooting algorithm
def coordinate_descent(X, y, lmda):
    D = X.shape[1]

    # initilize w 
    w = np.linalg.inv(np.matmul(np.matrix.transpose(X), X) + lmda * np.identity(D))
    w = np.matmul(w, np.matrix.transpose(X))
    w = np.matmul(w, y)

    tmp_X = X*X #np.multiply(X, X)
    a = tmp_X.sum(axis = 0)

    w1 = w
    w2 = np.zeros(D)
    # set tolerance
    tol = 10**-5
    while(np.sum(np.abs(w1 - w2)) > tol):
        w2 = w1
        for j in range(D):
            a_tmp = 2*a[j]
            c_tmp = 2*np.dot(X[:, j], y) - 2*np.sum(np.dot(X, w)) + 2*w1[j] * a[j]
            w1[j] = soft(a_tmp, c_tmp, lmda)

    return w1


def soft(a, c, lmda):
    if c/a > 0:
        return max(0, abs(c/a) - lmda/a)
    elif c/a < 0:
        return -max(0, abs(c/a) - lmda/a)
    else:
        return 0


X = np.loadtxt("data/X_train.txt")
y = np.loadtxt("data/y_train.txt")


print(coordinate_descent(X, y, 70))




