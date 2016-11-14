

# file to do GP Regression from scratch with some test examples. 
import numpy as np
from scipy.optimize import minimize, rosen
import matplotlib.pyplot as plt



KERNEL = 'noisyexp'
x_inp = np.asarray([-1.50, -1.00, -0.75, -0.40, -0.25, 0.00])
y_inp = np.asarray([-1.70, -1.10, -0.3, 0.20, 0.50, 0.95])


def negLogLike(gp_params):
	l= gp_params[0]
	sigma_fn = gp_params[1]
	sigma_noise = gp_params[2]
	K_mat = computeCovMat(x_inp, sigma_fn, l, sigma_noise)
	K_inv = np.linalg.inv(K_mat)

	return (-0.5 * np.dot(np.transpose(y_inp), np.dot(K_inv, y_inp)) - 0.5*np.log(np.linalg.det(K_mat)) - 0.5*n*2*np.pi)



def computeKernel(x1, x2, var=1.1, l=2.5, var_noise=0.3, fn='sqexp'):
	if fn == 'sqexp':
		val = var**2 * np.exp( -(x1-x2)**2 / 2 * l**2 )
	elif fn == 'noisyexp':
		val = var**2 * np.exp( -(x1-x2)**2 / 2 * l**2 )
		if x1 == x2:
			val += var_noise**2
	return val


def computeCovMat(x_vec, var=1.1, l=2.5, var_noise=0.3, fn='sqexp'):
	n = np.prod(x_vec.shape)
	cov = np.zeros((n,n))
	# if not isinstance(x_vec, ())
	for i in xrange(n):
		for j in xrange(n):
			cov[i,j] = computeKernel(x_vec[i], x_vec[j], var=1.1, l=2.5, var_noise=0.3, fn=KERNEL)

	# print cov
	return cov


def computeCovVec(x_vec, x_star):
	n = np.prod(x_vec.shape)
	n_test = np.prod(x_star.shape)
	covVec = np.zeros((n_test,n))

	for i in xrange(n_test):
		for j in xrange(n):
			covVec[i,j] = computeKernel(x_vec[j], x_star[i], fn=KERNEL)
	print covVec
	return covVec


x_inp = np.asarray([-1.50, -1.00, -0.75, -0.40, -0.25, 0.00])
y_inp = np.asarray([-1.70, -1.10, -0.3, 0.20, 0.50, 0.95])
y_inp = y_inp[:, np.newaxis]

x_star = np.asarray([0.20])

K_mat = computeCovMat(x_inp)
K_mat_star_star = computeCovMat(x_star)
K_mat_star = computeCovVec(x_inp, x_star)
K_inv = np.linalg.inv(K_mat)

print K_mat_star.shape, y_inp.shape, K_inv.shape
p1 =  np.dot(K_inv, y_inp)
print p1.shape
y_star_mean = np.dot(K_mat_star, p1)

y_star_cov = K_mat_star_star - np.dot(K_mat_star, np.dot(K_inv, np.transpose(K_mat_star)))

print "mean:", y_star_mean
print "covariance:", y_star_cov
x_plot = np.concatenate((x_inp, x_star), axis=0)
y_plot = np.concatenate((y_inp, y_star_mean), axis=0)
print x_plot.shape
# plt.plot(x_plot, y_plot, 'bo')
plt.plot(x_inp, y_inp, 'ro')
# plt.plot(x_star, y_star_mean, 'go')
plt.xlim((-1.6,0.5))
plt.ylim((-2,1.2))


# plot a solid line ....
x_test =  np.arange(-1.5, 0.5, 0.01)
K_test_star = computeCovVec(x_inp, x_test)
K_test_star_star = computeCovMat(x_test)

y_test_star_mean = np.dot(K_test_star, p1)
y_test_variance = K_test_star_star - np.dot(K_test_star, np.dot(K_inv, np.transpose(K_test_star)))

y_test_confidence_95_up = y_test_star_mean + 1.96*np.sqrt(y_test_variance)
y_test_confidence_95_down = y_test_star_mean - 1.96*np.sqrt(y_test_variance)



print y_test_star_mean.shape, x_test.shape
# print y_test_variance

# plt.plot(x_test, y_test_star_mean, 'g-')
# plt.plot(x_test, y_test_confidence_95_down, 'r-')
# plt.plot(x_test, y_test_confidence_95_up, 'r-')
# plt.plot(x_inp, y_inp, 'bo') 	
# plt.show()


l =2.5
sigma_fn =1.1
sigma_noise = 0.3

gp_params = [l, sigma_fn, sigma_noise]

# calculate data likelihood for these params
n = np.prod(y_inp.shape)

# nll = -0.5 * np.dot(np.transpose(y_inp), np.dot(K_inv, y_inp)) - 0.5*np.log(np.linalg.det(K_mat)) - 0.5*n*2*np.pi

nll = negLogLike(gp_params)
p0 = [4, 1.1, 0.7]
print nll

p_full = [p0, x_inp, y_inp]
# cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
# ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
# ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})


# constraints = ({'type':'ineq', 'fun': lambda x: })
bnds = ((0, None), (0, None), (0, None))
options={'xtol': 1e-8, 'disp': True}

results = minimize(negLogLike, p0, method='Newton-CG', bounds=bnds, options=options)
print results.x
# res = optimize.fmin(negLogLike, , args=p0)



# use optimisation function to minimise the likelihood wrt the parameters.







