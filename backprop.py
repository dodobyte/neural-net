import random, gzip, numpy as np

nn = {
	'w': [],	# weights of the network
	'b': [],	# biases of the network
	'a': [],	# activations from the last forward
	'z': [],	# weighted inputs from the last forward
	'gw': [],	# weight gradients accumulated through minibatch
	'gb': []	# bias gradients accumulated through minibatch
}

def init_net(shape):
	for i in range(len(shape)-1):
		m, n = shape[i], shape[i+1]
		nn['w'].append(np.random.randn(n,m))
		nn['b'].append(np.random.randn(n, 1))
		nn['a'].append(np.zeros([n, 1]))
		nn['z'].append(np.zeros([n, 1]))
		nn['gw'].append(np.zeros([n, m]))
		nn['gb'].append(np.zeros([n, 1]))
	nn['a'].insert(0, np.zeros([shape[0], 1]))

def forward(a):
	nn['a'][0] = a
	for i, w in enumerate(nn['w']):
		z = np.dot(w, a) + nn['b'][i]
		a = sigmoid(z)
		nn['z'][i] = z
		nn['a'][i+1] = a
	return a

def backward(y):
	n = len(nn['w'])
	l = n-1
	a = nn['a'][l+1]
	z = nn['z'][l]
	a_prev = nn['a'][l]

	dc_dz = 2*(a-y) * sig_prime(z)
	dc_dw = np.matmul(dc_dz, a_prev.T)
	dc_db = dc_dz
	nn['gw'][l] += dc_dw
	nn['gb'][l] += dc_db

	for i in range(1, n):
		l = n-i-1
		z = nn['z'][l]
		a_prev = nn['a'][l]
		w_next = nn['w'][l+1]
		dc_dz = np.dot(w_next.T, dc_dz) * sig_prime(z)
		dc_dw = np.matmul(dc_dz, a_prev.T)
		dc_db = dc_dz
		nn['gw'][l] += dc_dw
		nn['gb'][l] += dc_db

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def sig_prime(x):
	x = sigmoid(x)
	return x * (1. - x)

def zero_grad():
	for i in range(len(nn['w'])):
		nn['gw'][i].fill(0)
		nn['gb'][i].fill(0)

def optimize(n):
	rate = 3.0	# learning rate
	for i in range(len(nn['w'])):
		nn['w'][i] -= (nn['gw'][i] / n) * rate
		nn['b'][i] -= (nn['gb'][i] / n) * rate

def load_data():
	with open('train-images-idx3-ubyte.gz', 'rb') as f:
		data = gzip.decompress(f.read())[16:]
		data = np.frombuffer(data, dtype=np.uint8)
		train = data.reshape((-1, 784, 1))/255.0
	with open('train-labels-idx1-ubyte.gz', 'rb') as f:
		train_y = []
		for l in gzip.decompress(f.read())[8:]:
			z = np.zeros([10,1])
			z[l] = 1.0
			train_y.append(z)
	with open('t10k-images-idx3-ubyte.gz', 'rb') as f:
		data = gzip.decompress(f.read())[16:]
		data = np.frombuffer(data, dtype=np.uint8)
		test = data.reshape((-1,784, 1))/255.0
	with open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
		test_y = []
		for l in gzip.decompress(f.read())[8:]:
			z = np.zeros([10,1])
			z[l] = 1.0
			test_y.append(z)
	return list(zip(train, train_y)), list(zip(test, test_y))

def main():
	random.seed(42)
	train, test = load_data()
	init_net([784,40,10]) # network shape

	for epoch in range(10):
		random.shuffle(train)
		n = 10	# minibatch size
		batches = [train[i:i+n] for i in range(0, len(train), n)]
		for mini in batches:
			zero_grad()
			for x, y in mini:
				forward(x)
				backward(y)
			optimize(n)

		correct = 0
		for x, y in test:
			a = forward(x)
			if a.argmax() == y.argmax():
				correct += 1
		print("epoch", epoch+1, ":", correct, "/", len(test))

main()