import os
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import copy
from spn.root_node import RootNode
from spn.sum_node import SumNode
from spn.product_node import ProductNode, mult
from spn.normal_leaf_node import NormalLeafNode
from spn.multi_normal_leaf_node import MultiNormalLeafNode
from threading import Pool
def run_incremental_lspn(train_files, dtype=np.float32, batch_size=100):
	for train_file in train_files:
		obs = np.loadtxt(train_file, delimiter=",", dtype=dtype)
		spn = IncrementalLearnSPN()
		for i in range((len(obs)-1)//batch_size + 1):
			a = i*batch_size
			b = min((i+1)*batch_size, len(obs))
			spn.learn(obs[a:b])
		spn.model.check_valid()
	return spn
	#spn.model.display()


def run_lspn(dataset_name, folder, dtype=np.float32):
	train_file = os.path.join(folder, dataset_name + ".ts.data")
	test_file = os.path.join(folder, dataset_name + ".test.data")
	obs = np.loadtxt(train_file, delimiter=",", dtype=dtype)
	print (obs.shape)
	spn = LearnSPN()
	spn.learn(obs)
	spn.model.check_valid()
	test_obs = np.loadtxt(test_file, delimiter=",", dtype=dtype)
	print (np.mean(spn.evaluate(test_obs)))

def correlation(v1, v2):
	std = np.std(v1)*np.std(v2)
	covar = np.mean(v1*v2) - np.mean(v1)*np.mean(v2)
	return covar/std

def is_ind(v1, v2):
	return np.abs(correlation(v1, v2)) < 0.1

def cluster(data, n=2):
	kmeans = KMeans(n_clusters=n, max_iter=100).fit(data)
	#kmeans = DBSCAN(eps=0.1, min_samples=1).fit(data)
	labels = kmeans.labels_ 
#	print (labels.tolist())
	#print labels
	# if np.sum(labels) < data.shape[0]/100.:
	# 	return [data[:len(data)/2], data[len(data)/2:]]
	clusters = []
	for i in range(np.max(labels)+1):
		clusters.append(data[np.where(labels == i)])
	return clusters

def make_children(data, scope, n):
	children = []
	for c in range(n):
		batch_size = data.shape[0]//n
		node = ProductNode(0, scope, 'normal')
		for s in scope:
			D = data[batch_size*c:batch_size*(c+1), s:s+1]
			child = NormalLeafNode(0, s, np.mean(D), np.var(D))
			node.add_child(child)
		children.append(node)
	return children

def incremental_clustering(node, data, scope):
	likelihood = -np.inf
	children = 0
	while True:
		new_model = copy.deepcopy(node)
		new_children = make_children(data, scope, children)
		print (new_children)
		for c in new_children:
			new_model.add_child(c)
		new_model.evaluate(data, mpe=True)
		new_model.hard_em(data, np.arange(data.shape[0]))
		new_likelihood = np.mean(new_model.evaluate(data))
		if new_likelihood > (likelihood + 1e-43):
			likelihood = new_likelihood
			children += 1
		else:
			break
		del new_model
	if children > 0:
		new_children = make_children(data, scope, children-1)
		for c in new_children:
			node.add_child(c)
	return node

# def incremental_clustering(node, data, scope, eps=1.):
# 	likelihood = -np.inf
# 	children = 0
# 	while True:
# 		node.evaluate(data, mpe=True)
# 		node.hard_em(data, scope)
# 		new_lhd = np.mean(node.evaluate(data))
# 		if new_lhd > likelihood + eps:
# 			print ("yo")
# 			likelihood = new_lhd
# 			child = make_children(data, scope, 1)[0]
# 			node.add_child(child)
# 		else:
# 			break

def augment_spn(node, data, scope):
	if data.shape[0] < 10:
		node.hard_em(data, np.arange(data.shape[0]))
	elif isinstance(node, ProductNode):
		for c in node.children:
			augment_spn(c, data, c.scope)
	elif isinstance(node, SumNode):
		original_clusters = len(node.children)
		print (original_clusters)
		incremental_clustering(node, data, scope)
		print (len(node.children))
		node.evaluate(data, mpe=True)
		logprobs = np.array([c.value for c in node.children]).T
		print (logprobs.shape)
		childind = np.argmax(logprobs, axis=1)
		for i, c in enumerate(node.children):
			ind = np.where(childind==i)[0]
			if len(ind) > 0:
				if i < original_clusters:
					c.n += np.sum(ind)
					augment_spn(c, data[ind,:], scope)
				else:
					print ("hi")
					node.children[i] = learn(data[ind,:], scope)
	else:
		node.hard_em(data, np.arange(data.shape[0]))

def learn(data, scope=None):
	samples, variables = data.shape
	if scope is None:
		scope = np.arange(variables)
	groups = []
	for v1 in scope:
		ind = True
		for i, g in enumerate(groups):
			for v2 in g:
				ind = is_ind(data[:,v1], data[:, v2])
				if not ind:
					break;
			if not ind:
				groups[i].append(v1)
				break
		if ind:
			groups.append([v1])
#	print (groups)
	# if samples < 100:
	# 	groups = [[x] for x in scope]
	if len(groups) > 1:
		node = ProductNode(samples, scope, "normal")
		for i in range(len(groups)):
			if len(groups[i]) == 1:
				D = data[:, groups[i][0]]
				child = NormalLeafNode(samples, groups[i][0], np.mean(D), np.var(D))
			else:
				child = learn(data, scope=np.array(groups[i]))
			node.add_child(child)
	else:
		if samples < 100:
			node = MultiNormalLeafNode.create(0, groups[0])
			node.update(data, None)
			#node = ProductNode(len(samples), scope=np.array(groups[0]))
		else:
			node = SumNode(samples, scope)
			clusters = cluster(data)
			#print([len(c) for c in clusters])
			for c in clusters:
#				print (c.shape)
				child = learn(c, scope)
				node.add_child(child)

	return node

class LearnSPN:
	def __init__(self):
		self.model = None

	def learn(self, data):
		self.model = RootNode(learn(data))

	def evaluate(self, obs, mpe=False):
		assert self.model, "You gotta run learn on some data at least once!!"
		return self.model.evaluate(obs, mpe=mpe)

class IncrementalLearnSPN(LearnSPN):
    def __init__(self):
        super(IncrementalLearnSPN, self).__init__()

    def learn(self, data):
    	print ("Learning")
    	if self.model is None:
    		self.model = RootNode(learn(data))
    	else:
    		augment_spn(self.model.children[0], data, self.model.children[0].scope)
    		#self.model.evaluate(data, mpe=False)
    		#self.model.hard_em(data)
    	print (np.mean(self.model.evaluate(data, mpe=False)))

# v1 = np.array([0.2, 0.4, 0.7, 0.3])
# v2 = np.array([0.2, 0.4, 0.7, 0.3])*3
# v3 = np.array([0.3, 10, 2, 3])
# mult = False
# print (correlation(v1, v2))
# print (correlation(v1, v3))
# import time
# print ("started")
# spn = (run_incremental_lspn("abalone", "data/real/abalone", batch_size=10000))
# print ("finished")
# mult = True
# start = time.time()
# (np.mean(spn.evaluate(test_obs)))
# end = time.time() - start
# print (end)
