import numpy as np
from scipy.misc import logsumexp

from .node import Node

class SumNode(Node):
	def __init__(self, n, scope):
		super(SumNode, self).__init__(n, scope)

	def display(self, depth=0):
		print("{0}<+ {1} {2}>".format('-'*depth, self.n, self.scope))
		for child in self.children:
			child.display(depth+1)

	def evaluate(self, obs, mpe=False):
		logprobs = self.evaluate_children(obs, False, mpe)
		if mpe:
			self.value = np.max(logprobs, axis=1)
		else:
			self.value = logsumexp(logprobs, axis=1)
		#print( self.value.shape)
		return self.value

	def evaluate_children(self, obs, equal_weight, mpe=False):
		logprobs = np.vstack([child.evaluate(obs, mpe) for child in self.children]).T
		if equal_weight:
			return logprobs
		else:
			logweights = self.get_log_weights()
			return logprobs + logweights			

	def get_log_weights(self):
		counts = np.array([child.n for child in self.children])
		return np.log((counts+1e-8)/(np.sum(counts)+1e-8*len(self.children)))

	def update(self, obs, params):
		logprobs = self.evaluate_children(obs, params.equalweight)
		childind = np.argmax(logprobs, axis=1)
		for i in range(len(self.children)):
			ind = np.where(childind==i)[0]
			if len(ind) > 0:
				self.children[i].update(obs[ind,:], params)
		self.n += len(obs)

	def add_child(self, child):
		assert np.array_equal(child.scope, self.scope)
		child.parent = self
		self.children.append(child)

	def remove_child(self, child):
		self.children.remove(child)
		child.parent = None

	def normalize_nodes(self):
		summ = sum([x.n for x in self.children])
		for c in self.children:
			c.normalize_nodes()
			c.n = float(c.n)/summ
		self.n = 1.

	def check_valid(self):
		for c in self.children:
			assert c.check_valid()
			assert np.array_equal(c.scope, self.scope)
		return True

	def hard_em(self, data, inds):
		logprobs = np.array([c.value[inds] for c in self.children]).T
		#print (logprobs.shape, data.shape)
		childind = np.argmax(logprobs, axis=1)
		#print (childind)
		for i in range(len(self.children)):
			ind = np.where(childind==i)[0]
			if len(ind) > 0:
				self.children[i].hard_em(data[ind,:], inds[ind])
		self.n += len(data)