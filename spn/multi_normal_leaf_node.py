import copy

from .node import Node
from .multi_normal_stat import MultiNormalStat

class MultiNormalLeafNode(Node):

	@staticmethod
	def create(n, scope):
		node = MultiNormalLeafNode(n, scope)
		node.stat = MultiNormalStat.create(len(scope))
		return node

	@staticmethod
	def create_from_stat(n, scope, stat):
		node = MultiNormalLeafNode(n, scope)
		node.stat = copy.deepcopy(stat)
		return node

	def __repr__(self):
		return '<" {0} {1} {2}>'.format(self.n, self.scope, self.stat)

	def display(self, depth=0):
		print('{0}{1}'.format('-'*depth, self))

	def rep(self):
		obs = np.empty(np.max(self.scope)+1)
		obs[self.scope] = self.stat.rep()
		return obs

	def evaluate(self, obs, mpe=False):
		x = obs[:,self.scope]
		self.value = self.stat.evaluate(x)
		return self.value

	def hard_em(self, data, inds=None):
		self.update(data, None)

	def update(self, obs, params):
		self.stat.update(obs[:,self.scope], self.n)
		self.n += len(obs)

	def check_valid(self):
		return True

	def normalize_nodes(self):
		self.n = 0
