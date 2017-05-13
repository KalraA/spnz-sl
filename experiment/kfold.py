import os
import time
import pickle

import numpy as np

from spn.spn import *
from util.util import *
from experiment import Experiment, ILSPN_Experiment


from spn.sum_node import SumNode
from spn.product_node import ProductNode
from spn.multi_normal_leaf_node import MultiNormalLeafNode
from spn.normal_leaf_node import NormalLeafNode

def gen_random_node(scope, nodetype):
	if len(scope)==1:
		return NormalLeafNode(0, scope[0])
	elif len(scope) <= 2:
		return MultiNormalLeafNode.create(0, scope)
	if nodetype == 'P':
		i = np.random.randint(2, size=len(scope))
		i0 = np.where(i==0)[0]
		i1 = np.where(i==1)[0]
		if len(i0)==0 or len(i1)==0:
			return gen_random_node(scope, 'P')
		else:
			p = ProductNode(0, scope, 'normal')
			s0 = gen_random_node(scope[i0], 'S')
			s1 = gen_random_node(scope[i1], 'S')
			p.add_children(s0, s1)
			return p
	else:
		s = SumNode(0, scope)
		nc = np.random.randint(2,3)
		children = [None]*nc
		for i in range(nc):
			children[i] = gen_random_node(scope, 'P')
		s.add_children(*children)
		return s

def gen_random_spn(numvar, params):
	model = SPN(numvar, 2, params)
#	model.root = RootNode(gen_random_node(np.arange(numvar), 'S'))
	print(count_nodes(model))
	return model



DATADIR = 'data'
OUTDIR = 'output'

def make_kfold_filenames(vartype, name, k):
	filenames = [os.path.join(DATADIR, vartype, name, "{0}.{1}.data".format(name, i))
	             for i in ['ts', 'test']]
	return filenames

def make_train_test_filenames(vartype, name):
	trainfiles = [os.path.join(DATADIR, vartype, name, "{0}.train.data".format(name))]
	testfiles = [os.path.join(DATADIR, vartype, name, "{0}.test.data".format(name))]
	return trainfiles, testfiles

def run_train_test(trainfiles, testfiles, numvar, numcomp, params):
	model = gen_random_spn(numvar, params)
	experiment = Experiment(model, trainfiles, testfiles)
	t0 = time.clock()
	result = experiment.run()
	t1 = time.clock()
	return result, t1-t0, model

def run_ith_fold(i, filenames, numvar, numcomp, params):
	testfiles = filenames[i:i+1]
	trainfiles = filenames[i+1:] + filenames[:i]
	return run_train_test(trainfiles, testfiles, numvar, numcomp, params)

def run_kfold(vartype, name, k, numvar, numcomp, params):
	filenames = make_kfold_filenames(vartype, name, k)
	results = [None] * k
	times = [None] * k
	models = [None] * k
	numnodes = [None] * k
	numparams = [None] * k
	for i in range(k):
		results[i], times[i], models[i] = run_ith_fold(i, filenames, numvar, numcomp, params)
		numnodes[i] = count_nodes(models[i])
		numparams[i] = count_params(models[i])
		print(i, results[i], times[i], numnodes[i], numparams[i])
	print(np.mean(results), np.std(results))
	return results, times, models, numnodes, numparams

def run(vartype, traintest, name, numvar, numcomp, batchsize, mergebatch, corrthresh,
        equalweight, updatestruct, mvmaxscope, leaftype):
	outfile = "{0}_{1}_{2}_{3}_{4}_{5}".format(name, numcomp, batchsize, mergebatch,
                           corrthresh, mvmaxscope)
	resultpath = "{0}.txt".format(outfile)
	picklepath = "{0}.pkl".format(outfile)
	params = SPNParams(batchsize, mergebatch, corrthresh, equalweight, updatestruct,
	    mvmaxscope, leaftype)

	print('******{0}*******'.format(name))

	if traintest:
		trainfiles, testfiles = make_train_test_filenames(vartype, name)
		result, t, model = run_train_test(trainfiles, testfiles, numvar, numcomp, params)
		numnodes = count_nodes(model)
		numparams = count_params(model)
		print('Loglhd: {0:.3f}'.format(result))
		print('Time: {0:.3f}'.format(t))
		print('Number of nodes: {0}'.format(numnodes))
		print('Number of parameters: {0}'.format(numparams))
		with open(resultpath, 'w') as g:
			g.write('Loglhd: {0:.3f}\n'.format(result))
			g.write('Time: {0:.3f}\n'.format(t))
			g.write('Number of nodes: {0}\n'.format(numnodes))
			g.write('Number of parameters: {0}\n'.format(numparams))
			g.close()
		with open(picklepath, 'wb') as g:
			pickle.dump(model, g)
	else:
		results, times, models, numnodes, numparams = run_kfold(
		                    vartype, name, 10, numvar, numcomp, params)

		with open(resultpath, 'w') as g:
			g.write('Loglhd: {0:.3f}, {1:.3f}\n'.format(np.mean(results), np.std(results)))
			g.write('Times: {0:.3f}, {1:.3f}\n'.format(np.mean(times), np.std(times)))
			g.write('NumNodes: {0:.1f}, {1:.3f}\n'.format(np.mean(numnodes), np.std(numnodes)))
			g.write('NumParams: {0:.3f}, {1:.3f}\n'.format(np.mean(numparams), np.std(numparams)))
			g.write('\n')
			for r, t, a, b in zip(results, times, numnodes, numparams):
				g.write('{0:.3f} {1:.3f} {2} {3}\n'.format(r, t, a, b))
			g.close()
		with open(picklepath, 'wb') as g:
			pickle.dump(models, g)


def run_ilspn(vartype, traintest, folder, name, numvar, numcomp, batchsize, leaftype):
	outfile = "ilspn_{0}_{1}_{2}_{3}_{4}_{5}".format(name, numcomp, batchsize)
	resultpath = "{0}.txt".format(outfile)
	picklepath = "{0}.pkl".format(outfile)
	files = os.listdir(folder)
	results, times, numnodes, numparams = 0, 0, 0, 0
	for i in range(10):
		print('******{0}*******'.format(name))
		trainfiles = files[:i] + files[i+1:]
		testfiles = [files[i]]
		print (trainfiles, testfiles)
		experiment = ILSPN_Experiment(trainfiles, testfiles)
		start = time.time()
		result = experiment.run()
		end = time.time() - start
		spn = experiment.model
		numnode = count_nodes(model)
		numparam = count_params(model)		
		results += result
		times += end
		numparams += numparam
		numnodes += numnode
		print('Loglhd: {0:.3f}'.format(result))
		print('Time: {0:.3f}'.format(end))
		print('Number of nodes: {0}'.format(numnode))
		print('Number of parameters: {0}'.format(numparam))
	numnodes /= 10.
	times /= 10.
	results /= 10.
	numparams /= 10.
	print('Loglhd: {0:.3f}'.format(results))
	print('Time: {0:.3f}'.format(times))
	print('Number of nodes: {0}'.format(numnodes))
	print('Number of parameters: {0}'.format(numparams))
	with open(resultpath, 'w') as g:
		g.write('Loglhd: {0:.3f}\n'.format(results))
		g.write('Time: {0:.3f}\n'.format(times))
		g.write('Number of nodes: {0}\n'.format(numnodes))
		g.write('Number of parameters: {0}\n'.format(numparams))
		g.close()

	print('******{0}*******'.format(name))

