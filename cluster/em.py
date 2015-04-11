
import numpy as np
import sys
import glob
import multiprocessing
#N = number of observations
#d = dimensions of observation
#k = number of clusters

#Data: Nxdx4
#Responsibility matrix: Nxk
#rhos: kxdx4
#pis: kx1
#Conditional Prob matrix: Nxk


#@profile
def test_data(N,d):
	c1 = np.random.multinomial(1,[.7,.1,.1,.1],(N,d))
	c2 = np.random.multinomial(1,[.1,.1,.1,.7],(N,d))
	return np.concatenate((c1,c2))

def load_data(fnames):
	temp = np.loadtxt(fnames[0])
	temp = temp.astype(np.bool,copy=False)
	data = np.zeros(shape=(len(fnames),temp.shape[1],4),dtype=np.bool)
	for i,fname in enumerate(fnames):
		temp = np.loadtxt(fname)
		temp = temp.astype(np.bool,copy=False)
		data[i,:,:] = temp.T
	return data

def load_data2(fnames):
	#fnames = fnames[:5]
	results = multiprocessing.Pool(20).map(load_single_data,fnames)
	data = np.zeros(shape=(len(fnames),results[0].shape[0],4),dtype=np.bool)
	for i,fname in enumerate(fnames):
		data[i,:,:] = results[i]
	return data

def load_single_data(fname):
	t = np.loadtxt(fname)
	t = t.astype(np.bool,copy=False)
	t = t.T
	return t

#@profile
def fit(x,k,max_iter=1000):
	N = x.shape[0]
	d = x.shape[1]

	#Initialization
	pis = np.random.rand(k)
	pis = pis / np.sum(pis)

	rhos = np.random.rand(k,d,4)
	sums = np.sum(rhos,axis=2)
	rhos = rhos / sums[:,:,np.newaxis]
	rhos = np.log(rhos)
	
	old_resps = np.ones((N,k))
	resps = np.zeros((N,k))
	llh = 0
	old_llh = -np.inf 
	i = 0
	delta_llh = old_llh - llh	
	print "LLH","\t\tDelta LLH"
	#while np.sum(np.abs(old_resps-resps)) > 0.001 and i < max_iter:
	while delta_llh < -.0001 and i < max_iter:	
		#E Step
		old_resps = resps
		llhs = np.tensordot(x,rhos,axes=([1,2],[1,2]))	+np.log(pis)[np.newaxis,:]
		resps = np.exp(llhs - np.logaddexp.reduce(llhs,axis=1)[:,np.newaxis])
		llh = np.sum(np.logaddexp.reduce(np.add(llhs,np.log(resps)),axis=1))
		#M Step
		pis = np.sum(resps,axis=0) / N
		rhos = np.transpose(np.tensordot(x,resps,axes=([0],[0])),(2,0,1)) 
		rhos = rhos + np.ones(rhos.shape) #Pseudo counts
		sums = np.sum(rhos,axis=2)
		rhos = rhos / sums[:,:,np.newaxis]
		rhos = np.log(rhos)
		i = i + 1
		delta_llh = old_llh - llh
		print llh, old_llh - llh
		old_llh = llh 

	print "Iterations until convergence: %d" % i
	print('rhos', rhos)
	print('pis', pis)
	print('resps', resps)
	return(llh,(pis,rhos,resps))

def bic(N,d,k,llh):
	# k-1: this is for pi vector -- prior probability that datum originates
	#      from cluster
	# N(k - 1): responsibility matrix
	fp = k*d*3 + k-1 + N*(k-1)
	return -2*llh + fp * np.log(N)

def map_fit(args):
	return fit(args[0],args[1])

if __name__ == '__main__':
	fdir = sys.argv[1]
	maxk = 3
	flist = glob.glob(fdir+"/mutpairs_*")
	data = load_data2(flist)

	output = []
	ks = range(1,maxk+1)
	results = []
	for k in ks:
		results.append(map_fit((data,k)))

	#results = multiprocessing.Pool(maxk).map(map_fit,zip([data]*maxk,ks))
	for K in ks:
		i = K - 1
		llh,model = results[i]
		bicscore = bic(model[2].shape[0],model[1].shape[1],ks[i],llh)
		print ks[i],llh,bicscore
		print np.sum(model[2],axis=0)
		output.append((ks[i],llh,bicscore))

	pis = {str(i + 1): model[0] for (i, (llh, model)) in enumerate(results)}
	rhos = {str(i + 1): model[1] for (i, (llh, model)) in enumerate(results)}
	resps = {str(i + 1): model[2] for (i, (llh, model)) in enumerate(results)}
	np.savez_compressed('pi.npz', **pis)
	np.savez_compressed('rho.npz', **rhos)
	np.savez_compressed('resp.npz', **resps)

	f = open("em.txt","w")
	f.write("\n".join([str(x) for x in output]))
	f.close()
	
	#x = test_data(1000,500)
	#(pis,rhos,resps) = fit(x,2)
	#print np.sum(resps[0:999,:],axis=0)
	#print np.sum(resps[1000:1999,:],axis=0)
