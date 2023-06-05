# k means ++, better initial
# from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
import numpy as np
import random
import matplotlib.pyplot as plt

class KMeans(object):
	"""Standard k means algorithm. The optimal is sometimes found given random initial points."""
	def __init__(self, K, X=None, N=0):
		self.K = K
		if X.any() == None:
		    if N == 0:
		        raise Exception("If no data is provided, \
		                         a parameter N (number of points) is needed")
		    else:
		        self.N = N
		        self.X = self._init_board_gauss(N, K)
		else:
		    self.X = X
		    self.N = len(X)
		self.mu = None
		self.clusters = None
		self.method = None

	def find_centers(self, K, method='random'):
		self.method = method
		X = self.X
		K = self.K
		self.oldmu = random.sample(X, K)
		if method != '++':
		    # Initialize to K random centers
		    self.mu = random.sample(X, K)
		while not self._has_converged():
		    self.oldmu = self.mu
		    # Assign all points in X to clusters
		    self._cluster_points()
		    # Reevaluate centers
		    self._reevaluate_centers()

	def _init_board_gauss(self, N, k):
		n = float(N)/k
		X = []
		for i in range(k):
		    c = (random.uniform(-1,1), random.uniform(-1,1))
		    s = random.uniform(0.05,0.15)
		    x = []
		    while len(x) < n:
		        a,b = np.array([np.random.normal(c[0],s),np.random.normal(c[1],s)])
		        # Continue drawing points from the distribution in the range [-1,1]
		        if abs(a) and abs(b)<1:
		            x.append([a,b])
		    X.extend(x)
		X = np.array(X)[:N]
		return X

	def plot_board(self):
		X = self.X
		fig = plt.figure(figsize=(5,5))
		#plt.xlim(-1,1)
		#plt.ylim(-1,1)
		if self.mu and self.clusters:
		    mu = self.mu
		    clus = self.clusters
		    K = self.K
		    for m, clu in clus.items():
		        cs = plt.cm.spectral(1.*m/self.K)
		        plt.plot(mu[m][0], mu[m][1], 'o', marker='*', \
		                 markersize=12, color=cs)
		        plt.plot(zip(*clus[m])[0], zip(*clus[m])[1], '.', \
		                 markersize=8, color=cs, alpha=0.5)
		else:
		    plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
		if self.method == '++':
		    tit = 'K-means++'
		else:
		    tit = 'K-means with random initialization'
		pars = 'N=%s, K=%s' % (str(self.N), str(self.K))
		plt.title('\n'.join([pars, tit]), fontsize=16)
		plt.savefig('kpp_N%s_K%s.png' % (str(self.N), str(self.K)), \
		            bbox_inches='tight', dpi=200)

	def _cluster_points(self):
		mu = self.mu
		clusters  = {}
		for x in self.X:
		    bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
		                     for i in enumerate(mu)], key=lambda t:t[1])[0]
		    try:
		        clusters[bestmukey].append(x)
		    except KeyError:
		        clusters[bestmukey] = [x]
		self.clusters = clusters

	def _reevaluate_centers(self):
		clusters = self.clusters
		newmu = []
		keys = sorted(self.clusters.keys())
		for k in keys:
		    newmu.append(np.mean(clusters[k], axis = 0))
		self.mu = newmu

	def _has_converged(self):
		K = len(self.oldmu)
		return(set([tuple(a) for a in self.mu]) == \
		       set([tuple(a) for a in self.oldmu])\
		       and len(set([tuple(a) for a in self.mu])) == K)

	


#kmeans = KMeans(3, N=200)
#kmeans.find_centers()
#kmeans.plot_board()

class KPlusPlus(KMeans):
	"""Inherits the standard k means, with better intilization stage."""
	def _dist_from_centers(self):
	    cent = self.mu
	    X = self.X
	    D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
	    self.D2 = D2

	def _choose_next_center(self):
		#the next center is drawn from a distribution given by the normalized distance vector D_i^2 / (sum_j D_j^2)
	    self.probs = self.D2/self.D2.sum()
	    self.cumprobs = self.probs.cumsum()
	    r = random.random()
	    ind = np.where(self.cumprobs >= r)[0][0]
	    return(self.X[ind])

	def init_centers(self,n=1):
	    self.mu = random.sample(self.X, n)
	    while len(self.mu) < self.K:
	        self._dist_from_centers()
	        self.mu.append(self._choose_next_center())

	def plot_init_centers(self):
	    X = self.X
	    fig = plt.figure(figsize=(5,5))
	    #plt.xlim(-1,1)
	    #plt.ylim(-1,1)
	    plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
	    plt.plot(zip(*self.mu)[0], zip(*self.mu)[1], 'ro')
	    plt.savefig('kpp_init_N%s_K%s.png' % (str(self.N),str(self.K)), \
	                bbox_inches='tight', dpi=200)

#kplusplus = KPlusPlus(5, N=200)
#kplusplus.init_centers()
#kplusplus.plot_init_centers()

# # compare K means and Kplusplus
# Random initialization
#kplusplus.find_centers()
#kplusplus.plot_board()
# k-means++ initialization
#kplusplus.find_centers(method='++')
#kplusplus.plot_board()

#--------------Implementation of Pham et al 
#----ref: https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
class DetK(KPlusPlus):
    def fK(self, thisk, Skm1=0):
        X = self.X
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
        self.find_centers(thisk, method='++')
        mu, clusters = self.mu, self.clusters
        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk  
 
    def _bounding_box(self):
        X = self.X
        xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
        ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
        return (xmin,xmax), (ymin,ymax)        
         
    def gap(self, thisk):
        X = self.X
        (xmin,xmax), (ymin,ymax) = self._bounding_box()
        self.init_centers(thisk)
        self.find_centers(thisk, method='++')
        mu, clusters = self.mu, self.clusters
        Wk = np.log(sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
                    for i in range(thisk) for c in clusters[i]]))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax), \
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            kb = DetK(thisk, X=Xb)
            kb.init_centers(thisk)
            kb.find_centers(thisk, method='++')
            ms, cs = kb.mu, kb.clusters
            BWkbs[i] = np.log(sum([np.linalg.norm(ms[j]-c)**2/(2*len(c)) \
                              for j in range(thisk) for c in cs[j]]))
        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2)/float(B))*np.sqrt(1+1/B)
        return Wk, Wkb, sk
     
    def run(self, maxk, which='both'):
        ks = range(1,maxk)
        fs = np.zeros(len(ks))
        Wks,Wkbs,sks = np.zeros(len(ks)+1),np.zeros(len(ks)+1),np.zeros(len(ks)+1)
        # Special case K=1
        self.init_centers(1)
        if which == 'f':
            fs[0], Sk = self.fK(1)
        elif which == 'gap':
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        else:
            fs[0], Sk = self.fK(1)
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        # Rest of Ks
        for k in ks[1:]:
            self.init_centers(k)
            if which == 'f':
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
            elif which == 'gap':
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
            else:
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
        if which == 'f':
            self.fs = fs
        elif which == 'gap':
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
        else:
            self.fs = fs
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
        foundfK = np.where(self.fs == min(self.fs))[0][0] + 1
        return foundfK

     
    def plot_all(self):
        X = self.X
        ks = range(1, len(self.fs)+1)
        fig = plt.figure(figsize=(18,5))
        # Plot 1
        ax1 = fig.add_subplot(131)
        #ax1.set_xlim(-1,1)
        #ax1.set_ylim(-1,1)
        ax1.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        tit1 = 'N=%s' % (str(len(X)))
        ax1.set_title(tit1, fontsize=16)
        # Plot 2
        ax2 = fig.add_subplot(132)
        #ax2.set_ylim(0, 1.25)
        ax2.plot(ks, self.fs, 'ro-', alpha=0.6)
        ax2.set_xlabel('Number of clusters K', fontsize=16)
        ax2.set_ylabel('f(K)', fontsize=16) 
        foundfK = np.where(self.fs == min(self.fs))[0][0] + 1
        tit2 = 'f(K) finds %s clusters' % (foundfK)
        ax2.set_title(tit2, fontsize=16)
        # Plot 3
        ax3 = fig.add_subplot(133)
        ax3.bar(ks, self.G, alpha=0.5, color='g', align='center')
        ax3.set_xlabel('Number of clusters K', fontsize=16)
        ax3.set_ylabel('Gap', fontsize=16)
        foundG = np.where(self.G > 0)[0][0] + 1
        tit3 = 'Gap statistic finds %s clusters' % (foundG)
        ax3.set_title(tit3, fontsize=16)
        ax3.xaxis.set_ticks(range(1,len(ks)+1))
        plt.savefig('detK_N%s.png' % (str(len(X))), \
                     bbox_inches='tight', dpi=100)
        


#-----------find optimal K with gap statistics
#-----ref: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
	K = len(oldmu)
	return(set([tuple(a) for a in mu]) == \
		   set([tuple(a) for a in oldmu])\
		   and len(set([tuple(a) for a in mu])) == K)

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def Wk(mu, clusters):
	K = len(mu)
	return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
	           for i in range(K) for c in clusters[i]])

def bounding_box(X):
	xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
	ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
	return (xmin,xmax), (ymin,ymax)

def gap_statistic(X):
	(xmin,xmax), (ymin,ymax) = bounding_box(X)
	# Dispersion for real distribution
	ks = range(1,10)
	Wks = np.zeros(len(ks))
	Wkbs = np.zeros(len(ks))
	sk = np.zeros(len(ks))
	for indk, k in enumerate(ks):
	    mu, clusters = find_centers(X,k)
	    Wks[indk] = np.log(Wk(mu, clusters))
	    # Create B reference datasets
	    B = 10
	    BWkbs = np.zeros(B)
	    for i in range(B):
	        Xb = []
	        for n in range(len(X)):
	            Xb.append([random.uniform(xmin,xmax),
	                      random.uniform(ymin,ymax)])
	        Xb = np.array(Xb)
	        mu, clusters = find_centers(Xb,k)
	        BWkbs[i] = np.log(Wk(mu, clusters))
	    Wkbs[indk] = np.sum(BWkbs)/B
	    sk[indk] = np.sqrt(np.sum((BWkbs-Wkbs[indk])**2)/B)
	sk = sk*np.sqrt(1+1/B)
	return(ks, Wks, Wkbs, sk)


# example:
#X = init_board_gauss(200,3)
#ks, logWks, logWkbs, sk = gap_statistic(X)
##The correct K is the smallest for which the quantity plotted in blue bars becomes positive. 
