__author__ = 'Tom Schaul, tom@idsia.ch, Spyridon Samothrakis ssamot@essex.ac.uk, Sathish Sankarpandi'

## ssamot hacked ask/tell interface, algorithmic implementation is from Tom Schaul

from numpy import dot, exp, log, sqrt, ones, zeros_like, Inf, argmax
import numpy as np
from scipy.stats import entropy
import os
import cloudpickle as pickle
import random
random.seed(500)


def computeUtilities(fitnesses):
    L = len(fitnesses)
    ranks = zeros_like(fitnesses)
    l = list(zip(fitnesses, range(L)))
    l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = np.array([max(0., x) for x in log(L / 2. + 1.0) - log(L - np.array(ranks))])
    utilities = utilities/sum(utilities)       # make the utilities sum to 1
    utilities -= 1. / L  # baseline
    return utilities


# data loading

class SNES():
    def __init__(self, x0, learning_rate_mult, popsize):
        self.x0 = x0
        self.batchSize = popsize
        self.dim = len(x0)
        self.learningRate =  0.2 * (3 + log(self.dim)) / sqrt(self.dim)
        #print self.learningRate
        #self.learningRate = self.learningRate*learning_rate_mult
        #self.learningRate = 0.000001
        self.numEvals = 0
        self.bestFound = None
        self.sigmas = ones(self.dim)
        self.bestFitness = -Inf
        self.center = x0.copy()
        self.verbose = True

    def ask(self):
        self.samples = [np.random.randn(self.dim) for _ in range(self.batchSize)]
        asked = [(self.sigmas * s + self.center) for s in self.samples]
        self.asked = asked
        return asked

    def tell(self, asked, fitnesses):

        samples = self.samples
        assert(np.array_equal(asked, self.asked))
        if max(fitnesses) > self.bestFitness:
            self.bestFitness = max(fitnesses)
            self.bestFound = samples[argmax(fitnesses)]
        self.numEvals += self.batchSize
        if self.verbose: print ("Step", self.numEvals/self.batchSize, ":", max(fitnesses), "best:", self.bestFitness, len(fitnesses))

        # update center and variances
        utilities = computeUtilities(fitnesses)
        self.center = self.center + self.sigmas * dot(utilities, samples)
        covGradient = dot(utilities, [s ** 2 - 1 for s in samples])
        self.sigmas =  self.sigmas* exp(0.5 * self.learningRate * covGradient)
        best = self.bestFitness
        return best
        


if __name__ == "__main__":
    
    
    # loading the glove model and changing the list to array
    import gensim.models
    
    cwd = os.path.abspath(os.path.dirname('__file__'))
    cwd = r'C:\Users\Sathish\Desktop\Datasets'
    my_path = os.path.join(cwd, 'GoogleNews-vectors-negative300.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(my_path, binary=True)
    word_vectors = np.array(model.vectors)
    def close_wordfinder(vec):
        """ finding the closest words of the vector """    
          
        diff = word_vectors - vec
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        return i
    
    closest_word =  model.index2word[close_wordfinder(asked[300])]
    
         
    
    # 100-dimensional ellipsoid function
    dim = 300
#    A = np.array([np.power(1000, 2 * i / (dim - 1.)) for i in range(dim)])
#    def elli(x):
#        return -dot(A * x, x)
#    
    my_path = os.path.abspath(os.path.dirname('__file__'))

#    Initialisation (vector for what is cold)
    token = pickle.load(open(os.path.join(my_path, r'token.pkl'),'rb'))
      
    with open(os.path.join(my_path, r'intent_classifier.pkl'), 'rb') as f:
        trainedModel = pickle.load(f)
    
    def entropycalculator(x,trainedModel):
        proba = trainedModel.predict([x])
        entro = entropy(np.transpose([proba[1]]))
        return entro  
    snes = SNES(token, 1, 10)
    
#    snes = SNES(ones(dim), 1, 10)
    best= np.zeros(1000)
    for i in range(0,1000):
        asked = snes.ask()
        #print asked
#        told = [elli(a) for a in asked ]
        closest_word = [model.index2word[close_wordfinder(a)] for a in asked] 
        vec2word = ([model.get_vector(word) for word in closest_word])
#        mean_vec2word = np.mean(vec2word,axis=0)
#        told = [entropycalculator(a,trainedModel) for a in asked]
        told = [entropycalculator(a,trainedModel) for a in vec2word]
        snes.tell(asked,np.reshape(told,(10)))
        best[i] = snes.tell(asked,np.reshape(told,(10)))
    import matplotlib.pyplot as plt
    plt.plot(best)
        
#closest_word =  model.index2word[close_wordfinder(asked[3])]

    # # example run
    # print SNES(elli, ones(dim), verbose=True)
