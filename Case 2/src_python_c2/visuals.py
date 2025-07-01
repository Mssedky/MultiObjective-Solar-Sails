import numpy as np
import matplotlib.pyplot as plt

def make_plots(meanPi, minPi, meanParents):
    plt.figure()
    plt.semilogy(np.arange(len(meanPi)), meanPi, label='Mean $\Pi$')
    plt.semilogy(np.arange(len(minPi)), minPi, label='Min $\Pi$')
    plt.semilogy(np.arange(len(meanParents)), meanParents, label='Mean $\Pi_{parents}$')
    plt.title('Cost Evolution per Generation for $\Pi$',fontsize=22)
    plt.xlabel('Generation',fontsize=18)
    plt.ylabel('Value',fontsize=18)
    plt.legend()
    plt.show()

