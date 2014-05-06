import numpy as np
import matplotlib.pyplot as plt
import sys
sys.dont_write_bytecode = True

from testing_net import net, T, t0
from tools import gen_data
from direct_sample import Direct_Sample

NN = 100
nn = 100
small = []
large = []
for ep in range(NN):
    print 'ep', ep
    data = gen_data(T,net,t0)
    ests = []
    for i in range(nn):
        print i
        lhood = Direct_Sample(net,data,1000,T,t0)[0]
        ests.append(np.exp(lhood))
    small.append(np.log(np.mean(ests)))
    large.append(Direct_Sample(net,data,100000,T,t0)[0])

fig, ax = plt.subplots(1,1)
ax.plot(small,'b^-',label = 'Small N')
ax.plot(large,'r*-',label = 'Large N')
ax.set_ylabel('Log Likelihood')
ax.set_xticks(xrange(0,105,5))
fig.savefig('test_unbiased_MC.png')
