import numpy as np
import os
os.chdir('/home/justin/SFT-Nets/matlab/test_data')

def gen_py_data():
    """
    This takes the vectors from matlab sending times
    and converts them into a format that can be read
    by prob_model_given_data in python
    """
    vs = []
    for i in range(1,6,1):
        fname = 'vector'+str(i)+'.csv'
        data = np.genfromtxt(fname, delimiter =',', skip_header=True)[:,0]
        if i == 1:
            sender = 'A'
            receiver = 'B'
        if i==2 :
            sender ='A'
            receiver = 'C'
        if i==3 :
            sender ='B'
            receiver = 'C'
        if i == 4 :
            sender ='B'
            receiver = 'D'
        if i==5 :
            sender ='C'
            receiver = 'D'
        vs.extend(zip(data, [sender]*len(data), [receiver]*len(data)))
        print len(data)
    s = sorted(vs, key = lambda x: x[0])
    times = np.asarray([j[0] for j in s])
    senders = np.asarray([j[1] for j in s])
    receivers = np.asarray([j[2] for j in s])
    return times, senders, receivers




