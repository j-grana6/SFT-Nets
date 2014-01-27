"""
This script contains a function to uniformly sample the SFT net. The results 
will be used to compare to that from MH, Imp Samp, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

from orderings import *

def uniform_samp(SFTnet,s0):
	"""
	Returns a probability. Uniformly sample the SFT net and calculate the 
	likelihood.

	SFTnet: SFTnet instance
		The net where all allowable node infection orderings are computed

	s0: dict
		The state of the SFT net at 0. Key is the node name, value is the state.
	"""
