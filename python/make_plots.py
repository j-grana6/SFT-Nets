"""
This script makes plots for simulated results. 
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot likelihood difference against infection times
with open("out/output_att.csv","r") as fh:
	prob_no_att, prob_with_att, times_A, times_B, times_C, times_D = \
		np.loadtxt(fh,delimiter=',', unpack=True)
prob_diff = prob_with_att - prob_no_att
fig, ax = plt.subplots(1,1)
ax.plot(times_A,prob_diff,"yo",linestyle = " ",label = "A infection times")
ax.plot(times_B,prob_diff,"rs",linestyle = " ",label = "B infection times")
ax.plot(times_C,prob_diff,"b*",linestyle = " ",label = "C infection times")
ax.plot(times_D,prob_diff,"g^",linestyle = " ",label = "D infection times")
ax.set_xlabel("infection times")
ax.set_ylabel("likelihood difference")
ax.set_title("Attacker Existence")
ax.legend(bbox_to_anchor = (1.0,1.0))
ax.set_yticks(np.arange(-100,175,25))
fig.savefig("images/times_vs_probdiff.png")
