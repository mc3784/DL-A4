import matplotlib.pyplot as plt
listToPlot=[191.432,
139.289,
118.964,
108.117,
101.802,
93.502,
90.330,
89.181,
88.450,
88.156,
87.828,
87.658,
87.512,
87.441,
87.403,
87.384,
87.377,
87.372,
87.370,
87.368,
87.368,
87.368]
rangePot=range(len(listToPlot))

listToPlot2=[ 188.999,
139.717,
119.144,
107.786,
102.126,
93.855,
90.793,
89.649,
89.007,
88.642,
88.334,
88.037,
87.883,
87.810]

litGRU=[245.802,
168.402,
145.435,
132.561,
127.534,
110.841,
103.322,
100.191,
98.798,
97.914,
97.643,
97.437,
97.301,
97.162,
97.112]


rangePot2=range(len(litGRU))


plt.scatter(rangePot,listToPlot,color='red',label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.scatter(rangePot2,litGRU,color='blue',label="GRU")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.xlim(0)

plt.savefig("LSTMvsGRU3.pdf", format='pdf')


plt.show()
