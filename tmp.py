
#plt.imshow(X[3], cmap="Greys")
#plt.show()

#show data distribution
"""  #grafo già messo dentro a git. Ci sono pochi 8, ce ne freghiamo?
values, counts = np.unique(y, return_counts = True)
y_pos = np.arange(len(values))
plt.bar(y_pos, counts)  #creates the bars
plt.xticks(y_pos, values) #creates the names on the x-axis
plt.show()  #shows the graph
"""


"""
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)   #this can be done in the pipeline
X_val = scaler.transform(X_val)        #training time with this preprocessing step takes 1/3 of the time:
X_test = scaler.transform(X_test)       #train time : 53.98358082771301 sec vs 150.37893509864807 sec
                                        #prediction time : 9.5367431640625e-07 sec vs  0.0 sec
                                       #accuracy is also higher: 0.870992963252541 vs 0.7529319781078968
"""



