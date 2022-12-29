## Final Grade: A (95.54%)

### Homework 1. 10/10 pts

### Homework 2. 10/10 pts

### Quiz. 3.5/5 pts

### Final Project. 20/20 pts

Cross Validation wrt Hydrolyzed Polyacrylamide Research Paper

*The aim of the project is to test the results of various studies and to explore whether the published data and results can be extrapolated to another study. So let's say one has 4 research papers with input data (eg temperature, pressure, salinity) and report outputs (eg rheological parameters). You can use 3 of these given documents as a train subset and the fourth one as a test subset. By following this approach, you can perform cross-validate wrt a research article in sequence and see how the results of one experiment match up with others.*

**Results**

 - Data-mining from research papers
 - Cross valdiation wrt paper showed a really bad results which means that the data obtained from different conditions, experiments and with different study purposes do not match in terms of results with each other
 - Performance of general machine learning approach (mix all the data and perform train-test split) showed sustainable results. What is interesting that the best model is kNN with just *one* neighbour. This result shows that each data is distinguishable because of a gap in conditions of experiments (one with temp 298.15K, other wtih temp 323.15K)
 - According to PCA and results of TreeRegression with 2nd order polynomial features there might exist 2nd order connection in input parameters 