# sann_Safe-DNN_reproduce
Reproduce of sann benchmarks from the paper Synchronous neural networks for cyber-physical systems

About the Benchmarks:
All the benchmarks have the exact same NN structures as the benchmarks from the paper, except the darknet CNN. 
For the darknet benchmark, a simpler CNN is used to save training/validating time.
Hello_RNN is the only benchmark did not get reproduced, because Safe DNN framework does not support RNN at the moment.

About the Models:
To reproduce this work, all the relavant code are stored in the Code and Materials to generate benchmarks in python folder. Please use the MNN2C in Safe DNN tool to generate synchronous NNs from keras to c code. (Make sure the meta_structures and the NN structure/weight files are under the same folder)

About the Results:
The wcet_report is under NN models and reports folder for each benchmark
