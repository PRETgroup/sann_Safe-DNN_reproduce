/*
 * nn_wolf1.c
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.c written by keyan
 */

#include "nn_wolf1.h"
#include <stdbool.h>
#include <unistd.h>


// Dense init function
#include "wolf1_l0_dense_weights"
#if WOLF1_L0_DENSE_USE_BIAS
		#include "wolf1_l0_dense_bias"
#endif
void init_wolf1_l0_dense(void)
{
	for (int prev_neuron_index = 0; prev_neuron_index < WOLF1_INPUT_DIM_0; prev_neuron_index++)
	{
		for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L0_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_wolf1.wolf1_l0_dense_weights[prev_neuron_index][curr_neuron_index] =
				FROM_DBL(wolf1_l0_dense_weights[prev_neuron_index][curr_neuron_index]);
		}
	}
	#if WOLF1_L0_DENSE_USE_BIAS
		#include "wolf1_l0_dense_bias"
		for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L0_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_wolf1.wolf1_l0_dense_bias[curr_neuron_index] =
				FROM_DBL(wolf1_l0_dense_bias[curr_neuron_index]);
		}
	#endif
}
	

// Dense init function
#include "wolf1_l1_dense_weights"
#if WOLF1_L1_DENSE_USE_BIAS
		#include "wolf1_l1_dense_bias"
#endif
void init_wolf1_l1_dense(void)
{
	for (int prev_neuron_index = 0; prev_neuron_index < WOLF1_L0_DENSE_DIM_0; prev_neuron_index++)
	{
		for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L1_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_wolf1.wolf1_l1_dense_weights[prev_neuron_index][curr_neuron_index] =
				FROM_DBL(wolf1_l1_dense_weights[prev_neuron_index][curr_neuron_index]);
		}
	}
	#if WOLF1_L1_DENSE_USE_BIAS
		#include "wolf1_l1_dense_bias"
		for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L1_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_wolf1.wolf1_l1_dense_bias[curr_neuron_index] =
				FROM_DBL(wolf1_l1_dense_bias[curr_neuron_index]);
		}
	#endif
}
	

// create NN with pre-trained weights
void nn_init_wolf1(void)
{
	init_wolf1_l0_dense();
	init_wolf1_l1_dense();
}


// Layer run definitions
// Dense run function
// Applies activation function to a weighted sum of inputs for each neuron in this layer
void run_wolf1_l0_dense(NN_DATA_WOLF1 *nn_data)
{
	NN_NUM_TYPE weighted_sum;
	NN_NUM_TYPE prev_neuron_value;
	NN_NUM_TYPE weight;
	NN_NUM_TYPE bias;
	

	// Calculate weighted sums
	for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L0_DENSE_NEURON_COUNT; curr_neuron_index++)
	{
		weighted_sum = 0;
		for (int prev_neuron_index = 0; prev_neuron_index < WOLF1_INPUT_DIM_0; prev_neuron_index++)
		{
			prev_neuron_value = nn_data->inputs[prev_neuron_index];
			weight = nn_weights_wolf1.wolf1_l0_dense_weights[prev_neuron_index][curr_neuron_index];
			weighted_sum = ADD(weighted_sum, MUL(weight, prev_neuron_value));
		}
		#if WOLF1_L0_DENSE_USE_BIAS
			bias = nn_weights_wolf1.wolf1_l0_dense_bias[curr_neuron_index];
			weighted_sum = ADD(weighted_sum, bias);
		#endif
		nn_data->wolf1_l0_dense_outputs[curr_neuron_index] = weighted_sum;
	}

	// Apply activation
	NN_NUM_TYPE elem;
	#if WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
		NN_NUM_TYPE softmax_sum = 0;
		
		for (int dim_0_index = 0; dim_0_index < WOLF1_L0_DENSE_DIM_0; dim_0_index++)
		{
		    elem = nn_data->wolf1_l0_dense_outputs[dim_0_index];
		    elem = EXP(elem);
		    softmax_sum = ADD(softmax_sum, elem);
		    nn_data->wolf1_l0_dense_outputs[dim_0_index] = elem;
		}
	#endif
	
	for (int dim_0_index = 0; dim_0_index < WOLF1_L0_DENSE_DIM_0; dim_0_index++)
	{
	    elem = nn_data->wolf1_l0_dense_outputs[dim_0_index];

	    #if WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_SIGMOID
	    	elem = sigmoid(elem);
	    #elif WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_TANH
	    	elem = tanh(elem);
	    #elif WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_RELU
	    	elem = relu(elem);
	    #elif WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_LINEAR
	    	elem = linear(elem);
	    #elif WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
	    	elem = DIV(elem, softmax_sum);
      #elif WOLF1_L0_DENSE_ACTIVATION == ACT_ENUM_HARD_SIGMOID
      	elem = hard_sigmoid(elem);
      #else
	    	printf("Invalid activation function request - %d", activation);
	    	exit(1);
	    #endif
	    nn_data->wolf1_l0_dense_outputs[dim_0_index] = elem;
	}
}


// Dense run function
// Applies activation function to a weighted sum of inputs for each neuron in this layer
void run_wolf1_l1_dense(NN_DATA_WOLF1 *nn_data)
{
	NN_NUM_TYPE weighted_sum;
	NN_NUM_TYPE prev_neuron_value;
	NN_NUM_TYPE weight;
	NN_NUM_TYPE bias;
	int pl_out_index = nn_data->pl_index;

	// Calculate weighted sums
	for (int curr_neuron_index = 0; curr_neuron_index < WOLF1_L1_DENSE_NEURON_COUNT; curr_neuron_index++)
	{
		weighted_sum = 0;
		for (int prev_neuron_index = 0; prev_neuron_index < WOLF1_L0_DENSE_DIM_0; prev_neuron_index++)
		{
			prev_neuron_value = nn_data->wolf1_l0_dense_outputs[prev_neuron_index];
			weight = nn_weights_wolf1.wolf1_l1_dense_weights[prev_neuron_index][curr_neuron_index];
			weighted_sum = ADD(weighted_sum, MUL(weight, prev_neuron_value));
		}
		#if WOLF1_L1_DENSE_USE_BIAS
			bias = nn_weights_wolf1.wolf1_l1_dense_bias[curr_neuron_index];
			weighted_sum = ADD(weighted_sum, bias);
		#endif
		nn_data->outputs[pl_out_index][curr_neuron_index] = weighted_sum;
	}

	// Apply activation
	NN_NUM_TYPE elem;
	#if WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
		NN_NUM_TYPE softmax_sum = 0;
		
		for (int dim_0_index = 0; dim_0_index < WOLF1_L1_DENSE_DIM_0; dim_0_index++)
		{
		    elem = nn_data->outputs[pl_out_index][dim_0_index];
		    elem = EXP(elem);
		    softmax_sum = ADD(softmax_sum, elem);
		    nn_data->outputs[pl_out_index][dim_0_index] = elem;
		}
	#endif
	
	for (int dim_0_index = 0; dim_0_index < WOLF1_L1_DENSE_DIM_0; dim_0_index++)
	{
	    elem = nn_data->outputs[pl_out_index][dim_0_index];

	    #if WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_SIGMOID
	    	elem = sigmoid(elem);
	    #elif WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_TANH
	    	elem = tanh(elem);
	    #elif WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_RELU
	    	elem = relu(elem);
	    #elif WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_LINEAR
	    	elem = linear(elem);
	    #elif WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
	    	elem = DIV(elem, softmax_sum);
      #elif WOLF1_L1_DENSE_ACTIVATION == ACT_ENUM_HARD_SIGMOID
      	elem = hard_sigmoid(elem);
      #else
	    	printf("Invalid activation function request - %d", activation);
	    	exit(1);
	    #endif
	    nn_data->outputs[pl_out_index][dim_0_index] = elem;
	}
}


// Must be called after every nn_run for pipelining
void pl_update_wolf1(NN_DATA_WOLF1 *nn_data)
{
	if (++nn_data->pl_index >= WOLF1_MAX_PL_LEN)
	{
		nn_data->pl_index = 0;
	}
}


void lstm_dot_product(NN_NUM_TYPE v[], NN_NUM_TYPE u[4][4], int n, NN_NUM_TYPE results[]) {
	for(int i=0;i<n;i++) {
		for(int j=0;j<n;j++) results[j] += v[i] * u[j][i];
	}
}

void lstm_hadamard_product(NN_NUM_TYPE inputs1[], NN_NUM_TYPE inputs2[], int n, NN_NUM_TYPE results[]) {
	for(int i=0;i<n;i++) {
    results[i] = inputs1[i] * inputs2[i];
	}
}

void lstm_add(NN_NUM_TYPE v[], int n, NN_NUM_TYPE bias[], NN_NUM_TYPE results[]) {
  for(int i=0;i<n;i++){
    results[i]=v[i]+bias[i];
  }
}

void lstm_mul(NN_NUM_TYPE v, int n, NN_NUM_TYPE weights[], NN_NUM_TYPE results[]) {
  for(int i=0;i<n;i++){
    results[i]=v*weights[i];
  }
}

void lstm_hard_sigmoid(NN_NUM_TYPE inputs[], int n, NN_NUM_TYPE results[]) {
  for (int i=0;i<n;i++) {
    results[i] = hard_sigmoid(inputs[i]);
  }
}

void lstm_tanh(NN_NUM_TYPE inputs[], int n, NN_NUM_TYPE results[]) {
  for (int i=0;i<n;i++) {
    results[i] = tanh(inputs[i]);
  }
}


void nn_run_wolf1(NN_DATA_WOLF1 *nn_data)
{
	run_wolf1_l0_dense(nn_data);
	run_wolf1_l1_dense(nn_data);
	pl_update_wolf1(nn_data);
}