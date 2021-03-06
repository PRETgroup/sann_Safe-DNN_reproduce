/*
 * nn_aibro_b.h
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.h written by keyan
 */

#ifndef NN_AIBRO_B_H_
#define NN_AIBRO_B_H_

#pragma once

// includes
#include "nn_types.h"

// NOTE: only values marked "*" may be changed with defined behaviour

// Network input defines
#define AIBRO_B_INPUT_DIM_COUNT	1
#define AIBRO_B_INPUT_DIM_0	4

// Network output defines
#define AIBRO_B_MAX_PL_LEN 2
#define AIBRO_B_OUTPUT_DIM_COUNT	1
#define AIBRO_B_OUTPUT_DIM_0	2

// Dense layer defines
#define AIBRO_B_L0_DENSE_NEURON_COUNT	3
#define AIBRO_B_L0_DENSE_USE_BIAS	1
#define AIBRO_B_L0_DENSE_ACTIVATION	ACT_ENUM_TANH	// *
#define AIBRO_B_L0_DENSE_DIM_COUNT	1
#define AIBRO_B_L0_DENSE_DIM_0	3

// Dense layer defines
#define AIBRO_B_L1_DENSE_NEURON_COUNT	2
#define AIBRO_B_L1_DENSE_USE_BIAS	1
#define AIBRO_B_L1_DENSE_ACTIVATION	ACT_ENUM_TANH	// *
#define AIBRO_B_L1_DENSE_DIM_COUNT	1
#define AIBRO_B_L1_DENSE_DIM_0	2

// NN weights struct
typedef struct
{
	// Dense layer unit
	NN_NUM_TYPE aibro_b_l0_dense_weights[AIBRO_B_INPUT_DIM_0][AIBRO_B_L0_DENSE_NEURON_COUNT];
	NN_NUM_TYPE aibro_b_l0_dense_bias[AIBRO_B_L0_DENSE_NEURON_COUNT];
	// Dense layer unit
	NN_NUM_TYPE aibro_b_l1_dense_weights[AIBRO_B_L0_DENSE_DIM_0][AIBRO_B_L1_DENSE_NEURON_COUNT];
	NN_NUM_TYPE aibro_b_l1_dense_bias[AIBRO_B_L1_DENSE_NEURON_COUNT];
} NN_AIBRO_B;

// Static storage
NN_AIBRO_B nn_weights_aibro_b;

// Instance
typedef struct
{
	NN_NUM_TYPE inputs[AIBRO_B_INPUT_DIM_0];
	NN_NUM_TYPE aibro_b_l0_dense_outputs[AIBRO_B_L0_DENSE_DIM_0];
	NN_NUM_TYPE outputs[AIBRO_B_MAX_PL_LEN][AIBRO_B_L1_DENSE_DIM_0];
	

	// Output pipeline
	int pl_index;
} NN_DATA_AIBRO_B;

// Functions
void nn_init_aibro_b();
void run_aibro_b_l0_dense();
void run_aibro_b_l1_dense();

void nn_run_aibro_b(NN_DATA_AIBRO_B *nn_data);

#endif /* NN_AIBRO_B_H_ */