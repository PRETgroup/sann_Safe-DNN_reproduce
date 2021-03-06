/*
 * nn_wolf2.h
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.h written by keyan
 */

#ifndef NN_WOLF2_H_
#define NN_WOLF2_H_

#pragma once

// includes
#include "nn_types.h"

// NOTE: only values marked "*" may be changed with defined behaviour

// Network input defines
#define WOLF2_INPUT_DIM_COUNT	1
#define WOLF2_INPUT_DIM_0	20

// Network output defines
#define WOLF2_MAX_PL_LEN 2
#define WOLF2_OUTPUT_DIM_COUNT	1
#define WOLF2_OUTPUT_DIM_0	8

// Dense layer defines
#define WOLF2_L0_DENSE_NEURON_COUNT	30
#define WOLF2_L0_DENSE_USE_BIAS	1
#define WOLF2_L0_DENSE_ACTIVATION	ACT_ENUM_SIGMOID	// *
#define WOLF2_L0_DENSE_DIM_COUNT	1
#define WOLF2_L0_DENSE_DIM_0	30

// Dense layer defines
#define WOLF2_L1_DENSE_NEURON_COUNT	8
#define WOLF2_L1_DENSE_USE_BIAS	1
#define WOLF2_L1_DENSE_ACTIVATION	ACT_ENUM_SIGMOID	// *
#define WOLF2_L1_DENSE_DIM_COUNT	1
#define WOLF2_L1_DENSE_DIM_0	8

// NN weights struct
typedef struct
{
	// Dense layer unit
	NN_NUM_TYPE wolf2_l0_dense_weights[WOLF2_INPUT_DIM_0][WOLF2_L0_DENSE_NEURON_COUNT];
	NN_NUM_TYPE wolf2_l0_dense_bias[WOLF2_L0_DENSE_NEURON_COUNT];
	// Dense layer unit
	NN_NUM_TYPE wolf2_l1_dense_weights[WOLF2_L0_DENSE_DIM_0][WOLF2_L1_DENSE_NEURON_COUNT];
	NN_NUM_TYPE wolf2_l1_dense_bias[WOLF2_L1_DENSE_NEURON_COUNT];
} NN_WOLF2;

// Static storage
NN_WOLF2 nn_weights_wolf2;

// Instance
typedef struct
{
	NN_NUM_TYPE inputs[WOLF2_INPUT_DIM_0];
	NN_NUM_TYPE wolf2_l0_dense_outputs[WOLF2_L0_DENSE_DIM_0];
	NN_NUM_TYPE outputs[WOLF2_MAX_PL_LEN][WOLF2_L1_DENSE_DIM_0];
	

	// Output pipeline
	int pl_index;
} NN_DATA_WOLF2;

// Functions
void nn_init_wolf2();
void run_wolf2_l0_dense();
void run_wolf2_l1_dense();

void nn_run_wolf2(NN_DATA_WOLF2 *nn_data);

#endif /* NN_WOLF2_H_ */