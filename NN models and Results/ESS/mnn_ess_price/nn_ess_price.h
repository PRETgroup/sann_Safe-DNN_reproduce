/*
 * nn_ess_price.h
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.h written by keyan
 */

#ifndef NN_ESS_PRICE_H_
#define NN_ESS_PRICE_H_

#pragma once

// includes
#include "nn_types.h"

// NOTE: only values marked "*" may be changed with defined behaviour

// Network input defines
#define ESS_PRICE_INPUT_DIM_COUNT	1
#define ESS_PRICE_INPUT_DIM_0	10

// Network output defines
#define ESS_PRICE_MAX_PL_LEN 2
#define ESS_PRICE_OUTPUT_DIM_COUNT	1
#define ESS_PRICE_OUTPUT_DIM_0	1

// Dense layer defines
#define ESS_PRICE_L0_DENSE_NEURON_COUNT	5
#define ESS_PRICE_L0_DENSE_USE_BIAS	1
#define ESS_PRICE_L0_DENSE_ACTIVATION	ACT_ENUM_TANH	// *
#define ESS_PRICE_L0_DENSE_DIM_COUNT	1
#define ESS_PRICE_L0_DENSE_DIM_0	5

// Dense layer defines
#define ESS_PRICE_L1_DENSE_NEURON_COUNT	1
#define ESS_PRICE_L1_DENSE_USE_BIAS	1
#define ESS_PRICE_L1_DENSE_ACTIVATION	ACT_ENUM_TANH	// *
#define ESS_PRICE_L1_DENSE_DIM_COUNT	1
#define ESS_PRICE_L1_DENSE_DIM_0	1

// NN weights struct
typedef struct
{
	// Dense layer unit
	NN_NUM_TYPE ess_price_l0_dense_weights[ESS_PRICE_INPUT_DIM_0][ESS_PRICE_L0_DENSE_NEURON_COUNT];
	NN_NUM_TYPE ess_price_l0_dense_bias[ESS_PRICE_L0_DENSE_NEURON_COUNT];
	// Dense layer unit
	NN_NUM_TYPE ess_price_l1_dense_weights[ESS_PRICE_L0_DENSE_DIM_0][ESS_PRICE_L1_DENSE_NEURON_COUNT];
	NN_NUM_TYPE ess_price_l1_dense_bias[ESS_PRICE_L1_DENSE_NEURON_COUNT];
} NN_ESS_PRICE;

// Static storage
NN_ESS_PRICE nn_weights_ess_price;

// Instance
typedef struct
{
	NN_NUM_TYPE inputs[ESS_PRICE_INPUT_DIM_0];
	NN_NUM_TYPE ess_price_l0_dense_outputs[ESS_PRICE_L0_DENSE_DIM_0];
	NN_NUM_TYPE outputs[ESS_PRICE_MAX_PL_LEN][ESS_PRICE_L1_DENSE_DIM_0];
	

	// Output pipeline
	int pl_index;
} NN_DATA_ESS_PRICE;

// Functions
void nn_init_ess_price();
void run_ess_price_l0_dense();
void run_ess_price_l1_dense();

void nn_run_ess_price(NN_DATA_ESS_PRICE *nn_data);

#endif /* NN_ESS_PRICE_H_ */