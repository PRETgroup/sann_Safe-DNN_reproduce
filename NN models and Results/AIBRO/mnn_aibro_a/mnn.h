/*
 * mnn.h
 * The implementation of a pre-trained time-predictable meta neural network, generated by MNN2C.py.
 */

#include "nn_aibro_a.h"
#include "nn_types.h"

// Pipeline input
#define NN_META_IN_PL_LEN 2
// Pipeline length
#define NN_META_PL_LAG 2

// MNN I/O shapes
#define MNN_INPUT_DIM_0 3
#define MNN_OUTPUT_DIM_0 3

#define WORKER_COUNT 0

enum {STATUS_RUNNING, STATUS_FINISHED};

typedef struct
{
    volatile int status;
    volatile int worker_flags[WORKER_COUNT];
    volatile int tick_barrier;
    
} WORKER_DATA;


typedef struct
{
    NN_NUM_TYPE inputs[NN_META_IN_PL_LEN][MNN_INPUT_DIM_0];
    NN_DATA_AIBRO_A aibro_a;
    NN_NUM_TYPE outputs[MNN_OUTPUT_DIM_0];

    // Input pipeline
    int input_pl_index;
    // Cluster 0 pipeline sources
    // aibro_a
    int aibro_a_pl_conn_0_inputs;
    // Output pipelined sources
    int outputs_pl_conn_0_aibro_a;

    WORKER_DATA worker_data;
} NN_META_DATA;


int mnn_init(NN_META_DATA *mnn_data);

int __attribute__ ((noinline)) mnn_response(NN_META_DATA *nn_data);

void mnn_free(NN_META_DATA *nn);