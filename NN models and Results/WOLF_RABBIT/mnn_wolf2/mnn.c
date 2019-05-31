/*
 * mnn.c
 * The implementation of a pre-trained time-predictable meta neural network, generated by MNN2C.py.
 */

#include <stdbool.h>
#include "libcorethread/corethread.h"
#include "mnn.h"


#define DELAY_WOLF2_TO_OUTPUTS 1
#define DELAY_INPUTS_TO_WOLF2 1

// Cluster 0
// wolf2's connection 0, from inputs
// Source range(s)
#define CONN_INPUTS_TO_WOLF2_0_DIM_0_START 0
#define CONN_INPUTS_TO_WOLF2_0_DIM_0_STOP 20
// Destination range(s) and length(s)
#define CONN_WOLF2_FROM_INPUTS_0_DIM_0_START 0
#define CONN_WOLF2_FROM_INPUTS_0_DIM_0_STOP 20
#define CONN_WOLF2_FROM_INPUTS_0_DIM_0_LEN 20
// Outputs
// outputs's connection 0, from wolf2
// Source range(s)
#define CONN_WOLF2_TO_OUTPUTS_0_DIM_0_START 0
#define CONN_WOLF2_TO_OUTPUTS_0_DIM_0_STOP 8
// Destination range(s) and length(s)
#define CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_START 0
#define CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_STOP 8
#define CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_LEN 8

void increment_index(int *index, int length)
{
    if (++(*index) >= length)
    {
        *index = 0;
    }
}


// Load and run functions
// Cluster 0
void run_unit_wolf2(NN_META_DATA *mnn_data)
{
    int src_dim_0_index;
    int dst_dim_0_index;
    src_dim_0_index = CONN_INPUTS_TO_WOLF2_0_DIM_0_START;

    // Iterate over destination indices
    //#pragma loopbound min CONN_WOLF2_FROM_INPUTS_0_DIM_0_LEN max CONN_WOLF2_FROM_INPUTS_0_DIM_0_LEN
    for (dst_dim_0_index = CONN_WOLF2_FROM_INPUTS_0_DIM_0_START; dst_dim_0_index < CONN_WOLF2_FROM_INPUTS_0_DIM_0_STOP; dst_dim_0_index++)
    {
        mnn_data->wolf2.inputs[dst_dim_0_index] =
            mnn_data->inputs[mnn_data->wolf2_pl_conn_0_inputs][src_dim_0_index];
    
        src_dim_0_index++;
    }

    // assert(src_dim_0_index == CONN_INPUTS_TO_WOLF2_0_DIM_0_STOP);
    increment_index(&mnn_data->wolf2_pl_conn_0_inputs, NN_META_IN_PL_LEN);

    nn_run_wolf2(&mnn_data->wolf2);
}


void run_cluster_0(NN_META_DATA *mnn_data)
{
    run_unit_wolf2(mnn_data);
}


void copy_outputs(NN_META_DATA *mnn_data)
{
    int src_dim_0_index;
    int dst_dim_0_index;

    // Copy unit outputs to MNN outputs
    src_dim_0_index = CONN_WOLF2_TO_OUTPUTS_0_DIM_0_START;

    // Iterate over destination indices
    //#pragma loopbound min CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_LEN max CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_LEN
    for (dst_dim_0_index = CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_START; dst_dim_0_index < CONN_OUTPUTS_FROM_WOLF2_0_DIM_0_STOP; dst_dim_0_index++)
    {
        mnn_data->outputs[dst_dim_0_index] =
            mnn_data->wolf2.outputs[mnn_data->outputs_pl_conn_0_wolf2][src_dim_0_index];
    
        src_dim_0_index++;
    }

    // assert(src_dim_0_index == CONN_WOLF2_TO_OUTPUTS_0_DIM_0_STOP);
    increment_index(&mnn_data->outputs_pl_conn_0_wolf2, WOLF2_MAX_PL_LEN);
}


void tick_controller(WORKER_DATA *worker_data, int status)
{
    int worker_index;

    // Wait for all workers to arrive
    for (worker_index = 0; worker_index < WORKER_COUNT; worker_index++)
    {
        while (!worker_data->worker_flags[worker_index])
        {
            inval_dcache();
        }
    }  

    // Set status
    worker_data->status = status;

    // Reset flags
    for (worker_index = 0; worker_index < WORKER_COUNT; worker_index++)
    {
        worker_data->worker_flags[worker_index] = false;
    }

    // Toggle barrier, releasing workers
    worker_data->tick_barrier = !worker_data->tick_barrier;
}


void tick_worker(WORKER_DATA *worker_data, int id)
{
    int barrier_sample = worker_data->tick_barrier;

    // Mark self as arrived
    worker_data->worker_flags[id] = true;

    // Wait for a signal from the controller (a change in tick_barrier value)
    while (worker_data->tick_barrier == barrier_sample)
    {
        inval_dcache();
    }
}


int mnn_init(NN_META_DATA *mnn_data)
{
    nn_init_wolf2();
    
    // Input
    mnn_data->input_pl_index = 0;
    // Cluster 0
    mnn_data->wolf2_pl_conn_0_inputs = (NN_META_IN_PL_LEN - DELAY_INPUTS_TO_WOLF2) % NN_META_IN_PL_LEN;
    // Output
    mnn_data->outputs_pl_conn_0_wolf2 = WOLF2_MAX_PL_LEN - DELAY_WOLF2_TO_OUTPUTS;

    mnn_data->worker_data.status = STATUS_RUNNING;

    return 0;
}


int mnn_response(NN_META_DATA *nn_data)
{
    increment_index(&nn_data->input_pl_index, NN_META_IN_PL_LEN);
    copy_outputs(nn_data);
    
    run_cluster_0(nn_data);

    tick_controller(&nn_data->worker_data, STATUS_RUNNING);
    return nn_data->input_pl_index; // The buffer to which new input data must be written
}


void mnn_free(NN_META_DATA *mnn_data)
{
    tick_controller(&mnn_data->worker_data, STATUS_FINISHED);

    int *ret;
    for (int worker_index = 0; worker_index < WORKER_COUNT; worker_index++)
    {
        corethread_join(worker_index + 1, (void **)&ret);
        if (ret != NULL)
        {
            printf("Worker exited with unexpected return status: %p", ret);
            exit(1);
        }
    }
}