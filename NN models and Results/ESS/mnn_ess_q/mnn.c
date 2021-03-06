/*
 * mnn.c
 * The implementation of a pre-trained time-predictable meta neural network, generated by MNN2C.py.
 */

#include <stdbool.h>
#include "libcorethread/corethread.h"
#include "mnn.h"


#define DELAY_ESS_Q_TO_OUTPUTS 1
#define DELAY_INPUTS_TO_ESS_Q 1

// Cluster 0
// ess_q's connection 0, from inputs
// Source range(s)
#define CONN_INPUTS_TO_ESS_Q_0_DIM_0_START 0
#define CONN_INPUTS_TO_ESS_Q_0_DIM_0_STOP 21
// Destination range(s) and length(s)
#define CONN_ESS_Q_FROM_INPUTS_0_DIM_0_START 0
#define CONN_ESS_Q_FROM_INPUTS_0_DIM_0_STOP 21
#define CONN_ESS_Q_FROM_INPUTS_0_DIM_0_LEN 21
// Outputs
// outputs's connection 0, from ess_q
// Source range(s)
#define CONN_ESS_Q_TO_OUTPUTS_0_DIM_0_START 0
#define CONN_ESS_Q_TO_OUTPUTS_0_DIM_0_STOP 41
// Destination range(s) and length(s)
#define CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_START 0
#define CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_STOP 41
#define CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_LEN 41

void increment_index(int *index, int length)
{
    if (++(*index) >= length)
    {
        *index = 0;
    }
}


// Load and run functions
// Cluster 0
void run_unit_ess_q(NN_META_DATA *mnn_data)
{
    int src_dim_0_index;
    int dst_dim_0_index;
    src_dim_0_index = CONN_INPUTS_TO_ESS_Q_0_DIM_0_START;

    // Iterate over destination indices
    //#pragma loopbound min CONN_ESS_Q_FROM_INPUTS_0_DIM_0_LEN max CONN_ESS_Q_FROM_INPUTS_0_DIM_0_LEN
    for (dst_dim_0_index = CONN_ESS_Q_FROM_INPUTS_0_DIM_0_START; dst_dim_0_index < CONN_ESS_Q_FROM_INPUTS_0_DIM_0_STOP; dst_dim_0_index++)
    {
        mnn_data->ess_q.inputs[dst_dim_0_index] =
            mnn_data->inputs[mnn_data->ess_q_pl_conn_0_inputs][src_dim_0_index];
    
        src_dim_0_index++;
    }

    // assert(src_dim_0_index == CONN_INPUTS_TO_ESS_Q_0_DIM_0_STOP);
    increment_index(&mnn_data->ess_q_pl_conn_0_inputs, NN_META_IN_PL_LEN);

    nn_run_ess_q(&mnn_data->ess_q);
}


void run_cluster_0(NN_META_DATA *mnn_data)
{
    run_unit_ess_q(mnn_data);
}


void copy_outputs(NN_META_DATA *mnn_data)
{
    int src_dim_0_index;
    int dst_dim_0_index;

    // Copy unit outputs to MNN outputs
    src_dim_0_index = CONN_ESS_Q_TO_OUTPUTS_0_DIM_0_START;

    // Iterate over destination indices
    //#pragma loopbound min CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_LEN max CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_LEN
    for (dst_dim_0_index = CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_START; dst_dim_0_index < CONN_OUTPUTS_FROM_ESS_Q_0_DIM_0_STOP; dst_dim_0_index++)
    {
        mnn_data->outputs[dst_dim_0_index] =
            mnn_data->ess_q.outputs[mnn_data->outputs_pl_conn_0_ess_q][src_dim_0_index];
    
        src_dim_0_index++;
    }

    // assert(src_dim_0_index == CONN_ESS_Q_TO_OUTPUTS_0_DIM_0_STOP);
    increment_index(&mnn_data->outputs_pl_conn_0_ess_q, ESS_Q_MAX_PL_LEN);
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
    nn_init_ess_q();
    
    // Input
    mnn_data->input_pl_index = 0;
    // Cluster 0
    mnn_data->ess_q_pl_conn_0_inputs = (NN_META_IN_PL_LEN - DELAY_INPUTS_TO_ESS_Q) % NN_META_IN_PL_LEN;
    // Output
    mnn_data->outputs_pl_conn_0_ess_q = ESS_Q_MAX_PL_LEN - DELAY_ESS_Q_TO_OUTPUTS;

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