/*
 * main.c
 * The interface to a pre-trained time-predictable meta neural network, generated by MNN2C.py.
 * Expects input from stdin in the following form:
 *   One integer (the number of inputs to feed into the meta NN)
 *   For each test case, one long space-delimited string, with one entry per network input (read in row-major order).
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "mnn.h"


void print_output_arr(NN_NUM_TYPE output_arr[MNN_OUTPUT_DIM_0])
{
    double write_var;
    int dim_0_index;
    
    for (dim_0_index = 0; dim_0_index < MNN_OUTPUT_DIM_0; dim_0_index++)
    {
        write_var = TO_DBL(output_arr[dim_0_index]);
        printf("%lf ", write_var);
    }

    printf("\n");
}


double tmp_input[MNN_INPUT_DIM_0];
void read_input_data(NN_NUM_TYPE dst[MNN_INPUT_DIM_0])
{
    scanf("%lf %lf %lf %lf %lf", &tmp_input[0], &tmp_input[1], &tmp_input[2], &tmp_input[3], &tmp_input[4]);
    for (int input_dim_0 = 0; input_dim_0 < MNN_INPUT_DIM_0; input_dim_0++)
    {
        dst[input_dim_0] = FROM_DBL(tmp_input[input_dim_0]);
    }
}


void copy_input_arr(NN_NUM_TYPE src[MNN_INPUT_DIM_0], NN_NUM_TYPE dst[MNN_INPUT_DIM_0])
{
    for (int input_dim_0 = 0; input_dim_0 < MNN_INPUT_DIM_0; input_dim_0++)
    {
        dst[input_dim_0] = src[input_dim_0];
    }
}


void copy_output_arr(NN_NUM_TYPE src[MNN_OUTPUT_DIM_0], NN_NUM_TYPE dst[MNN_OUTPUT_DIM_0])
{
    for (int output_dim_0 = 0; output_dim_0 < MNN_OUTPUT_DIM_0; output_dim_0++)
    {
        dst[output_dim_0] = src[output_dim_0];
    }
}





int run_single(
    NN_META_DATA *mnn_data,
    NN_NUM_TYPE test_case_data[MNN_INPUT_DIM_0],
    int input_buffer_index,
    NN_NUM_TYPE results[MNN_OUTPUT_DIM_0],
    bool copy_input,
    bool copy_output
)
{
    // Only load new input for the first test_case_num ticks 
    if (copy_input)
    {
        copy_input_arr(test_case_data, (void *)&mnn_data->inputs[input_buffer_index]);
    }

    // Wait for MNN to finish current step in pipeline, get the next safe writing index for the input buffer
    input_buffer_index = mnn_response(mnn_data);

    // Only print output at (and after) NN_META_PL_LAG ticks
    if (copy_output)
    {
        copy_output_arr(mnn_data->outputs, results);
    }

    return input_buffer_index;
}


// A demonstration of how data should be passed to and retrieved from the MNN during pipelined operation
void run_batch(
    NN_META_DATA *mnn_data,
    int test_case_num,
    NN_NUM_TYPE test_data[test_case_num][MNN_INPUT_DIM_0],
    int input_buffer_index,
    NN_NUM_TYPE results[test_case_num][MNN_OUTPUT_DIM_0]
)
{
    bool copy_input, copy_output;
    for (int tick_index = 0; tick_index < test_case_num + NN_META_PL_LAG; tick_index++)
    {
        copy_input = tick_index < test_case_num;
        copy_output = tick_index >= NN_META_PL_LAG;
        input_buffer_index = run_single(mnn_data, test_data[tick_index], input_buffer_index, results[tick_index - NN_META_PL_LAG], copy_input, copy_output);
    }
}


NN_META_DATA mnn_data;
int main(void)
{
    int test_index;
    int test_case_num;
    NN_NUM_TYPE *test_data;
    NN_NUM_TYPE *results;
    int input_buffer_index;
    int output_buffer_index = 0;
    int input_len = MNN_INPUT_DIM_0;
    int output_len = MNN_OUTPUT_DIM_0;

    scanf("%d", &test_case_num);
    test_data = malloc(test_case_num * input_len * sizeof(NN_NUM_TYPE));
    
    for (test_index = 0; test_index < test_case_num; test_index++)
    {
        read_input_data((void *)&test_data[test_index * input_len]);
    }

    results = malloc(test_case_num * output_len * sizeof(NN_NUM_TYPE));
    input_buffer_index = mnn_init(&mnn_data);
    run_batch(&mnn_data, test_case_num, (void *)test_data, input_buffer_index, (void *)results);

    for (output_buffer_index = 0; output_buffer_index < test_case_num; output_buffer_index++)
    {
        print_output_arr((void *)&results[output_buffer_index * output_len]);
    }

    

    mnn_free(&mnn_data);
    free(test_data);
    free(results);
    
}