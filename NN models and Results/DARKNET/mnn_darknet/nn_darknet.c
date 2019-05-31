/*
 * nn_darknet.c
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.c written by keyan
 */

#include "nn_darknet.h"
#include <stdbool.h>
#include <unistd.h>


// Conv2D init function
#include "darknet_l0_conv2d_weights"
#if DARKNET_L0_CONV2D_USE_BIAS
	#include "darknet_l0_conv2d_bias"
#endif
void init_darknet_l0_conv2d(void)
{
	#include "darknet_l0_conv2d_weights"
	for (int kernel_row_index = 0; kernel_row_index < DARKNET_L0_CONV2D_KERNEL_HEIGHT; kernel_row_index++)
	{
		for (int kernel_col_index = 0; kernel_col_index < DARKNET_L0_CONV2D_KERNEL_WIDTH; kernel_col_index++)
		{
			for (int prev_layer_index = 0; prev_layer_index < DARKNET_INPUT_DIM_2; prev_layer_index++)
			{
				for (int kernel_index = 0; kernel_index < DARKNET_L0_CONV2D_KERNEL_COUNT; kernel_index++)
				{
					nn_weights_darknet.darknet_l0_conv2d_kernel_weights[kernel_row_index][kernel_col_index][prev_layer_index][kernel_index] =
						FROM_DBL(darknet_l0_conv2d_weights[kernel_row_index][kernel_col_index][prev_layer_index][kernel_index]);
				}
			}
		}
	}
	#if DARKNET_L0_CONV2D_USE_BIAS
		#include "darknet_l0_conv2d_bias"
		for (int kernel_index = 0; kernel_index < DARKNET_L0_CONV2D_KERNEL_COUNT; kernel_index++)
		{
			nn_weights_darknet.darknet_l0_conv2d_kernel_bias[kernel_index] =
				FROM_DBL(darknet_l0_conv2d_bias[kernel_index]);
		}
	#endif
}
  

// MaxPooling2D init function
void init_darknet_l1_maxpooling2d(void)
{
}
	

// Conv2D init function
#include "darknet_l2_conv2d_weights"
#if DARKNET_L2_CONV2D_USE_BIAS
	#include "darknet_l2_conv2d_bias"
#endif
void init_darknet_l2_conv2d(void)
{
	#include "darknet_l2_conv2d_weights"
	for (int kernel_row_index = 0; kernel_row_index < DARKNET_L2_CONV2D_KERNEL_HEIGHT; kernel_row_index++)
	{
		for (int kernel_col_index = 0; kernel_col_index < DARKNET_L2_CONV2D_KERNEL_WIDTH; kernel_col_index++)
		{
			for (int prev_layer_index = 0; prev_layer_index < DARKNET_L1_MAXPOOLING2D_DIM_2; prev_layer_index++)
			{
				for (int kernel_index = 0; kernel_index < DARKNET_L2_CONV2D_KERNEL_COUNT; kernel_index++)
				{
					nn_weights_darknet.darknet_l2_conv2d_kernel_weights[kernel_row_index][kernel_col_index][prev_layer_index][kernel_index] =
						FROM_DBL(darknet_l2_conv2d_weights[kernel_row_index][kernel_col_index][prev_layer_index][kernel_index]);
				}
			}
		}
	}
	#if DARKNET_L2_CONV2D_USE_BIAS
		#include "darknet_l2_conv2d_bias"
		for (int kernel_index = 0; kernel_index < DARKNET_L2_CONV2D_KERNEL_COUNT; kernel_index++)
		{
			nn_weights_darknet.darknet_l2_conv2d_kernel_bias[kernel_index] =
				FROM_DBL(darknet_l2_conv2d_bias[kernel_index]);
		}
	#endif
}
  

// MaxPooling2D init function
void init_darknet_l3_maxpooling2d(void)
{
}
	

// Flatten init function
void init_darknet_l4_flatten(void)
{
}
	

// Dense init function
#include "darknet_l5_dense_weights"
#if DARKNET_L5_DENSE_USE_BIAS
		#include "darknet_l5_dense_bias"
#endif
void init_darknet_l5_dense(void)
{
	for (int prev_neuron_index = 0; prev_neuron_index < DARKNET_L4_FLATTEN_DIM_0; prev_neuron_index++)
	{
		for (int curr_neuron_index = 0; curr_neuron_index < DARKNET_L5_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_darknet.darknet_l5_dense_weights[prev_neuron_index][curr_neuron_index] =
				FROM_DBL(darknet_l5_dense_weights[prev_neuron_index][curr_neuron_index]);
		}
	}
	#if DARKNET_L5_DENSE_USE_BIAS
		#include "darknet_l5_dense_bias"
		for (int curr_neuron_index = 0; curr_neuron_index < DARKNET_L5_DENSE_NEURON_COUNT; curr_neuron_index++)
		{
			nn_weights_darknet.darknet_l5_dense_bias[curr_neuron_index] =
				FROM_DBL(darknet_l5_dense_bias[curr_neuron_index]);
		}
	#endif
}
	

// create NN with pre-trained weights
void nn_init_darknet(void)
{
	init_darknet_l0_conv2d();
	init_darknet_l1_maxpooling2d();
	init_darknet_l2_conv2d();
	init_darknet_l3_maxpooling2d();
	init_darknet_l4_flatten();
	init_darknet_l5_dense();
}


// Layer run definitions
// Conv2D run function
// Convolves the input (with 2 spatial and one channel dimension(s)) with a
// series of 3D kernels, and applies an activation function to the result
void run_darknet_l0_conv2d(NN_DATA_DARKNET *nn_data)
{
	NN_NUM_TYPE in_value;
	NN_NUM_TYPE kernel_value;
	NN_NUM_TYPE running_total;
	NN_NUM_TYPE bias;
	int kernel_radius_y, kernel_radius_x;
	int input_row_offset, input_col_offset;
	int ref_row, ref_col;
	int read_row, read_col;
	

	// Using padding calc from http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
	int padding_top, padding_left;
	if (DARKNET_L0_CONV2D_PADDING_VALID)
	{
		padding_top = 0;
		padding_left = 0;
	}
	else
	{
		int padding_vert = (DARKNET_L0_CONV2D_DIM_0 - 1) * DARKNET_L0_CONV2D_STRIDE_Y + DARKNET_L0_CONV2D_KERNEL_HEIGHT - DARKNET_INPUT_DIM_0;
		padding_vert = padding_vert > 0 ? padding_vert : 0;
		padding_top = padding_vert / 2;

		int padding_horiz = (DARKNET_L0_CONV2D_DIM_1 - 1) * DARKNET_L0_CONV2D_STRIDE_X + DARKNET_L0_CONV2D_KERNEL_WIDTH - DARKNET_INPUT_DIM_1;
		padding_horiz = padding_horiz > 0 ? padding_horiz : 0;
		padding_left = padding_horiz / 2;
	}

	kernel_radius_y = (DARKNET_L0_CONV2D_KERNEL_HEIGHT - 1) / 2;
	kernel_radius_x = (DARKNET_L0_CONV2D_KERNEL_WIDTH - 1) / 2;
	input_row_offset = kernel_radius_y - padding_top;
	input_col_offset = kernel_radius_x - padding_left;

	// Uses kernel center for position
	for (int kernel_index = 0; kernel_index < DARKNET_L0_CONV2D_KERNEL_COUNT; kernel_index++)
	{
		for (int output_row = 0; output_row < DARKNET_L0_CONV2D_DIM_0; output_row++)
		{
			ref_row = input_row_offset + output_row * DARKNET_L0_CONV2D_STRIDE_Y;

			for (int output_col = 0; output_col < DARKNET_L0_CONV2D_DIM_1; output_col++)
			{
				ref_col = input_col_offset + output_col * DARKNET_L0_CONV2D_STRIDE_X;

				// new kernel position
				running_total = 0;

				for (int in_layer_index = 0; in_layer_index < DARKNET_INPUT_DIM_2; in_layer_index++)
				{
					for (int kernel_cell_y = 0; kernel_cell_y < DARKNET_L0_CONV2D_KERNEL_HEIGHT; kernel_cell_y++)
					{
						for (int kernel_cell_x = 0; kernel_cell_x < DARKNET_L0_CONV2D_KERNEL_WIDTH; kernel_cell_x++)
						{
							read_row = ref_row + kernel_cell_y - kernel_radius_y;
							read_col = ref_col + kernel_cell_x - kernel_radius_x;

							// Skip overhanging kernel entries (same as zero-padding)
							if ((read_row >= 0 && read_row < DARKNET_INPUT_DIM_0) &&
								(read_col >= 0 && read_col < DARKNET_INPUT_DIM_1))
							{
								in_value = nn_data->inputs[read_row][read_col][in_layer_index];
								kernel_value = nn_weights_darknet.darknet_l0_conv2d_kernel_weights[kernel_cell_y][kernel_cell_x][in_layer_index][kernel_index];
								running_total = ADD(running_total, MUL(kernel_value, in_value));
							}
						}
					}
				}
				#if DARKNET_L0_CONV2D_USE_BIAS
					bias = nn_weights_darknet.darknet_l0_conv2d_kernel_bias[kernel_index];
					running_total = ADD(running_total, bias);
				#endif
				nn_data->darknet_l0_conv2d_outputs[output_row][output_col][kernel_index] = running_total;
			}
		}
	}

	// Apply activation
	NN_NUM_TYPE elem;
	#if DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_SOFTMAX
		NN_NUM_TYPE softmax_sum = 0;
		
		for (int dim_0_index = 0; dim_0_index < DARKNET_L0_CONV2D_DIM_0; dim_0_index++)
		{
		    for (int dim_1_index = 0; dim_1_index < DARKNET_L0_CONV2D_DIM_1; dim_1_index++)
		    {
		        for (int dim_2_index = 0; dim_2_index < DARKNET_L0_CONV2D_DIM_2; dim_2_index++)
		        {
		            elem = nn_data->darknet_l0_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index];
		            elem = EXP(elem);
		            softmax_sum = ADD(softmax_sum, elem);
		            nn_data->darknet_l0_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index] = elem;
		        }
		    }
		}
	#endif
	
	for (int dim_0_index = 0; dim_0_index < DARKNET_L0_CONV2D_DIM_0; dim_0_index++)
	{
	    for (int dim_1_index = 0; dim_1_index < DARKNET_L0_CONV2D_DIM_1; dim_1_index++)
	    {
	        for (int dim_2_index = 0; dim_2_index < DARKNET_L0_CONV2D_DIM_2; dim_2_index++)
	        {
	            elem = nn_data->darknet_l0_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index];

	            #if DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_SIGMOID
	            	elem = sigmoid(elem);
	            #elif DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_TANH
	            	elem = tanh(elem);
	            #elif DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_RELU
	            	elem = relu(elem);
	            #elif DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_LINEAR
	            	elem = linear(elem);
	            #elif DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_SOFTMAX
	            	elem = DIV(elem, softmax_sum);
              #elif DARKNET_L0_CONV2D_ACTIVATION == ACT_ENUM_HARD_SIGMOID
              	elem = hard_sigmoid(elem);
              #else
	            	printf("Invalid activation function request - %d", activation);
	            	exit(1);
	            #endif
	            nn_data->darknet_l0_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index] = elem;
	        }
	    }
	}
}



// MaxPooling2D run function
// Applies function to receptive regions on each layer
void run_darknet_l1_maxpooling2d(NN_DATA_DARKNET *nn_data)
{
	NN_NUM_TYPE focus_val;
	int ref_row, ref_col, read_row, read_col;
	
	NN_NUM_TYPE local_max;
	int sample_count;

	// Using padding calc from http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
	int padding_top, padding_left;
	if (DARKNET_L1_MAXPOOLING2D_PADDING_VALID)
	{
		padding_top = 0;
		padding_left = 0;
	}
	else
	{
		int padding_vert = (DARKNET_L1_MAXPOOLING2D_DIM_0 - 1) * DARKNET_L1_MAXPOOLING2D_STRIDE_Y + DARKNET_L1_MAXPOOLING2D_HEIGHT - DARKNET_L0_CONV2D_DIM_0;
		padding_vert = padding_vert > 0 ? padding_vert : 0;
		padding_top = padding_vert / 2;

		int padding_horiz = (DARKNET_L1_MAXPOOLING2D_DIM_1 - 1) * DARKNET_L1_MAXPOOLING2D_STRIDE_X + DARKNET_L1_MAXPOOLING2D_WIDTH - DARKNET_L0_CONV2D_DIM_1;
		padding_horiz = padding_horiz > 0 ? padding_horiz : 0;
		padding_left = padding_horiz / 2;
	}

	for (int layer_index = 0; layer_index < DARKNET_L1_MAXPOOLING2D_DIM_2; layer_index++)
	{
		for (int out_row_index = 0; out_row_index < DARKNET_L1_MAXPOOLING2D_DIM_0; out_row_index++)
		{
			for (int out_col_index = 0; out_col_index < DARKNET_L1_MAXPOOLING2D_DIM_1; out_col_index++)
			{
				// First entry in region
				sample_count = 0;
				ref_row = out_row_index * DARKNET_L1_MAXPOOLING2D_STRIDE_Y - padding_top;
				ref_col = out_col_index * DARKNET_L1_MAXPOOLING2D_STRIDE_X - padding_left;

				for (int region_row_index = 0; region_row_index < DARKNET_L1_MAXPOOLING2D_HEIGHT; region_row_index++)
				{
					for (int region_col_index = 0; region_col_index < DARKNET_L1_MAXPOOLING2D_WIDTH; region_col_index++)
					{
						read_row = ref_row + region_row_index;
						read_col = ref_col + region_col_index;

						// Skip overhanging kernel entries (same as zero-padding)
						if ((read_row >= 0 && read_row < DARKNET_L0_CONV2D_DIM_0) &&
							(read_col >= 0 && read_col < DARKNET_L0_CONV2D_DIM_1))
						{
							focus_val = nn_data->darknet_l0_conv2d_outputs[read_row][read_col][layer_index];
							if (focus_val > local_max || sample_count == 0)
							{
								local_max = focus_val;
							}
							sample_count++;
						}
					}
				}

				focus_val = local_max;
				nn_data->darknet_l1_maxpooling2d_outputs[out_row_index][out_col_index][layer_index] = focus_val;
			}
		}
	}
}


// Conv2D run function
// Convolves the input (with 2 spatial and one channel dimension(s)) with a
// series of 3D kernels, and applies an activation function to the result
void run_darknet_l2_conv2d(NN_DATA_DARKNET *nn_data)
{
	NN_NUM_TYPE in_value;
	NN_NUM_TYPE kernel_value;
	NN_NUM_TYPE running_total;
	NN_NUM_TYPE bias;
	int kernel_radius_y, kernel_radius_x;
	int input_row_offset, input_col_offset;
	int ref_row, ref_col;
	int read_row, read_col;
	

	// Using padding calc from http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
	int padding_top, padding_left;
	if (DARKNET_L2_CONV2D_PADDING_VALID)
	{
		padding_top = 0;
		padding_left = 0;
	}
	else
	{
		int padding_vert = (DARKNET_L2_CONV2D_DIM_0 - 1) * DARKNET_L2_CONV2D_STRIDE_Y + DARKNET_L2_CONV2D_KERNEL_HEIGHT - DARKNET_L1_MAXPOOLING2D_DIM_0;
		padding_vert = padding_vert > 0 ? padding_vert : 0;
		padding_top = padding_vert / 2;

		int padding_horiz = (DARKNET_L2_CONV2D_DIM_1 - 1) * DARKNET_L2_CONV2D_STRIDE_X + DARKNET_L2_CONV2D_KERNEL_WIDTH - DARKNET_L1_MAXPOOLING2D_DIM_1;
		padding_horiz = padding_horiz > 0 ? padding_horiz : 0;
		padding_left = padding_horiz / 2;
	}

	kernel_radius_y = (DARKNET_L2_CONV2D_KERNEL_HEIGHT - 1) / 2;
	kernel_radius_x = (DARKNET_L2_CONV2D_KERNEL_WIDTH - 1) / 2;
	input_row_offset = kernel_radius_y - padding_top;
	input_col_offset = kernel_radius_x - padding_left;

	// Uses kernel center for position
	for (int kernel_index = 0; kernel_index < DARKNET_L2_CONV2D_KERNEL_COUNT; kernel_index++)
	{
		for (int output_row = 0; output_row < DARKNET_L2_CONV2D_DIM_0; output_row++)
		{
			ref_row = input_row_offset + output_row * DARKNET_L2_CONV2D_STRIDE_Y;

			for (int output_col = 0; output_col < DARKNET_L2_CONV2D_DIM_1; output_col++)
			{
				ref_col = input_col_offset + output_col * DARKNET_L2_CONV2D_STRIDE_X;

				// new kernel position
				running_total = 0;

				for (int in_layer_index = 0; in_layer_index < DARKNET_L1_MAXPOOLING2D_DIM_2; in_layer_index++)
				{
					for (int kernel_cell_y = 0; kernel_cell_y < DARKNET_L2_CONV2D_KERNEL_HEIGHT; kernel_cell_y++)
					{
						for (int kernel_cell_x = 0; kernel_cell_x < DARKNET_L2_CONV2D_KERNEL_WIDTH; kernel_cell_x++)
						{
							read_row = ref_row + kernel_cell_y - kernel_radius_y;
							read_col = ref_col + kernel_cell_x - kernel_radius_x;

							// Skip overhanging kernel entries (same as zero-padding)
							if ((read_row >= 0 && read_row < DARKNET_L1_MAXPOOLING2D_DIM_0) &&
								(read_col >= 0 && read_col < DARKNET_L1_MAXPOOLING2D_DIM_1))
							{
								in_value = nn_data->darknet_l1_maxpooling2d_outputs[read_row][read_col][in_layer_index];
								kernel_value = nn_weights_darknet.darknet_l2_conv2d_kernel_weights[kernel_cell_y][kernel_cell_x][in_layer_index][kernel_index];
								running_total = ADD(running_total, MUL(kernel_value, in_value));
							}
						}
					}
				}
				#if DARKNET_L2_CONV2D_USE_BIAS
					bias = nn_weights_darknet.darknet_l2_conv2d_kernel_bias[kernel_index];
					running_total = ADD(running_total, bias);
				#endif
				nn_data->darknet_l2_conv2d_outputs[output_row][output_col][kernel_index] = running_total;
			}
		}
	}

	// Apply activation
	NN_NUM_TYPE elem;
	#if DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_SOFTMAX
		NN_NUM_TYPE softmax_sum = 0;
		
		for (int dim_0_index = 0; dim_0_index < DARKNET_L2_CONV2D_DIM_0; dim_0_index++)
		{
		    for (int dim_1_index = 0; dim_1_index < DARKNET_L2_CONV2D_DIM_1; dim_1_index++)
		    {
		        for (int dim_2_index = 0; dim_2_index < DARKNET_L2_CONV2D_DIM_2; dim_2_index++)
		        {
		            elem = nn_data->darknet_l2_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index];
		            elem = EXP(elem);
		            softmax_sum = ADD(softmax_sum, elem);
		            nn_data->darknet_l2_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index] = elem;
		        }
		    }
		}
	#endif
	
	for (int dim_0_index = 0; dim_0_index < DARKNET_L2_CONV2D_DIM_0; dim_0_index++)
	{
	    for (int dim_1_index = 0; dim_1_index < DARKNET_L2_CONV2D_DIM_1; dim_1_index++)
	    {
	        for (int dim_2_index = 0; dim_2_index < DARKNET_L2_CONV2D_DIM_2; dim_2_index++)
	        {
	            elem = nn_data->darknet_l2_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index];

	            #if DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_SIGMOID
	            	elem = sigmoid(elem);
	            #elif DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_TANH
	            	elem = tanh(elem);
	            #elif DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_RELU
	            	elem = relu(elem);
	            #elif DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_LINEAR
	            	elem = linear(elem);
	            #elif DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_SOFTMAX
	            	elem = DIV(elem, softmax_sum);
              #elif DARKNET_L2_CONV2D_ACTIVATION == ACT_ENUM_HARD_SIGMOID
              	elem = hard_sigmoid(elem);
              #else
	            	printf("Invalid activation function request - %d", activation);
	            	exit(1);
	            #endif
	            nn_data->darknet_l2_conv2d_outputs[dim_0_index][dim_1_index][dim_2_index] = elem;
	        }
	    }
	}
}



// MaxPooling2D run function
// Applies function to receptive regions on each layer
void run_darknet_l3_maxpooling2d(NN_DATA_DARKNET *nn_data)
{
	NN_NUM_TYPE focus_val;
	int ref_row, ref_col, read_row, read_col;
	
	NN_NUM_TYPE local_max;
	int sample_count;

	// Using padding calc from http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
	int padding_top, padding_left;
	if (DARKNET_L3_MAXPOOLING2D_PADDING_VALID)
	{
		padding_top = 0;
		padding_left = 0;
	}
	else
	{
		int padding_vert = (DARKNET_L3_MAXPOOLING2D_DIM_0 - 1) * DARKNET_L3_MAXPOOLING2D_STRIDE_Y + DARKNET_L3_MAXPOOLING2D_HEIGHT - DARKNET_L2_CONV2D_DIM_0;
		padding_vert = padding_vert > 0 ? padding_vert : 0;
		padding_top = padding_vert / 2;

		int padding_horiz = (DARKNET_L3_MAXPOOLING2D_DIM_1 - 1) * DARKNET_L3_MAXPOOLING2D_STRIDE_X + DARKNET_L3_MAXPOOLING2D_WIDTH - DARKNET_L2_CONV2D_DIM_1;
		padding_horiz = padding_horiz > 0 ? padding_horiz : 0;
		padding_left = padding_horiz / 2;
	}

	for (int layer_index = 0; layer_index < DARKNET_L3_MAXPOOLING2D_DIM_2; layer_index++)
	{
		for (int out_row_index = 0; out_row_index < DARKNET_L3_MAXPOOLING2D_DIM_0; out_row_index++)
		{
			for (int out_col_index = 0; out_col_index < DARKNET_L3_MAXPOOLING2D_DIM_1; out_col_index++)
			{
				// First entry in region
				sample_count = 0;
				ref_row = out_row_index * DARKNET_L3_MAXPOOLING2D_STRIDE_Y - padding_top;
				ref_col = out_col_index * DARKNET_L3_MAXPOOLING2D_STRIDE_X - padding_left;

				for (int region_row_index = 0; region_row_index < DARKNET_L3_MAXPOOLING2D_HEIGHT; region_row_index++)
				{
					for (int region_col_index = 0; region_col_index < DARKNET_L3_MAXPOOLING2D_WIDTH; region_col_index++)
					{
						read_row = ref_row + region_row_index;
						read_col = ref_col + region_col_index;

						// Skip overhanging kernel entries (same as zero-padding)
						if ((read_row >= 0 && read_row < DARKNET_L2_CONV2D_DIM_0) &&
							(read_col >= 0 && read_col < DARKNET_L2_CONV2D_DIM_1))
						{
							focus_val = nn_data->darknet_l2_conv2d_outputs[read_row][read_col][layer_index];
							if (focus_val > local_max || sample_count == 0)
							{
								local_max = focus_val;
							}
							sample_count++;
						}
					}
				}

				focus_val = local_max;
				nn_data->darknet_l3_maxpooling2d_outputs[out_row_index][out_col_index][layer_index] = focus_val;
			}
		}
	}
}


// Flatten run function
// Flattens 3D array iterating over layers, then columns, then rows
void run_darknet_l4_flatten(NN_DATA_DARKNET *nn_data)
{
	int current_out_index = 0;
	

	for (int row_index = 0; row_index < DARKNET_L3_MAXPOOLING2D_DIM_0; row_index++)
	{
		for (int col_index = 0; col_index < DARKNET_L3_MAXPOOLING2D_DIM_1; col_index++)
		{
			for (int layer_index = 0; layer_index < DARKNET_L3_MAXPOOLING2D_DIM_2; layer_index++)
			{
				nn_data->darknet_l4_flatten_outputs[current_out_index++] =
					nn_data->darknet_l3_maxpooling2d_outputs[row_index][col_index][layer_index];
			}
		}
	}
}


// Dense run function
// Applies activation function to a weighted sum of inputs for each neuron in this layer
void run_darknet_l5_dense(NN_DATA_DARKNET *nn_data)
{
	NN_NUM_TYPE weighted_sum;
	NN_NUM_TYPE prev_neuron_value;
	NN_NUM_TYPE weight;
	NN_NUM_TYPE bias;
	int pl_out_index = nn_data->pl_index;

	// Calculate weighted sums
	for (int curr_neuron_index = 0; curr_neuron_index < DARKNET_L5_DENSE_NEURON_COUNT; curr_neuron_index++)
	{
		weighted_sum = 0;
		for (int prev_neuron_index = 0; prev_neuron_index < DARKNET_L4_FLATTEN_DIM_0; prev_neuron_index++)
		{
			prev_neuron_value = nn_data->darknet_l4_flatten_outputs[prev_neuron_index];
			weight = nn_weights_darknet.darknet_l5_dense_weights[prev_neuron_index][curr_neuron_index];
			weighted_sum = ADD(weighted_sum, MUL(weight, prev_neuron_value));
		}
		#if DARKNET_L5_DENSE_USE_BIAS
			bias = nn_weights_darknet.darknet_l5_dense_bias[curr_neuron_index];
			weighted_sum = ADD(weighted_sum, bias);
		#endif
		nn_data->outputs[pl_out_index][curr_neuron_index] = weighted_sum;
	}

	// Apply activation
	NN_NUM_TYPE elem;
	#if DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
		NN_NUM_TYPE softmax_sum = 0;
		
		for (int dim_0_index = 0; dim_0_index < DARKNET_L5_DENSE_DIM_0; dim_0_index++)
		{
		    elem = nn_data->outputs[pl_out_index][dim_0_index];
		    elem = EXP(elem);
		    softmax_sum = ADD(softmax_sum, elem);
		    nn_data->outputs[pl_out_index][dim_0_index] = elem;
		}
	#endif
	
	for (int dim_0_index = 0; dim_0_index < DARKNET_L5_DENSE_DIM_0; dim_0_index++)
	{
	    elem = nn_data->outputs[pl_out_index][dim_0_index];

	    #if DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_SIGMOID
	    	elem = sigmoid(elem);
	    #elif DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_TANH
	    	elem = tanh(elem);
	    #elif DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_RELU
	    	elem = relu(elem);
	    #elif DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_LINEAR
	    	elem = linear(elem);
	    #elif DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_SOFTMAX
	    	elem = DIV(elem, softmax_sum);
      #elif DARKNET_L5_DENSE_ACTIVATION == ACT_ENUM_HARD_SIGMOID
      	elem = hard_sigmoid(elem);
      #else
	    	printf("Invalid activation function request - %d", activation);
	    	exit(1);
	    #endif
	    nn_data->outputs[pl_out_index][dim_0_index] = elem;
	}
}


// Must be called after every nn_run for pipelining
void pl_update_darknet(NN_DATA_DARKNET *nn_data)
{
	if (++nn_data->pl_index >= DARKNET_MAX_PL_LEN)
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


void nn_run_darknet(NN_DATA_DARKNET *nn_data)
{
	run_darknet_l0_conv2d(nn_data);
	run_darknet_l1_maxpooling2d(nn_data);
	run_darknet_l2_conv2d(nn_data);
	run_darknet_l3_maxpooling2d(nn_data);
	run_darknet_l4_flatten(nn_data);
	run_darknet_l5_dense(nn_data);
	pl_update_darknet(nn_data);
}