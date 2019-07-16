/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include <stdio.h>

#include "tiny_graph.h"
#include "tiny_param_generated.h"

/* first conv node */
static const struct tiny_tensor conv_0_input = {
    .dims = {1, 99, 10, 1},
    .dim_num = 4,
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_INPUT,
    .data = NULL,
};

/* for conv weight, the layout is hwio */
static const struct tiny_tensor conv_0_weight = {
    .dim_num = 4,
    .dims = {10, 4, 1, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_0_weight_data,
};

static const struct tiny_tensor conv_0_bias = {
    .dim_num = 1,
    .dims = {32},
    .shift = FIRST_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_0_bias_data,
};

static const struct tiny_tensor conv_0_output = {
    .dim_num = 4,
    .dims = {1, 45, 7, 32},
    .shift = FIRST_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_0_param = {
    .kernel_h = 10,
    .kernel_w = 4,
    .stride_h = 2,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_0_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_0_param,
    .input = {&conv_0_input, &conv_0_weight, &conv_0_bias},
    .output = &conv_0_output,
};

/* the relu node */
static const struct tiny_tensor relu_1_output = {
    .dim_num = 4,
    .dims = {1, 45, 7, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_1_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_0_output},
    .output = &relu_1_output,
};

/* the pool node */

static const struct tiny_tensor pool_2_output = {
    .dim_num = 4,
    .dims = {1, 23, 4, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_pool_param pool_2_param = {
    .pool_method = NN_POOL_MAX,
    .kernel_h = 2,
    .kernel_w = 2,
    .stride_h = 2,
    .stride_w = 2,
    .pad_h = NN_PAD_SAME,
    .pad_w = NN_PAD_SAME,
};

static const struct tiny_node pool_2_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_POOL,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &pool_2_param,
    .input = {&relu_1_output},
    .output = &pool_2_output,
};

/* the conv node */
static const struct tiny_tensor conv_3_weight = {
    .dim_num = 4,
    .dims = {8, 4, 32, 48},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_3_weight_data,
};

static const struct tiny_tensor conv_3_bias = {
    .dim_num = 1,
    .dims = {48},
    .shift = SECOND_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_3_bias_data,
};

static const struct tiny_tensor conv_3_output = {
    .dim_num = 4,
    .dims = {1, 16, 1, 48},
    .shift = SECOND_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_3_param = {
    .kernel_h = 8,
    .kernel_w = 4,
    .stride_h = 1,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_3_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_3_param,
    .input = {&pool_2_output, &conv_3_weight, &conv_3_bias},
    .output = &conv_3_output,
};

/* the relu node */
static const struct tiny_tensor relu_4_output = {
    .dim_num = 4,
    .dims = {1, 16, 1, 48},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_4_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_3_output},
    .output = &relu_4_output,
};

/* for conv weight, the layout is hwio */
static const struct tiny_tensor conv_5_weight = {
    .dim_num = 4,
    .dims = {4, 1, 48, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_5_weight_data,
};

static const struct tiny_tensor conv_5_bias = {
    .dim_num = 1,
    .dims = {32},
    .shift = THIRD_CONV_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = conv_5_bias_data,
};

static const struct tiny_tensor conv_5_output = {
    .dim_num = 4,
    .dims = {1, 7, 1, 32},
    .shift = THIRD_CONV_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_conv_param conv_5_param = {
    .kernel_h = 4,
    .kernel_w = 1,
    .stride_h = 2,
    .stride_w = 1,
    .pad_h = NN_PAD_VALID,
    .pad_w = NN_PAD_VALID,
    .activation = -1,
};

static const struct tiny_node conv_5_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_CONV,
    .op_ver = NN_OP_VERSION_1,
    .op_param = &conv_5_param,
    .input = {&relu_4_output, &conv_5_weight, &conv_5_bias},
    .output = &conv_5_output,
};

/* the relu node */
static const struct tiny_tensor relu_6_output = {
    .dim_num = 4,
    .dims = {1, 7, 1, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_6_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&conv_5_output},
    .output = &relu_6_output,
};

/* the fc node */
static const struct tiny_tensor fc_7_weight = {
    .dim_num = 2,
    .dims = {32, 7 * 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_7_weight_data,
};

static const struct tiny_tensor fc_7_bias = {
    .dim_num = 1,
    .dims = {32},
    .shift = LINEAR_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_7_bias_data,
};

static const struct tiny_tensor fc_7_output = {
    .dim_num = 2,
    .dims = {1, 32},
    .shift = LINEAR_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_7_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&relu_6_output, &fc_7_weight, &fc_7_bias},
    .output = &fc_7_output,
};

/* fc node */
static const struct tiny_tensor fc_8_weight = {
    .dim_num = 2,
    .dims = {128, 32},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_8_weight_data,
};

static const struct tiny_tensor fc_8_bias = {
    .dim_num = 1,
    .dims = {128},
    .shift = FIRST_FC_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_8_bias_data,
};

static const struct tiny_tensor fc_8_output = {
    .dim_num = 2,
    .dims = {1, 128},
    .shift = FIRST_FC_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_8_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_7_output, &fc_8_weight, &fc_8_bias},
    .output = &fc_8_output,
};

/* the relu node */
static const struct tiny_tensor relu_9_output = {
    .dim_num = 2,
    .dims = {1, 128},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node relu_9_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_RELU,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_8_output},
    .output = &relu_9_output,
};

/* fc node */
static const struct tiny_tensor fc_10_weight = {
    .dim_num = 2,
    .dims = {2, 128},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_10_weight_data,
};

static const struct tiny_tensor fc_10_bias = {
    .dim_num = 1,
    .dims = {2},
    .shift = FINAL_FC_BIAS_LSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_CONST,
    .data = fc_10_bias_data,
};

static const struct tiny_tensor fc_10_output = {
    .dim_num = 2,
    .dims = {1, 2},
    .shift = FINAL_FC_OUTPUT_RSHIFT,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node fc_10_node = {
    .input_num = 3,
    .output_num = 1,
    .op_type = NN_OP_FC,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&relu_9_output, &fc_10_weight, &fc_10_bias},
    .output = &fc_10_output,
};

/* SoftMax node */

static const struct tiny_tensor softmax_11_output = {
    .dim_num = 2,
    .dims = {1, 2},
    .shift = 0,
    .data_type = NN_DT_Q7,
    .tensor_type = NN_TENSOR_VAR,
    .data = NULL,
};

static const struct tiny_node softmax_11_node = {
    .input_num = 1,
    .output_num = 1,
    .op_type = NN_OP_SOFTMAX,
    .op_ver = NN_OP_VERSION_1,
    .op_param = NULL,
    .input = {&fc_10_output},
    .output = &softmax_11_output,
};

/* the graph node list */
static const struct tiny_node* node_list[] = {
    &conv_0_node, &relu_1_node, &pool_2_node, &conv_3_node, &relu_4_node, &conv_5_node,
    &relu_6_node, &fc_7_node,   &fc_8_node,   &relu_9_node, &fc_10_node,  &softmax_11_node,
};

static const struct tiny_graph tiny_graph = {
    .name = "test",
    .tiny_version = NN_TINY_VERSION_1,
    .nn_id = 0xdeadbeaf,
    .create_time = 0,
    .layout = NN_LAYOUT_NHWC,
    .node_num = sizeof(node_list) / sizeof(void*),
    .node_list = node_list,
};

const struct tiny_graph* get_tiny_graph(void)
{
    return &tiny_graph;
}

void free_tiny_graph(const struct tiny_graph* tiny_graph)
{
    /* NOTHING NEEDS TO DO */
}
