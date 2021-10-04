#include <torch/torch.h>

#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/function.h>

#include "Common.hpp"
#include "Config.hpp"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/cpu/ExtendOPs.h"

namespace torch_ipex {
namespace autocast {
namespace int8 {

#define OP_TYPE_DEFAULT 0
#define OP_TYPE_INPLACE 1
#define OP_TYPE_POOLING 2

struct params {
  std::vector<std::vector<quant_utils::TensorQuantizationParams>> qparams;
  std::vector<at::ScalarType> input_quantized_dtypes;
  std::vector<at::ScalarType> output_quantized_dtypes;
  std::vector<bool> inputs_quantized;
  std::vector<bool> outputs_quantized;
};

/**Get the parameters needed for quantization
 *
 * @param op_id:   the return value of
 * torch_ipex::Int8OptConfig::fetch_and_add_ops_id()
 */
params get_params(int64_t op_id);

/**Calibrate inputs, weights and outputs
 *
 * @param inputs:  the input tensors of an operator
 * @param weights: the weight tensors of an operator.
 *                 If the weight tensors do not need to be calibrated, then pass
 *                 empty list
 * @param outputs: the output tensors of an operator
 * @param op_name: the name of an operator
 * @param op_id:   the return value of
 *                 torch_ipex::Int8OptConfig::fetch_and_add_ops_id()
 * @param op_type:
 *           OP_TYPE_DEFAULT: calibrate inputs, weights and outputs respectively
 *           OP_TYPE_INPLACE: inplace operator
 *           OP_TYPE_POOLING: using the same scale for inputs and outputs
 */
void calibrate(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& outputs,
    const std::string& op_name,
    int64_t op_id,
    int op_type);

/**Quantize and dequantize inputs and weights
 * It will quantize and dequantize input and weight tensors according to
 * quantization parameters, and return new input and weight tensors after
 * quantization and dequantization.
 * For example: given inputs and weights, this function will do:
 *              inputs  -> quantization -> dequantization -> new_inputs
 *              weights -> quantization -> dequantization -> new_weights
 *
 * @param inputs:                   the input tensors to be quantized and
 *                                  dequantized
 * @param weights:                  the weight tensors to be quantized and
 *                                  dequantized
 * @param inputs_qparams:           quantization parameters for inputs
 * @param inputs_quantized_dtypes:  the quantization data types of input tensors
 * @param inputs_quantized:         whether the input tensors need to be
 *                                  quantized and dequantized
 * @param op_id:                    the return value of
 *                                  torch_ipex::Int8OptConfig::fetch_and_add_ops_id()
 */
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> insert_q_dq_inputs(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& weights,
    const std::vector<quant_utils::TensorQuantizationParams>& inputs_qparams,
    const std::vector<at::ScalarType>& inputs_quantized_dtypes,
    const std::vector<bool>& inputs_quantized,
    int64_t op_id);

/**Quantize and dequantize outputs
 * It will quantize and dequantize output tensors according to quantization
 * parameters, and return new output tensors after quantization and
 * dequantization.
 * For example: give outputs, this function will do:
 *              outputs -> quantization -> dequantization -> new_outputs
 *
 * @param outputs:                  the output tensors to be quantized and
 *                                  dequantized
 * @param outputs_qparams:          quantization parameters for outputs
 * @param outputs_quantized_dtypes: the quantization data types of output
 *                                  tensors
 * @param outputs_quantized:        whether the output tensors need to be
 *                                  quantized and dequantized
 */
std::vector<at::Tensor> insert_q_dq_outputs(
    const std::vector<at::Tensor>& outputs,
    const std::vector<quant_utils::TensorQuantizationParams>& outputs_qparams,
    const std::vector<at::ScalarType>& outputs_quantized_dtypes,
    const std::vector<bool>& outputs_quantized);

} // namespace int8
} // namespace autocast
} // namespace torch_ipex
