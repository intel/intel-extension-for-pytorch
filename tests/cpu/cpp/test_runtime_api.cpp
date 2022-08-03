#include <torch/torch.h>
#include "gtest/gtest.h"
#include "intel_extension_for_pytorch/csrc/cpu/runtime/CPUPool.h"
#include "intel_extension_for_pytorch/csrc/cpu/runtime/Task.h"
#include "intel_extension_for_pytorch/csrc/cpu/runtime/TaskExecutor.h"

#define ASSERT_VARIABLE_EQ(a, b) ASSERT_TRUE(torch::allclose((a), (b)))
#define EXPECT_VARIABLE_EQ(a, b) EXPECT_TRUE(torch::allclose((a), (b)))

TEST(TestRuntimeAPI, TestMainThreadCoreBind) {
  // 1. Get the default thread affinity information in main thread.
  // 2. Set the new thread affinity information in main thread.
  // 3. Run the function in main thread.
  // 4. Restore the default thread affinity information in main thread.
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeAPI::TestMainThreadCoreBind. Didn't preload IOMP.";
  }
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result.
  auto res_ref = at::softmax(input_tensor, -1);
  // Get current cpu_pool information.
  torch_ipex::runtime::CPUPool previous_cpu_pool =
      torch_ipex::runtime::get_cpu_pool_from_mask_affinity();
  // Ping CPU Cores.
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  torch_ipex::runtime::_pin_cpu_cores(cpu_pool);
  auto res = at::softmax(input_tensor, -1);
  ASSERT_VARIABLE_EQ(res, res_ref);
  // restore the cpu pool information.
  torch_ipex::runtime::set_mask_affinity_from_cpu_pool(previous_cpu_pool);
}

TEST(TestRuntimeAPI, TestMainThreadCoreBindWithCPUPool) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeAPI::TestMainThreadCoreBindWithCPUPool. Didn't preload IOMP.";
  }
  at::Tensor input_tensor = at::rand({100, 8276});
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  {
    torch_ipex::runtime::WithCPUPool with_cpu_pool(std::move(cpu_pool));
    auto res = at::softmax(input_tensor, -1);
  }
  auto res_ = at::softmax(input_tensor, -1);
}

TEST(TestRuntimeTaskAPI, TestTaskAPINativeTorchOperation) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPINativeTorchOperation. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = at::softmax(input_tensor, -1);
  // Create the task
  torch_ipex::runtime::Task<
      at::Tensor (*)(const at::Tensor&, int64_t, c10::optional<at::ScalarType>),
      const at::Tensor&,
      int64_t,
      c10::optional<at::ScalarType>&>
      task(at::softmax, task_executor);
  c10::optional<at::ScalarType> dtype = c10::nullopt;
  // or
  // c10::optional<at::ScalarType> dtype = input_tensor.scalar_type();
  auto res_future = task(input_tensor, -1, dtype);
  auto res = res_future.get();

  // Test Rvalue Input Tensor
  auto res_future_rinput = task(std::move(input_tensor), -1, dtype);
  auto res_rinput = res_future_rinput.get();

  // Test Const Input Tensor
  const at::Tensor input_tensor2 = at::rand({100, 8276});
  auto res_ref_const_input = at::softmax(input_tensor2, -1);
  auto res_future_const_input = task(input_tensor2, -1, dtype);
  auto res_const_input = res_future_const_input.get();

  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
  ASSERT_VARIABLE_EQ(res_rinput, res_ref);
  ASSERT_VARIABLE_EQ(res_const_input, res_ref_const_input);
}

TEST(TestRuntimeTaskAPI, TestTaskAPILambdaFunction) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPILambdaFunction. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = at::softmax(input_tensor, -1);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor (*)(const at::Tensor&), const at::Tensor&>
          task(
              [](const at::Tensor& input) -> at::Tensor {
                return at::softmax(input, -1);
              },
              task_executor);
  auto res_future = task(input_tensor);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

at::Tensor taskfunction_native_input(at::Tensor input) {
  at::Tensor output;
  output = at::softmax(input, -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionNativeInput) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionNativeInput. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = taskfunction_native_input(input_tensor);
  // Create the task
  torch_ipex::runtime::Task<at::Tensor (*)(at::Tensor), at::Tensor> task(
      taskfunction_native_input, task_executor);
  auto res_future = task(std::move(input_tensor));
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionNativeInputLValue) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionNativeInput. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = taskfunction_native_input(input_tensor);
  // Create the task
  torch_ipex::runtime::Task<at::Tensor (*)(at::Tensor), at::Tensor&> task(
      taskfunction_native_input, task_executor);
  auto res_future = task(input_tensor);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

at::Tensor taskfunction_lvalue_reference(at::Tensor& input) {
  at::Tensor output;
  output = at::softmax(input, -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionLValueReference) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionLValueReference. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = taskfunction_lvalue_reference(input_tensor);
  // Create the task
  torch_ipex::runtime::Task<at::Tensor (*)(at::Tensor&), at::Tensor&> task(
      taskfunction_lvalue_reference, task_executor);
  auto res_future = task(input_tensor);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

at::Tensor taskfunction_const_lvalue_reference(const at::Tensor& input) {
  at::Tensor output;
  output = at::softmax(input, -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionConstLValueReference) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionConstLValueReference. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = taskfunction_const_lvalue_reference(input_tensor);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor (*)(const at::Tensor&), const at::Tensor&>
          task(taskfunction_const_lvalue_reference, task_executor);
  auto res_future = task(input_tensor);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

at::Tensor taskfunction_rvalue_reference(at::Tensor&& input) {
  at::Tensor output;
  output = at::softmax(input, -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionRvalueReference) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionRvalueReference. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  at::Tensor input_tensor2 = input_tensor;
  // Get the reference result
  auto res_ref = taskfunction_rvalue_reference(std::move(input_tensor));
  // Create the task
  torch_ipex::runtime::Task<at::Tensor (*)(at::Tensor &&), at::Tensor&&> task(
      taskfunction_rvalue_reference, task_executor);
  auto res_future = task(std::move(input_tensor2));
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
taskfunction_mix_lvalue_rvalue_reference(
    at::Tensor input1,
    at::Tensor& input2,
    const at::Tensor& input3,
    at::Tensor&& input4) {
  at::Tensor output1 = at::softmax(input1, -1);
  at::Tensor output2 = at::softmax(input2, -1);
  at::Tensor output3 = at::softmax(input3, -1);
  at::Tensor output4 = at::softmax(input4, -1);
  return std::make_tuple(output1, output2, output3, output4);
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionMixLvalueRvalueReference) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionMixLvalueRvalueReference. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  at::Tensor input_tensor2 = input_tensor;
  at::Tensor input_tensor3 = input_tensor;
  at::Tensor input_tensor4 = input_tensor;
  // Get the reference result
  auto res_ref = taskfunction_mix_lvalue_rvalue_reference(
      std::move(input_tensor),
      input_tensor2,
      input_tensor2,
      std::move(input_tensor2));
  // Create the task
  torch_ipex::runtime::Task<
      std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> (*)(
          at::Tensor, at::Tensor&, const at::Tensor&, at::Tensor&&),
      at::Tensor,
      at::Tensor&,
      const at::Tensor&,
      at::Tensor&&>
      task(taskfunction_mix_lvalue_rvalue_reference, task_executor);
  auto res_future = task(
      std::move(input_tensor3),
      input_tensor4,
      input_tensor4,
      std::move(input_tensor4));
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(std::get<0>(res), std::get<0>(res_ref));
  ASSERT_VARIABLE_EQ(std::get<1>(res), std::get<1>(res_ref));
  ASSERT_VARIABLE_EQ(std::get<2>(res), std::get<2>(res_ref));
  ASSERT_VARIABLE_EQ(std::get<3>(res), std::get<3>(res_ref));
}

at::Tensor taskfunction_input_vector(std::vector<at::Tensor>& inputs) {
  at::Tensor output;
  output = at::softmax(inputs[0], -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionInputVectorTensor) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionInputVectorTensor. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  std::vector<at::Tensor> input_tenosrs;
  input_tenosrs.emplace_back(input_tensor);
  // Get the reference result
  auto res_ref = taskfunction_input_vector(input_tenosrs);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor (*)(std::vector<at::Tensor>&), std::vector<at::Tensor>&>
          task(taskfunction_input_vector, task_executor);
  auto res_future = task(input_tenosrs);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
}

at::Tensor& taskfunction_input_reference_output_lvalue_reference(
    at::Tensor& input,
    at::Tensor& output) {
  output = at::softmax(input, -1);
  return output;
}

TEST(TestRuntimeTaskAPI, TestTaskAPICPPFunctionOutputTensorLValueReference) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPICPPFunctionOutputTensorLValueReference. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);
  at::Tensor input_tensor = at::rand({100, 8276});
  at::Tensor output_tensor;
  at::Tensor output_tensor2;
  // Get the reference result
  auto res_ref = taskfunction_input_reference_output_lvalue_reference(
      input_tensor, output_tensor);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor& (*)(at::Tensor&, at::Tensor&), at::Tensor&, at::Tensor&>
          task(
              taskfunction_input_reference_output_lvalue_reference,
              task_executor);
  auto res_future = task(input_tensor, output_tensor2);
  auto res = res_future.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
  ASSERT_VARIABLE_EQ(output_tensor, res_ref);
  ASSERT_VARIABLE_EQ(output_tensor2, res_ref);
  ASSERT_VARIABLE_EQ(output_tensor2, output_tensor);
}

TEST(TestRuntimeTaskAPI, TestTaskAPIMultiTasksSameTensorInput) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPIMultiTasksSameTensorInput. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);

  std::vector<int32_t> cpu_core_list2({1});
  torch_ipex::runtime::CPUPool cpu_pool2(cpu_core_list2);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor2 =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool2);

  at::Tensor input_tensor = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = at::softmax(input_tensor, -1);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor (*)(const at::Tensor&), const at::Tensor&>
          task(taskfunction_const_lvalue_reference, task_executor);

  torch_ipex::runtime::
      Task<at::Tensor (*)(const at::Tensor&), const at::Tensor&>
          task2(taskfunction_const_lvalue_reference, task_executor2);

  auto res_future = task(input_tensor);
  auto res_future2 = task2(input_tensor);
  auto res = res_future.get();
  auto res2 = res_future2.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
  ASSERT_VARIABLE_EQ(res2, res_ref);
}

TEST(TestRuntimeTaskAPI, TestTaskAPISameTasksMultiTensorInputs) {
  if (!torch_ipex::runtime::is_runtime_ext_enabled()) {
    GTEST_SKIP()
        << "Skip TestRuntimeTaskAPI::TestTaskAPISameTasksMultiTensorInputs. Didn't preload IOMP.";
  }
  std::vector<int32_t> cpu_core_list({0});
  torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
  std::shared_ptr<torch_ipex::runtime::TaskExecutor> task_executor =
      std::make_shared<torch_ipex::runtime::TaskExecutor>(cpu_pool);

  at::Tensor input_tensor = at::rand({100, 8276});
  at::Tensor input_tensor2 = at::rand({100, 8276});
  // Get the reference result
  auto res_ref = at::softmax(input_tensor, -1);
  auto res_ref2 = at::softmax(input_tensor2, -1);
  // Create the task
  torch_ipex::runtime::
      Task<at::Tensor (*)(const at::Tensor&), const at::Tensor&>
          task(taskfunction_const_lvalue_reference, task_executor);

  auto res_future = task(input_tensor);
  auto res_future2 = task(input_tensor2);
  auto res = res_future.get();
  auto res2 = res_future2.get();
  // Assert the result
  ASSERT_VARIABLE_EQ(res, res_ref);
  ASSERT_VARIABLE_EQ(res2, res_ref2);
}
