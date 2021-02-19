//
// Created by liangan1 on 2020/11/12.
//

#include <torch/csrc/autograd/record_function_ops.h>
//#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <utils.h>
#include <auto_opt_config.h>

namespace torch_ccl
{

namespace {

// Get the device type from list of tensors
at::Device get_dev_type(const std::vector<at::Tensor>& tensors) {
  return tensors[0].device();
}

// Get the list of devices from list of tensors
at::Device get_dev_type(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  return tensor_lists[0][0].device();
}

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  std::vector<at::Device> res;
  res.reserve(tensor_lists.size());
  for (auto& tensors : tensor_lists) {
    res.push_back(tensors[0].device());
  }
  return res;
}

} //namespace anonymous

class XPUCCLStubs final: public DispatchStub {

public:
  XPUCCLStubs() {}

  bool enabled() override {
    return true;
  }

  ~XPUCCLStubs() {}

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;


  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const GatherOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> scatter_(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<std::vector<at::Tensor>>& inputTensors,
                                                             const ScatterOptions& opts,
                                                             ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base_(at::Tensor& outputTensor,
                                                               at::Tensor& inputTensor,
                                                               std::vector<int64_t>& outputSplitSizes,
                                                               std::vector<int64_t>& inputSplitSizes,
                                                               const AllToAllOptions& opts,
                                                               ProcessGroupCCL& pg_ccl) override;
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<at::Tensor>& inputTensors,
                                                             const AllToAllOptions& opts,
                                                             ProcessGroupCCL& pg_ccl) override;
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier_(const BarrierOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) override;

};

struct RegisterDPCPPPMethods {
  RegisterDPCPPPMethods() {
    static XPUCCLStubs methods;
    DispatchStub::register_ccl_stub(c10::DeviceType::XPU, &methods);
  }
};


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allreduce_(std::vector<at::Tensor>& tensors,
                                                                         const AllreduceOptions& opts,
                                                                         ProcessGroupCCL& pg_ccl) {
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->allreduce_(tensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                      const ReduceOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl) {
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->reduce_(tensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                         const BroadcastOptions &opts,
                                                                         ProcessGroupCCL& pg_ccl) {
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->broadcast_(tensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

  
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                         std::vector<at::Tensor>& inputTensors,
                                                                         const AllgatherOptions& opts,
                                                                         ProcessGroupCCL& pg_ccl) {
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU; 
           return stubs_[to_int(dev_type)]->allgather_(outputTensors, inputTensors, opts, pg_ccl); 
           
      }default :
           std::runtime_error("unsorpported xpu mode");
  }
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                    std::vector<at::Tensor>& inputTensors,
                                                                    const GatherOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl){
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->gather_(outputTensors, inputTensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::scatter_(std::vector<at::Tensor>& outputTensors,
                                                                     std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                     const ScatterOptions& opts,
                                                                     ProcessGroupCCL& pg_ccl){
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->scatter_(outputTensors, inputTensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::alltoall_base_(at::Tensor& outputTensor,
                                                                           at::Tensor& inputTensor,
                                                                           std::vector<int64_t>& outputSplitSizes,
                                                                           std::vector<int64_t>& inputSplitSizes,
                                                                           const AllToAllOptions& opts,
                                                                           ProcessGroupCCL& pg_ccl){
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->alltoall_base_(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::alltoall_(std::vector<at::Tensor>& outputTensors,
                                                                      std::vector<at::Tensor>& inputTensors,
                                                                      const AllToAllOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl){
  auto xpu_mode = torch_ipex::AutoOptConfig::singleton().get_xpu_mode();
  switch(xpu_mode){
      case torch_ipex::XPUMode::CPU :{
           auto dev_type = c10::DeviceType::CPU;
           return stubs_[to_int(dev_type)]->alltoall_(outputTensors, inputTensors, opts, pg_ccl);

      }default :
           std::runtime_error("unsorpported xpu mode");
  }

}
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::barrier_(const BarrierOptions& opts,
                                                                     ProcessGroupCCL& pg_ccl){
  auto dev_type = c10::DeviceType::CPU;
  return stubs_[to_int(dev_type)]->barrier_(opts, pg_ccl);
}
RegisterDPCPPPMethods dpcpp_register;

}


