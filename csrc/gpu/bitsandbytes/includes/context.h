/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include <cassert>
#include <iostream>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>

#include <torch/library.h>

#ifndef BF16_AVAILABLE
#define BF16_AVAILABLE
#endif

// define OP register macro

namespace at {
namespace bnb {

template <typename Func>
void construct_function_schema_and_register(
    const char* name,
    Func&& func,
    torch::Library& m) {
  std::unique_ptr<c10::FunctionSchema> infer_schema =
      c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>();
  auto parse_name = torch::jit::parseSchemaOrName(name);
  c10::OperatorName op_name =
      std::get<c10::OperatorName>(std::move(parse_name));
  c10::FunctionSchema s = infer_schema->cloneWithName(
      std::move(op_name.name), std::move(op_name.overload_name));
  s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
  c10::OperatorName find_name = s.operator_name();
  const auto found = c10::Dispatcher::realSingleton().findOp(find_name);
  if (found == c10::nullopt) {
    m.def(std::move(s));
  }
}

template <typename T, int N>
struct TypeSelector {
  template <typename... Args>
  void extract_type(Args... args) {
    return;
  }

  template <typename... Args>
  void extract_type(T& type, Args... args) {
    container_.push_back(type);
    extract_type(args...);
  }

  template <typename U, typename... Args>
  void extract_type(U type, Args... args) {
    extract_type(args...);
  }

  at::ArrayRef<T> retrive_types() {
    return at::ArrayRef<T>(container_.begin(), container_.end());
  }

  at::SmallVector<T, N> container_;
};

template <typename Signature, Signature* Func, typename Ret, typename TypeList>
struct BNBFunctionWarpper_ {};

template <typename Signature, Signature* Func, typename Ret, typename... Args>
struct BNBFunctionWarpper_<
    Signature,
    Func,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    TypeSelector<at::Tensor, sizeof...(args)> selector;
    selector.extract_type(args...);
    const at::OptionalDeviceGuard dev_guard(
        device_of(selector.retrive_types()));
    return (*Func)(args...);
  }
};

template <typename Signature, Signature* Func>
struct BNBFunctionWarpper {
  using type = BNBFunctionWarpper_<
      Signature,
      Func,
      typename guts::function_traits<Signature>::return_type,
      typename guts::function_traits<Signature>::parameter_types>;
};
} // namespace bnb
} // namespace at

#define BNB_LIBRARY_FRAGMENT() TORCH_LIBRARY_FRAGMENT(torch_ipex, m)

#define BNB_OP_REGISTER(NAME, Func, Dispatchkey)           \
  at::bnb::construct_function_schema_and_register(         \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME), Func, m); \
  m.impl(                                                  \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),           \
      Dispatchkey,                                         \
      &at::bnb::BNBFunctionWarpper<decltype(Func), &Func>::type::call);

#ifndef SYCL_CUDA_STREAM
#define SYCL_CUDA_STREAM
namespace at {
inline sycl::queue* getCurrentSYCLStream() {
  auto& dpcpp_queue = at::xpu::getCurrentXPUStream().queue();
  return &dpcpp_queue;
}

} // namespace at
#endif