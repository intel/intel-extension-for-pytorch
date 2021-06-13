#pragma once

#include <ATen/Context.h>
#include <ATen/core/ATenGeneral.h>

#include <runtime/DPCPPUtils.h>
#include <core/Device.h>


namespace xpu {
namespace dpcpp {

// Provide a caching allocator for host allocation by USM malloc_host
Allocator* dpcpp_getCachingHostAllocator();

// Record the event on queue where the host allocation is using
void dpcpp_recordEventInCachingHostAllocator(void* ptr, DPCPP::event& e);

// Releases all cached host memory allocations
void dpcpp_emptyCacheInCachingHostAllocator();

bool dpcpp_isAllocatedByCachingHostAllocator(void* ptr);
}} // namespace xpu::dpcpp
