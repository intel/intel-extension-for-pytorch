#pragma once
#include <c10/util/Exception.h>

/*
 * All available launch options for IPEX
 * IPEX_VERBOSE:                Default = 0, Set verbose level in IPEX
 * IPEX_FORCE_SYNC:             Default = 0, Set 1 to enforce blocked/sync execution mode
 * IPEX_DISABLE_PROFILING:      Default = 0, Set 1 to disable IPEX event profiling
 * IPEX_LAZY_REORDER:           Default = 0, Set 1 to enable lazy reorder to avoid unnecessary reorders
 * IPEX_WEIGHT_CACHE:           Default = 0, Set 1 to cache the packed weight in original weight Tensor
 * IPEX_DISABLE_TILE_PARTITION: Default = 0, Set 1 to disable tile partition and map device per physical device.
 */

int dpcpp_env(int env);

int dpcpp_verbose();

int dpcpp_force_sync();

bool dpcpp_profiling();

int lazy_reorder_enabled();

int weight_cache_enabled();

bool disable_tile_partition();
