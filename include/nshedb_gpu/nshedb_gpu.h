#pragma once
// nshedb_gpu.h — master include for the GPU backend of NSHEDB.
//
// Usage:
//   #include "nshedb_gpu/nshedb_gpu.h"
//
// Requires linking against: nshedb_gpu (this library) and Phantom (GPU FHE).
//
// For COUNT_GPU / SUM_GPU the caller needs a BfvRotationKeyStream:
//   using nshedb_gpu::utils::BfvRotationKeyStream;
//   BfvRotationKeyStream rks(ctx, gk);   // gk = PhantomGaloisKey
//   COUNT_GPU(ctx, ct, slot_count, rks);

#include "nshedb_gpu/utils/math_utils_gpu.h"
#include "nshedb_gpu/utils/conversion_gpu.h"
#include "nshedb_gpu/utils/cache_bridge.h"
#include "nshedb_gpu/core/comparator_gpu.cuh"
#include "nshedb_gpu/predicates/predicates_gpu.cuh"
