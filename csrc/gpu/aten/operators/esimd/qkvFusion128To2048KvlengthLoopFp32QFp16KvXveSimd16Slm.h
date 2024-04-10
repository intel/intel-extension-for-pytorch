ESIMD_INLINE void qkvFusion128To2048KvlengthLoopFp32QFp16KvXveSimd16Slm_ipex(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* out,
    int kvSeqLen,
    int vCacheStride,
    nd_item<2>& ndi) {
  constexpr float matMulQuantCoeff =
      0.08838834764831844f; // 1.0f / sqrt(128.0f);
  // constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  // 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(32 * 128 * sizeof(fp16) + 32 * sizeof(float)); //
  int localLinearId = ndi.get_local_linear_id(); // [0, 32)
  int hh = localLinearId & 0x3; // [0, 4)
  int vv = localLinearId >> 2; // [0, 8)
  int h = ndi.get_group(0); // [0, 32)
  int v = ndi.get_group(1); // [0, batch)
  int kvSeqOutLoopCount = (kvSeqLen + 0x1f) >> 5; // seqLen / 64
  simd<float, 128> qqFp32;
  simd<fp16, 128> qq;
  simd<fp16, 128> kk;
  simd<fp16, 128> vvState;
  simd<float, 128> fp32Kv;
  simd<float, 16> kvCacheOut = 0;
  simd<fp16, 16> kvCacheOut_fp16 = 0;
  simd<float, 16> softMaxSumTemp = 0;
  simd<fp16, 32 * 16> vvCache;
  simd<float, 32> softMaxCache;
  simd<float, 1> softMax;
  float softMaxPadding = 0;
  simd<float, 16> output = 0;
  unsigned int outputOffset = (v * 32 + h) * 128 + localLinearId * 16;
  unsigned int outputVOffsetSlm = localLinearId * 128 * sizeof(fp16);
  unsigned int outputSoftmaxOffsetSlm = 32 * 128 * sizeof(fp16);
  unsigned int offsetQ = h * 128 * sizeof(fp16); // Q is fp16
  unsigned int offsetK = (h * 128 + localLinearId * 128 * 32) * sizeof(fp16);
  unsigned int offsetVBase =
      (h * 128 + vv * 16 + hh * 8 * 128 * 32) * sizeof(fp16);
  unsigned int offsetV = offsetVBase;
  unsigned int offsetV_temp = 0;
  int kvSeqOffset = localLinearId;

  qq.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
      __ESIMD_ENS::lsc_block_load<
          unsigned char,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((unsigned char*)qState + offsetQ);

  qq.template bit_cast_view<unsigned char>().template select<128, 1>(128) =
      __ESIMD_ENS::lsc_block_load<
          unsigned char,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(
          (unsigned char*)qState + offsetQ + 128);

  qqFp32.select<128, 1>(0) = qq.select<128, 1>(0);

  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    offsetV_temp = offsetV;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      vvState.template bit_cast_view<unsigned char>().template select<32, 1>(
          32 * k) =
          __ESIMD_ENS::lsc_block_load<
              unsigned char,
              32,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>(
              (unsigned char*)vState + offsetV_temp);
      offsetV_temp += 128 * 32 * sizeof(fp16);
    }

    if (kvSeqOffset < kvSeqLen) {
      output = 0;
      kk.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
          __ESIMD_ENS::lsc_block_load<
              unsigned char,
              256,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>(
              (unsigned char*)kState + offsetK);

      fp32Kv = kk;

#pragma unroll
      for (int ll = 0; ll < 8; ll++) {
        output += qqFp32.select<16, 1>(16 * ll) * fp32Kv.select<16, 1>(16 * ll);
      }

      output.select<8, 1>(0) = output.select<8, 1>(0) + output.select<8, 1>(8);
      output.select<4, 1>(0) = output.select<4, 1>(0) + output.select<4, 1>(4);
      output.select<2, 1>(0) = output.select<2, 1>(0) + output.select<2, 1>(2);
      softMax = output[0] + output[1];
      softMax = softMax * matMulQuantCoeff;
      softMax = pow<float, 1, float>(2.718f, softMax);
    } else {
      softMax = 0;
    }

    // slm_scalar_store(outputSoftmaxOffsetSlm + localLinearId * sizeof(float),
    // softMax);
    slm_block_store<float, 1>(
        outputSoftmaxOffsetSlm + localLinearId * sizeof(float), softMax);
// #pragma unroll
// for (int ll = 0; ll < 4; ll++) {
//   simd<fp16, 32> shuffleTemp;
//   shuffleTemp = vvState.select<32, 1>(32 * ll);
//   vvState.select<16, 1>(32 * ll) = shuffleTemp.select<16, 2>(0);
//   vvState.select<16, 1>(32 * ll + 16) = shuffleTemp.select<16, 2>(1);
// }
#pragma unroll
    for (int ll = 0; ll < 2; ll++) {
      slm_block_store<fp16, 64>(
          outputVOffsetSlm + ll * 64 * sizeof(fp16),
          vvState.select<64, 1>(64 * ll));
    }

    barrier();
    if (localLinearId < 8) {
      int slmVLoadOffset = localLinearId * 16 * 32 * sizeof(fp16);
      int slmSoftMaxLoadOffset = outputSoftmaxOffsetSlm;
      softMaxCache.select<32, 1>(0) =
          slm_block_load<float, 32>(slmSoftMaxLoadOffset);
#pragma unroll
      for (int cc = 0; cc < 4; cc++) {
        vvCache.select<128, 1>(128 * cc) =
            slm_block_load<fp16, 128>(slmVLoadOffset + 128 * cc * sizeof(fp16));
      }

#pragma unroll
      for (int nn = 0; nn < 4; nn++) {
        fp32Kv = vvCache.select<128, 1>(128 * nn);
#pragma unroll
        for (int nnn = 0; nnn < 8; nnn++) {
          if (32 * loopIdx + nn * 8 + nnn < kvSeqLen) {
            kvCacheOut.select<16, 1>(0) +=
                fp32Kv.select<16, 1>(16 * nnn) * softMaxCache[nn * 8 + nnn];
          }
        }
      }
#pragma unroll
      for (int mm = 0; mm < 2; mm++) {
        softMaxSumTemp += softMaxCache.select<16, 1>(16 * mm);
      }
    }

    offsetK += 32 * 128 * 32 * sizeof(fp16);
    offsetV += 32 * 128 * 32 * sizeof(fp16);
    kvSeqOffset += 32;
    barrier();
  }

  if (localLinearId < 8) {
    softMaxSumTemp.select<8, 1>(0) =
        softMaxSumTemp.select<8, 1>(0) + softMaxSumTemp.select<8, 1>(8);
    softMaxSumTemp.select<4, 1>(0) =
        softMaxSumTemp.select<4, 1>(0) + softMaxSumTemp.select<4, 1>(4);
    softMaxSumTemp.select<2, 1>(0) =
        softMaxSumTemp.select<2, 1>(0) + softMaxSumTemp.select<2, 1>(2);
    softMaxSumTemp[0] = softMaxSumTemp[0] + softMaxSumTemp[1];
    softMaxSumTemp[0] = 1.0f / softMaxSumTemp[0];
    kvCacheOut.select<16, 1>(0) =
        kvCacheOut.select<16, 1>(0) * softMaxSumTemp[0];

    kvCacheOut_fp16.select<16, 1>(0) = kvCacheOut.select<16, 1>(0);
    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)out + outputOffset, kvCacheOut_fp16);
  }
}

// Shape Q [activation token length, 32, 128]  FP16,
// Shape K:  [kv len, 32, 128] FP16,
// Shape V : [kv len, 32, 128] FP16,
// output: [activation token length, 32, 128]  FP16,
// atten = mat_mul(q, k), shape = [first token length, kv len]
// atten = softmax(atten)
// output = mat_mul(atten, V)
