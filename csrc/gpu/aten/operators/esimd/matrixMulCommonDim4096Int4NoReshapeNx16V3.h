// FP16 internal p

// template <uint32_t pixelPerGroupShift>
// ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2(
//     uint8_t* a,
//     uint8_t* b,
//     uint8_t* c,
//     uint8_t* d,
//     nd_item<1>& ndi) {
//   constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
//   constexpr uint32_t quantPerGroup = 4096 / 32 * pixelPerGroup;
//   constexpr uint32_t baseOffsetInc16[16] = {
//       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//   constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
//   __ESIMD_NS::slm_init(16 * 16 * sizeof(fp16));
//   int hh = ndi.get_local_id(0); // [0, 64)
//   int h = ndi.get_group(0); // [0, 256)
//   // int rowSize = ndi.get_group_range(0) * pixelPerGroup;
//   int offsetABase = (h * pixelPerGroup * 4096 + hh * 8 * 8) >> 1;
//   int offsetQuanBase = /*rowSize * 2048 +*/ h * quantPerGroup * sizeof(fp16)
//   +
//       hh * 2 * sizeof(fp16);
//   int offsetB = hh * 64 * sizeof(fp16);
//   int outputOffset = pixelPerGroup * h;
//   simd<int8_t, 128> aaa;
//   simd<fp16, 16> quant;
//   simd<fp16, 8> fp32Quant;
//   simd<fp16, 256> bb;
//   simd<fp16, 256> bb_fp16;
//   simd<fp16, 16 * 16> aa;
//   simd<fp16, 16> cc(0.0f);
//   simd<uint32_t, 8> offsetA(baseOffsetInc8);
//   simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
//   simd_mask<8> quantPred = 1;
//   quantPred[4] = 0;
//   quantPred[5] = 0;
//   quantPred[6] = 0;
//   quantPred[7] = 0;
//   offsetA = offsetA * sizeof(uint32_t) + offsetABase;
//   offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

// #pragma unroll
//   for (int k = 0; k < 4; k++) {
//     bb_fp16.template bit_cast_view<unsigned char>().template select<128, 1>(
//         128 * k) =
//         __ESIMD_ENS::lsc_block_load<
//             uint8_t,
//             128,
//             __ESIMD_ENS::lsc_data_size::default_size,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

//     // bb.template bit_cast_view<unsigned char>().template select<256, 1>(512
//     *
//     // k + 256) =
//     //  __ESIMD_ENS::lsc_block_load<
//     //  uint8_t,
//     //  256,
//     //  __ESIMD_ENS::lsc_data_size::default_size,
//     //  __ESIMD_ENS::cache_hint::cached,
//     //  __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 256);

//     offsetB += 1024 * sizeof(fp16);
//   }

//   bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

//   for (int n = 0; n < pixelPerGroup; n++) {
//     cc = 0.0f;
//     quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)d, offsetQuan, quantPred);

//     aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)a, offsetA);
//     offsetA += 512;

//     aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)a, offsetA);
//     offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

//     aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)a, offsetA);
//     offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

//     aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)a, offsetA);
//     offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

// #pragma unroll
//     for (int k = 0; k < 8; k++) {

//       simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * k) & 0xf;
//       #pragma unroll
//       for (int l = 0; l < 16; l++)
//       {
//         if ((aaaa[l] & 0x8) != 0)
//         {
//           aaaa[l] = aaaa[l] | 0xf0;
//         }
//       }
//       aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

//       aaaa = aaa.select<16, 1>(16 * k) >> 4;
//       #pragma unroll
//       for (int l = 0; l < 16; l++)
//       {
//         if ((aaaa[l] & 0x8) != 0)
//         {
//           aaaa[l] = aaaa[l] | 0xf0;
//         }
//       }
//       aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);
//     }

//     // aa = aa - 8.0f;
//     fp32Quant = quant.select<8, 1>(0);
// #pragma unroll
//     for (int k = 0; k < 8; k++) {
//       aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
//     }

// #pragma unroll
//     for (int k = 0; k < 16; k++) {
//       cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
//     }

//     cc.select<8, 1>(0) += cc.select<8, 1>(8);
//     cc.select<4, 1>(0) += cc.select<4, 1>(4);
//     cc.select<2, 1>(0) += cc.select<2, 1>(2);
//     simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
//     uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(fp16);
//     // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
//     slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
//     offsetQuan += 128 * sizeof(fp16);
//   }
//   barrier();

//   if (hh == 0) {
//     if constexpr (pixelPerGroupShift == 4) {
// #pragma unroll
//       for (int k = 0; k < 4; k++) {
//         bb.select<64, 1>(64 * k) =
//             slm_block_load<fp16, 64>(64 * k * sizeof(fp16));
//       }
// #pragma unroll
//       for (int k = 1; k < 16; k++) {
//         bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
//       }

//     } else if constexpr (pixelPerGroupShift == 3) {
// #pragma unroll
//       for (int k = 0; k < 2; k++) {
//         bb.select<64, 1>(64 * k) =
//             slm_block_load<fp16, 64>(64 * k * sizeof(fp16));
//       }
// #pragma unroll
//       for (int k = 1; k < 8; k++) {
//         bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
//       }
//       bb.select<8, 1>(0) += bb.select<8, 1>(8);
//     } else if constexpr (pixelPerGroupShift == 2) {
//       bb.select<64, 1>(0) = slm_block_load<fp16, 64>(0);
// #pragma unroll
//       for (int k = 1; k < 4; k++) {
//         bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
//       }
//       bb.select<8, 1>(0) += bb.select<8, 1>(8);
//       bb.select<4, 1>(0) += bb.select<4, 1>(4);
//     } else if constexpr (pixelPerGroupShift == 1) {
//       bb.select<32, 1>(0) = slm_block_load<fp16, 32>(0);
//       bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
//       bb.select<8, 1>(0) += bb.select<8, 1>(8);
//       bb.select<4, 1>(0) += bb.select<4, 1>(4);
//       bb.select<2, 1>(0) += bb.select<2, 1>(2);
//     } else if constexpr (pixelPerGroupShift == 0) {
//       bb.select<16, 1>(0) = slm_block_load<fp16, 16>(0);
//       bb.select<8, 1>(0) += bb.select<8, 1>(8);
//       bb.select<4, 1>(0) += bb.select<4, 1>(4);
//       bb.select<2, 1>(0) += bb.select<2, 1>(2);
//       bb.select<1, 1>(0) += bb.select<1, 1>(1);
//     }

//     bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

//     __ESIMD_ENS::lsc_block_store<
//         fp16,
//         pixelPerGroup,
//         __ESIMD_ENS::lsc_data_size::default_size,
//         __ESIMD_ENS::cache_hint::write_back,
//         __ESIMD_ENS::cache_hint::write_back>(
//         (fp16*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
//   }
// }

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 4096 / 32 * pixelPerGroup;
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 16 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 4096 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 2048 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 128> aaa;
  simd<fp16, 16> quant;
  simd<float, 8> fp32Quant;
  simd<float, 256> bb;
  simd<fp16, 256> bb_fp16;
  simd<float, 16 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int k = 0; k < 4; k++) {
    bb_fp16.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    // bb.template bit_cast_view<unsigned char>().template select<256, 1>(512 *
    // k + 256) =
    //  __ESIMD_ENS::lsc_block_load<
    //  uint8_t,
    //  256,
    //  __ESIMD_ENS::lsc_data_size::default_size,
    //  __ESIMD_ENS::cache_hint::cached,
    //  __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 256);

    offsetB += 1024 * sizeof(fp16);
  }

  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

#pragma unroll
    for (int k = 0; k < 8; k++) {
      simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * k) & 0xf;
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) << 4;
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

      aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

      aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * k);
      aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

      aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);

      // simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * k) & 0xf;
      // #pragma unroll
      // for (int l = 0; l < 16; l++)
      // {
      //   if ((aaaa[l] & 0x8) != 0)
      //   {
      //     aaaa[l] = aaaa[l] | 0xf0;
      //   }
      // }
      // aa.select<16, 2>(32 * k) = aaaa.select<16, 1>(0);

      // aaaa = aaa.select<16, 1>(16 * k) >> 4;
      // #pragma unroll
      // for (int l = 0; l < 16; l++)
      // {
      //   if ((aaaa[l] & 0x8) != 0)
      //   {
      //     aaaa[l] = aaaa[l] | 0xf0;
      //   }
      // }
      // aa.select<16, 2>(32 * k + 1) = aaaa.select<16, 1>(0);
    }

    // aa = aa - 8.0f;
    fp32Quant = quant.select<8, 1>(0);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 128 * sizeof(fp16);
  }
  barrier();

  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
  }
}

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshapeNx16V2_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 11008 / 32 * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 11008 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 5504 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  simd<int8_t, 64> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 704> bb;
  simd<float, 256> bb_f32;
  simd<float, 8 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int k = 0; k < 10; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  if (hh < 12) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
  } else {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) = 0;
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 344 * sizeof(fp16);
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetQuan += 32 * sizeof(fp16) * 8;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 5504;

#pragma unroll
    for (int k = 0; k < 5; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
        simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) << 4;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }

      // aa = aa - 8.0f;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
      }
    }

    if (hh < 12) {
      simd<float, 2> fp32Q = quant.select<2, 1>(4 * 5);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
        // aa.select<32, 1>(32 * kk) = aa.select<32, 1>(32 * kk) - 8.0f;

        simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) << 4;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 5);
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb_f32.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb_f32.select<16, 1>(0) =
          bb_f32.select<16, 1>(0) + bb_f32.select<16, 1>(16 * k);
    }

    bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}

// template <uint32_t pixelPerGroupShift>
// ESIMD_INLINE void matrixMulCommonDim14336Int4NoReshapeNx16V2_ipex(
//     uint8_t* a,
//     uint8_t* b,
//     uint8_t* c,
//     uint8_t* d,
//     nd_item<1>& ndi) {
//   constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
//   constexpr uint32_t quantPerGroup = 14336 / 32 * pixelPerGroup;
//   constexpr uint32_t sumThreads = pixelPerGroup / 16;
//   constexpr uint32_t baseOffsetInc16[16] = {
//       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//   constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
//   __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
//   int hh = ndi.get_local_id(0); // [0, 64)
//   int h = ndi.get_group(0); // [0, 256)
//   int rowSize = ndi.get_group_range(0) * pixelPerGroup;
//   int offsetABase = (h * pixelPerGroup * 14336 + hh * 8 * 8) >> 1;
//   int offsetQuanBase = /*rowSize * 7168 +*/ h * quantPerGroup * sizeof(fp16)
//   +
//       hh * 2 * sizeof(fp16);
//   int offsetB = hh * 64 * sizeof(fp16);
//   int outputOffset = pixelPerGroup * h;
//   simd<int8_t, 64> aaa;
//   simd<fp16, 32> quant;
//   simd<fp16, 896> bb;
//   //simd<float, 256> bb_f32;
//   simd<float, 8 * 16> aa;
//   simd<float, 16> cc(0.0f);
//   simd<uint32_t, 8> offsetA(baseOffsetInc8);
//   simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
//   offsetA = offsetA * sizeof(uint32_t) + offsetABase;
//   offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

// #pragma unroll
//   for (int k = 0; k < 14; k++) {
//     bb.template bit_cast_view<unsigned char>().template select<128, 1>(
//         128 * k) =
//         __ESIMD_ENS::lsc_block_load<
//             uint8_t,
//             128,
//             __ESIMD_ENS::lsc_data_size::default_size,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

//     offsetB += 1024 * sizeof(fp16);
//   }

//   for (int n = 0; n < pixelPerGroup; n++) {
//     cc = 0.0f;
//     offsetQuan = baseOffsetInc8;
//     offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
//         n * 448 * sizeof(fp16);
//     quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)d, offsetQuan);

//     offsetQuan += 32 * sizeof(fp16) * 8;
//     quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
//         __ESIMD_ENS::lsc_gather<
//             uint32_t,
//             1,
//             __ESIMD_ENS::lsc_data_size::u32,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached,
//             8,
//             uint32_t>((uint32_t*)d, offsetQuan);

//     offsetA = baseOffsetInc8;
//     offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 7168;

// #pragma unroll
//     for (int k = 0; k < 7; k++) {
//       simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
//       aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
//           __ESIMD_ENS::lsc_gather<
//               uint32_t,
//               1,
//               __ESIMD_ENS::lsc_data_size::u32,
//               __ESIMD_ENS::cache_hint::cached,
//               __ESIMD_ENS::cache_hint::cached,
//               8,
//               uint32_t>((uint32_t*)a, offsetA);
//       offsetA += 512;

//       aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
//           __ESIMD_ENS::lsc_gather<
//               uint32_t,
//               1,
//               __ESIMD_ENS::lsc_data_size::u32,
//               __ESIMD_ENS::cache_hint::cached,
//               __ESIMD_ENS::cache_hint::cached,
//               8,
//               uint32_t>((uint32_t*)a, offsetA);
//       offsetA += 512;

// #pragma unroll
//       for (int kk = 0; kk < 4; kk++) {
//         // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
//         // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
//         simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
//         aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) << 4;
//         aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

//         aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

//         aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
//         aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

//         aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
//       }

//       // aa = aa - 8.0f;
// #pragma unroll
//       for (int kk = 0; kk < 4; kk++) {
//         aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
//       }

// #pragma unroll
//       for (int kk = 0; kk < 8; kk++) {
//         cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 *
//         k);
//       }
//     }

//     cc.select<8, 1>(0) += cc.select<8, 1>(8);
//     cc.select<4, 1>(0) += cc.select<4, 1>(4);
//     cc.select<2, 1>(0) += cc.select<2, 1>(2);
//     simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
//     uint32_t slmGroup = n >> 4;
//     uint32_t slmInnerGroupOffset = n & 0xf;
//     uint32_t slmAccumulationOffset =
//         (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
//     // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
//     slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
//   }
//   barrier();
//   if (hh < sumThreads) {
//     uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
// #pragma unroll
//     for (int k = 0; k < 4; k++) {
//       bb.select<64, 1>(64 * k) = slm_block_load<float, 64>(
//           slmSumPhase1LoadOffset + 64 * k * sizeof(float));
//     }

// #pragma unroll
//     for (int k = 1; k < 16; k++) {
//       bb.select<16, 1>(0) =
//           bb.select<16, 1>(0) + bb.select<16, 1>(16 * k);
//     }

//     //bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

//     __ESIMD_ENS::lsc_block_store<
//         fp16,
//         16,
//         __ESIMD_ENS::lsc_data_size::default_size,
//         __ESIMD_ENS::cache_hint::write_back,
//         __ESIMD_ENS::cache_hint::write_back>(
//         (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
//   }
// }

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim14336Int4NoReshapeNx16V2_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 14336 / 32 * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  constexpr uint32_t offsetSLM = 16 * 128 * sizeof(float);
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float) + 16 * 2 * 64 * sizeof(fp16));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 14336 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 7168 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  int offsetSLMThread = hh * 2 * 64 * sizeof(fp16);
  simd<int8_t, 64> aaa;
  simd<fp16, 32> quant;
  // simd<fp16, 896> bb;
  simd<fp16, 768> bb; // avoid grf spill
  simd<fp16, 64> bbb;
  simd<float, 8 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int k = 0; k < 12; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

#pragma unroll
  for (int k = 0; k < 2; k++) {
    bbb.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
    slm_block_store<fp16, 64>(
        offsetSLM + offsetSLMThread + k * 64 * sizeof(fp16),
        bbb.select<64, 1>(0));

    offsetB += 1024 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 448 * sizeof(fp16);
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetQuan += 32 * sizeof(fp16) * 8;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 7168;

#pragma unroll
    for (int k = 0; k < 7; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
        simd<int8_t, 16> aaaa = aaa.select<16, 1>(16 * kk) & 0xf;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) << 4;
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk) = aaaa.select<16, 1>(0);

        aaaa.select<16, 1>(0) = aaa.select<16, 1>(16 * kk);
        aaaa.select<16, 1>(0) = aaaa.select<16, 1>(0) >> 4;

        aa.select<16, 2>(32 * kk + 1) = aaaa.select<16, 1>(0);
      }

      // aa = aa - 8.0f;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

      if (k < 6) {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          bbb.select<16, 1>(0) = slm_block_load<fp16, 16>(
              offsetSLM + offsetSLMThread + 16 * kk * sizeof(fp16));
          cc += aa.select<16, 1>(16 * kk) * bbb.select<16, 1>(0);
        }
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb.select<16, 1>(0) = bb.select<16, 1>(0) + bb.select<16, 1>(16 * k);
    }

    // bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}