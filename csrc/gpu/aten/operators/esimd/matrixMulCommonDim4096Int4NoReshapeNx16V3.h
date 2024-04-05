
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
  simd<unsigned char, 128> aaa;
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
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
      aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
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