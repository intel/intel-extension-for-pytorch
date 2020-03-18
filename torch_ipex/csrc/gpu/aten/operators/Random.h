#include <core/DPCPP.h>
#include <core/Memory.h>

using namespace DPCPP;
using namespace at::dpcpp;
using ACC_1D_RW_G = accessor<buffer_data_type_t, 1, access::mode::read_write, DPCPP::access::target::global_buffer>;
using ACC_1D_W_G = accessor<buffer_data_type_t, 1, access::mode::write, DPCPP::access::target::global_buffer>;

#define RANDOM_NUM (200000)
#define NUM_PER_RND (624)

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397

#define MT_WMASK 0xFFFFFFFFU
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static const double DOUBLE_DIVISOR = 18446744073709551616.0;
static const float FLOAT_DIVISOR = 4294967296.0f;

template<typename META_TYPE>
class MTRandomEngine {
public:
    MTRandomEngine(ACC_1D_W_G ptr, int64_t ele_num, uint64_t seed, META_TYPE min, META_TYPE max) : m_ptr(ptr) {
        m_ele_num  = ele_num;
        m_seed     = seed;
        m_real_min = static_cast<META_TYPE>(min);
        m_real_max = static_cast<META_TYPE>(max);
    }

    // mersenne_twister_engine
    void generate_random_numbers(const int& global_linear_id, const int& seed_offset, uint32_t *mt) {
        mt[0] = m_seed + global_linear_id * 100 + seed_offset;
        for (int i = 1; i < NUM_PER_RND; i++) {
            mt[i] = (1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & MT_WMASK;
        }

        uint32_t *p = mt;
        for(int j = (_MERSENNE_STATE_N - _MERSENNE_STATE_M + 1); --j; p++)
            *p = p[_MERSENNE_STATE_M] ^ TWIST(p[0], p[1]);

        for(int j = _MERSENNE_STATE_M; --j; p++)
            *p = p[_MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ TWIST(p[0], p[1]);

        *p = p[_MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ TWIST(p[0], mt[0]);

        for (int i = 0; i < NUM_PER_RND; i++) {
            uint32_t &y = mt[i];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680UL;
            y ^= (y << 15) & 0xefc60000UL;
            y ^= (y >> 18);
        }
    }

protected:
  ACC_1D_W_G  m_ptr;
  int64_t     m_ele_num;
  uint64_t    m_seed;
  META_TYPE   m_real_min;
  META_TYPE   m_real_max;
};

class HalfRandomFiller : public MTRandomEngine<unsigned short> {
public:
    HalfRandomFiller(ACC_1D_W_G ptr, int64_t ele_num, uint64_t seed, float min, float max)
        : MTRandomEngine<unsigned short>(ptr, ele_num, seed, min, max) {
    }

    void operator()(nd_item<1> item) {
        unsigned short *real_ptr = SyclConvertToActualTypePtr(unsigned short, m_ptr);
        uint32_t mt[NUM_PER_RND];

        int g_id = item.get_global_linear_id();
        generate_random_numbers(g_id, 0, mt);

        for (int i = 0; i < NUM_PER_RND; i++) {
            int idx = i + g_id * NUM_PER_RND;

            if (idx >= m_ele_num)
                return;
            uint32_t y = mt[i];

            float tmp = m_real_min + ((float)y + 1.0f) / FLOAT_DIVISOR * (m_real_max - m_real_min);
            real_ptr[idx] = static_cast<unsigned short>(static_cast<at::Half>(tmp));
        }
    }
};

class FloatRandomFiller : public MTRandomEngine<float> {
public:
    FloatRandomFiller(ACC_1D_W_G ptr, int64_t ele_num, uint64_t seed, float min, float max)
        : MTRandomEngine<float>(ptr, ele_num, seed, min, max) {
    }

    void operator()(nd_item<1> item) {
        float *real_ptr = SyclConvertToActualTypePtr(float, m_ptr);
        uint32_t mt[NUM_PER_RND];

        int g_id = item.get_global_linear_id();
        generate_random_numbers(g_id, 0, mt);

        for (int i = 0; i < NUM_PER_RND; i++) {
            int idx = i + g_id * NUM_PER_RND;

            if (idx >= m_ele_num)
                return;
            uint32_t y = mt[i];
            real_ptr[idx] = m_real_min + ((float)y + 1.0f) / FLOAT_DIVISOR * (m_real_max - m_real_min);
        }
    }
};

class DoubleRandomFiller : public MTRandomEngine<double> {
public:
    DoubleRandomFiller(ACC_1D_W_G ptr, int64_t ele_num, uint64_t seed, double min, double max)
        : MTRandomEngine<double>(ptr, ele_num, seed, min, max) {
    }

    void operator()(nd_item<1> item) {
        double *real_ptr = SyclConvertToActualTypePtr(double, m_ptr);
        uint32_t mt_hi[NUM_PER_RND];
        uint32_t mt_lo[NUM_PER_RND];

        int g_id = item.get_global_linear_id();
        generate_random_numbers(g_id, 0, mt_hi);
        generate_random_numbers(g_id, 50, mt_lo);

        for (int i = 0; i < NUM_PER_RND; i++) {
            int idx = i + g_id * NUM_PER_RND;
            if (idx >= m_ele_num)
                return;
            uint64_t y = (((uint64_t)mt_hi[i]) << 32) | mt_lo[i];
            real_ptr[idx] = m_real_min + ((double)y + 1.0) / DOUBLE_DIVISOR * (m_real_max - m_real_min);
        }
    }
};

template<typename META_TYPE>
class NormalRandomFiller {
public:
  NormalRandomFiller(ACC_1D_RW_G ptr, int64_t compute_num, double stdv, double mean) : m_ptr(ptr) {
    m_compute_num = compute_num;
    m_stdv = stdv;
    m_mean = mean;
  }

  void operator()(nd_item<1> item) {
    META_TYPE *real_ptr = SyclConvertToActualTypePtr(META_TYPE, m_ptr);
    int g_id = item.get_global_linear_id();

    if (g_id < m_compute_num) {
      const META_TYPE u1 = 1 - real_ptr[g_id]; // [0, 1) -> (0, 1] for log.
      const META_TYPE u2 = real_ptr[g_id + m_compute_num];

      const META_TYPE radius = DPCPP::sqrt(-2 * DPCPP::log(u1));
      const META_TYPE theta = 2.0f * M_PI * u2;

      real_ptr[g_id] = radius * DPCPP::cos(theta) * m_stdv + m_mean;
      real_ptr[g_id + m_compute_num] = radius * DPCPP::sin(theta) * m_stdv + m_mean;
    }
  }

private:
  ACC_1D_RW_G m_ptr;
  double m_stdv;
  double m_mean;
  int64_t m_compute_num;
};
