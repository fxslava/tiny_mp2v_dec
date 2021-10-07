#include "mc.h"
#include "core/common/cpu.hpp"

#define MC_ARRAYS(prefix) \
mc_pred_func_t mc_pred_16xh[4]    = { mc_pred00_16xh_##prefix,    mc_pred01_16xh_##prefix,    mc_pred10_16xh_##prefix,    mc_pred11_16xh_##prefix }; \
mc_pred_func_t mc_pred_8xh[4]     = { mc_pred00_8xh_##prefix,     mc_pred01_8xh_##prefix,     mc_pred10_8xh_##prefix,     mc_pred11_8xh_##prefix }; \
mc_bidir_func_t mc_bidir_16xh[16] = { mc_bidir0000_16xh_##prefix, mc_bidir0001_16xh_##prefix, mc_bidir0010_16xh_##prefix, mc_bidir0011_16xh_##prefix, \
                                      mc_bidir0100_16xh_##prefix, mc_bidir0101_16xh_##prefix, mc_bidir0110_16xh_##prefix, mc_bidir0111_16xh_##prefix, \
                                      mc_bidir1000_16xh_##prefix, mc_bidir1001_16xh_##prefix, mc_bidir1010_16xh_##prefix, mc_bidir1011_16xh_##prefix, \
                                      mc_bidir1100_16xh_##prefix, mc_bidir1101_16xh_##prefix, mc_bidir1110_16xh_##prefix, mc_bidir1111_16xh_##prefix }; \
mc_bidir_func_t mc_bidir_8xh[16] = {  mc_bidir0000_8xh_##prefix,  mc_bidir0001_8xh_##prefix,  mc_bidir0010_8xh_##prefix,  mc_bidir0011_8xh_##prefix, \
                                      mc_bidir0100_8xh_##prefix,  mc_bidir0101_8xh_##prefix,  mc_bidir0110_8xh_##prefix,  mc_bidir0111_8xh_##prefix, \
                                      mc_bidir1000_8xh_##prefix,  mc_bidir1001_8xh_##prefix,  mc_bidir1010_8xh_##prefix,  mc_bidir1011_8xh_##prefix, \
                                      mc_bidir1100_8xh_##prefix,  mc_bidir1101_8xh_##prefix,  mc_bidir1110_8xh_##prefix,  mc_bidir1111_8xh_##prefix };

#if defined(CPU_PLATFORM_AARCH64)
#include "mc_aarch64.hpp"
MC_ARRAYS(aarch64)
#elif defined(CPU_PLATFORM_X64)
#include "mc_sse2.hpp"
MC_ARRAYS(sse2)
#else
#include "mc_c.hpp"
MC_ARRAYS(c)
#endif
