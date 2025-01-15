#pragma once

#ifdef _WIN32
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
