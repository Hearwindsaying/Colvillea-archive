#pragma once

#ifndef NDEBUG
#if defined(_MSC_VER) 
#define TW_ASSERT(expr) if (!(expr)) __debugbreak();
#else
#define TW_ASSERT(expr)
#endif
#endif

#define RT_CHECK_ERROR_NO_CONTEXT( func ) \
  do { \
    RTresult code = func; \
    if (code != RT_SUCCESS) \
      std::cerr << "ERROR: Function " << #func << std::endl; \
  } while (0)