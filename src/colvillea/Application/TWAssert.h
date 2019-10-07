#pragma once

#if !defined(NDEBUG)
#if defined(_WIN32) // Windows 32/64-bit
#include <windows.h>
#define DBGBREAK() DebugBreak();
#else
#define DBGBREAK() __builtin_trap();
#endif
#define TW_ASSERT(expr) if (!(expr)) DBGBREAK()
#else
#define TW_ASSERT(expr)
#endif

#define RT_CHECK_ERROR_NO_CONTEXT( func ) \
  do { \
    RTresult code = func; \
    if (code != RT_SUCCESS) \
      std::cerr << "ERROR: Function " << #func << std::endl; \
  } while (0)