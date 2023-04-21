#pragma once

#ifndef _MSC_VER
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define sys_getpid() getpid()
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define sys_gettid() syscall(SYS_gettid)
#else // __GLIBC__
#define sys_gettid() gettid()
#endif // __GLIBC__
#else // _MSC_VER
#include <windows.h>
#define sys_getpid() GetCurrentProcessId()
#define sys_gettid() GetCurrentThreadId()
#endif // _MSC_VER
