#include <immintrin.h>
#include <stdio.h>
#include <iostream>

#ifdef RTM_DEBUG
extern int rtm_stats[1000][16];
#endif
#define ATTEMPTS 0
#define ABORTS 1
#define LOCKS 2
#define COUNTS 3
#define ABORTS_RETRY 4
#define ABORTS_NORETRY 5
#define ABORTS_TIMEOUT 6
#define ABORTS_EXPLICIT 7
#define ABORTS_ZERO 8

inline void clear_rtm_stats() {
#ifdef RTM_DEBUG
  int rtm_max_threads = omp_get_max_threads();
  for (int i = 0; i < rtm_max_threads; i++) {
    for (int j = 0; j < 16; j++) {
      rtm_stats[i][j] = 0;
    }
  }
#endif
}

inline void print_rtm_stats() {
#ifdef RTM_DEBUG
  int rtm_max_threads = omp_get_max_threads();
  int total[16] = {0};
  for (int i = 0; i < rtm_max_threads; i++) {
    printf(
        "Tid %3d: RTM_STATS C: %8d   AT: %8d   AB: %8d   L:  %6d (E: %6d Z: %6d R: %6d O: %6d T: %6d)\n",
        i,
        rtm_stats[i][COUNTS],
        rtm_stats[i][ATTEMPTS],
        rtm_stats[i][ABORTS],
        rtm_stats[i][LOCKS],
        rtm_stats[i][ABORTS_EXPLICIT],
        rtm_stats[i][ABORTS_ZERO],
        rtm_stats[i][ABORTS_RETRY],
        rtm_stats[i][ABORTS_NORETRY],
        rtm_stats[i][ABORTS_TIMEOUT]);
    total[COUNTS] += rtm_stats[i][COUNTS];
    total[ATTEMPTS] += rtm_stats[i][ATTEMPTS];
    total[ABORTS] += rtm_stats[i][ABORTS];
    total[LOCKS] += rtm_stats[i][LOCKS];
    total[ABORTS_EXPLICIT] += rtm_stats[i][ABORTS_EXPLICIT];
    total[ABORTS_ZERO] += rtm_stats[i][ABORTS_ZERO];
    total[ABORTS_RETRY] += rtm_stats[i][ABORTS_RETRY];
    total[ABORTS_NORETRY] += rtm_stats[i][ABORTS_NORETRY];
    total[ABORTS_TIMEOUT] += rtm_stats[i][ABORTS_TIMEOUT];
  }
  printf(
      "Total:   RTM_STATS C: %8d   AT: %8d   AB: %8d   L:  %6d (E: %6d Z: %6d R: %6d O: %6d T: %6d)\n",
      total[COUNTS],
      total[ATTEMPTS],
      total[ABORTS],
      total[LOCKS],
      total[ABORTS_EXPLICIT],
      total[ABORTS_ZERO],
      total[ABORTS_RETRY],
      total[ABORTS_NORETRY],
      total[ABORTS_TIMEOUT]);
#endif
}

class SimpleSpinLock {
  volatile unsigned int state;
  enum { Free = 0, Busy = 1 };

 public:
  SimpleSpinLock() : state(Free) {}
  void lock() {
    while (__sync_val_compare_and_swap(&state, Free, Busy) != Free) {
      do {
        _mm_pause();
      } while (state == Busy);
    }
  }
  void unlock() {
    state = Free;
  }
  bool isLocked() const {
    return state == Busy;
  }
};
#ifdef RTM_DEBUG
#define INC_RTM_DEBUG_COUNT(tid, x) rtm_stats[tid][x]++
#else
#define INC_RTM_DEBUG_COUNT(tid, x)
#endif

class TransactionScope {
  SimpleSpinLock& fallBackLock;

  TransactionScope(); // forbidden
 public:
  TransactionScope(
      SimpleSpinLock& fallBackLock_,
      int max_retries = 10,
      int tid = 0)
      : fallBackLock(fallBackLock_) {
    int nretries = 0;
    INC_RTM_DEBUG_COUNT(tid, COUNTS);
    while (1) {
      ++nretries;
      INC_RTM_DEBUG_COUNT(tid, ATTEMPTS);
      unsigned status = _xbegin();
      if (status == _XBEGIN_STARTED) {
        if (!fallBackLock.isLocked())
          return; // successfully started transaction
        _xabort(0xff); // abort with code 0xff
      }
      // abort handler

      INC_RTM_DEBUG_COUNT(tid, ABORTS);

      // handle _xabort(0xff) from above
      if ((status & _XABORT_EXPLICIT) && _XABORT_CODE(status) == 0xff &&
          !(status & _XABORT_NESTED)) {
        while (fallBackLock.isLocked())
          _mm_pause(); // wait until lock is free
        INC_RTM_DEBUG_COUNT(tid, ABORTS_EXPLICIT);
      } else if (status == 0) {
        INC_RTM_DEBUG_COUNT(tid, ABORTS_ZERO);
      } else if ((status & _XABORT_RETRY) || (status & _XABORT_CONFLICT)) {
        INC_RTM_DEBUG_COUNT(tid, ABORTS_RETRY);
      } else {
        INC_RTM_DEBUG_COUNT(tid, ABORTS_NORETRY);
        // break; // take the fall-back lock if the retry abort flag is not set
      }
      if (nretries >= max_retries) {
        INC_RTM_DEBUG_COUNT(tid, ABORTS_TIMEOUT);
        break; // too many retries, take the fall-back lock
      }
    }
    fallBackLock.lock();
    INC_RTM_DEBUG_COUNT(tid, LOCKS);
  }

  ~TransactionScope() {
    if (fallBackLock.isLocked()) {
      fallBackLock.unlock();
    } else {
      _xend();
    }
  }
};

#undef INC_RTM_DEBUG_COUNT
