#pragma once

#include <condition_variable>
#include <mutex>

/*

Usage:

torch_ipex::ReadWriteMutex rwmutex;

void read_function() {
    torch_ipex::UniqueReadLock<torch_ipex::ReadWriteMutex> lock( rwmutex );
    // reading
}

void write_function() {
    torch_ipex::UniqueWriteLock<torch_ipex::ReadWriteMutex> lock( rwmutex );
    // writing
}
*/

namespace torch_ipex {
class ReadWriteMutex {
public:
  ReadWriteMutex() = default;
  ~ReadWriteMutex() = default;

  ReadWriteMutex(const ReadWriteMutex &) = delete;
  ReadWriteMutex &operator=(const ReadWriteMutex &) = delete;

  ReadWriteMutex(const ReadWriteMutex &&) = delete;
  ReadWriteMutex &operator=(const ReadWriteMutex &&) = delete;

  void lock_read() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond_read.wait(lock, [this]() -> bool { return m_write_count == 0; });
    ++m_read_count;
  }

  void unlock_read() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (--m_read_count == 0 && m_write_count > 0) {
      m_cond_write.notify_one();
    }
  }

  void lock_write() {
    std::unique_lock<std::mutex> lock(m_mutex);
    ++m_write_count;
    m_cond_write.wait(
        lock, [this]() -> bool { return m_read_count == 0 && !m_writing; });
    m_writing = true;
  }

  void unlock_write() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (--m_write_count == 0) {
      m_cond_read.notify_all();
    } else {
      m_cond_write.notify_one();
    }
    m_writing = false;
  }

private:
  volatile size_t m_read_count = 0;
  volatile size_t m_write_count = 0;
  volatile bool m_writing = false;
  std::mutex m_mutex;
  std::condition_variable m_cond_read;
  std::condition_variable m_cond_write;
};

template <typename _ReadWriteLock> class UniqueReadLock {
public:
  explicit UniqueReadLock(_ReadWriteLock &rwLock) : m_ptr_rw_lock(&rwLock) {
    m_ptr_rw_lock->lock_read();
  }

  ~UniqueReadLock() {
    if (m_ptr_rw_lock) {
      m_ptr_rw_lock->unlock_read();
    }
  }

  UniqueReadLock() = delete;
  UniqueReadLock(const UniqueReadLock &) = delete;
  UniqueReadLock &operator=(const UniqueReadLock &) = delete;
  UniqueReadLock(const UniqueReadLock &&) = delete;
  UniqueReadLock &operator=(const UniqueReadLock &&) = delete;

private:
  _ReadWriteLock *m_ptr_rw_lock = nullptr;
};

template <typename _ReadWriteLock> class UniqueWriteLock {
public:
  explicit UniqueWriteLock(_ReadWriteLock &rwLock) : m_ptr_rw_lock(&rwLock) {
    m_ptr_rw_lock->lock_write();
  }

  ~UniqueWriteLock() {
    if (m_ptr_rw_lock) {
      m_ptr_rw_lock->unlock_write();
    }
  }

  UniqueWriteLock() = delete;
  UniqueWriteLock(const UniqueWriteLock &) = delete;
  UniqueWriteLock &operator=(const UniqueWriteLock &) = delete;
  UniqueWriteLock(const UniqueWriteLock &&) = delete;
  UniqueWriteLock &operator=(const UniqueWriteLock &&) = delete;

private:
  _ReadWriteLock *m_ptr_rw_lock = nullptr;
};
} // namespace torch_ipex