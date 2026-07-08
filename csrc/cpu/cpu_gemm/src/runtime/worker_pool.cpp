#include "runtime/worker_pool.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CG_POOL_PAUSE() _mm_pause()
#else
#define CG_POOL_PAUSE() std::this_thread::yield()
#endif

namespace cpu_gemm {

namespace {

int spin_budget_us() {
  static const int v = [] {
    const char* s = std::getenv("CG_POOL_SPIN_US");
    return s ? std::atoi(s) : 200;
  }();
  return v;
}

/* Busy-poll `pred` for up to the spin budget; true if it fired. The clock is
 * only sampled every 128 pauses — steady_clock::now() itself costs ~20ns. */
template <class Pred>
bool spin_for(const Pred& pred) {
  const int budget = spin_budget_us();
  if (budget <= 0) return pred();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(budget);
  int k = 0;
  for (;;) {
    if (pred()) return true;
    CG_POOL_PAUSE();
    if ((++k & 127) == 0 && std::chrono::steady_clock::now() >= deadline)
      return pred();
  }
}

}  // namespace

WorkerPool::WorkerPool(int n_threads) : total_threads_(std::max(1, n_threads)) {
  // Worker count == total_threads - 1; the calling thread also participates
  // when parallel_for() runs, so we get total_threads_ of parallelism.
  int spawn = total_threads_ - 1;
  workers_.reserve(spawn);
  for (int i = 0; i < spawn; ++i) {
    workers_.emplace_back([this, i] { worker_loop(i + 1); });
  }
}

WorkerPool::~WorkerPool() {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    exit_.store(true, std::memory_order_release);
    job_epoch_.fetch_add(1, std::memory_order_release);
  }
  cv_start_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
}

bool WorkerPool::drain_spin() const {
  return spin_for(
      [this] { return in_flight_.load(std::memory_order_acquire) == 0; });
}

void WorkerPool::parallel_for(int count, const std::function<void(int)>& body) {
  if (count <= 0) return;

  if (total_threads_ == 1 || count == 1) {
    for (int i = 0; i < count; ++i) body(i);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(mtx_);
    job_count_ = count;
    job_body_ = &body;
    next_.store(0, std::memory_order_release);
    in_flight_.store(total_threads_ - 1, std::memory_order_release);
    job_epoch_.fetch_add(1, std::memory_order_release);
  }
  cv_start_.notify_all();

  // Calling thread participates.
  int idx;
  while ((idx = next_.fetch_add(1, std::memory_order_acq_rel)) < count) {
    body(idx);
  }

  // Wait for workers to drain — spin first (they finish within microseconds
  // of the caller at decode-sized jobs), condvar only past the budget.
  if (!drain_spin()) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [this] {
      return in_flight_.load(std::memory_order_acquire) == 0;
    });
  }
  job_body_ = nullptr;
}

void WorkerPool::submit(int count, const std::function<void(int)>& body) {
  if (count <= 0) {
    in_flight_.store(0, std::memory_order_release);
    return;
  }
  if (total_threads_ == 1) {
    // No spawned workers: nothing would ever pick the job up. Run inline
    // (the caller blocks here instead of in wait_done).
    for (int i = 0; i < count; ++i) body(i);
    in_flight_.store(0, std::memory_order_release);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(mtx_);
    job_count_ = count;
    job_body_ = &body;
    next_.store(0, std::memory_order_release);
    in_flight_.store(total_threads_ - 1, std::memory_order_release);
    job_epoch_.fetch_add(1, std::memory_order_release);
  }
  cv_start_.notify_all();
}

void WorkerPool::wait_done() {
  if (!drain_spin()) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [this] {
      return in_flight_.load(std::memory_order_acquire) == 0;
    });
  }
  job_body_ = nullptr;
}

void WorkerPool::worker_loop(int /*id*/) {
  uint64_t last_seen = 0;
  for (;;) {
    // Spin-then-sleep: a new job usually lands within the spin budget at
    // serving time; the condvar (with its futex wake latency) is the idle
    // fallback, not the steady-state path.
    const bool woke = spin_for([&] {
      return job_epoch_.load(std::memory_order_acquire) != last_seen ||
             exit_.load(std::memory_order_acquire);
    });
    if (!woke) {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_start_.wait(lk, [&] {
        return job_epoch_.load(std::memory_order_acquire) != last_seen ||
               exit_.load(std::memory_order_acquire);
      });
    }
    last_seen = job_epoch_.load(std::memory_order_acquire);
    if (exit_.load(std::memory_order_acquire)) return;

    const auto* body = job_body_;
    int count = job_count_;
    if (body && count > 0) {
      int idx;
      while ((idx = next_.fetch_add(1, std::memory_order_acq_rel)) < count) {
        (*body)(idx);
      }
    }

    if (in_flight_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lk(mtx_);
      cv_done_.notify_one();
    }
  }
}

}  // namespace cpu_gemm
