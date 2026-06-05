#include "runtime/worker_pool.h"

#include <algorithm>
#include <cassert>

namespace cpu_gemm {

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

  // Wait for workers to drain.
  std::unique_lock<std::mutex> lk(mtx_);
  cv_done_.wait(lk, [this] {
    return in_flight_.load(std::memory_order_acquire) == 0;
  });
  job_body_ = nullptr;
}

void WorkerPool::worker_loop(int /*id*/) {
  uint64_t last_seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_start_.wait(lk, [&] {
        return job_epoch_.load(std::memory_order_acquire) != last_seen ||
               exit_.load(std::memory_order_acquire);
      });
      last_seen = job_epoch_.load(std::memory_order_acquire);
      if (exit_.load(std::memory_order_acquire)) return;
    }

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
