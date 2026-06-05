/*
 * Portable work-stealing thread pool — std::thread + std::atomic only.
 *
 * Mirrors the shape of InNumaPool from ktransformers/cpu_backend/worker_pool.h
 * (the do_work_stealing_job(N, fn) primitive) but drops the NUMA / hwloc
 * dependency. A NUMA-aware variant can be added later behind
 * CPU_GEMM_WITH_NUMA without changing the dispatcher.
 */
#ifndef CPU_GEMM_RUNTIME_WORKER_POOL_H
#define CPU_GEMM_RUNTIME_WORKER_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cpu_gemm {

class WorkerPool {
 public:
  explicit WorkerPool(int n_threads);
  ~WorkerPool();

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  int threads() const noexcept { return total_threads_; }

  /* Fan a counted parallel-for out to all workers and block until done.
   * The main caller participates as thread 0 (no extra core needed). */
  void parallel_for(int count, const std::function<void(int)>& body);

 private:
  void worker_loop(int id);

  int total_threads_;
  std::atomic<bool>                exit_{false};

  /* Job state — set by parallel_for, consumed by workers. */
  std::atomic<int>                 next_{0};
  int                              job_count_{0};
  const std::function<void(int)>*  job_body_{nullptr};
  std::atomic<int>                 in_flight_{0};

  /* Wake-up. */
  std::atomic<uint64_t>            job_epoch_{0};
  std::mutex                       mtx_;
  std::condition_variable          cv_start_;
  std::condition_variable          cv_done_;

  std::vector<std::thread>         workers_;
};

}  // namespace cpu_gemm
#endif
