/*
 * Portable work-stealing thread pool — std::thread + std::atomic only.
 *
 * Mirrors the shape of InNumaPool from ktransformers/cpu_backend/worker_pool.h
 * (the do_work_stealing_job(N, fn) primitive) but drops the NUMA / hwloc
 * dependency. A NUMA-aware variant can be added later behind
 * CPU_GEMM_WITH_NUMA without changing the dispatcher.
 *
 * Waiting is spin-then-sleep: workers (and the parallel_for caller) busy-poll
 * for CG_POOL_SPIN_US microseconds (default 200) before falling back to the
 * condition variable. MoE decode publishes a job every few hundred
 * microseconds and a condvar wake costs tens of them per thread; within the
 * spin window dispatch costs ~1us. Idle pools fall asleep after the window,
 * so an idle server does not burn cores. CG_POOL_SPIN_US=0 restores pure
 * condvar waiting.
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

  /* Publish a job WITHOUT participating or blocking: only the pool's spawned
   * workers run it; pair with wait_done(). Lets one thread drive two pools
   * concurrently (submit to B, parallel_for on A, wait_done on B) without a
   * per-call helper thread. One job in flight per pool at a time. `body`
   * must outlive the wait_done() call — pass a named object, NOT a
   * temporary lambda (submit stores a pointer and returns immediately).
   * A 1-thread pool has no workers: submit runs the job inline instead. */
  void submit(int count, const std::function<void(int)>& body);
  void wait_done();

 private:
  void worker_loop(int id);
  bool drain_spin() const;   // true if in_flight_ hit 0 within the budget

  int total_threads_;
  std::atomic<bool>                exit_{false};

  /* Job state — set by parallel_for/submit, consumed by workers. */
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
