# os_demo.py
import sys
import threading
import time
import socket
import json
import random
from queue import Queue, Empty
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------
# Data models
# ---------------------------
_process_id_counter = 1


def next_pid():
    global _process_id_counter
    pid = _process_id_counter
    _process_id_counter += 1
    return pid


@dataclass
class SimProcess:
    pid: int
    name: str
    arrival: int
    burst: int
    remaining: int
    priority: int = 0
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    state: str = "NEW"  # NEW, READY, RUNNING, BLOCKED, TERMINATED
    wait_time: int = 0
    turnaround: int = 0
    history: List[tuple] = field(default_factory=list)  # (time, state)

    def to_dict(self):
        return {
            "pid": self.pid,
            "name": self.name,
            "arrival": self.arrival,
            "burst": self.burst,
            "remaining": self.remaining,
            "priority": self.priority,
            "state": self.state,
            "start_time": self.start_time,
            "finish_time": self.finish_time
        }


# ---------------------------
# Scheduling algorithms (simulator-only)
# ---------------------------
class Scheduler:
    """
    Given a list of processes (SimProcess), run scheduling simulation to produce Gantt chart segments.
    Supports FCFS, RR, SJF (non-preemptive), SJF-preemptive optional via flag.
    This scheduler is independent of the GUI real-time threads; it's used to compute scheduling results for visualization.
    """

    def __init__(self, processes: List[SimProcess], algo="FCFS", time_slice=2, preemptive=False):
        self.procs = [self._copy_proc(p) for p in processes]
        self.algo = algo
        self.time_slice = time_slice
        self.preemptive = preemptive
        self.time = 0
        self.gantt = []  # list of (start, finish, pid)
        self.log = []

    def _copy_proc(self, p: SimProcess):
        return SimProcess(pid=p.pid, name=p.name, arrival=p.arrival, burst=p.burst, remaining=p.burst,
                          priority=p.priority)

    def run(self):
        if self.algo == "FCFS":
            self._run_fcfs()
        elif self.algo == "RR":
            self._run_rr()
        elif self.algo == "SJF":
            if self.preemptive:
                self._run_sjf_preemptive()
            else:
                self._run_sjf_nonpreemptive()
        else:
            self._run_fcfs()
        return self.gantt, self._compute_metrics()

    def _run_fcfs(self):
        procs = sorted(self.procs, key=lambda p: p.arrival)
        t = 0
        for p in procs:
            if t < p.arrival:
                t = p.arrival
            start = t
            finish = t + p.burst
            self.gantt.append((start, finish, p.pid))
            p.start_time = start
            p.finish_time = finish
            t = finish

    def _run_rr(self):
        procs = sorted(self.procs, key=lambda p: p.arrival)
        t = 0
        i = 0
        q = deque()
        while True:
            # enqueue newly arrived
            while i < len(procs) and procs[i].arrival <= t:
                q.append(procs[i])
                i += 1
            if not q and i < len(procs):
                t = procs[i].arrival
                continue
            if not q:
                break
            p = q.popleft()
            execute = min(self.time_slice, p.remaining)
            start = t
            finish = t + execute
            self.gantt.append((start, finish, p.pid))
            p.remaining -= execute
            if p.start_time is None:
                p.start_time = start
            t = finish
            # enqueue newly arrived during execution
            while i < len(procs) and procs[i].arrival <= t:
                q.append(procs[i])
                i += 1
            if p.remaining > 0:
                q.append(p)
            else:
                p.finish_time = t

    def _run_sjf_nonpreemptive(self):
        procs = sorted(self.procs, key=lambda p: p.arrival)
        t = 0
        ready = []
        i = 0
        import heapq
        while i < len(procs) or ready:
            while i < len(procs) and procs[i].arrival <= t:
                heapq.heappush(ready, (procs[i].burst, procs[i]))
                i += 1
            if not ready:
                t = procs[i].arrival
                continue
            burst, p = heapq.heappop(ready)
            start = t
            finish = t + p.burst
            self.gantt.append((start, finish, p.pid))
            p.start_time = start
            p.finish_time = finish
            t = finish

    def _run_sjf_preemptive(self):
        procs = sorted(self.procs, key=lambda p: p.arrival)
        t = 0
        i = 0
        import heapq
        ready = []
        last_pid = None
        while i < len(procs) or ready:
            while i < len(procs) and procs[i].arrival <= t:
                heapq.heappush(ready, (procs[i].remaining, procs[i]))
                i += 1
            if not ready:
                if i < len(procs):
                    t = procs[i].arrival
                    continue
                else:
                    break
            rem, p = heapq.heappop(ready)
            # execute 1 unit
            start = t
            finish = t + 1
            self.gantt.append((start, finish, p.pid))
            p.remaining -= 1
            t = finish
            # push newly arrived that arrived at t
            while i < len(procs) and procs[i].arrival <= t:
                heapq.heappush(ready, (procs[i].remaining, procs[i]))
                i += 1
            if p.remaining > 0:
                heapq.heappush(ready, (p.remaining, p))
            else:
                p.finish_time = t

    def _compute_metrics(self):
        # compute average waiting time and turnaround from simulated finish/start
        metrics = {}
        total_wait = 0
        total_turn = 0
        n = len(self.procs)
        for p in self.procs:
            if p.finish_time is None:
                # compute as if ran to completion last
                p.finish_time = max(p.arrival, max([g[1] for g in self.gantt]) if self.gantt else p.arrival) + 0
            p.turnaround = p.finish_time - p.arrival
            p.wait_time = p.turnaround - p.burst
            total_wait += p.wait_time
            total_turn += p.turnaround
        metrics['avg_wait'] = total_wait / n if n else 0
        metrics['avg_turnaround'] = total_turn / n if n else 0
        return metrics


# ---------------------------
# GUI: Matplotlib canvas for Gantt and charts
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


# ---------------------------
# Simulator thread (drives states in "real time" for demo)
# ---------------------------
class Simulator(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(dict)  # send snapshot to GUI
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = 0
        self.processes: List[SimProcess] = []
        self.running = False
        self.step_mode = False
        self.algo = "FCFS"
        self.time_slice = 3
        self.speed = 1.0
        self.lock = threading.Lock()
        self._stop_req = False

        # RR调度专用队列
        self.rr_queue = deque()
        self.current_pid = None
        self.running_since = 0
        self.time_in_slice = 0

    def run(self):
        self.log("Simulator started.")
        self.running = True
        while not self._stop_req:
            if not self.running:
                time.sleep(0.05)
                continue
            with self.lock:
                self._advance_one_unit()
            snapshot = self._make_snapshot()
            self.update_signal.emit(snapshot)
            if self.step_mode:
                self.running = False
            time.sleep(max(0.01, 1.0 / max(0.0001, self.speed)))
        self.log("Simulator stopped.")

    def stop(self):
        self._stop_req = True

    def log(self, msg):
        self.log_signal.emit(f"[t={self.time}] {msg}")

    def add_process(self, p: SimProcess):
        with self.lock:
            self.processes.append(p)
            self.processes.sort(key=lambda x: x.arrival)
            self.log(f"Added process {p.pid} (arrival={p.arrival}, burst={p.burst})")

    def reset(self):
        with self.lock:
            self.time = 0
            self.processes.clear()
            self.rr_queue.clear()
            self.current_pid = None
            self.running_since = 0
            self.time_in_slice = 0
        self.log("Simulator reset.")

    def set_algo(self, algo, time_slice=3):
        with self.lock:
            self.algo = algo
            self.time_slice = time_slice
            self.rr_queue.clear()
            self.current_pid = None
            self.running_since = 0
            self.time_in_slice = 0
            if algo == "RR":
                ready_procs = [p for p in self.processes if p.state in ["READY", "NEW"] and p.remaining > 0]
                ready_procs.sort(key=lambda x: x.arrival)
                for p in ready_procs:
                    p.state = "READY"
                    self.rr_queue.append(p)
            self.log(f"Algorithm changed to {algo}, time_slice={time_slice}")

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed

    def _find_process_by_pid(self, pid):
        for p in self.processes:
            if p.pid == pid:
                return p
        return None

    def _advance_one_unit(self):
        t = self.time

        # 处理新到达进程
        for p in self.processes:
            if p.state == "NEW" and p.arrival <= t:
                p.state = "READY"
                p.history.append((t, "READY"))
                if self.algo == "RR" and p not in self.rr_queue:
                    self.rr_queue.append(p)
                    self.log(f"Process {p.pid} arrived and added to RR queue")

        # 根据算法选择要运行的进程
        current = None

        if self.algo == "RR":
            # 修复：先检查当前运行进程是否应该被抢占
            if self.current_pid is not None:
                current = self._find_process_by_pid(self.current_pid)
                if current and current.state == "RUNNING":
                    # 修复：这里需要先增加时间片计数
                    self.time_in_slice += 1

                    # 检查时间片是否用完
                    if self.time_in_slice > self.time_slice:  # 修复：使用 > 而不是 >=
                        # 时间片用完，让出CPU
                        if current.remaining > 0:
                            current.state = "READY"
                            self.rr_queue.append(current)
                            self.log(
                                f"Process {current.pid} time slice expired ({self.time_in_slice}/{self.time_slice}), requeued")
                        self.current_pid = None
                        self.time_in_slice = 0
                        current = None
                    elif current.remaining <= 0:
                        # 进程已完成
                        current.state = "TERMINATED"
                        current.finish_time = t
                        self.log(f"Process {current.pid} terminated (finish={t})")
                        self.current_pid = None
                        self.time_in_slice = 0
                        current = None
                else:
                    self.current_pid = None
                    self.time_in_slice = 0

            # 从RR队列中选择下一个进程
            if self.current_pid is None:
                # 清理队列
                temp_queue = deque()
                while self.rr_queue:
                    p = self.rr_queue.popleft()
                    if p.state == "READY" and p.remaining > 0:
                        temp_queue.append(p)
                self.rr_queue = temp_queue

                if self.rr_queue:
                    # 从队头取出进程
                    current = self.rr_queue.popleft()
                    if current.remaining > 0 and current.state == "READY":
                        current.state = "RUNNING"
                        self.current_pid = current.pid
                        self.running_since = t
                        self.time_in_slice = 1  # 注意：这里初始化为1，因为即将执行1个单位时间
                        if current.start_time is None:
                            current.start_time = t
                        self.log(f"Process {current.pid} started running, time_slice={self.time_slice}")
                    else:
                        current = None
                        self.current_pid = None

        elif self.algo == "FCFS":
            ready = [p for p in self.processes if p.state == "READY"]
            running = [p for p in self.processes if p.state == "RUNNING"]

            if not running and ready:
                ready.sort(key=lambda x: x.arrival)
                current = ready[0]
                current.state = "RUNNING"
                if current.start_time is None:
                    current.start_time = t
            elif running:
                current = running[0]

        elif self.algo == "SJF":
            ready = [p for p in self.processes if p.state == "READY"]
            running = [p for p in self.processes if p.state == "RUNNING"]

            if not running and ready:
                ready.sort(key=lambda x: x.remaining)
                current = ready[0]
                current.state = "RUNNING"
                if current.start_time is None:
                    current.start_time = t
            elif running:
                current = running[0]

        # 执行当前进程
        if current and current.state == "RUNNING":
            # 执行1个单位时间
            current.remaining -= 1

            if self.algo == "RR" and random.random() < 0.02:
                current.state = "BLOCKED"
                self.current_pid = None
                self.time_in_slice = 0
                self.log(f"Process {current.pid} blocked")

            elif current.remaining <= 0:
                current.state = "TERMINATED"
                current.finish_time = t + 1
                if self.algo == "RR":
                    self.current_pid = None
                    self.time_in_slice = 0
                self.log(f"Process {current.pid} terminated (finish={t + 1})")

            if not current.history or current.history[-1][1] != "RUNNING":
                current.history.append((t, "RUNNING"))

        # 阻塞进程恢复
        for p in self.processes:
            if p.state == "BLOCKED" and random.random() < 0.3:
                p.state = "READY"
                p.history.append((t + 1, "READY"))
                if self.algo == "RR" and p not in self.rr_queue:
                    self.rr_queue.append(p)
                self.log(f"Process {p.pid} unblocked")

        self.time += 1

    def _make_snapshot(self):
        with self.lock:
            procs = [p.to_dict() for p in self.processes]
        return {"time": self.time, "processes": procs, "algo": self.algo, "time_slice": self.time_slice}


# ---------------------------
# IPC Demo: TCP producer-consumer and in-process queue demo
# ---------------------------
class IPCDemo:
    def __init__(self, gui_log_callback=None):
        self.server_thread = None
        self.server_socket = None
        self.port = 50007
        self.running = False
        self.queue = Queue(maxsize=10)
        self.gui_log = gui_log_callback or (lambda s: None)
        self.producer_thread = None
        self.consumer_thread = None

    def start_tcp_server(self):
        if self.running: return
        self.running = True
        self.server_thread = threading.Thread(target=self._tcp_server, daemon=True)
        self.server_thread.start()
        self.gui_log("IPC TCP server started.")

    def _tcp_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', self.port))
        s.listen(1)
        self.server_socket = s
        while self.running:
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
            except socket.timeout:
                continue
            with conn:
                self.gui_log(f"IPC server: connection from {addr}")
                while self.running:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        msg = data.decode('utf-8')
                        self.gui_log(f"IPC server recv: {msg}")
                    except Exception:
                        break

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        self.gui_log("IPC server stopped.")

    def start_producer_consumer_queue(self):
        if self.producer_thread and self.producer_thread.is_alive():
            return
        self.producer_thread = threading.Thread(target=self._producer_worker, daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer_worker, daemon=True)
        self.producer_thread.start()
        self.consumer_thread.start()
        self.gui_log("IPC queue producer/consumer started.")

    def _producer_worker(self):
        i = 0
        while True:
            try:
                item = f"msg-{i}"
                self.queue.put(item, timeout=1)
                self.gui_log(f"Producer put {item} (size={self.queue.qsize()})")
                i += 1
            except Exception as e:
                self.gui_log("Producer blocked (queue full).")
            time.sleep(0.5)

    def _consumer_worker(self):
        while True:
            try:
                item = self.queue.get(timeout=2)
                self.gui_log(f"Consumer got {item} (size={self.queue.qsize()})")
            except Empty:
                self.gui_log("Consumer waiting (queue empty).")
            time.sleep(0.8)


# ---------------------------
# Semaphore demo (Philosopher style)
# ---------------------------
class SemaphoreDemo:
    def __init__(self, gui_log_callback=None):
        self.gui_log = gui_log_callback or (lambda s: None)
        self.forks = [threading.Semaphore(1) for _ in range(5)]
        self.philosophers = []
        self.running = False

    def start(self):
        self.running = True
        for i in range(5):
            t = threading.Thread(target=self._philosopher, args=(i,), daemon=True)
            t.start()
            self.philosophers.append(t)
        self.gui_log("Semaphore (Dining Philosophers) started.")

    def _philosopher(self, idx):
        left = idx
        right = (idx + 1) % 5
        while self.running:
            self.gui_log(f"Philosopher {idx} thinking")
            time.sleep(random.uniform(0.5, 1.5))
            a, b = (left, right) if idx % 2 == 0 else (right, left)
            self.gui_log(f"Philosopher {idx} attempting to acquire fork {a}")
            self.forks[a].acquire()
            self.gui_log(f"Philosopher {idx} got fork {a}")
            time.sleep(0.1)
            self.gui_log(f"Philosopher {idx} attempting to acquire fork {b}")
            self.forks[b].acquire()
            self.gui_log(f"Philosopher {idx} got fork {b} - eating")
            time.sleep(random.uniform(0.5, 1.2))
            self.forks[a].release()
            self.forks[b].release()
            self.gui_log(f"Philosopher {idx} released forks {a},{b}")

    def stop(self):
        self.running = False
        self.gui_log("Semaphore demo stopped.")


# ---------------------------
# Reader-Writer Problem Demo (修复版)
# ---------------------------
class ReaderWriterDemo:
    """读者-写者问题的经典实现（修复死锁版）"""

    def __init__(self, gui_log_callback=None):
        self.gui_log = gui_log_callback or (lambda s: None)

        # 共享数据
        self.data = 0
        self.data_lock = threading.Lock()

        # 读者统计
        self.read_count = 0
        self.read_count_lock = threading.Lock()

        # 写者统计
        self.write_count = 0
        self.write_count_lock = threading.Lock()

        # 读者优先的信号量
        self.read_mutex = threading.Semaphore(1)  # 保护read_count
        self.resource_lock = threading.Semaphore(1)  # 保护共享数据

        # 写者优先的信号量
        self.write_mutex = threading.Semaphore(1)  # 保护write_count
        self.read_room_empty = threading.Semaphore(1)  # 读者进入控制
        self.write_lock = threading.Semaphore(1)  # 写者互斥

        # 线程管理
        self.readers = []
        self.writers = []
        self.running = False

        # 统计信息
        self.reads_completed = 0
        self.writes_completed = 0
        self.waiting_readers = 0
        self.waiting_writers = 0

    def start_reader_priority(self, num_readers=3, num_writers=2):
        """启动读者优先的读者-写者问题演示"""
        if self.running:
            return

        self.running = True
        self.data = 0
        self.read_count = 0
        self.write_count = 0
        self.reads_completed = 0
        self.writes_completed = 0
        self.waiting_readers = 0
        self.waiting_writers = 0

        # 创建读者线程
        self.readers = []
        for i in range(num_readers):
            reader = threading.Thread(
                target=self._reader_worker_reader_priority,
                args=(i,),
                daemon=True
            )
            reader.start()
            self.readers.append(reader)

        # 创建写者线程
        self.writers = []
        for i in range(num_writers):
            writer = threading.Thread(
                target=self._writer_worker_reader_priority,
                args=(i,),
                daemon=True
            )
            writer.start()
            self.writers.append(writer)

        self.gui_log(f"读者-写者问题（读者优先）启动: {num_readers} 读者, {num_writers} 写者")

    def start_writer_priority(self, num_readers=3, num_writers=2):
        """启动写者优先的读者-写者问题演示"""
        if self.running:
            return

        self.running = True
        self.data = 0
        self.read_count = 0
        self.write_count = 0
        self.reads_completed = 0
        self.writes_completed = 0
        self.waiting_readers = 0
        self.waiting_writers = 0

        # 创建读者线程
        self.readers = []
        for i in range(num_readers):
            reader = threading.Thread(
                target=self._reader_worker_writer_priority,
                args=(i,),
                daemon=True
            )
            reader.start()
            self.readers.append(reader)

        # 创建写者线程
        self.writers = []
        for i in range(num_writers):
            writer = threading.Thread(
                target=self._writer_worker_writer_priority,
                args=(i,),
                daemon=True
            )
            writer.start()
            self.writers.append(writer)

        self.gui_log(f"读者-写者问题（写者优先）启动: {num_readers} 读者, {num_writers} 写者")

    def _reader_worker_reader_priority(self, reader_id):
        """读者优先的读者线程"""
        while self.running:
            # 登记等待读者
            with self.read_count_lock:
                self.waiting_readers += 1

            # 读者进入协议
            self.read_mutex.acquire()
            if self.read_count == 0:
                self.resource_lock.acquire()  # 第一个读者锁住资源
            self.read_count += 1

            with self.read_count_lock:
                self.waiting_readers -= 1

            self.read_mutex.release()

            # 读取数据
            with self.data_lock:
                data_copy = self.data
            time.sleep(random.uniform(0.2, 0.8))
            self.gui_log(f"读者 {reader_id} 读取数据: {data_copy}")

            # 读者离开协议
            self.read_mutex.acquire()
            self.read_count -= 1
            if self.read_count == 0:
                self.resource_lock.release()  # 最后一个读者释放资源
            self.read_mutex.release()

            with self.read_count_lock:
                self.reads_completed += 1

            # 休息
            time.sleep(random.uniform(1.0, 2.0))

    def _writer_worker_reader_priority(self, writer_id):
        """读者优先的写者线程"""
        while self.running:
            # 登记等待写者
            with self.write_count_lock:
                self.waiting_writers += 1

            # 写者必须独占资源
            self.resource_lock.acquire()

            with self.write_count_lock:
                self.waiting_writers -= 1

            # 写入数据
            with self.data_lock:
                new_value = self.data + 1
                time.sleep(random.uniform(0.3, 0.7))
                self.data = new_value
            self.gui_log(f"写者 {writer_id} 写入数据: {new_value}")

            # 释放资源
            self.resource_lock.release()

            with self.write_count_lock:
                self.writes_completed += 1

            # 休息
            time.sleep(random.uniform(1.5, 2.5))

    def _reader_worker_writer_priority(self, reader_id):
        """写者优先的读者线程"""
        while self.running:
            # 登记等待读者
            with self.read_count_lock:
                self.waiting_readers += 1

            # 检查是否有写者在等待
            self.read_room_empty.acquire()
            self.read_room_empty.release()

            # 读者进入协议
            self.read_mutex.acquire()
            if self.read_count == 0:
                self.write_lock.acquire()  # 第一个读者锁住写者
            self.read_count += 1

            with self.read_count_lock:
                self.waiting_readers -= 1

            self.read_mutex.release()

            # 读取数据
            with self.data_lock:
                data_copy = self.data
            time.sleep(random.uniform(0.2, 0.8))
            self.gui_log(f"读者 {reader_id} 读取数据: {data_copy} (写者优先)")

            # 读者离开协议
            self.read_mutex.acquire()
            self.read_count -= 1
            if self.read_count == 0:
                self.write_lock.release()  # 最后一个读者允许写者
            self.read_mutex.release()

            with self.read_count_lock:
                self.reads_completed += 1

            # 休息
            time.sleep(random.uniform(1.0, 2.0))

    def _writer_worker_writer_priority(self, writer_id):
        """写者优先的写者线程（修复死锁版）"""
        while self.running:
            # 登记等待写者
            with self.write_count_lock:
                self.waiting_writers += 1
                if self.waiting_writers == 1:
                    # 第一个写者阻止新读者
                    self.read_room_empty.acquire()

            # 获取写锁
            self.write_lock.acquire()

            with self.write_count_lock:
                self.waiting_writers -= 1

            # 写入数据
            with self.data_lock:
                new_value = self.data + 1
                time.sleep(random.uniform(0.3, 0.7))
                self.data = new_value
            self.gui_log(f"写者 {writer_id} 写入数据: {new_value} (写者优先)")

            # 释放写锁
            self.write_lock.release()

            # 检查是否还有写者在等待
            with self.write_count_lock:
                self.writes_completed += 1
                if self.waiting_writers == 0:
                    # 没有写者等待，允许读者进入
                    self.read_room_empty.release()

            # 休息
            time.sleep(random.uniform(1.5, 2.5))

    def stop(self):
        """停止读者-写者演示"""
        self.running = False
        self.gui_log("读者-写者问题演示已停止")

    def get_stats(self):
        """获取统计信息"""
        with self.read_count_lock:
            with self.write_count_lock:
                return {
                    "reads": self.reads_completed,
                    "writes": self.writes_completed,
                    "waiting_readers": self.waiting_readers,
                    "waiting_writers": self.waiting_writers,
                    "active_readers": self.read_count
                }


# ---------------------------
# Main GUI Application
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("操作系统实验展示软件 - Demo")
        self.resize(1200, 900)  # 增加高度以容纳新控件
        self.sim = Simulator()
        self.sim.update_signal.connect(self.on_sim_update)
        self.sim.log_signal.connect(self.append_log)
        self.ipc = IPCDemo(gui_log_callback=self.append_log)
        self.sem = SemaphoreDemo(gui_log_callback=self.append_log)
        self.rw = ReaderWriterDemo(gui_log_callback=self.append_log)

        self.init_ui()
        self.sim.start()
        self.rw_stats_timer = QtCore.QTimer()
        self.rw_stats_timer.timeout.connect(self.update_rw_stats)
        self.rw_stats_timer.start(1000)  # 每秒更新一次统计信息

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: Controls and status
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 2)

        # Process creation panel
        grp_proc = QtWidgets.QGroupBox("创建进程 / 线程")
        left.addWidget(grp_proc)
        form = QtWidgets.QFormLayout(grp_proc)
        self.input_name = QtWidgets.QLineEdit("P")
        self.input_arrival = QtWidgets.QSpinBox();
        self.input_arrival.setRange(0, 1000)
        self.input_burst = QtWidgets.QSpinBox();
        self.input_burst.setRange(1, 1000);
        self.input_burst.setValue(5)
        self.input_priority = QtWidgets.QSpinBox();
        self.input_priority.setRange(0, 10)
        form.addRow("名称前缀:", self.input_name)
        form.addRow("到达时间:", self.input_arrival)
        form.addRow("执行时间:", self.input_burst)
        form.addRow("优先级:", self.input_priority)
        hbtn = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("添加进程")
        self.btn_add.clicked.connect(self.add_process)
        hbtn.addWidget(self.btn_add)
        self.btn_reset = QtWidgets.QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_sim)
        hbtn.addWidget(self.btn_reset)
        form.addRow(hbtn)

        # Scheduler controls
        grp_sched = QtWidgets.QGroupBox("调度算法与控制")
        left.addWidget(grp_sched)
        v = QtWidgets.QVBoxLayout(grp_sched)
        self.combo_algo = QtWidgets.QComboBox()
        self.combo_algo.addItems(["FCFS", "RR", "SJF"])
        self.combo_algo.currentIndexChanged.connect(self.change_algo)
        v.addWidget(self.combo_algo)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("时间片 (RR):"))
        self.spin_ts = QtWidgets.QSpinBox();
        self.spin_ts.setRange(1, 100);
        self.spin_ts.setValue(3)
        self.spin_ts.valueChanged.connect(self.on_time_slice_changed)
        h2.addWidget(self.spin_ts)
        v.addLayout(h2)
        h3 = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("运行")
        self.btn_run.clicked.connect(self.run_sim)
        self.btn_pause = QtWidgets.QPushButton("暂停")
        self.btn_pause.clicked.connect(self.pause_sim)
        self.btn_step = QtWidgets.QPushButton("单步")
        self.btn_step.clicked.connect(self.step_sim)
        h3.addWidget(self.btn_run);
        h3.addWidget(self.btn_pause);
        h3.addWidget(self.btn_step)
        v.addLayout(h3)
        v.addWidget(QtWidgets.QLabel("模拟速度 (越大越快)"))
        self.slider_speed = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 20);
        self.slider_speed.setValue(5)
        self.slider_speed.valueChanged.connect(self.change_speed)
        v.addWidget(self.slider_speed)

        # IPC & Semaphore controls
        grp_ipc = QtWidgets.QGroupBox("IPC 与 信号量 演示")
        left.addWidget(grp_ipc)
        v2 = QtWidgets.QVBoxLayout(grp_ipc)
        self.btn_start_ipc = QtWidgets.QPushButton("启动 IPC (TCP server)")
        self.btn_start_ipc.clicked.connect(lambda: self.ipc.start_tcp_server())
        v2.addWidget(self.btn_start_ipc)
        self.btn_start_queue = QtWidgets.QPushButton("启动 队列 Producer/Consumer")
        self.btn_start_queue.clicked.connect(lambda: self.ipc.start_producer_consumer_queue())
        v2.addWidget(self.btn_start_queue)
        self.btn_start_sem = QtWidgets.QPushButton("启动 哲学家信号量演示")
        self.btn_start_sem.clicked.connect(lambda: self.sem.start())
        v2.addWidget(self.btn_start_sem)

        # Reader-Writer Problem Controls
        grp_rw = QtWidgets.QGroupBox("读者-写者问题演示")
        left.addWidget(grp_rw)
        v3 = QtWidgets.QVBoxLayout(grp_rw)

        # 读者优先按钮
        h_rw1 = QtWidgets.QHBoxLayout()
        h_rw1.addWidget(QtWidgets.QLabel("读者优先:"))
        self.btn_start_reader_priority = QtWidgets.QPushButton("启动")
        self.btn_start_reader_priority.clicked.connect(self.start_reader_priority)
        h_rw1.addWidget(self.btn_start_reader_priority)
        v3.addLayout(h_rw1)

        # 写者优先按钮
        h_rw2 = QtWidgets.QHBoxLayout()
        h_rw2.addWidget(QtWidgets.QLabel("写者优先:"))
        self.btn_start_writer_priority = QtWidgets.QPushButton("启动")
        self.btn_start_writer_priority.clicked.connect(self.start_writer_priority)
        h_rw2.addWidget(self.btn_start_writer_priority)
        v3.addLayout(h_rw2)

        # 停止按钮
        self.btn_stop_rw = QtWidgets.QPushButton("停止读者-写者演示")
        self.btn_stop_rw.clicked.connect(self.stop_reader_writer)
        v3.addWidget(self.btn_stop_rw)

        # 读者-写者统计信息显示
        self.rw_stats_label = QtWidgets.QLabel("读者: 0, 写者: 0, 等待读者: 0, 等待写者: 0")
        self.rw_stats_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        v3.addWidget(self.rw_stats_label)

        # Log
        grp_log = QtWidgets.QGroupBox("日志 / 指标")
        left.addWidget(grp_log, 2)
        v4 = QtWidgets.QVBoxLayout(grp_log)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        v4.addWidget(self.log_text)

        # Right: Visualizations
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 5)

        # Process table
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["PID", "Name", "Arrival", "Burst", "Remaining", "State", "Priority"])
        right.addWidget(self.table, 2)

        # Gantt canvas
        self.canvas = MplCanvas(self, width=8, height=3, dpi=100)
        right.addWidget(self.canvas, 3)

        # Metrics display
        metrics_box = QtWidgets.QGroupBox("性能指标")
        right.addWidget(metrics_box)
        self.metrics_label = QtWidgets.QLabel("avg wait: -, avg turnaround: -")
        ml = QtWidgets.QVBoxLayout(metrics_box)
        ml.addWidget(self.metrics_label)

    def add_process(self):
        name_pref = self.input_name.text().strip() or "P"
        arrival = self.input_arrival.value()
        burst = self.input_burst.value()
        prio = self.input_priority.value()
        pid = next_pid()
        proc = SimProcess(pid=pid, name=f"{name_pref}{pid}", arrival=arrival, burst=burst,
                          remaining=burst, priority=prio)
        self.sim.add_process(proc)
        self.append_log(f"Added process {proc.name} (pid={pid})")
        self.refresh_table()

    def reset_sim(self):
        self.sim.reset()
        global _process_id_counter
        _process_id_counter = 1
        self.append_log("Reset simulation and PID counter.")
        self.refresh_table()
        self.canvas.axes.clear()
        self.canvas.draw()

    def change_algo(self):
        algo = self.combo_algo.currentText()
        ts = self.spin_ts.value()
        self.sim.set_algo(algo, time_slice=ts)
        self.append_log(f"Algorithm changed to {algo}, time_slice={ts}")

    def on_time_slice_changed(self, value):
        if self.combo_algo.currentText() == "RR":
            ts = value
            self.sim.set_algo("RR", time_slice=ts)
            self.append_log(f"Time slice changed to {ts}")

    def run_sim(self):
        self.sim.running = True
        self.append_log("Simulator running.")

    def pause_sim(self):
        self.sim.running = False
        self.append_log("Simulator paused.")

    def step_sim(self):
        self.sim.step_mode = True
        self.sim.running = True

    def change_speed(self, val):
        self.sim.set_speed(val)
        self.append_log(f"Speed set to {val}")

    def append_log(self, text):
        t = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{t}] {text}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_rw_stats(self):
        """更新读者-写者统计信息"""
        stats = self.rw.get_stats()
        self.rw_stats_label.setText(
            f"已完成读取: {stats['reads']}, 已完成写入: {stats['writes']}, "
            f"等待读者: {stats['waiting_readers']}, 等待写者: {stats['waiting_writers']}"
        )

    def start_reader_priority(self):
        """启动读者优先演示"""
        self.rw.start_reader_priority(num_readers=3, num_writers=2)
        self.append_log("启动读者-写者问题演示（读者优先模式）")

    def start_writer_priority(self):
        """启动写者优先演示"""
        self.rw.start_writer_priority(num_readers=3, num_writers=2)
        self.append_log("启动读者-写者问题演示（写者优先模式）")

    def stop_reader_writer(self):
        """停止读者-写者演示"""
        self.rw.stop()
        self.append_log("读者-写者问题演示已停止")

    def on_sim_update(self, snapshot):
        self.update_process_table(snapshot["processes"])
        procs = []
        for p in snapshot["processes"]:
            sp = SimProcess(pid=p["pid"], name=p["name"], arrival=p["arrival"],
                            burst=p["burst"], remaining=p["remaining"],
                            priority=p.get("priority", 0))
            procs.append(sp)
        algo = snapshot.get("algo", "FCFS")
        ts = snapshot.get("time_slice", self.spin_ts.value())
        if ts is None:
            ts = self.spin_ts.value()
        sched = Scheduler(procs, algo=algo, time_slice=ts, preemptive=False)
        gantt, metrics = sched.run()
        self.plot_gantt(gantt, procs)
        self.metrics_label.setText(
            f"avg wait: {metrics['avg_wait']:.2f}, avg turnaround: {metrics['avg_turnaround']:.2f}")

    def update_process_table(self, procs):
        self.table.setRowCount(len(procs))
        for i, p in enumerate(procs):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(p["pid"])))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(p["name"])))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(p["arrival"])))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(p["burst"])))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(p["remaining"])))
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(p["state"])))
            self.table.setItem(i, 6, QtWidgets.QTableWidgetItem(str(p.get("priority", 0))))
        self.table.resizeColumnsToContents()

    def refresh_table(self):
        procs = [p.to_dict() for p in self.sim.processes]
        self.update_process_table(procs)

    def plot_gantt(self, gantt, procs):
        ax = self.canvas.axes
        ax.clear()
        if not gantt:
            ax.text(0.5, 0.5, "No schedule yet", ha='center')
            self.canvas.draw()
            return
        pids = sorted({g[2] for g in gantt})
        pid_to_row = {pid: i for i, pid in enumerate(pids)}
        for (start, end, pid) in gantt:
            row = pid_to_row[pid]
            ax.broken_barh([(start, end - start)], (row * 10, 9), facecolors=('tab:blue'))
            ax.text((start + end) / 2, row * 10 + 4.5, f"P{pid}", ha='center', va='center', color='white', fontsize=8)
        ax.set_ylim(0, len(pids) * 10 + 10)
        ax.set_xlim(0, max([g[1] for g in gantt]) + 1)
        ax.set_xlabel("时间")
        ax.set_yticks([i * 10 + 4.5 for i in range(len(pids))])
        ax.set_yticklabels([f"P{pid}" for pid in pids])
        ax.grid(True)
        self.canvas.draw()

    def closeEvent(self, event):
        self.sim.stop()
        self.ipc.stop()
        self.sem.stop()
        self.rw.stop()
        self.rw_stats_timer.stop()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()