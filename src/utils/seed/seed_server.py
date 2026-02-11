# Copyright 2025 POLAR Team and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
SEED Score Server - 管理器版本

设计：
- N个worker进程，每个有独立的任务队列和结果队列
- 1个管理器线程轮询所有worker
- 管理器逻辑：
  - 空闲worker + 任务数<100 → 分发任务
  - 空闲worker + 任务数>=100 → 重启worker，再分发
  - 工作中 + 超时 → kill并重启，返回默认分数
  - 工作中 + 未超时 → 跳过
"""

import argparse
import sys
import os
import json
import logging
import gc
import time
import signal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import traceback
import multiprocessing
from multiprocessing import Process, Queue
from queue import Empty
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('SEEDServer')

MAX_EXPR_LENGTH = 1000
MAX_TASKS_PER_WORKER = 100


# ============================================================================
# Worker进程
# ============================================================================

def _worker_loop(task_queue: Queue, result_queue: Queue, worker_id: int):
    """Worker主循环"""
    # 初始化
    try:
        import src.utils.seed as seed_module
        import src.utils.latex_pre_process as latex_module
        seed_module._DISABLE_SIGNAL_TIMEOUT = True
        latex_module._DISABLE_SIGNAL_TIMEOUT = True
        from src.utils.seed import SEED
        seed_func = SEED
    except Exception as e:
        logger.error(f"Worker {worker_id} init failed: {e}")
        seed_func = None
    
    while True:
        try:
            # 阻塞等待任务
            task = task_queue.get()
            if task is None:  # 退出信号
                break
            
            task_id, reference, output, expr_type = task
            
            # 计算
            try:
                if seed_func is None:
                    result = (task_id, 0.0, -1.0, -1.0, -1.0, "Worker not initialized")
                elif output is None or len(str(output).strip()) == 0:
                    result = (task_id, 0.0, -1.0, -1.0, -1.0, "Empty output")
                elif len(str(output)) > MAX_EXPR_LENGTH:
                    result = (task_id, 0.0, -1.0, -1.0, -1.0, "Output too long")
                else:
                    score, rel_dist, tree_size, distance = seed_func(reference, output, expr_type)
                    result = (
                        task_id,
                        float(score) if score is not None else 0.0,
                        float(rel_dist) if rel_dist is not None else -1.0,
                        float(tree_size) if tree_size is not None else -1.0,
                        float(distance) if distance is not None else -1.0,
                        None
                    )
            except Exception as e:
                result = (task_id, 0.0, -1.0, -1.0, -1.0, str(e)[:100])
            
            result_queue.put(result)
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")


@dataclass
class WorkerState:
    """Worker状态"""
    worker_id: int
    process: Optional[Process] = None
    task_queue: Optional[Queue] = None
    result_queue: Optional[Queue] = None
    task_count: int = 0  # 已完成任务数
    current_task_id: Optional[str] = None  # 当前任务ID
    task_start_time: float = 0.0  # 当前任务开始时间
    
    def is_idle(self) -> bool:
        return self.current_task_id is None
    
    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


@dataclass
class SEEDRequest:
    reference: str
    output: str
    expr_type: str = "Expression"


@dataclass
class SEEDResponse:
    score: float
    relative_distance: float
    tree_size: float
    distance: float
    error: Optional[str] = None


@dataclass
class PendingTask:
    """待处理任务"""
    task_id: str
    request: SEEDRequest
    result_event: threading.Event = field(default_factory=threading.Event)
    response: Optional[SEEDResponse] = None


class SEEDComputer:
    """管理器版本的SEED计算引擎"""
    
    def __init__(self, max_workers: int = 32, task_timeout: float = 300.0):
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        
        # Worker状态
        self.workers: List[WorkerState] = []
        
        # 全局任务队列
        self.pending_tasks: Dict[str, PendingTask] = {}  # task_id -> PendingTask
        self.task_queue: List[PendingTask] = []  # 等待分发的任务
        self._lock = threading.Lock()
        
        # 统计
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'timeout': 0, 'restarts': 0}
        
        # 初始化workers
        for i in range(max_workers):
            ws = WorkerState(worker_id=i)
            self._start_worker(ws)
            self.workers.append(ws)
        
        # 启动管理器线程
        self._running = True
        self._manager_thread = threading.Thread(target=self._manager_loop, daemon=True)
        self._manager_thread.start()
        
        logger.info(f"SEEDComputer(Managed): {max_workers} workers, timeout={task_timeout}s")
    
    def _start_worker(self, ws: WorkerState):
        """启动一个worker"""
        ws.task_queue = multiprocessing.Queue()
        ws.result_queue = multiprocessing.Queue()
        ws.process = Process(
            target=_worker_loop,
            args=(ws.task_queue, ws.result_queue, ws.worker_id),
            daemon=True
        )
        ws.process.start()
        ws.task_count = 0
        ws.current_task_id = None
        ws.task_start_time = 0.0
    
    def _restart_worker(self, ws: WorkerState):
        """重启一个worker"""
        # 终止旧进程
        if ws.process and ws.process.is_alive():
            ws.process.terminate()
            ws.process.join(timeout=1)
            if ws.process.is_alive():
                try:
                    os.kill(ws.process.pid, signal.SIGKILL)
                except:
                    pass
        
        # 清理队列
        if ws.task_queue:
            ws.task_queue.close()
        if ws.result_queue:
            ws.result_queue.close()
        
        # 启动新进程
        self._start_worker(ws)
        self.stats['restarts'] += 1
    
    def _manager_loop(self):
        """管理器主循环"""
        while self._running:
            try:
                with self._lock:
                    for ws in self.workers:
                        # 1. 检查结果队列
                        self._collect_results(ws)
                        
                        # 2. 处理worker状态
                        if ws.is_idle():
                            # 空闲worker
                            if ws.task_count >= MAX_TASKS_PER_WORKER:
                                # 需要重启
                                self._restart_worker(ws)
                            
                            # 分发任务
                            if self.task_queue:
                                task = self.task_queue.pop(0)
                                self._dispatch_task(ws, task)
                        else:
                            # 工作中的worker
                            elapsed = time.time() - ws.task_start_time
                            if elapsed > self.task_timeout:
                                # 超时
                                self._handle_timeout(ws)
                
                # 短暂sleep避免CPU空转（100ms轮询）
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Manager error: {e}\n{traceback.format_exc()}")
                time.sleep(0.1)
    
    def _collect_results(self, ws: WorkerState):
        """收集worker的结果"""
        try:
            while True:
                result = ws.result_queue.get_nowait()
                task_id, score, rel_dist, tree_size, distance, error = result
                
                if task_id in self.pending_tasks:
                    task = self.pending_tasks[task_id]
                    task.response = SEEDResponse(score, rel_dist, tree_size, distance, error)
                    task.result_event.set()
                    del self.pending_tasks[task_id]
                    
                    if error:
                        self.stats['failed'] += 1
                    else:
                        self.stats['success'] += 1
                
                # 任务完成，worker变为空闲
                if ws.current_task_id == task_id:
                    ws.current_task_id = None
                    ws.task_count += 1
                    
        except Empty:
            pass
    
    def _dispatch_task(self, ws: WorkerState, task: PendingTask):
        """分发任务到worker"""
        ws.current_task_id = task.task_id
        ws.task_start_time = time.time()
        ws.task_queue.put((
            task.task_id,
            task.request.reference,
            task.request.output,
            task.request.expr_type
        ))
    
    def _handle_timeout(self, ws: WorkerState):
        """处理超时"""
        task_id = ws.current_task_id
        
        # 返回默认分数
        if task_id and task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            task.response = SEEDResponse(0.0, -1.0, -1.0, -1.0, f"Timeout({self.task_timeout}s)")
            task.result_event.set()
            del self.pending_tasks[task_id]
            self.stats['timeout'] += 1
        
        # 重启worker
        logger.warning(f"Worker {ws.worker_id} timeout, restarting")
        self._restart_worker(ws)
    
    def compute_batch(self, requests: List[SEEDRequest]) -> List[SEEDResponse]:
        """批量计算"""
        n = len(requests)
        
        # 创建任务
        tasks = []
        for req in requests:
            task_id = str(uuid.uuid4())
            task = PendingTask(task_id=task_id, request=req)
            tasks.append(task)
        
        # 提交任务
        with self._lock:
            self.stats['total'] += n
            for task in tasks:
                self.pending_tasks[task.task_id] = task
                self.task_queue.append(task)
        
        # 等待所有结果
        for task in tasks:
            task.result_event.wait()
        
        return [task.response for task in tasks]
    
    def get_stats(self) -> Dict[str, Any]:
        import psutil
        proc = psutil.Process()
        children = proc.children(recursive=True)
        
        with self._lock:
            stats = dict(self.stats)
            stats['pending'] = len(self.task_queue)
            stats['in_progress'] = sum(1 for ws in self.workers if not ws.is_idle())
            stats['idle_workers'] = sum(1 for ws in self.workers if ws.is_idle())
        
        stats['memory_mb'] = proc.memory_info().rss / 1024 / 1024
        stats['workers'] = len(children)
        stats['children_memory_mb'] = sum(c.memory_info().rss for c in children) / 1024 / 1024 if children else 0
        return stats
    
    def shutdown(self):
        self._running = False
        for ws in self.workers:
            if ws.task_queue:
                ws.task_queue.put(None)  # 退出信号
            if ws.process and ws.process.is_alive():
                ws.process.terminate()
                ws.process.join(timeout=1)


_computer: Optional[SEEDComputer] = None


def get_computer(max_workers: int = 32, task_timeout: float = 300.0) -> SEEDComputer:
    global _computer
    if _computer is None:
        _computer = SEEDComputer(max_workers, task_timeout)
    return _computer


class Handler(BaseHTTPRequestHandler):
    max_workers = 32
    task_timeout = 300.0
    
    def log_message(self, fmt, *args):
        pass
    
    def _json(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except:
            pass
    
    def do_GET(self):
        if self.path == '/health':
            self._json({'status': 'ok'})
        elif self.path == '/stats':
            self._json(get_computer(self.max_workers, self.task_timeout).get_stats())
        elif self.path == '/gc':
            gc.collect()
            self._json({'status': 'gc done'})
        else:
            self._json({'error': 'not found'}, 404)
    
    def do_POST(self):
        if self.path == '/compute':
            self._compute_single()
        elif self.path == '/compute_batch':
            self._compute_batch()
        else:
            self._json({'error': 'not found'}, 404)
    
    def _compute_single(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            req = SEEDRequest(data['reference'], data['output'], data.get('type', 'Expression'))
            resp = get_computer(self.max_workers, self.task_timeout).compute_batch([req])[0]
            self._json({
                'score': resp.score,
                'relative_distance': resp.relative_distance,
                'tree_size': resp.tree_size,
                'distance': resp.distance,
                'error': resp.error
            })
        except Exception as e:
            self._json({'error': str(e)}, 500)
    
    def _compute_batch(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            
            reqs = []
            if 'items' in data:
                for item in data['items']:
                    reqs.append(SEEDRequest(item['reference'], item['output'], item.get('type', 'Expression')))
            elif 'references' in data:
                refs = data['references']
                outs = data['outputs']
                types = data.get('types', ['Expression'] * len(refs))
                for r, o, t in zip(refs, outs, types):
                    reqs.append(SEEDRequest(r, o, t))
            
            responses = get_computer(self.max_workers, self.task_timeout).compute_batch(reqs)
            
            results = []
            scores = []
            for resp in responses:
                results.append({
                    'score': resp.score,
                    'relative_distance': resp.relative_distance,
                    'tree_size': resp.tree_size,
                    'distance': resp.distance,
                    'error': resp.error
                })
                scores.append(resp.score)
            
            self._json({'results': results, 'scores': scores})
        except Exception as e:
            logger.error(f"Batch error: {e}\n{traceback.format_exc()}")
            self._json({'error': str(e)}, 500)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=30001)
    parser.add_argument('--max-workers', type=int, default=32)
    parser.add_argument('--task-timeout', type=float, default=300.0)
    parser.add_argument('--timeout', type=float, default=None)
    parser.add_argument('--max-tasks-per-child', type=int, default=None)
    args = parser.parse_args()
    
    timeout = args.task_timeout if args.timeout is None else args.timeout
    
    Handler.max_workers = args.max_workers
    Handler.task_timeout = timeout
    
    global _computer
    _computer = SEEDComputer(args.max_workers, timeout)
    
    server = ThreadedServer((args.host, args.port), Handler)
    logger.info(f"SEED Server(Managed): {args.host}:{args.port}, workers={args.max_workers}, timeout={timeout}s")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _computer.shutdown()


if __name__ == '__main__':
    main()
