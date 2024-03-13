from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from functools import wraps, partial
import threading
import queue
import qactor
import pickle

class ActorSystem:
    def __init__(self):
        signal.signal(signal.SIGINT, signal_handler)
        self.tasks = []
        self.actorInstances = dict()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.curr_dev = 0
    
    def new_actor(self, actorId, cls, model, use_gpu=True):
        moduleName = cls.__module__
        className = cls.__qualname__
    
        myqueue = queue.Queue()
        # if deploy on GPU, device should start from 0, if deploy on cpu, should set to -1
        if use_gpu:
            dev = self.curr_dev
            self.curr_dev = self.curr_dev + 1
        else:
            dev = -1
        qactor.new_py_actor(actorId, moduleName, className, myqueue, dev)
        actorInst = ActorProxy(actorId, cls, myqueue, model)
        thread = threading.Thread(target = actorInst.process, args=[])
        self.tasks.append(thread)

    def new_http_actor(self, actorId, gatewayActorId, gatewayFunc, httpPort):
        qactor.new_http_actor(actorId, gatewayActorId, gatewayFunc, httpPort)

    def send(self, target, funcName, reqId, data, auto_data_movement=True):
        tensor_dev = data.get_device()
        worker_dev = qactor.get_actor_dev(target)
        # print("worker dev is", worker_dev)
        # print("tensor_dev,", tensor_dev)
        new_data = data
        if auto_data_movement:
            if worker_dev == tensor_dev:
                print("one same device, do nothing")
            else:
                dev_str = "cuda:"+str(worker_dev)
                new_data = data.detach().to(dev_str)
        else:
            pass
            # print("Auto data movement is disabled")
        serialized_tensor = pickle.dumps(new_data)
        qactor.sendto(target, funcName, reqId, serialized_tensor)
    
    def http_return(self, target, funcName, reqId, data):
        #serialized_tensor = pickle.dumps(new_data)
        qactor.sendto(target, funcName, reqId, data)

    def wait(self):
        qactor.depolyment()
        for t in self.tasks:
            t.start()
        for t in self.tasks:
            t.join()
        

class ActorProxy:
    def __init__(self, actorName, cls, queue, model):
        self.actorName = actorName
        self.actorInst = cls(model)
        self.queue = queue

    def process(self):
        while True:
            (func, req_id, data) = self.queue.get()
            func = getattr(self.actorInst, func)
            run = partial(func, req_id, data)
            run()
            
def signal_handler(signal, frame):
    sys.exit(0)

