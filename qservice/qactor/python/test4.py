
import signal
import sys
import time
import worker
import torch
import torch.nn  as nn
import pickle
from gptmodel import GPT, GPT_P1, GPT_P2
from transformers import GPT2Tokenizer
from torch.nn import functional as F
from queue import Queue, Empty 
from threading import Thread

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print("Dummpy model, do nothing")
        return x

class Preprocessor:
    def __init__(self, model):
        self.model = model
        print("Preprocessor: init ...")

    def execute(self, reqid, data):
        input_str = ""
        for i in range(len(data)):
            c = chr(data[i])
            input_str = input_str+c
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        indexed_tokens = tokenizer.encode(input_str)
        print("Preprocessor: input phrase:",tokenizer.decode(indexed_tokens), "sending to worker1...")
        tokens_tensor = torch.tensor([indexed_tokens]) 
        print("tokens_tensor:", tokens_tensor)
        worker.ActorSystem.send(worker, "Scheduler", "recv", reqid, tokens_tensor)

class Postprocessor:
    def __init__(self, model):
        self.model = model
        print("Preprocessor: init ...")

    def execute(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        str = tokenizer.decode(deserializaed_tensor[0])
        ascii_str = str.encode('ascii')
        worker.ActorSystem.http_return(worker, "httpactor", "send", reqid, ascii_str)
        #worker.ActorSystem.send(worker, "worker1", "run_model", reqid, tokens_tensor)

class Scheduler:
    def __init__(self, model, max_new_tokens=20, temperature=1.0, do_sample=False, top_k=None):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_k = top_k
        self.queue = Queue() # item: tuple or (reqid, tensor)
        self.thread = Thread(target=self.generate)
        self.thread.start()

    def recv(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        self.queue.put((reqid, deserializaed_tensor))

    def try_fetch_queue_item(self):
        return self.queue.get(True)

    def generate(self):# make it a thread
        (reqid, idx) = self.try_fetch_queue_item()
        for i in range(int(self.max_new_tokens)):
            start_time = time.time()
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.model.block_size else idx[:, -self.model.block_size:]
            # forward the model to get the logits for the index in the sequence
            print("Model_GPT: send to worker1")
            worker.ActorSystem.send(worker, "worker1", "run_model", reqid, idx_cond)
            
            (reqid, logits) = self.try_fetch_queue_item()
            # logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature
            # optionally crop the logits to only the top k options
            if self.top_k is not None:
                v, _ = torch.topk(logits, self.top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if self.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = idx.to("cuda:1")
            idx = torch.cat((idx, idx_next), dim=1)
            print("i is:", i, "time elapse is:", time.time()-start_time)

        worker.ActorSystem.send(worker, "Postprocessor", "execute", reqid, idx, auto_data_movement=False)

class Worker1:
    def __init__(self, model):
        self.model = model
        print("worker1: init ...")

    def run_model(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        print("worker1: run_model ...[req:", reqid,"]")
        output = self.model(deserializaed_tensor)
        worker.ActorSystem.send(worker, "worker2", "run_model", reqid, output)
        print("worker1: finish ", reqid, ", sending to worker2")
    
class Worker2:
    def __init__(self, model):
        self.model = model
        print("worker2: init ...")

    def run_model(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        print("worker2: run_model ...[req:", reqid,"]")
        output = self.model(deserializaed_tensor)
        worker.ActorSystem.send(worker, "Scheduler", "recv", reqid, output, auto_data_movement=False)
        print("worker2: finish ", reqid, ", sending to Scheduler")
        # print(logits.shape)
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # decoded_output = tokenizer.batch_decode(logits.cpu().detach().numpy().argmax(-1).tolist())
        # print(decoded_output)
        # str = ""
        # for i in range(len(decoded_output)):
        #     str = str+decoded_output[i]
        # ascii_str = str.encode('ascii')
        # worker.ActorSystem.http_return(worker, "httpactor", "send", reqid, ascii_str)

# create C1 and C2 first, run model first part in C1, then C1 will send to C2 to continue execute second part
# 给C1, C2 模型结构 （各执行一半），actor接受tensor

def main():
    torch.manual_seed(0)
    system = worker.ActorSystem()
    dummy_model = Dummy
    model = GPT.from_pretrained("gpt2")
    model_p1 = GPT_P1.from_pretrained("gpt2").to("cuda:0")
    model_p2 = GPT_P2.from_pretrained("gpt2").to("cuda:1")

    system.new_actor("Preprocessor", Preprocessor, dummy_model, use_gpu=False)
    system.new_actor("Postprocessor", Postprocessor, dummy_model, use_gpu=False)
    system.new_actor("Scheduler", Scheduler, model, use_gpu=False)
    system.new_actor("worker1", Worker1, model_p1)
    system.new_actor("worker2", Worker2, model_p2)
    # system.new_actor("Model_GPT_HELPER", Model_GPT_HELPER, dummy_model, use_gpu=False)
    system.new_http_actor("httpactor", "Preprocessor", "execute", 9876)
    system.wait()

main()