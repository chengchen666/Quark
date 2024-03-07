
import signal
import sys

import worker
import torch
import torch.nn  as nn
import pickle

#@qactor(memory=1000， )
class C1:
    def __init__(self, model):
        self.model = model
        print("c1.init ...")

    def run_model(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        print("c1.run_model ...", reqid, deserializaed_tensor)
        output = self.model(deserializaed_tensor)
        worker.ActorSystem.send(worker, "worker2", "run_model", reqid, output)
        #worker.ActorSystem.send("httpactor", "send", reqid, data)
    
class C2:
    def __init__(self, model):
        self.model = model

    def run_model(self, reqid, data):
        deserializaed_tensor = pickle.loads(bytes(data))
        print("c2.run_model ...", reqid, deserializaed_tensor)
        output = self.model(deserializaed_tensor)
        
        print("finish", output)
        #worker.ActorSystem.send("httpactor", "send", reqid, output)

# create C1 and C2 first, run model first part in C1, then C1 will send to C2 to continue execute second part
# 给C1, C2 模型结构 （各执行一半），actor接受tensor
# worker 与模型绑定？ 或是只是纯粹的worker？
def main():
    tensor = torch.randn(1, 784)
    system = worker.ActorSystem()
    model1 = MyNet1().cuda()
    model2 = MyNet2().cuda()
    system.new_actor("worker1", C1, model1)
    #system.set_actor_dev("worker1", 0)
    system.new_actor("worker2", C2, model2)
    #system.set_actor_dev("worker1", 1)
    #system.new_http_actor("httpactor", "worker1", "run_model", 9876)
    system.send("worker1", "run_model", 1, tensor)
    system.wait()


class MyNet1(nn.Module):
    def __init__(self):
        super(MyNet1, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.fc1.weight.data = torch.full((784, 512), 0.5)
        self.fc1.weight.data = torch.full((512, 10), 0.8)
 
    def forward(self, x):
        #x = x.view(-1, 784)
        #x = self.fc1(x)
        x = self.relu(x)
        #x = self.fc2(x)
        return x

class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.fc1.weight.data = torch.full((784, 512), 0.5)
        self.fc1.weight.data = torch.full((512, 10), 0.8)
 
    def forward(self, x):
        #x = x.view(-1, 784)
        #x = self.fc1(x)
        x = self.relu(x)
        #x = self.fc2(x)
        return x

main()