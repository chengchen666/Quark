// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeMap;
use std::sync::{Mutex, Arc};
use std::result::Result as SResult;
use std::time::SystemTime;
use qobjs::func::func_agent_service_server::FuncAgentServiceServer;
use qobjs::utility::SystemTimeProto;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};
use tokio::sync::{oneshot, mpsc};
use core::ops::Deref;

use qobjs::{common::*, func::{self, func_agent_msg::EventBody}};

use crate::{FUNC_SVC_CLIENT, FUNC_AGENT};

use super::funcpod::FuncPod;
use super::funcpod_mgr::FuncPodMgr;

#[derive(Debug)]
pub struct FuncCallInner {
    pub id: String,
    pub namespace: String,
    pub packageName: String,
    pub funcName: String,
    pub priority: usize,
    pub parameters: String,
    pub createTime: SystemTime,

    pub callerNodeId: String,
    pub callerFuncPodId: String,
    pub calleeNodeId: String,
    pub calleeFuncPodId: String,
}

impl FuncCallInner {
    pub fn ToGrpcType(&self) -> func::FuncSvcCallReq {
        return func::FuncSvcCallReq {
            id: self.id.clone(),
            namespace: self.namespace.clone(),
            package_name: self.packageName.clone(),
            func_name: self.funcName.clone(),
            priority: self.priority as u64,
            parameters: self.parameters.clone(),
            createtime: Some(SystemTimeProto::FromSystemTime(self.createTime).ToTimeStamp()),
            caller_node_id: self.callerNodeId.clone(),
            calller_pod_id: self.callerFuncPodId.clone(),
            callee_node_id: self.calleeNodeId.clone(),
            callee_pod_id: self.calleeFuncPodId.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuncCall(Arc<FuncCallInner>);

impl Deref for FuncCall {
    type Target = Arc<FuncCallInner>;

    fn deref(&self) -> &Arc<FuncCallInner> {
        &self.0
    }
}

#[derive(Debug)]
pub struct FuncCallContext {
    pub respChann: oneshot::Sender<SResult<String, String>>,
}

#[derive(Debug, Default)]
pub struct FuncAgentInner {
    pub nodeId: String,
    pub callContexts: Mutex<BTreeMap<String, FuncCallContext>>,
    pub funcPodMgr: FuncPodMgr,
    // func instance id to funcCall
    pub callerCalls: Mutex<BTreeMap<String, FuncCall>>,
    // func instance id to funcCall
    pub calleeCalls: Mutex<BTreeMap<String, FuncCall>>,
}

#[derive(Debug, Default, Clone)]
pub struct FuncAgent(Arc<FuncAgentInner>);

impl Deref for FuncAgent {
    type Target = Arc<FuncAgentInner>;

    fn deref(&self) -> &Arc<FuncAgentInner> {
        &self.0
    }
}

impl FuncAgent {
    pub fn New(nodeId: &str) -> Self {
        let inner = FuncAgentInner {
            nodeId: nodeId.to_string(),
            ..Default::default()
        };

        return Self(Arc::new(inner))
    }

    pub fn CallResponse(&self, funcCallId: &str, response: SResult<String, String>) -> Result<()> {
        match self.callContexts.lock().unwrap().remove(funcCallId) {
            None => return Err(Error::CommonError(format!("get unepxecet callid {}", funcCallId))),
            Some(context) => {
                context.respChann.send(response).expect("CallResponse fail...");
            }
        };

        return Ok(())
    }

    pub fn ToGrpcType(&self) -> func::FuncAgentRegisterReq {
        let mut callerCalls = Vec::new();
        let mut calleeCalls = Vec::new();
        for (_, call) in self.callerCalls.lock().unwrap().iter() {
            callerCalls.push(call.ToGrpcType());
        }
        for (_, call) in self.calleeCalls.lock().unwrap().iter() {
            calleeCalls.push(call.ToGrpcType());
        }
        return func::FuncAgentRegisterReq {
            node_id: self.nodeId.clone(),
            caller_calls: callerCalls,
            callee_calls: calleeCalls,
            func_pods: self.funcPodMgr.ToGrpcType(),
        }
    }

    pub async fn FuncCall(&self, funcCallReq: func::FuncAgentCallReq) -> SResult<String, String> {
        let (tx, rx) = oneshot::channel();
        let id = funcCallReq.id.clone();
        self.callContexts.lock().unwrap().insert(id.clone(), FuncCallContext {
            respChann: tx,
        });

        // if callerId is "", it is from grpc call directly
        match self.OnFuncAgentCallReq("", funcCallReq) {
            Ok(()) => (),
            Err(e) => {
                self.callContexts.lock().unwrap().remove(&id);
                return Err(format!("{:?}", e));
            }
        }

        let ret = rx.await.expect("get unepxecet oneshot result");
        return ret;
    }

    pub fn OnNewConnection(
        &self, 
        registerMsg: &func::FuncPodRegisterReq, 
        stream: tonic::Streaming<func::FuncAgentMsg>,
        agentTx: mpsc::Sender<SResult<func::FuncAgentMsg, Status>>) 
    -> Result<()> {
        let funcPodId = registerMsg.func_pod_id.clone();
        let instance = FuncPod::New(registerMsg, stream, agentTx)?;
        self.funcPodMgr.AddPod(&funcPodId, &instance)?;
        return Ok(());
    }

    pub fn OnFuncAgentCallReq(&self, callerFuncPodId: &str, req: func::FuncAgentCallReq) -> Result<()> {
        let createTime = SystemTime::now();
        let inner = FuncCallInner {
            id: req.id.clone(),
            calleeNodeId: self.nodeId.clone(),
            callerFuncPodId: callerFuncPodId.to_string(),
            callerNodeId: String::new(),
            calleeFuncPodId: String::new(), // not assigned,
            namespace: req.namespace.clone(),
            packageName: req.package_name.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            priority: req.priority as usize,
            createTime: createTime,
        };

        let funcCall = FuncCall(Arc::new(inner));
        self.callerCalls.lock().unwrap().insert(req.id.clone(), funcCall);

        let protoTime = SystemTimeProto::FromSystemTime(createTime);
        let req = func::FuncSvcCallReq {
            id: req.id.clone(),
            namespace: req.namespace.to_string(),
            package_name: req.package_name.to_string(),
            func_name: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            priority: req.priority,
            createtime: Some(protoTime.ToTimeStamp()),
            caller_node_id: self.nodeId.clone(),
            calller_pod_id: callerFuncPodId.to_string(),
            callee_node_id: String::new(),
            callee_pod_id: String::new(),
        };

        FUNC_SVC_CLIENT.get().unwrap().Send(func::FuncSvcMsg {
            event_body: Some(func::func_svc_msg::EventBody::FuncSvcCallReq(req))
        })?;

        return Ok(())
    }

    pub fn OnFuncSvcCallReq(&self, req: func::FuncSvcCallReq) -> Result<()> {
        let funcPod = self.funcPodMgr.GetPod(&req.callee_pod_id)?;

        let createTimeProto = SystemTimeProto::FromTimestamp(req.createtime.as_ref().unwrap());

        let inner = FuncCallInner {
            id: req.id.clone(),
            callerNodeId: req.caller_node_id.clone(),
            callerFuncPodId: req.callee_pod_id.clone(),
            calleeNodeId: req.callee_node_id.clone(),
            calleeFuncPodId: String::new(), // not assigned,
            namespace: req.namespace.clone(),
            packageName: req.package_name.clone(),
            funcName: req.func_name.clone(),
            parameters: req.parameters.clone(), 
            priority: req.priority as usize,
            createTime: createTimeProto.ToSystemTime(),
        }; 

        let funcCall = FuncCall(Arc::new(inner));
        self.calleeCalls.lock().unwrap().insert(req.id.clone(), funcCall);

        let req = func::FuncAgentCallReq {
            id: req.id.clone(),
            namespace: req.namespace.clone(),
            package_name: req.package_name.clone(),
            func_name: req.func_name.clone(),
            parameters: req.parameters.clone(),
            priority: req.priority,
        };

        funcPod.Send(func::FuncAgentMsg {
            event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallReq(req))
        })?;

        return Ok(())
        
    }

    pub fn OnFuncSvcCallResp(&self, resp: func::FuncSvcCallResp) -> Result<()> {
        let id = resp.id.clone();
        let callerPodId = resp.callee_pod_id.clone();
        
        if callerPodId.len() == 0 { // it is from grpc gateway direct call
            let resp = if resp.error.len() == 0 {
                Ok(resp.resp)
            } else {
                Err(resp.error)
            };
            self.CallResponse(&id, resp)?;
        } else {
            let resp = func::FuncAgentCallResp {
                id: id,
                resp: resp.resp,
                error: resp.error,
            };
            self.funcPodMgr.SendTo(&callerPodId, func::FuncAgentMsg {
                event_body: Some(func::func_agent_msg::EventBody::FuncAgentCallResp(resp)),
            })?;
        }
        
        return Ok(())
    }

    pub async fn OneFuncSvcMgr(&self, msg: func::FuncSvcMsg) -> Result<()> {
        let body = match msg.event_body {
            None => return Err(Error::EINVAL),
            Some(b) => b,
        };

        match body {
            func::func_svc_msg::EventBody::FuncSvcCallReq(msg) => {
                self.OnFuncSvcCallReq(msg)?;
            }
            _ => {

            }
        };

        return Ok(())    
    }

    pub async fn OnFuncAgentMsg(&self, callerId: &str, msg: func::FuncAgentMsg) -> Result<()> {
        let body = match msg.event_body {
            None => return Err(Error::EINVAL),
            Some(b) => b,
        };

        match body {
            EventBody::FuncAgentCallReq(msg) => {
                self.OnFuncAgentCallReq(callerId, msg)?;
            }
            _ => {

            }
        };

        return Ok(())
    }
}

#[tonic::async_trait]
impl func::func_agent_service_server::FuncAgentService for FuncAgent {
    type StreamProcessStream = ReceiverStream<SResult<func::FuncAgentMsg, Status>>;
    
    async fn stream_process(
        &self,
        request: tonic::Request<tonic::Streaming<func::FuncAgentMsg>>,
    ) -> SResult<tonic::Response<Self::StreamProcessStream>, tonic::Status> {
        let mut stream = request.into_inner();
        let msg = stream.message().await.unwrap().unwrap();
        let body = match msg.event_body {
            None => return Err(Status::aborted("get non event_body")),
            Some(body) => body,
        };

        let registerMsg = match body {
            EventBody::FuncPodRegisterReq(req) => {
                req
            }
            _  => {
                return Err(Status::aborted("first message should be FuncPodRegisterResp"));
            }
        };

        let (tx, rx) = mpsc::channel(30);
        self.OnNewConnection(&registerMsg, stream, tx).unwrap();
        return Ok(Response::new(ReceiverStream::new(rx)));
    }

    async fn func_call(
        &self,
        request: tonic::Request<func::FuncAgentCallReq>,
    ) -> SResult<tonic::Response<func::FuncAgentCallResp>, tonic::Status> {
        let req = request.into_inner();
        let id = req.id.clone();
        let res = self.FuncCall(req).await;
        let resp = match res {
            Ok(response) => func::FuncAgentCallResp {
                id: id,
                resp: response,
                error: String::new(),
            },
            Err(err) => func::FuncAgentCallResp {
                id: id,
                resp: String::new(),
                error: err,
            }
        };
        
        return Ok(Response::new(resp));
    }
}

pub async fn FuncAgentGrpcService() -> Result<()> {
    use tonic::transport::Server;
    //let svc = FuncAgent::default();
    let funcSvcFuture = Server::builder()
        .add_service(FuncAgentServiceServer::new(FUNC_AGENT.clone()))
        .serve("127.0.0.1:8892".parse().unwrap());

    info!("func service start ...");
    tokio::select! {
        _ = funcSvcFuture => {}
    }

    Ok(())
}