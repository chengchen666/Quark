# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: func.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nfunc.proto\x12\x04\x66unc\"\x85\x02\n\nBlobSvcReq\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12)\n\x0b\x42lobOpenReq\x18\xf5\x03 \x01(\x0b\x32\x11.func.BlobOpenReqH\x00\x12)\n\x0b\x42lobReadReq\x18\xf9\x03 \x01(\x0b\x32\x11.func.BlobReadReqH\x00\x12)\n\x0b\x42lobSeekReq\x18\xfb\x03 \x01(\x0b\x32\x11.func.BlobSeekReqH\x00\x12+\n\x0c\x42lobCloseReq\x18\x81\x04 \x01(\x0b\x32\x12.func.BlobCloseReqH\x00\x12-\n\rBlobDeleteReq\x18\x83\x04 \x01(\x0b\x32\x13.func.BlobDeleteReqH\x00\x42\x0b\n\tEventBody\"\x90\x02\n\x0b\x42lobSvcResp\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12+\n\x0c\x42lobOpenResp\x18\xf6\x03 \x01(\x0b\x32\x12.func.BlobOpenRespH\x00\x12+\n\x0c\x42lobReadResp\x18\xfa\x03 \x01(\x0b\x32\x12.func.BlobReadRespH\x00\x12+\n\x0c\x42lobSeekResp\x18\xfc\x03 \x01(\x0b\x32\x12.func.BlobSeekRespH\x00\x12-\n\rBlobCloseResp\x18\x82\x04 \x01(\x0b\x32\x13.func.BlobCloseRespH\x00\x12/\n\x0e\x42lobDeleteResp\x18\x84\x04 \x01(\x0b\x32\x14.func.BlobDeleteRespH\x00\x42\x0b\n\tEventBody\"\x89\x07\n\x0c\x46uncAgentMsg\x12\r\n\x05msgId\x18\x01 \x01(\x04\x12\x36\n\x12\x46uncPodRegisterReq\x18\x64 \x01(\x0b\x32\x18.func.FuncPodRegisterReqH\x00\x12\x39\n\x13\x46uncPodRegisterResp\x18\xc8\x01 \x01(\x0b\x32\x19.func.FuncPodRegisterRespH\x00\x12\x33\n\x10\x46uncAgentCallReq\x18\xac\x02 \x01(\x0b\x32\x16.func.FuncAgentCallReqH\x00\x12\x35\n\x11\x46uncAgentCallResp\x18\x90\x03 \x01(\x0b\x32\x17.func.FuncAgentCallRespH\x00\x12)\n\x0b\x42lobOpenReq\x18\xf5\x03 \x01(\x0b\x32\x11.func.BlobOpenReqH\x00\x12+\n\x0c\x42lobOpenResp\x18\xf6\x03 \x01(\x0b\x32\x12.func.BlobOpenRespH\x00\x12-\n\rBlobCreateReq\x18\xf7\x03 \x01(\x0b\x32\x13.func.BlobCreateReqH\x00\x12/\n\x0e\x42lobCreateResp\x18\xf8\x03 \x01(\x0b\x32\x14.func.BlobCreateRespH\x00\x12)\n\x0b\x42lobReadReq\x18\xf9\x03 \x01(\x0b\x32\x11.func.BlobReadReqH\x00\x12+\n\x0c\x42lobReadResp\x18\xfa\x03 \x01(\x0b\x32\x12.func.BlobReadRespH\x00\x12)\n\x0b\x42lobSeekReq\x18\xfb\x03 \x01(\x0b\x32\x11.func.BlobSeekReqH\x00\x12+\n\x0c\x42lobSeekResp\x18\xfc\x03 \x01(\x0b\x32\x12.func.BlobSeekRespH\x00\x12+\n\x0c\x42lobWriteReq\x18\xfd\x03 \x01(\x0b\x32\x12.func.BlobWriteReqH\x00\x12-\n\rBlobWriteResp\x18\xfe\x03 \x01(\x0b\x32\x13.func.BlobWriteRespH\x00\x12+\n\x0c\x42lobCloseReq\x18\x81\x04 \x01(\x0b\x32\x12.func.BlobCloseReqH\x00\x12-\n\rBlobCloseResp\x18\x82\x04 \x01(\x0b\x32\x13.func.BlobCloseRespH\x00\x12-\n\rBlobDeleteReq\x18\x83\x04 \x01(\x0b\x32\x13.func.BlobDeleteReqH\x00\x12/\n\x0e\x42lobDeleteResp\x18\x84\x04 \x01(\x0b\x32\x14.func.BlobDeleteRespH\x00\x42\x0b\n\tEventBody\"?\n\x0b\x42lobOpenReq\x12\x0f\n\x07svcAddr\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"\xb8\x01\n\x0c\x42lobOpenResp\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x0c\n\x04size\x18\x05 \x01(\x04\x12\x10\n\x08\x63hecksum\x18\x06 \x01(\t\x12#\n\ncreateTime\x18\x07 \x01(\x0b\x32\x0f.func.Timestamp\x12\'\n\x0elastAccessTime\x18\x08 \x01(\x0b\x32\x0f.func.Timestamp\x12\r\n\x05\x65rror\x18\t \x01(\t\"A\n\rBlobDeleteReq\x12\x0f\n\x07svcAddr\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"\x1f\n\x0e\x42lobDeleteResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"0\n\rBlobCreateReq\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\"<\n\x0e\x42lobCreateResp\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0f\n\x07svcAddr\x18\x03 \x01(\t\x12\r\n\x05\x65rror\x18\t \x01(\t\"&\n\x0b\x42lobReadReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0b\n\x03len\x18\x03 \x01(\x04\"+\n\x0c\x42lobReadResp\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\r\n\x05\x65rror\x18\x04 \x01(\t\"8\n\x0b\x42lobSeekReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0b\n\x03pos\x18\x03 \x01(\x03\x12\x10\n\x08seekType\x18\x04 \x01(\r\"-\n\x0c\x42lobSeekResp\x12\x0e\n\x06offset\x18\x02 \x01(\x04\x12\r\n\x05\x65rror\x18\x03 \x01(\t\"\x1a\n\x0c\x42lobCloseReq\x12\n\n\x02id\x18\x02 \x01(\x04\"\x1e\n\rBlobCloseResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"(\n\x0c\x42lobWriteReq\x12\n\n\x02id\x18\x02 \x01(\x04\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"\x1e\n\rBlobWriteResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"\x19\n\x0b\x42lobSealReq\x12\n\n\x02id\x18\x02 \x01(\x04\"\x1d\n\x0c\x42lobSealResp\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"c\n\x12\x46uncPodRegisterReq\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12\x12\n\nclientMode\x18\x04 \x01(\x08\"$\n\x13\x46uncPodRegisterResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"~\n\x10\x46uncAgentCallReq\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12\x10\n\x08\x66uncName\x18\x04 \x01(\t\x12\x12\n\nparameters\x18\x05 \x01(\t\x12\x10\n\x08priority\x18\x06 \x01(\x04\"<\n\x11\x46uncAgentCallResp\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04resp\x18\x03 \x01(\t\"\xcc\x03\n\nFuncSvcMsg\x12:\n\x14\x46uncAgentRegisterReq\x18\x64 \x01(\x0b\x32\x1a.func.FuncAgentRegisterReqH\x00\x12=\n\x15\x46uncAgentRegisterResp\x18\xc8\x01 \x01(\x0b\x32\x1b.func.FuncAgentRegisterRespH\x00\x12/\n\x0e\x46uncPodConnReq\x18\xac\x02 \x01(\x0b\x32\x14.func.FuncPodConnReqH\x00\x12\x31\n\x0f\x46uncPodConnResp\x18\x90\x03 \x01(\x0b\x32\x15.func.FuncPodConnRespH\x00\x12\x35\n\x11\x46uncPodDisconnReq\x18\xf4\x03 \x01(\x0b\x32\x17.func.FuncPodDisconnReqH\x00\x12\x37\n\x12\x46uncPodDisconnResp\x18\xd8\x04 \x01(\x0b\x32\x18.func.FuncPodDisconnRespH\x00\x12/\n\x0e\x46uncSvcCallReq\x18\xbc\x05 \x01(\x0b\x32\x14.func.FuncSvcCallReqH\x00\x12\x31\n\x0f\x46uncSvcCallResp\x18\xa0\x06 \x01(\x0b\x32\x15.func.FuncSvcCallRespH\x00\x42\x0b\n\tEventBody\"\xa3\x01\n\x14\x46uncAgentRegisterReq\x12\x0e\n\x06nodeId\x18\x01 \x01(\t\x12)\n\x0b\x63\x61llerCalls\x18\x02 \x03(\x0b\x32\x14.func.FuncSvcCallReq\x12)\n\x0b\x63\x61lleeCalls\x18\x03 \x03(\x0b\x32\x14.func.FuncSvcCallReq\x12%\n\x08\x66uncPods\x18\x04 \x03(\x0b\x32\x13.func.FuncPodStatus\"&\n\x15\x46uncAgentRegisterResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"_\n\x0e\x46uncPodConnReq\x12\x11\n\tfuncPodId\x18\x02 \x01(\t\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x13\n\x0bpackageName\x18\x04 \x01(\t\x12\x12\n\nclientMode\x18\x05 \x01(\x08\"3\n\x0f\x46uncPodConnResp\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\"&\n\x11\x46uncPodDisconnReq\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\"#\n\x12\x46uncPodDisconnResp\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"\xf7\x01\n\x0e\x46uncSvcCallReq\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12\x10\n\x08\x66uncName\x18\x04 \x01(\t\x12\x12\n\nparameters\x18\x05 \x01(\t\x12\x10\n\x08priority\x18\x06 \x01(\x04\x12#\n\ncreatetime\x18\x07 \x01(\x0b\x32\x0f.func.Timestamp\x12\x14\n\x0c\x63\x61llerNodeId\x18\x08 \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\t \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\n \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\x0b \x01(\t\"\x90\x01\n\x0f\x46uncSvcCallResp\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04resp\x18\x03 \x01(\t\x12\x14\n\x0c\x63\x61llerNodeId\x18\x08 \x01(\t\x12\x13\n\x0b\x63\x61llerPodId\x18\t \x01(\t\x12\x14\n\x0c\x63\x61lleeNodeId\x18\n \x01(\t\x12\x13\n\x0b\x63\x61lleePodId\x18\x0b \x01(\t\"\x95\x01\n\rFuncPodStatus\x12\x11\n\tfuncPodId\x18\x01 \x01(\t\x12\x11\n\tnamespace\x18\x02 \x01(\t\x12\x13\n\x0bpackageName\x18\x03 \x01(\t\x12!\n\x05state\x18\x04 \x01(\x0e\x32\x12.func.FuncPodState\x12\x12\n\nfuncCallId\x18\x05 \x01(\t\x12\x12\n\nclientMode\x18\x06 \x01(\x08\"+\n\tTimestamp\x12\x0f\n\x07seconds\x18\x01 \x01(\x04\x12\r\n\x05nanos\x18\x02 \x01(\r*%\n\x0c\x46uncPodState\x12\x08\n\x04Idle\x10\x00\x12\x0b\n\x07Running\x10\x01\x32G\n\x0b\x42lobService\x12\x38\n\rStreamProcess\x12\x10.func.BlobSvcReq\x1a\x11.func.BlobSvcResp(\x01\x30\x01\x32\x8c\x01\n\x10\x46uncAgentService\x12;\n\rStreamProcess\x12\x12.func.FuncAgentMsg\x1a\x12.func.FuncAgentMsg(\x01\x30\x01\x12;\n\x08\x46uncCall\x12\x16.func.FuncAgentCallReq\x1a\x17.func.FuncAgentCallResp2I\n\x0e\x46uncSvcService\x12\x37\n\rStreamProcess\x12\x10.func.FuncSvcMsg\x1a\x10.func.FuncSvcMsg(\x01\x30\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'func_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FUNCPODSTATE._serialized_start=4132
  _FUNCPODSTATE._serialized_end=4169
  _BLOBSVCREQ._serialized_start=21
  _BLOBSVCREQ._serialized_end=282
  _BLOBSVCRESP._serialized_start=285
  _BLOBSVCRESP._serialized_end=557
  _FUNCAGENTMSG._serialized_start=560
  _FUNCAGENTMSG._serialized_end=1465
  _BLOBOPENREQ._serialized_start=1467
  _BLOBOPENREQ._serialized_end=1530
  _BLOBOPENRESP._serialized_start=1533
  _BLOBOPENRESP._serialized_end=1717
  _BLOBDELETEREQ._serialized_start=1719
  _BLOBDELETEREQ._serialized_end=1784
  _BLOBDELETERESP._serialized_start=1786
  _BLOBDELETERESP._serialized_end=1817
  _BLOBCREATEREQ._serialized_start=1819
  _BLOBCREATEREQ._serialized_end=1867
  _BLOBCREATERESP._serialized_start=1869
  _BLOBCREATERESP._serialized_end=1929
  _BLOBREADREQ._serialized_start=1931
  _BLOBREADREQ._serialized_end=1969
  _BLOBREADRESP._serialized_start=1971
  _BLOBREADRESP._serialized_end=2014
  _BLOBSEEKREQ._serialized_start=2016
  _BLOBSEEKREQ._serialized_end=2072
  _BLOBSEEKRESP._serialized_start=2074
  _BLOBSEEKRESP._serialized_end=2119
  _BLOBCLOSEREQ._serialized_start=2121
  _BLOBCLOSEREQ._serialized_end=2147
  _BLOBCLOSERESP._serialized_start=2149
  _BLOBCLOSERESP._serialized_end=2179
  _BLOBWRITEREQ._serialized_start=2181
  _BLOBWRITEREQ._serialized_end=2221
  _BLOBWRITERESP._serialized_start=2223
  _BLOBWRITERESP._serialized_end=2253
  _BLOBSEALREQ._serialized_start=2255
  _BLOBSEALREQ._serialized_end=2280
  _BLOBSEALRESP._serialized_start=2282
  _BLOBSEALRESP._serialized_end=2311
  _FUNCPODREGISTERREQ._serialized_start=2313
  _FUNCPODREGISTERREQ._serialized_end=2412
  _FUNCPODREGISTERRESP._serialized_start=2414
  _FUNCPODREGISTERRESP._serialized_end=2450
  _FUNCAGENTCALLREQ._serialized_start=2452
  _FUNCAGENTCALLREQ._serialized_end=2578
  _FUNCAGENTCALLRESP._serialized_start=2580
  _FUNCAGENTCALLRESP._serialized_end=2640
  _FUNCSVCMSG._serialized_start=2643
  _FUNCSVCMSG._serialized_end=3103
  _FUNCAGENTREGISTERREQ._serialized_start=3106
  _FUNCAGENTREGISTERREQ._serialized_end=3269
  _FUNCAGENTREGISTERRESP._serialized_start=3271
  _FUNCAGENTREGISTERRESP._serialized_end=3309
  _FUNCPODCONNREQ._serialized_start=3311
  _FUNCPODCONNREQ._serialized_end=3406
  _FUNCPODCONNRESP._serialized_start=3408
  _FUNCPODCONNRESP._serialized_end=3459
  _FUNCPODDISCONNREQ._serialized_start=3461
  _FUNCPODDISCONNREQ._serialized_end=3499
  _FUNCPODDISCONNRESP._serialized_start=3501
  _FUNCPODDISCONNRESP._serialized_end=3536
  _FUNCSVCCALLREQ._serialized_start=3539
  _FUNCSVCCALLREQ._serialized_end=3786
  _FUNCSVCCALLRESP._serialized_start=3789
  _FUNCSVCCALLRESP._serialized_end=3933
  _FUNCPODSTATUS._serialized_start=3936
  _FUNCPODSTATUS._serialized_end=4085
  _TIMESTAMP._serialized_start=4087
  _TIMESTAMP._serialized_end=4130
  _BLOBSERVICE._serialized_start=4171
  _BLOBSERVICE._serialized_end=4242
  _FUNCAGENTSERVICE._serialized_start=4245
  _FUNCAGENTSERVICE._serialized_end=4385
  _FUNCSVCSERVICE._serialized_start=4387
  _FUNCSVCSERVICE._serialized_end=4460
# @@protoc_insertion_point(module_scope)
