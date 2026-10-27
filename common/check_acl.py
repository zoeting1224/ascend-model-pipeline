import ctypes
c=ctypes.CDLL("libascendcl.so")
c.aclInit.argtypes=[ctypes.c_char_p]
def check(name,r):
 print(name,r,flush=True)
 if r: raise RuntimeError(name+" failed: "+str(r))
check("aclInit",c.aclInit(None))
n=ctypes.c_uint()
check("aclrtGetDeviceCount",c.aclrtGetDeviceCount(ctypes.byref(n)))
print("device_count",n.value,flush=True)
check("aclrtSetDevice",c.aclrtSetDevice(0))
p=ctypes.c_void_p()
check("aclrtMalloc",c.aclrtMalloc(ctypes.byref(p),ctypes.c_size_t(1048576),0))
check("aclrtFree",c.aclrtFree(p))
check("aclrtResetDevice",c.aclrtResetDevice(0))
check("aclFinalize",c.aclFinalize())
