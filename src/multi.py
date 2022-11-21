import time 
import os 
import logging 

try:
    import simplejson as json 
except ImportError:
    import json 

#Separate process to handle the queue of operations 
#This will be called by the main process
#This will push entries to the database 
#This ensures multiple programs dont write to the database together and fail SQLite constraints  
def process_handler(operations , src , dest , f , queue, real_src, holder_code , *args, **kwargs):
    
    t = time.perf_counter()
    ext= os.path.splitext(src)[1]
    
    for operation, func in operations:
    
        rand_name = os.urandom(16).hex() + ext 
        src = func(src  = src , dest_folder = dest , fname = rand_name ).stabilize() 
    
    e = time.perf_counter()    
    
    os.rename(src , os.path.join(dest , f))
    
    ops = [op for op , _ in operations]
    
    queue.put({"file" : f , 
               "operations" : json.dumps(ops) ,
               "n" : len(ops), 
               "time" : e-t , 
               "src" : real_src , 
               "holder_code" : holder_code 
               })
    
    #print(f'File: {f} , Operations: {ops} , Time: {e-t}')
    return None 