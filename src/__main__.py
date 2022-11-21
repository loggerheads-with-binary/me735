##Takes source video input, performs some permutations, creates a file and writes corresponding information to the local database 
import os

PROG_PATH = os.path.abspath(os.path.dirname(__file__))
SEMAPHORE_COUNT = 4 #4 cores of max concurrency 
DEST_FOLDER = os.path.join(PROG_PATH , "..", "results")

#Instantiating a SQLite3 database
import sqlite3 
conn = sqlite3.connect(os.path.join(PROG_PATH , '..' ,  'db.sqlite3'))
cursor = conn.cursor() 

#Init SQL to create table if not exists 
with open(os.path.join(PROG_PATH , 'init.sql') , "r" , encoding = 'utf-8') as handle:
    cursor.executescript(handle.read())
    conn.commit()

import cv2 
import itertools 
from typing import List, Callable, Dict, Iterable   
import time 
import multiprocessing
from . import multi 
from .dbput import queue_runner, get_code 

try:
    import simplejson as json 
except ImportError:
    import json 

#Standard logging operations from various handles 
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(os.path.join(PROG_PATH , '..' , 'project.log')))

#For destination file name 
def get_op_str(src , operations):
    
    ops = "+".join(operations)
    base , ext = os.path.splitext(os.path.basename(src))
    return f'{base}_({ops}){ext}'
    
def handler(operations : Dict[str , Callable[[cv2.VideoCapture , ], str ]] , src : os.path , dest_folder : os.path , queue : multiprocessing.Queue  ) -> None:
    
    logger.info('handler')
    logger.info(f'operations: {operations}')
    
    holder_code = get_code()
    
    N = len(operations)
    keys = operations.keys()
    
    dp = dict() #Using dynamic programming for successively larger cascading operations 
    
    procs = list()
    
    
        
    for i in range(1 , N+1):
        
        pool_procs = list() 
        
        t = list(itertools.combinations(keys , i))
        for combination in t:    
            for permutation in itertools.permutations(combination):
                
                logger.info(f'Setting permutation: {permutation}')
                                    
                curr_operations = tuple((op , operations[op]) for op in permutation)
                dest = get_op_str(src , permutation)
                at_src = src 
            
                if tuple(permutation[:-1]) in dp:
                    
                    curr_operations = tuple((op , operations[op]) for op in permutation[-1:])
                    at_src = os.path.join( dest_folder , dp[tuple(permutation[:-1])])                        
                    
                #pool.map( multi.process_handler,  args = 
                pool_procs.append((curr_operations , at_src , dest_folder , dest,  queue , os.path.basename(src) , holder_code ) )
                dp[tuple(permutation)] = dest 
        
        #Concurrently executing all operations of a given length(ex: 3c1*1! operations, 3c2*2! operations etc)
        #Semaphore is used to ensure system resources are not entirely drained
        with multiprocessing.Pool(SEMAPHORE_COUNT) as pool:
            logger.info(f'Executing combination: {t} in parallel')            
            pool.starmap( multi.process_handler, pool_procs)
            pool.close()
            pool.join()

    return procs 
        
if __name__ == "__main__":
    
    import pretty_traceback 
    pretty_traceback.install()

    from . import pointfeature
    from . import pointspline  
    from . import l1 
        
    operations = {
                    "point-feature" : pointfeature.Stabilizer,
                    "pointspline" : pointspline.Stabilizer , 
                    "l1" : l1.Stabilizer , 
                    #"mesh_flow" : mesh.Stabilizer ,
                }     
    
    ##Running from bash file instead of standard python execution 
    if '1' in os.environ.get('REPO_DYNAMIC_RUN'  , '0' ):
        
        if '1' in os.environ.get('REPO_BATCH_RUN' , '0'):
            
            import time 
            time.sleep(15)  ##To execute from within batch file 
        
        srcs =  [
                #os.path.join(PROG_PATH , '..' , "data-src" , 'Static.mp4') ,
                os .path.join(PROG_PATH , '..' , "data-src" , 'Dynamic.mp4') ,
                ] 

    else:
        
        srcs =  [
                os.path.join(PROG_PATH , '..' , "data-src" , 'Static.mp4') ,
                #os .path.join(PROG_PATH , '..' , "data-src" , 'Dynamic.mp4') ,
                ] 


    queue = multiprocessing.Manager().Queue()    #queue to communicate between database and main process
    
    proc = multiprocessing.Process(name = 'queue-handler' , target = queue_runner , args = (queue , ) , daemon = True) #Pushes data to database 
    proc.start()
    
    
    for src in srcs:
        print(f'Processing {src}')
        handler(operations, src , DEST_FOLDER , queue)

    queue.put('STOP') #Queue stops when a string value is received 
    proc.terminate()
