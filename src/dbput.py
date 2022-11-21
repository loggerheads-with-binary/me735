import sqlite3 
import os 
import json 

PROG_PATH = os.path.dirname(os.path.abspath(__file__))
conn = sqlite3.connect(os.path.join(PROG_PATH , '..' ,  'db.sqlite3'))
cursor = conn.cursor() 

get_code = lambda  : cursor.execute('SELECT ifnull(MAX(holder_code) , 0) FROM entries; ').fetchone()[0] + 1 
#Gets what is called a holder code to give each running script a unique instance 
#This is done because we use multiple processes and multiple scripts but the database is same 

def queue_runner(queue):
    
    while True:
    
        item = queue.get() 
        print('Found queue item')
        
        #A string 'STOP' is pushed into the queue as a stop instruction 
        if isinstance(item , str):
            break     
            
        #A dictionary of values is pushed into  the queue as results of an operation 
        #These values are then pushed to the database 
        f , ops , t , src, n, holder_code  = item['file'] , item['operations'] , item['time'], item['src'], item['n'], item['holder_code']

        ops = f[f.find('(') + 1 : f.find(')')].split('+')
        n = len(ops)
        
        
        if len(ops) > 1 :
            if cursor.execute("SELECT COUNT(*) FROM entries WHERE trim(lower(operations)) = trim(lower(?)) AND holder_code = ?" , 
                            (json.dumps(ops[:-1]) , holder_code) ).fetchone()[0] != 0:
                
                t += cursor.execute('SELECT ifnull(time_taken, 0) FROM entries wHERE operations = ?\
                                                                                        AND \
                                                                                    holder_code = ?' , 
                                    (json.dumps(ops[:-1]) , holder_code ) ).fetchone()[0]
                #This is done to imbibe dynamic programming 
                #Since if operation a,b is completed already 
                #We apply c to a,b to get operation a,b,c
                #This reports time taken as time(c) instead of time(a,b)
                #So we fetch time(a,b) from the database 
                #And then add the time(c) to get time(a,b,c)
                
        #Convert list of operations to a json format 
        ops = json.dumps(ops)
        
        while 1:
            try:
                #Push entry to the database 
                cursor.execute("""INSERT INTO entries 
                               (holder_code , src, no_of_operations ,
                               operations, time_taken , fpath)
                               VALUES 
                               (?,?,?,?,?,?)
                               """ , (
                                    holder_code , 
                                    os.path.basename(src) , 
                                    n , 
                                    ops,
                                    t , 
                                    f))
                conn.commit() #Commit changes instantly to avoid any shutdown effects 
                break 
                
            except (sqlite3.IntegrityError, sqlite3.OperationalError, sqlite3.ProgrammingError):
                
                import traceback 
                print(traceback.format_exc()) #Print exception information 
                
                cursor.execute('rollback') #Rollback in case of an error 
                continue

                
        print("done")