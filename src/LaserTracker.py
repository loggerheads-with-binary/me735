import cv2 
import numpy as np 
import pandas as pd
from dataclasses import dataclass, field
import sqlite3 

#Data structure to contain laser point information 
@dataclass(  eq = True  )
class LaserPoint:

    x : int  = field(default = 0)
    y : int = field(default = 0)
    radius : int = field(default = 0)
    abs_r : int = field(default = 0 )

    
    def assign(self, x , y , radius):
        
        self.x = x
        self.y = y
        self.radius = radius 
        self.abs_r = np.sqrt(x**2 + y**2)
        
        return self 

#Rotation tracker method 
#NOTE: Not used in final project 
class RotationTracker:
    
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        
        self.w , self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def render(self):
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        transforms_modified = np.zeros((self.n_frames-1, 3), np.float32)
        
        _, prev_mod = self.cap.read()
        prev_gray_mod = cv2.cvtColor(prev_mod, cv2.COLOR_BGR2GRAY)
        
        for i in range(self.n_frames-2):
            # Detect feature points in previous frame
            prev_pts_mod = cv2.goodFeaturesToTrack(prev_gray_mod,
                                                maxCorners=5000,
                                                qualityLevel=0.01,
                                                minDistance=20,
                                                blockSize=3)

            # Read next frame
            success_mod, curr_mod = self.cap.read()
            if not success_mod:
                break 

            # Convert to grayscale
            curr_gray_mod = cv2.cvtColor(curr_mod, cv2.COLOR_BGR2GRAY) 

            # Calculate optical flow (i.e. track feature points)
            curr_pts_mod, status_mod, err_mod = cv2.calcOpticalFlowPyrLK(prev_gray_mod, curr_gray_mod, prev_pts_mod, None) 

            # Sanity check
            assert prev_pts_mod.shape == curr_pts_mod.shape 

            # Filter only valid points
            idx_m = np.where(status_mod==1)[0]
            prev_pts_mod = prev_pts_mod[idx_m]
            curr_pts_mod = curr_pts_mod[idx_m]

            #Find transformation matrix
            m_mod,_ = cv2.estimateAffinePartial2D(prev_pts_mod, curr_pts_mod) #will only work with OpenCV-3 or less
            #print(m)
            #break
            # Extract traslation
            dx_mod = m_mod[0,2]
            dy_mod = m_mod[1,2]

            # Extract rotation angle
            #da = np.arctan2(m[1,0], m[0,0])
            da_mod = np.arctan2(m_mod[1,0],m_mod[1,1])
            # Store transformation
            transforms_modified[i] = [dx_mod,dy_mod,da_mod]

            # Move to next frame
            prev_gray_mod = curr_gray_mod

            print(f'\rFrame: {i}/{self.n_frames} analyzed. Tracked points::{len(prev_pts_mod)}' , end = '')

            trajectory_mod = np.cumsum(transforms_modified, axis=0)

        return trajectory_mod
    
    def produce_contour(self):
        
        self.trajectory_mod = self.render()
        df = pd.DataFrame(self.trajectory_mod[: , 2] , columns = ['angle'])

        df['angle_degrees'] = df['angle'].apply(lambda x: np.degrees(x))
        # df['abs_angle'] = df['angle'].apply(lambda x: np.abs(x))
        df['total_angle'] = (df['angle']).cumsum()
        df['total_angle_normalized'] = df['total_angle'] / self.fps

        self.df = df 
        
        return self 
    
    def to_xlsx(self, fname):
        
        self.df.to_excel(fname)
        return self
         
    def export_csv(self , fname):
        
        self.df.to_csv(fname)
        return self 
    
    def export_sqlite(self, fname , table):
        
        import sqlite3
        conn = sqlite3.connect(fname)
        self.df.to_sql(table, conn)
        conn.close()
        
        return self  

#Laser Tracker across video 
class LaserTracker:
    
    def __init__(self , fname : str , hue_min=0, hue_max=256,
                 sat_min=0, sat_max=255, val_min=0, val_max=256):
        
        self.file = fname 
        self.cap = cv2.VideoCapture(self.file)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.value_min = val_min
        self.value_max = val_max
        self.saturation_min = sat_min
        self.saturation_max = sat_max
        
        self.prev_center = 0 , 0 
        self.prev_radius = 0 
        
        self.channels = {'hue' : None , 'saturation' : None , 'value' : None } #Using channels to propagate the HSV values

    #Sets up a threshold value to identify the laser and then eventually track it 
    def threshold_image(self , channel):
        
        maximum = getattr(self, channel + '_max') #Ex: sat_max
        minimum = getattr(self, channel + '_min') #Ex: sat_min
        
        (t , alpha) = cv2.threshold(self.channels[channel], maximum, 0, cv2.THRESH_TOZERO_INV) #Thresholding the image upper bound
        (t , self.channels[channel]) = cv2.threshold(alpha, minimum, 255, cv2.THRESH_BINARY) #Thresholding the image lower bound 

        if channel == 'hue':
            # only works for filtering red color because the range for the hue is split across red due to weird CV2 properties 
            self.channels[channel] = cv2.bitwise_not(self.channels[channel])
        
    #Tracks the laser point in a given frame 
    def track(self ,  mask):
        
        (x,y), radius = self.prev_center , self.prev_radius #In case laser is msising in the frame
        contours= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] #Obtaining contours of high HSV discrepancy(ex : a bright laser spot)
        
        if len(contours)>0:
            
            max_contour = max(contours, key=cv2.contourArea) #Get contour by max contour area, to avoid any sharp edges which might be 1-2 pixels thick 
            
            #Find largest contour, and then obtain minium enclosing cirlce for the same. 
            (x , y ), radius = cv2.minEnclosingCircle(max_contour) #Find min enclosing circle of the contour and use its centre as x,y coordinates 
            
            #Moment contouring method 
            #Better than min enclosing circle method
            #But may not yield results at times 
            #Hence using both methods and then prioritizing the moment method
            moments = cv2.moments(max_contour)
            
            if moments["m00"] > 0:
                x , y = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
                         
            else:
                x , y = int(x), int(y)
                
            self.prev_center = x , y
            self.prev_radius = radius
            
        
        return x , y , radius  
    
    ##Produces x,y,r values     
    def detect(self , frame):
        
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Hue, Saturation , Value
        h, s, v = cv2.split(hsv_img)
        
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v #Channeling the value 
    
        #Thresholding the image across different channels 
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        #The below bitwise anding obtains us a mask which is the intersection of the three channels
        #Where the values are very high to obtain the laser spot and thresholded to maintain the laser spot
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        ) #Finding a match in both channels 
        
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        ) #Finding a match across all channels 
        
        
        #Return the tracked laser spot across the frame
        return self.track(self.channels['laser'])
    
    #Produce a contour of x,y,r across frames 
    def produce_contour(self):
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.mov = list()   #Will store contours 
        
        i = 0
        while True:
            
            success, frame = self.cap.read() 
            
            if not success:
                break 
            
            x , y , radius = self.detect(frame) #Get x,y,r values
            self.mov.append(LaserPoint().assign(x , y , radius)) #This data structure is used to ensure no misordering occurs of the values 
            i += 1
        
        #Create a dataframe to store the contours 
        #This dataframe is later exported as an excel file which is submitted 
        self.df = pd.DataFrame(map(vars , self.mov) , columns = ['x' , 'y' , 'radius' , 'abs_r'])
        
        #Mean coordinates for laser spot 
        xmean = self.df['x'].mean()
        ymean = self.df['y'].mean()
        
        #Find delta_x and delta_y for each frame as compared to the mean coordinates
        self.df['delta_x'] = self.df['x'] - xmean
        self.df['delta_y'] = self.df['y'] - ymean
        
        self.df['r']  = np.sqrt(np.array(self.df['delta_x'])**2 + np.array(self.df['delta_y'])**2)
        
        #Find integral[rdt] 
        self.df['cumsum'] = self.df['r'].cumsum()/self.fps #Finding the cumulative sum of radius values and dividing by the fps 
        #The above is equivalent to finding area under the r vs t curve 
        
        #Find maximum displacement from mean coordinates 
        self.maxr = self.df['r'].max()
        
        #Find the area under the curve 
        self.auc = self.df['cumsum'].max()
        
        #Storing the result in a excel file
        self.df['Result'] = [self.auc , self.maxr  ] + [np.nan]*(len(self.df)-2) 
        return self
        
    def __export_pandas(self ):
        return self.df 
        
    def export_xlsx(self , fname, sheet_name = None):
        
        df = self.__export_pandas()
        
        if sheet_name:
            df.to_excel(fname, sheet_name = sheet_name)
            
        else:
            df.to_excel(fname)
            
        return self
    
    export_excel = export_xlsx
        
    def export_csv(self , fname):
        
        df = self.__export_pandas()
        df.to_csv(fname)
        return self
    
    def export_sqlite(self , fname , table):
        
        
        df = self.__export_pandas()
        b = "\\"
        df.to_sql(table, f'sqlite:///{fname.replace(b , "/")}' )
        return self

import os, glob  
PROG_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER  = os.path.join(PROG_PATH , ".." , 'results')

if __name__ == '__main__':
    
    
    static_src = os.path.join(PROG_PATH , '..' , 'data-src', 'Static.mp4')
    dynamic_src = os.path.join(PROG_PATH , '..' , 'data-src', 'Dynamic.mp4')

    #Also editing the stability values in the database as 1/auc 
    DB = os.path.join(PROG_PATH , '..' , 'db.sqlite3')
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute('begin')
    
    query = """
        UPDATE entries 
        SET stability_1 = ? 
        WHERE LOWER(fpath) = LOWER(?)
    """
    
    
    try:    
        
        for vid, src in zip(['Static' , 'Dynamic'] , [static_src , dynamic_src]):

            print('Handling::' , vid)
            
            res = list() 
            
            with pd.ExcelWriter(os.path.join(RESULTS_FOLDER , f'{vid}.xlsx') ) as f:
                
                pd.DataFrame().to_excel(f , sheet_name = 'Summary') ##Dummy sheet to put summary first  
              
                t = LaserTracker(src).produce_contour().export_excel(f , vid[:31]) #Source file     
                
                #Summary item                 
                res.append({'file' : os.path.splitext(os.path.basename(src))[0] , 
                           'maxr' : t.maxr , 'auc' : t.auc , 
                           'fps' : t.fps , 'n_frames' : t.n_frames}) 
                
                for file in glob.glob(os.path.join(RESULTS_FOLDER , f'{vid}*.mp4')):
                    
                    print(f"Processing {file}")
                    
                    #Processing each cascaded operation destination file 
                    t = LaserTracker(file).produce_contour().export_excel(f , sheet_name = os.path.splitext(os.path.split(file)[-1])[0][:31] )
                    cur.execute(query , (1/t.auc , os.path.split(file)[-1] )) 
                    
                    #Summary item 
                    res.append({'file' : os.path.splitext(os.path.basename(file))[0] ,  
                                'maxr' : t.maxr , 'auc' : t.auc , 
                                'fps' : t.fps , 'n_frames' : t.n_frames})
                    
                    print("Done")
            
                #Exporting the summary of each source and its cascaded stabilized output videos
                df = pd.DataFrame(res)
                df.to_excel(f , sheet_name = 'Summary')
                
        cur.execute('commit')
        
        ##Exporting the database to a new excel file 
        df = pd.read_sql_table('entries' , 'sqlite:///' + DB)
        df.to_excel(os.path.join(RESULTS_FOLDER , 'results.xlsx') , sheet_name = 'Summary')
        
    except Exception as e:
        
        cur.execute('rollback')
        raise 