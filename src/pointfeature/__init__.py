import cv2 
import os 
import numpy as np 

SMOOTHING_RADIUS = 25       ##Variable Parameter 

#Here everything is placed inside one object 
#Which is useful in a multi algorithm paradigm 
#But its more advisable to break useful functions wherever data binding is not primary 

class Stabilizer():
    
    def __init__(self , src , dest_folder, fname):
        
        self.src = src 
        self.dest = os.path.join(dest_folder , fname) 
        
        self.cap = self.vid = cv2.VideoCapture(self.src) 
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #Defining video codec
        self.out = cv2.VideoWriter( self.dest , fourcc , self.fps, (self.w , self.h))
     
    #Create transformation matrix for LK optical flow 
    def maketransformmatrix(self):
        
        self.transforms = np.zeros((self.n_frames-1, 3), np.float32) 
        #3 is for x,y,theta
        
        # Read first frame
        _, ff = self.vid.read() 
    
        # Converting frame to grayscale
        previous_gray = cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY) #We use previous_gray in a loop 
        
        for i in range(self.n_frames-2):
            trackable_features = cv2.goodFeaturesToTrack(previous_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3) #These parameters can be modified. Picked from reference 
                        
            success, curr = self.vid.read()
            
            if not success:
                break 
            
            current_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            #Tracking feature points 
            current_features , status , _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, trackable_features, None) 

            #Lucas-Kanade optical flow
            #Status is an array with 1 for successfully identified points and 0 for others.
            #So we pick only successful indices
            
            idx = np.where(status==1)[0]    #Indices of moved points
            trackable_features = trackable_features[idx]
            current_features = current_features[idx]
            
            m = cv2.estimateAffinePartial2D(trackable_features, current_features)[0] #Get affine displacement matrix of frame to frame movemeent 
            
            dx = m[0,2] #Translation vectors
            dy = m[1,2]
            da = np.arctan2(m[1,0], m[0,0]) #Rotation angle
            
            self.transforms[i] = [dx,dy,da]
            previous_gray = current_gray    #For next frame 

    #Moving average trajectory smoothing 
    @staticmethod 
    def movingAverage(curve , radius):
        
        #Obtain moving average across frames for smoothening the trajectory 
        window_size = 2*radius + 1 
        frames = np.ones(window_size)/window_size   #Creating convolution kernel of equal weights 
        padded_curve = np.lib.pad(curve , (radius , radius) , 'edge')   #Padding the curve to avoid edge effects
        smoothened_curve = np.convolve(padded_curve , frames , mode = 'same') #Create convolusion for moving average 
        
        smoothened_curve = smoothened_curve[radius : -radius] #Smoothen within bounds 
        return smoothened_curve 

    #Create smooth trajectory 
    @staticmethod 
    def smooth(trajectory):
        
        s = np.copy(trajectory)
        
        for i in range(3): #from -2 -> 2 
            s[: , i] = Stabilizer.movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS) 
            #Smoothening trajectory 

        return s 

    #Helper function 
    def smoothen(self):
        
        self.trajectory = np.cumsum(self.transforms,  axis = 0) #Trajectory is the sum of motions 
        self.smoothened = Stabilizer.smooth(self.trajectory)    #Smoothen said trajectory 
        
        delta = self.smoothened - self.trajectory   #Find difference between smoothened trajectory and actual trajectory
        self.smooth_transform_matrix = self.transforms + delta #Using the differnece matrix to stabilize the video

    #Final trajectory updated 
    def finalize(self):
        
        # Reset stream to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        
        
        # Write n-2 transformed frames
        for i in range(self.n_frames-2):
        
            success, frame = self.cap.read()
            if not success:
                break
            
            # Extract transformations from the new transformation array
            dx = self.smooth_transform_matrix[i,0]
            dy = self.smooth_transform_matrix[i,1]
            da = self.smooth_transform_matrix[i,2]
            
            # Reconstruct transformation matrix accordingly to new values
            m = np.zeros((2,3), np.float32)
            m[0,0] = np.cos(da)
            m[0,1] = -np.sin(da)
            m[1,0] = np.sin(da)
            m[1,1] = np.cos(da)
            m[0,2] = dx
            m[1,2] = dy
            #Affine transformation using the 2x2 rotation matrix 
            
            # Apply affine wrapping to the given frame
            frame_stabilized = cv2.warpAffine(frame, m, (self.w,self.h))
            
            # Fix border artifacts
            frame_stabilized = cv2.rotate(self.fixBorder(frame_stabilized) , cv2.ROTATE_180) 
            
            # Write the frame to the file
            self.out.write(frame_stabilized)
     
    #Fixes borders for out of bound frames 
    def fixBorder(self,frame):
        s = frame.shape
        
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.08)
        frame = cv2.warpAffine(frame, T, (s[1], s[0])) #Standard cv2 affine warping to avoid edge effects 
        return frame
              
    def stabilize(self ) -> str : 
        
        self.maketransformmatrix()
        print('Transformation Matrix Complete')
        self.smoothen()
        print('Trajectory smoothened')
        self.finalize() 
        print('Finalized')
        
        self.cap.release()
        self.out.release()
        
        return self.dest 