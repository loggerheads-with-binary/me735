import cv2 
import os 
import numpy as np 
from . import lpp 

CROP_RATIO = 0.9 #This is the ratio of the cropped image to the original image.

class Stabilizer():
    
    def __init__(self, src : str , dest_folder : str , fname : str ):
        
        self.src = src 
        self.dest = os.path.join(dest_folder , fname)
        
        self.cap = cv2.VideoCapture(self.src)
        
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shape = (self.w , self.h)
        self.crop_ratio = CROP_RATIO 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #Defining video codec
        #self.output_path = os.path.join( self.tmpdir , f'{os.urandom(16).hex()}_output.mp4')     ##Temporary file 
        self.out = cv2.VideoWriter( self.dest , fourcc , self.fps, (self.w , self.h))
          
    def write_output(self):
        
        F_transforms = np.zeros((self.n_frames, 3, 3), np.float32) #This will hold the transformation matrix for each frame
        F_transforms[:, :, :] = np.eye(3)  #Identity matrix 
        
        #Continue finding the transformation matrix across frames 
        _, prev = self.cap.read() 
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        lpp.get_inter_frame_transforms(self.cap, F_transforms, prev_gray)
        #The above method transforms the F_transforms matrix in place by looping across frames 
        #Finally C= C*F will give us the transformed frames as mentioned in the report 
        
        
        B_transforms = lpp.lppsolve(F_transforms, prev.shape, self.crop_ratio) 
        #B_transforms is a matrix such that it minimizes the camera displacements by solving it as a LPP 
        
        C_trajectory = F_transforms.copy() #Camera trajectory 
        
        for i in range(1, self.n_frames):
            
            # Right multiply transformations to accumulate all the changes into camera trajectory
            C_trajectory[i, :, :] = C_trajectory[i - 1, :, :] @ F_transforms[i, :, :]
        
        P_trajectory = C_trajectory.copy() #P will accumulate the final stabilized camera trajectory
        
        for i in range(self.n_frames):
            P_trajectory[i, :, :] = C_trajectory[i, :, :] @ B_transforms[i, :, :] #P = C*B for stabilization processes 

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n_frames = B_transforms.shape[0]
    
        for i in range(n_frames):
            
            success, frame = self.cap.read()
            
            if not success:
                break 
            
            scale_x = 1 / self.crop_ratio
            scale_y = 1 / self.crop_ratio
            
            #Creating a scaling matrix 
            #[scale_x 0 0  
            # 0 scale_y 0 
            # 0 0 1 ]
            scaling_matrix = np.eye(3, dtype=float)
            scaling_matrix[0][0] = scale_x
            scaling_matrix[1][1] = scale_y
            
            
            #Creating a translation matrix 
            #T = [
            #1 0 -sel.w/2
            #0 1 -self.h/2    
            #0 0 1 ]
            shifting_to_center_matrix = np.eye(3, dtype=float)
            shifting_to_center_matrix[0][2] = -self.w / 2.0
            shifting_to_center_matrix[1][2] = -self.h / 2.0
            
            #Creating a shift matrix 
            #T = [ 
            # 1 0 self.w/2
            # 0 1 self.h/2
            # 0 0 1 
            # ]
            shifting_back_matrix = np.eye(3, dtype=float)
            shifting_back_matrix[0][2] = self.w / 2.0
            shifting_back_matrix[1][2] = self.h / 2.0
            
            
            B_matrix = np.eye(3, dtype=float)
            B_matrix[:2][:] = B_transforms[i, :, :2].T #B_matrix is the transformation matrix for each frame 
            #While B_transforms is the cumulative transformation matrix across all frames
                   
            #Creating a final matrix by the below operations
            final_matrix = shifting_back_matrix @ scaling_matrix @ shifting_to_center_matrix @ np.linalg.inv(B_matrix)

            #Applying the transformation matrix to the frame
            frame_stabilized = cv2.warpAffine(frame, final_matrix[:2, :], self.shape)
            
            #Rotate 180 degree due to weird CV2 behavior 
            frame_out = cv2.rotate(frame_stabilized , cv2.ROTATE_180)
            self.out.write(frame_out)    
                
    def stabilize(self):
        
        self.write_output() 
        self.cap.release()
        self.out.release()
        
        return self.dest 