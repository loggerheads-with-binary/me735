import numpy as np
import cv2 as cv
import os
from scipy.interpolate import UnivariateSpline

#Resizes individual frames 
##NOTE: This function was used after wierd scaling found in direct cv2 list comprehension 
def resize_to_fit(frame, size):
    
    w_d, h_d = size #To crop to 
    actual_height, actual_width = tuple(frame.shape[:2]) #actual shape 
    factor = min(w_d / actual_width, h_d / actual_height)
    return cv.resize(frame, (0,0), fx=factor, fy=factor) #To not make a non uniform scaling of the frame we choose min scaler 

#Standard fix_border operation across every method 
def fix_border(frame  ):
    
    s = frame.shape
    
    # Scale the image 10% without moving the center
    T = cv.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.1)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))       ##Standard cv2 warping
    return frame   

#Returns smoothened trajectory using an FFT based method
def smooth(vals, fps, freq_cutoff):
    # take the points, mirror them and add them to the end to complete a loop
    # which is necessary for the fourier transform to work.
    
    vals_mirrored = np.array(list(vals) + list(reversed(list(vals)))[1:-1])
    N = vals_mirrored.shape[0]
    f = np.fft.fftfreq(N) * fps #Standard FFT frequency calculation

    vals_mirrored_fft = np.fft.fft(vals_mirrored) #Mirroring the values for FFT calculation as is standard procedure 
    vals_mirrored_fft[np.abs(f) > freq_cutoff] = 0 #Discarding high freqency values 
    vals_mirrored_smoothed = np.real(np.fft.ifft(vals_mirrored_fft)) #Inverse FFT to get the smoothened trajectory
    return vals_mirrored_smoothed[:vals.shape[0]] #Returning the smoothened trajectory

#Creating univariate B-spline with uniform periodic knots. 
#Returning a reverse normalized of normalized trajectory to not let the spline go out of bounds
def spline(vals, factor=None):
    
    #Normalizing the spline 
    v_min = np.min(vals)
    v_max = np.max(vals)
    v_norm = (vals - v_min) / (v_max - v_min)
    
    #Creating the uniform periodic knots 
    ts = np.arange(len(vals))            
               
    spl = UnivariateSpline(ts, v_norm, k = 3 , s=factor) #s is used to choose number of knots for B-spline smoothing. 
    v_norm_smoothed = spl(ts) #Applying spline 
    return v_norm_smoothed * (v_max - v_min) + v_min #De normalizing the spline

class Stabilizer():
    
    def __init__(self , src, dest_folder , fname):
        
        self.video = src
        self.dest = os.path.join(dest_folder, fname )
 
    def stabilize(self):    

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 17,
                            blockSize = 7 ) #Standard values from references 
        
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) #Standard values from references 
        
        # Take first frame in grayscale and find features to track 
        cap = cv.VideoCapture(self.video)
        ret, frame = cap.read()
        old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        frame_transforms = list()

        while(1):

            ret, frame = cap.read()

            if not ret:
                break

            # calculate LK optical flow
            new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]    #St==1 indicates trackable feature 
            good_old = p0[st==1]

            ##No points to track if either is zero 
            if good_new.size == 0 or good_old.size == 0:
                print('points lost')
                break

            transform, _ = cv.estimateAffinePartial2D(p0, p1) #Estimate displacement  
            frame_transforms.append(transform) #Create list of trajectories

            old_gray = new_gray.copy()
            

        freq_cutoff = 0.2       #Cut frequency for low pass filter 
        fps = cap.get(cv.CAP_PROP_FPS)                
        
        frame_transforms = np.array(frame_transforms)

        #Decomposing displacements into x,y,theta components
        tx = frame_transforms[:, 0, 2]
        ty = frame_transforms[:, 1, 2]
        t_theta = np.arctan2(frame_transforms[:, 1, 0], frame_transforms[:, 0, 0])
        
        # calculate trajectory from transforms
        trajectory_x = np.cumsum(tx)
        trajectory_y = np.cumsum(ty)
        trajectory_theta = np.cumsum(t_theta)
        
        # splined trajectory
        tx_splined = spline(trajectory_x, 0.7)
        ty_splined = spline(trajectory_y, 0.7)
        t_theta_splined = spline(trajectory_theta, 0.7)

        # Curve out trajectory using spline values 
        #tx added to get individual frame values since trajectory_x is cumsum 
        tx_splined = tx + tx_splined - trajectory_x
        ty_splined = ty + ty_splined - trajectory_y
        t_theta_splined = t_theta + t_theta_splined - trajectory_theta


        s_splined = 1 # otherwise the video will be very sheared (from reference)
        smooth_transform = np.zeros(frame_transforms.shape)
        smooth_transform[:, 0, 0] = s_splined *  np.cos(t_theta_splined) # s *  cos
        smooth_transform[:, 1, 0] = s_splined *  np.sin(t_theta_splined) # s *  sin
        smooth_transform[:, 0, 1] = s_splined * -np.sin(t_theta_splined) # s * -sin
        smooth_transform[:, 1, 1] = s_splined *  np.cos(t_theta_splined) # s *  cos
        smooth_transform[:, 0, 2] = tx_splined # tx
        smooth_transform[:, 1, 2] = ty_splined # ty

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        output_writer = None 

        # Read first frame to get an idea about frame size
        # and reset stream back to the start

        cap = cv.VideoCapture(self.video)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        for frame in smooth_transform:
    
            ret, frame = cap.read()
    
            if not ret:
                break 

            frame_size = frame.shape 
            
            stabilized = fix_border(cv.warpAffine(frame, frame, frame_size))
            ##Fix border fixes the black border that appears after warping the image.            

            if output_writer is None:
                output_writer = cv.VideoWriter(
                            self.dest,
                            fourcc,
                            fps,
                            tuple(reversed(list(stabilized.shape[:2]))),
                            True
                        )   #Instantiate the writer object 


            output_writer.write(stabilized) #Write to the destination file 

        output_writer.release()
        print('\ndone')
        return self.dest 
