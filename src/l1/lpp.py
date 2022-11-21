import pulp
import numpy as np 
import cv2 as cv 

w1 , w2 , w3 = 10 , 1 , 100 #Weights for 1,2,3rd derivatives respectively
N = 6 #Dimensions => 2 translational, 1 rotational , 1 scaling, 1 shear and 1 aspect ratio 

#L1 Format for parameter vector (dx_t, dy_t, a_t, b_t, c_t, d_t)
c1 = c2 = c3 = [1, 1, 100, 100, 100, 100 ] #Can create different for different derivatives too 

# Takes im_shape, a tuple and crop ratio, a float < 1.0
#And returns the four corners of the image 
def get_corners(im_shape, crop_ratio):
	
	# Get center of original frames
	img_ctr_x = round(im_shape[1] / 2)
	img_ctr_y = round(im_shape[0] / 2)
	
	crop_w = round(im_shape[1] * crop_ratio)
	crop_h = round(im_shape[0] * crop_ratio)
	
	# Get upper left corner of centered crop window
	crop_x = round(img_ctr_x - crop_w / 2)
	crop_y = round(img_ctr_y - crop_h / 2)
	
	return crop_x, crop_w + crop_x, crop_y, crop_h + crop_y

## NOTE: Updates the F_transforms array -> make sure to be careful 
def get_inter_frame_transforms(cap, F_transforms, prev_gray):

	n_frames = F_transforms.shape[0]

	for i in range(n_frames):
		
		# Detect feature points in previous frame (or 1st frame in 1st iteration)
		# This will be used to apply Lucas Kanade Optical Flow to 
		prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
										  minDistance=30, blockSize=3)
		# Read next frame
		success, curr = cap.read()
		
		if not success:
			break
		
		# Convert to grayscale
		curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
		
		# Calculate optical flow (i.e. track feature points)
		curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

		# Filter out and use only valid points
		idx = np.where(status == 1)[0]
		
		# Update which points we should continue to maintain state for
		prev_pts = prev_pts[idx]
		curr_pts = curr_pts[idx]

		# Find transformation matrix for full 6 DOF affine transform
		m, T = cv.estimateAffine2D(curr_pts, prev_pts)  #Getting a 2x3 matrix for the involved affine transformations 
		F_transforms[i + 1, :, :2] = m.T #Multiplying with each other to get the transformation matrix 
		 
		# Move to next frame
		prev_gray = curr_gray
		
	return None 

##Direct emulation of the transformation formula from literature 
def transform(F_t : list[np.array] , p : np.array , t : float ):
	
	return [ p[t, 0] + F_t[2, 0] * p[t, 2] + F_t[2, 1] * p[t, 3],
				p[t, 1] + F_t[2, 0] * p[t, 4] + F_t[2, 1] * p[t, 5],
				F_t[0, 0] * p[t, 2] + F_t[0, 1] * p[t, 3],
				F_t[1, 0] * p[t, 2] + F_t[1, 1] * p[t, 3],
				F_t[0, 0] * p[t, 4] + F_t[0, 1] * p[t, 5],
				F_t[1, 0] * p[t, 4] + F_t[1, 1] * p[t, 5]
			] #L1 Transform 

##Crops the window based on shape and crop ratio 
def get_crop_window(im_shape : tuple , crop_ratio : float):
	
	assert crop_ratio <= 1, "Crop ratio must be less than or equal to 1"
	
	# Get center of original frames
	img_ctr_x = round(im_shape[1] / 2)
	img_ctr_y = round(im_shape[0] / 2)
	
	crop_w = round(im_shape[1] * crop_ratio)
	crop_h = round(im_shape[0] * crop_ratio)
	
	crop_x = round(img_ctr_x - crop_w / 2)
	crop_y = round(img_ctr_y - crop_h / 2)
	
	#Rendering the corner points from obtained crop widths and crop positions 
	corner_points = [
		(crop_x, crop_y),
		(crop_x + crop_w, crop_y),
		(crop_x, crop_y + crop_h),
		(crop_x + crop_w, crop_y + crop_h)
	]
  
	return corner_points  
	
def lppsolve(F_transforms : list[np.array] , frame_shape : tuple[int] , 
			crop_ratio  : float = 0.8):
	
	 # Create lpp minimization problem object
	prob = pulp.LpProblem("stabilize", pulp.LpMinimize)
	
	# Get the number of frames in sequence to be stabilized
	n_frames = len(F_transforms)
	
	# Get corners of decided crop window for inclusion constraints
	corner_points = get_crop_window(frame_shape, crop_ratio)
	
	#Creating all variables in a n_frames*N matrix 
	##Delcaring the decision variables 
 	# Matrix terms, all positive due to mod constraint  
	first_derivative_terms = pulp.LpVariable.dicts("e1", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
	
 	# Matrix terms for the second derivative,
	second_derivative_terms = pulp.LpVariable.dicts("e2", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
	
 	# Matrix terms for the third derivative,
	third_derivative_terms = pulp.LpVariable.dicts("e3", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
 
	# Stabilization parameters for each frame 
	p = pulp.LpVariable.dicts("p", ((i, j) for i in range(n_frames) for j in range(N)))

	#Formulation of LPP objective to be minimized 
	prob += w1 * pulp.lpSum([first_derivative_terms[i, j] * c1[j] for i in range(n_frames) for j in range(N)]) + \
			w2 * pulp.lpSum([second_derivative_terms[i, j] * c2[j] for i in range(n_frames) for j in range(N)]) + \
			w3 * pulp.lpSum([third_derivative_terms[i, j] * c3[j] for i in range(n_frames) for j in range(N)])
	
	# Apply smoothness constraints on the slack variables e1, e2 and e3 using params p
	for t in range(n_frames - 3):
		# Depending on in what form F_transforms come to us use raw p vectors to create smoothness constraints
		# No need to assemble p in matrix form
		translation_c1 = transform(F_transforms[t + 1], p, t + 1)
		translation_first_derivative_c1 = transform(F_transforms[t + 2], p, t + 2)
		translation_second_derivative_c1 = transform(F_transforms[t + 3], p, t + 3)
		
		constraint_term_t = [translation_c1[j] - p[t, j] for j in range(N)]
		constraint_term_d1 = [translation_first_derivative_c1[j] - p[t + 1, j] for j in range(N)]
		constraint_term_d2 = [translation_second_derivative_c1[j] - p[t + 2, j] for j in range(N)]
		
  		# Apply the smoothness constraints on the slack variables e1, e2 and e3
		#Creating constraints so that in a discrete frame of a continous variable, variables are smooth 
		#Essentially no weird jumps across frames for e1,e2,e3 matrix terms 
  
		for j in range(N):
			prob += -1*first_derivative_terms[t, j] <= constraint_term_t[j]
			prob += first_derivative_terms[t, j] >= constraint_term_t[j]
			prob += -1 * second_derivative_terms[t, j] <= constraint_term_d1[j] - constraint_term_t[j]
			prob += second_derivative_terms[t, j] >= constraint_term_d1[j] - constraint_term_t[j]
			prob += -1 * third_derivative_terms[t, j] <= constraint_term_d2[j] - 2*constraint_term_d1[j] + constraint_term_t[j]
			prob += third_derivative_terms[t, j] >= constraint_term_d2[j] - 2*constraint_term_d1[j] + constraint_term_t[j]
	
	# Constraints
	for t1 in range(n_frames):
		
		#Proximity constraints, ensuring that the center of the transformed frame is not too far away from the center of the original frame 
		for i in range(2 , 6):
			prob += (p[t1, i] >= 0.9) 
			prob += (p[t1, i] <= 1.1) 
   
		# Inclusion constraints, ensuring that the transformed frame is not too far away from the original frame
		prob += p[t1, 3] + p[t1, 4] >= -0.1
		prob += p[t1, 3] + p[t1, 4] <= 0.1
		prob += p[t1, 2] - p[t1, 5] >= -0.05
		prob += p[t1, 2] - p[t1, 5] <= 0.05
  
		# Inclusion Constraints
		#Only include what is supposed to be in the frame 
  		# Loop over all 4 corner points of centered crop window
		for (cx, cy) in corner_points:
			prob += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy >= 0 #Coordinate frame 0-width, 0-height 
			prob += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy <= frame_shape[1]
			prob += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy >= 0
			prob += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy <= frame_shape[0]
	
 
	# Pre allocate array for holding computed optimal stabilization transforms
	B_transforms = np.zeros((n_frames, 3, 3), np.float32)
	# Initialise all transformations with Identity matrix
	B_transforms[:, :, :] = np.eye(3)
	
	#Solve the LPP problem 
	prob.solve()
	
 	
 	#Check if the  solution successfully converges 
	if prob.status:
	
		print("Solution does converge")
		# Return the computed stabilization transforms
		for i in range(n_frames):
			B_transforms[i, :, :2] = np.array([[p[i, 2].varValue, p[i, 4].varValue],
											   [p[i, 3].varValue, p[i, 5].varValue],
											   [p[i, 0].varValue, p[i, 1].varValue]])
	else:
		
		print("Solution does not converge")
		#Return the identity matrices which will essentially generate the same old video 
		#Since the video cannot be stabilized 
  
	return B_transforms