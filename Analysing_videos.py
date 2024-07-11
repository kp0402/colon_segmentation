import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from focal_loss import BinaryFocalLoss
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve2d,resample,detrend,butter,lfilter
from skimage.filters import threshold_otsu,threshold_multiotsu
from skimage import exposure
from tracking import tracking,define_tracking_region
from freqency import plot_fft_plus_power
from utilities import *
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2 


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# Set GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU configured successfully.")
else:
    print("No GPU devices found.")

def load_simple_unet_model(model_path):
    model = load_model(model_path, custom_objects={'BinaryFocalLoss': BinaryFocalLoss, 'jacard_coef': jacard_coef})
    return model

simple_model = load_simple_unet_model('.../colon_10kimages_model_best.hdf5')

#Global settings
SIZE_X,SIZE_Y = 224,224
gradient = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
M = None #rotation matrix

def register_tracking_points(event,x,y,flags,param):
    global pts_queue
    if event == cv.EVENT_RBUTTONDOWN:
        pts_queue.append((x, y))

def OnInit(video_path):
    global cap, ROI, ptr_scores, p0
    global frame_count,LEFT_BOUND,RIGHT_BOUND 
    global M
    frame_count = 0; p0=None

    cap = cv.VideoCapture(video_path)
    if (cap.isOpened()== False):
        print(video_path)
        print("Error opening video stream or file")
        quit()
    size, fps = (int(cap.get(3)),int(cap.get(4))),cap.get(5)
    print(f"video size: {size}, fps: {fps}")

    print("-------------please select the RoI-------------")
    ret, frame = cap.read()
    cv.imshow('ROI selection',frame)
    #M,frame = Utilities.rotate(frame,'ROI selection')
    ROI = cv.selectROI(windowName='ROI selection',img=frame)
    print(ROI)
            
    bounding_box = cv.selectROI(windowName='ROI selection',img=frame)
    if bounding_box != (0,0,0,0):
        LEFT_BOUND,RIGHT_BOUND = int(bounding_box[0]),int(bounding_box[0]+bounding_box[2]-ROI[2])
        p0 = define_tracking_region(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), ROI)
        if p0 is not None:
            p0 = p0.reshape((p0.shape[0],2))
            ptr_scores = np.zeros((p0.shape[0]))
    else:
        print("Tracking disabled")

    cv.destroyAllWindows()
    return frame[:,:,0],size, fps

def moving_average(data,window_size=15):
    flat_data = np.asarray(data).ravel()
    smoothed_data = np.convolve(flat_data, np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def readframe(skip=0,as_gray=False):
    global cap, frame_count, M
    ret, frame = cap.read()
    if M is not None:
        frame = cv.warpAffine(frame, M, (frame.shape[0],frame.shape[1]))

    if not ret:
         print("End of video")
    if as_gray:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    frame_count = frame_count + skip + 1
    return ret, frame

def map1d(distance):
    global STmap
    if STmap is None:
        STmap = [np.float32(distance)]
    else:
        STmap.append(np.float32(distance))

def diamap(top,bottom):
    global sample_locations
    if top is not None and bottom is not None:
        upper_mean = np.mean(top)
        lower_mean = np.mean(bottom)
        if sample_locations is None:
            sample_locations = [(upper_mean,lower_mean)]
        else:
            sample_locations.append((upper_mean,lower_mean))   


def plot(map_type = 1,filename=None):
    global time_scale,STmap,num_of_pts,pixel_scale_y,pixel_scale_x
    def remap(transistion_point,array):
        k1 = transistion_point[1]/transistion_point[0]
        k2 = (1-transistion_point[1])/(1-transistion_point[0])
        cutoff = transistion_point[0]
        return np.where(array<cutoff, k1*array,k2*array-k2+1)

    STmap = np.vstack(STmap)
    STmap = np.negative(STmap)
    STmap_detrend = detrend(STmap,axis=0)
    
    d = np.median(STmap,axis=1)
    if filename is not None:
        data_to_save = {"STmap":STmap,
                        "mean_distance":d,
                        "scales":np.array([pixel_scale_x,pixel_scale_y]),
                        "breathing":breathing_movement,
                        "sample_rate": Fs}
        np.savez(filename,**data_to_save)

    smoothed_data = moving_average(d)
    fig1,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("diameter(cm)")
    ax2.set_xlabel("Frequency(s)")
    ax2.set_ylabel("Normalized PSD")
    ax1.plot(time_scale,smoothed_data)
    plot_fft_plus_power(ax2,time_scale, detrend(d),True)
    DiaMap = np.ones((SIZE_X,len(sample_locations)),dtype=np.uint8)

    if map_type == 1:
        for col in range(DiaMap.shape[1]):
            y1,y2 = sample_locations[col]
            DiaMap[int(y1):int(y2),col] = 0
        plt.imshow(DiaMap,cmap='gray')
        plt.title("Diameter map")
        plt.axis('off')
        plt.show()
    else:
        b, a = butter(11, 0.3, fs=Fs, btype='low', analog=False)
        STmap = lfilter(b, a, STmap_detrend, axis=0)
        min_img = np.min(STmap)
        max_img = np.max(STmap)
        normalized = (STmap-min_img)/(max_img - min_img)
        p2, p98 = np.percentile(normalized, (2, 98))
        normalized = exposure.rescale_intensity(normalized, in_range=(p2, p98))
        normalized = remap((0.5,0.25),normalized)
        upscaled = resample(normalized,SIZE_Y,axis=1)
        yticks = ["{:.1f}".format(i) for i in pixel_scale_y * np.linspace(0,SIZE_Y,10)]
        ST_MAP = Interactive_frequencyplot(unit_time)
        ST_MAP.plot_stmap(upscaled,yticks)
        ST_MAP = Interactive_frequencyplot(unit_time)
        ST_MAP.plot_wavelet(STmap_detrend,d,wavelet_window,yticks)

def measure_distance(lower_edge,upper_edge,sensitivity=5):
    global pixel_scale_x
    if lower_edge is not None and len(lower_edge) > 0:
        ret = np.zeros_like(lower_edge)
        for i in range(len(lower_edge)):
            dis = (lower_edge[i] - upper_edge[i])//sensitivity * sensitivity
            ret[i] = dis*pixel_scale_x
        return ret

def preprocess_frame(frame):
    # Resize and normalize the frame to match the input format expected by your model
    frame = cv.resize(frame, (256, 256))
    frame = frame / 255.0  # Normalize to the range [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def draw_upper_and_lower_edges(image, upper_edges, lower_edges, color_upper=(0, 0, 255), color_lower=(0, 255, 0), thickness=4):
    # Make a copy of the image to draw the edges on
    image_with_edges = image.copy()

    # Draw upper edges in red
    if upper_edges is not None:
        if upper_edges.ndim == 1:
            upper_edges = upper_edges.reshape(-1, 2)  # Reshape as a 2D array with two columns
        for point in upper_edges:
            x, y = point
            cv2.circle(image_with_edges, (int(x), int(y)), thickness, color_upper, -1)

    # Draw lower edges in green
    if lower_edges is not None:
        if lower_edges.ndim == 1:
            lower_edges = lower_edges.reshape(-1, 2)  # Reshape as a 2D array with two columns
        for point in lower_edges:
            x, y = point
            cv2.circle(image_with_edges, (int(x), int(y)), thickness, color_lower, -1)

    return image_with_edges

def find_corners_from_segmented_mask(largest_contour):
    # Attempt to find convex hull
    hull = cv2.convexHull(largest_contour)

    # Get the corner points of the convex hull
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if hull is not None and len(approx) == 4:
        corners = approx

        # Sort corners by their x-coordinate (left to right)
        corners = sorted(corners, key=lambda point: point[0][0])

        # Split sorted corners into top and bottom halves
        left_corners = sorted(corners[:2], key=lambda point: point[0][1])
        right_corners = sorted(corners[2:], key=lambda point: point[0][1])

        top_left, bottom_left = left_corners
        top_right, bottom_right = right_corners

        return top_left, top_right, bottom_left, bottom_right

    else:
        # Fit a rotated rectangle to the largest contour
        rect = cv2.minAreaRect(largest_contour)
    
        # Get the box points of the rotated rectangle
        box = cv2.boxPoints(rect).astype(int)
    
        # Sort the box points by their x-coordinate (left to right)
        box = sorted(box, key=lambda point: point[0])
    
        # Split sorted box points into top and bottom halves
        left_corners = sorted(box[:2], key=lambda point: point[1])
        right_corners = sorted(box[2:], key=lambda point: point[1])
    
        top_left, bottom_left = left_corners
        top_right, bottom_right = right_corners
    
        return top_left, top_right, bottom_left, bottom_right

    
def find_nearest_point_on_contour(point, contour):
    # Calculate the distances from the point to all points on the contour
    distances = [np.linalg.norm(point - contour_point[0]) for contour_point in contour]

    # Find the index of the nearest point
    nearest_index = np.argmin(distances)

    return contour[nearest_index][0]

def extract_edges_with_points(top_left, top_right, bottom_left, bottom_right, contour, num_points=10):
    if top_left.ndim == 1:  # Check if corner points are 1D arrays
        # Access the coordinates of the corner points
        top_left_x, top_left_y = top_left[0], top_left[1]
        top_right_x, top_right_y = top_right[0], top_right[1]
        bottom_left_x, bottom_left_y = bottom_left[0], bottom_left[1]
        bottom_right_x, bottom_right_y = bottom_right[0], bottom_right[1]
    else:
        # Access the coordinates of the corner points
        top_left_x, top_left_y = top_left[0][0], top_left[0][1]
        top_right_x, top_right_y = top_right[0][0], top_right[0][1]
        bottom_left_x, bottom_left_y = bottom_left[0][0], bottom_left[0][1]
        bottom_right_x, bottom_right_y = bottom_right[0][0], bottom_right[0][1]

    # Initialize arrays for the upper and lower edges
    upper_edge = []
    lower_edge = []

    # Interpolate points along the upper edge
    for i in range(num_points):
        alpha = i / (num_points - 1)
        x = int((1 - alpha) * top_left_x + alpha * top_right_x)
        y = int((1 - alpha) * top_left_y + alpha * top_right_y)

        # Find the nearest point on the contour
        nearest_point = find_nearest_point_on_contour((x, y), contour)

        upper_edge.append(nearest_point)

    # Interpolate points along the lower edge
    for i in range(num_points):
        alpha = i / (num_points - 1)
        x = int((1 - alpha) * bottom_left_x + alpha * bottom_right_x)
        y = int((1 - alpha) * bottom_left_y + alpha * bottom_right_y)

        # Find the nearest point on the contour
        nearest_point = find_nearest_point_on_contour((x, y), contour)

        lower_edge.append(nearest_point)

    return upper_edge, lower_edge
    
def detect_boundaries(segmentation_mask):
    # Convert the segmentation mask to a binary image by thresholding
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255

    # Find contours in the binary mask
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Assuming you want the first and last contours as the upper and lower boundaries
    if contours:
        upper_boundary = contours[0]
        lower_boundary = contours[-1]
        return upper_boundary, lower_boundary
    else:
        return None, None

def draw_boundaries(frame, upper_boundary, lower_boundary):
    # Create a copy of the frame to draw boundaries on
    frame_with_boundaries = frame.copy()

    # Draw upper boundary in green
    if upper_boundary is not None:
        cv.drawContours(frame_with_boundaries, [upper_boundary], -1, (0, 255, 0), 2)

    # Draw lower boundary in red
    if lower_boundary is not None:
        cv.drawContours(frame_with_boundaries, [lower_boundary], -1, (0, 0, 255), 2)

    return frame_with_boundaries

def extract_edges_from_segmented_mask_1(segmented_mask, num_points=10):
    # Find the contours in the segmented mask
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None  # No contours found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the corner points of the largest contour
    corner_points = find_corners_from_segmented_mask(largest_contour)

    if corner_points is None or any(point is None for point in corner_points):
        return None, None  # The largest contour doesn't have four corners or some corners are not found

    top_left, top_right, bottom_left, bottom_right = corner_points

    # Calculate the upper and lower edges with interpolated points matched to the contour
    upper_edge, lower_edge = extract_edges_with_points(top_left, top_right, bottom_left, bottom_right, largest_contour, num_points)

    return upper_edge, lower_edge

def segmentation(img_bgr,method,args=0.9):
    global VGG_filters, model

    if method == 'RF':
        img_input = cv.resize(img_bgr,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        img_input=cv.medianBlur(img_input,7)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = preprocess_input(img_input)
        features = VGG_filters.predict(img_input)
        x = features.reshape(-1,features.shape[-1])
        prob = (model.predict_proba(x)[:,1]).reshape((SIZE_X,SIZE_Y))
        mask = mean_field_approx(prob[4:-4,:],3)
    elif method == 'Multiotsu':
        img_input = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        img_input = cv.resize(img_input,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        mask = threhold_multilevel(img_input,thred=args)* np.uint8(255)
        mask = cv.medianBlur(mask,11)
    elif method == 'Otsu':
        img_input = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        img_input = cv.resize(img_input,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        mask = threhold(img_input)
        mask = cv.medianBlur(mask,11)
    elif method == 'simple_model':
        image = cv.resize(img_bgr, (256, 256))
        img_norm = image/255.0
        img_input = np.expand_dims(img_norm, axis=0)
        mask = simple_model.predict(img_input)
        mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8)
        mask = (mask > 0.5) 
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))

    return mask
        
def draw_edges_as_circles(frame, upper_edge, lower_edge):
    frame_with_circles = frame.copy()

    for point in upper_edge:
        point = tuple(map(int, point))
        cv2.circle(frame_with_circles, point, 4, (0, 0, 255), -1)

    for point in lower_edge:
        point = tuple(map(int, point))
        cv2.circle(frame_with_circles, point, 4, (0, 255, 0), -1)

    return frame_with_circles

if __name__ == "__main__":
#Parameters (EVERYTHING THAT NEEDS TO BE FILLED IN FOR CODE TO RUN)  
    video_path = r".../video_path/video.avi"
    map_type = 2            #1d: diameter map + freqency; 2d:2d color map
    skip_frame = 2          #number of frame skipped
    num_of_pts = 250         #number of points for edge tracking   
    wavelet_window = np.linspace(8,256,200)  #200 steps

#initialization
    method = "simple_model"
    old_frame_gray,_,fps = OnInit(video_path)
    unit_time = (1+skip_frame)/fps
    Fs = 1/unit_time
    print(f"sample rate {Fs}")
    anchor = float(ROI[0])
    sample_locations =None
    
#scale, cm per pixel
    cm_per_pix = Utilities.get_pixscale(old_frame_gray,"draw 1cm box") # type: ignore
    pixel_scale_y = ROI[2]/(cm_per_pix*SIZE_Y)    
    pixel_scale_x = ROI[3]/(cm_per_pix*SIZE_X)

#collect data
    STmap = None
    breathing_movement = []
    pts_queue = []
    dx = 0
    init_state=True
    
    cv.namedWindow('frame')
    cv.setMouseCallback('frame',register_tracking_points)
    while(1):
        ret, frame = readframe(skip_frame,as_gray=False)
        if ret:
            frame_gray = frame[:,:,0]
            key = cv.waitKey(1)
            if key & 0xff == 27:
                break 
            if key & 0xff == 97:#A
                anchor-=10
            if key & 0xff == 100:#D
                anchor+=10
            frame_1 = frame.copy()
            overlay = frame.copy()

            imCrop = frame[int(ROI[1]):int(ROI[1]+ROI[3]), int(anchor):int(anchor+ROI[2])] 
            mask = segmentation(frame,method)
            
            # Resize the binary mask to match the overlay dimensions
            mask_resized = cv.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)

            # Set the transparency of the resized binary mask in the overlay
            overlay[mask_resized == 1] = [0, 255, 0]
            
            u, l = detect_boundaries(mask_resized)
    
            frame_with_contour = draw_boundaries(frame, u, l)
        
            if p0 is not None:
                p0, ptr_scores,dx = tracking(frame_gray,old_frame_gray,p0,ptr_scores,len(pts_queue)!=0, pts_queue)
                for i,(x,y) in enumerate(p0):
                    cv.circle(frame,(int(x),int(y)),1,(255,0,0),4)      
                anchor = min(RIGHT_BOUND,max(LEFT_BOUND,dx+anchor))
            breathing_movement.append(anchor-ROI[0])   
                
            upper_edge, lower_edge = extract_edges_from_segmented_mask_1(mask_resized, num_of_pts)

            if upper_edge is not None and lower_edge is not None:
                frame = draw_edges_as_circles(frame, upper_edge, lower_edge)
            
             #Convert to one-dimensional arrays
            if upper_edge is not None and lower_edge is not None:
                upper_edge_flat = np.concatenate([edge.flatten() for edge in upper_edge])
                lower_edge_flat = np.concatenate([edge.flatten() for edge in lower_edge])

                distance = measure_distance(lower_edge_flat, upper_edge_flat, 1)
                diamap(upper_edge_flat, lower_edge_flat)
                map1d(distance)
            
            else:
                # Handle the case where either upper_edge or lower_edge is None
                print("Edges not found in the current frame.")

            old_frame_gray = frame_gray.copy()
            init_state = False
            
            cv.imshow('frame_with_contour', frame_with_contour)
            cv.imshow('frame', frame)
        else:
            break 

    cv.destroyAllWindows()
    cap.release()
    
    STmap = np.array(STmap)

    d = np.median(STmap.astype(float), axis=1)
    
    data_to_save = {
    "STmap": STmap,
    "mean_distance": d,
    "scales": np.array([pixel_scale_x, pixel_scale_y]),
    "breathing": breathing_movement,
    "sample_rate": Fs
}
    
    filename = '.../video.npz' # saving the npz file to save the constraints if you wanna plot or get the figures again
    np.savez(filename, **data_to_save)
    
    breathing_movement = np.array(breathing_movement) * pixel_scale_y
    time_scale = np.arange(0,len(STmap))*unit_time

    plot(map_type,"temp")
    fig3,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
    ax1.plot(breathing_movement)
    plt.title("breathing movement tracking")
    plot_fft_plus_power(ax2, time_scale, breathing_movement,True)




