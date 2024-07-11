import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import roberts
from scipy.ndimage import convolve1d
from matplotlib.widgets import Slider,RangeSlider
import ipywidgets as widgets
from scipy.signal import resample

class BinaryMaskAnalyser:
    def __init__(self):
        self.sample_map = None
        self.indexing_array = None
        self.mask = None

    def register_mask(self,mask):
        assert mask.ndim == 2 and mask.dtype == np.uint8
        kernel = np.ones((3,3))
        self.mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    def register_sample_map(self,sample_map:np.ndarray,index_array):
        self.sample_map = sample_map
        self.indexing_array = index_array.astype(np.int32)

    def scanline_sampling(self,n_regions:int,k:float):
        global previous_1, previous_2, win_size

        increments = self.mask.shape[1]//n_regions
        mid = self.mask.shape[0]//2
        ret1,ret2 = [],[]
        edges = roberts(self.mask)>0
        c1 = convolve1d(previous_1,np.ones(3)/3)
        c2 = convolve1d(previous_2,np.ones(3)/3)
        for i in range(n_regions):
            y1,y2 = c1[i],c2[i]
            search_win1 = edges[y1-win_size:y1+win_size,i*increments:increments*(i+1)]
            search_win2 = edges[y2-win_size:y2+win_size,i*increments:increments*(i+1)]
            temp1=temp2= 0
            weight1=weight2=0
            for s in np.argwhere(search_win1>0):
                w = 1/(abs(s[0]-win_size)+1e-9)
                temp1 += w * (s[0]+y1-win_size)
                weight1 += w
            for s in np.argwhere(search_win2>0):
                w = 1/(abs(s[0]-win_size)+1e-9)
                temp2 += w * (s[0]+y2-win_size)
                weight2 += w
            
            if weight1 > 0:
                ret1.append(int(temp1/weight1))
                win_size +=1
            else:
                ret1.append(y1)
            if weight2 > 0:    
                ret2.append(int(temp2/weight2))
                win_size +=1
            else:
                ret2.append(y2)
            win_size = max(win_size-1,5)
        
        return np.array(ret1), np.array(ret2)
             
    def draw_contour(self,level,canvas):
        contours = self.get_contour(level)
        cv.drawContours(canvas, contours, -1, color=(0, 255, 0), thickness=2)
    
    def get_contour(self,level):
        contours, _ = cv.findContours(image=self.mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        s = np.array([len(i) for i in contours])
        rank = np.argsort(s)
        total = len(contours)-1
        level = max(total,level)
        ret = []
        for i in range(level):
            ret.append(contours[rank[total-i]])
        return tuple(ret)

    def refine_mask(self,level):
        new_mask = np.zeros_like(self.mask)
        contours = self.get_contour(level)
        cv.drawContours(new_mask, contours, -1, color=(127), thickness=5)


        x,y = new_mask.shape[0]//2,new_mask.shape[1]//2
        cv.floodFill(new_mask, None, (x,y), 255)
        return cv.threshold(new_mask , 128, 255, cv.THRESH_BINARY)[1]
    
    def sample_edge_location(self,trackingpts):
        contours = self.get_contour(2)
        counts = np.zeros((trackingpts.shape[0],trackingpts.shape[1]))
        new_trackingpts = np.zeros_like(trackingpts)

        for contour in contours:
            for i in range(contour.shape[0]):
                x,y = contour[i,0,:]
                g = self.sample_map[x,y]
                if g == -1: 
                    continue
                index_x,index_y = self.indexing_array[g]
                counts[index_x,index_y]+=1
                new_trackingpts[index_x,index_y,:] += np.array([x,y])

        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                new_trackingpts/=counts[i,j]
        return new_trackingpts


    def DiceLoss(self,mask2):
        IoU = np.bitwise_and(self.mask,mask2)
        denom = np.sum(self.mask) + np.sum(mask2)
        return 2*np.sum(IoU)/denom

class Utilities:
    mask = None
    button_down = False
    mode = True
    radius = 20
    mid_coord = None
    start_coord = None
    angle = 0
    angle_change = 0
    refPt = []

    def moving_average(a: np.ndarray,n:int):
        'a: input 1d array, n:average windows(odd)'
        if n%2!=1:
            n=n+1
        N = len(a)
        cumsum_vec = np.cumsum(np.insert(np.pad(a,(n-1,n-1),'constant'), 0, 0)) 
        d = np.hstack((np.arange(n//2+1,n),np.ones(N-n)*n,np.arange(n,n//2,-1)))  
        return (cumsum_vec[n+n//2:-n//2+1] - cumsum_vec[n//2:-n-n//2]) / d

    def rotate(original_img,windows_name="window"):
        M = None
        cv.namedWindow(windows_name)
        cv.setMouseCallback(windows_name,Utilities._click_to_rotate)
        Utilities.mid_coord = original_img.shape[0]//2, original_img.shape[1]//2
        img = original_img.copy()
        f = Utilities.angle
        while(1):
            new_angle = Utilities.angle_change + Utilities.angle
            if new_angle != f:
                f = new_angle
                rad = np.deg2rad(f)
                scale = max(abs(np.sin(rad)),abs(np.cos(rad)))

                M = cv.getRotationMatrix2D((Utilities.mid_coord), new_angle, scale)
                img = cv.warpAffine(original_img, M, (original_img.shape[0],original_img.shape[1]))
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            cv.imshow(windows_name,img)
        cv.destroyAllWindows()
        return M , img

    def get_pixscale(original_img,windows_name="window"):
        ROI = cv.selectROI(windowName=windows_name,img=original_img)
        return max(ROI[3],1)

    def manual_segmentation(original_img,reset=True):
        if Utilities.mask is None or reset == True:
            Utilities.mask = np.zeros_like(original_img)
        cv.namedWindow('Draw')
        cv.setMouseCallback('Draw',Utilities._draw_circle)
        
        while(1):
            dst = cv.addWeighted(original_img, 0.5, Utilities.mask, 0.5, 0.0)
            cv.imshow('Draw',dst)
            k = cv.waitKey(1) & 0xFF
            if k == ord('e'):
                Utilities.mode = not Utilities.mode
            elif k == 27:
                break
        cv.destroyAllWindows()
        return Utilities.mask  

    def register_regions(original_img):
        Utilities.mask = original_img
        cv.namedWindow('window')
        cv.setMouseCallback('window',Utilities._register_pts)
        img_cpy = Utilities.mask.copy() 
        while(1):
            if Utilities.button_down == False:
                cv.imshow('window',Utilities.mask)
            k = cv.waitKey(1) & 0xFF
            if k == ord('r'):
                Utilities.mask = img_cpy.copy()
                Utilities.refPt.clear()
            elif k == 27:
                break
        return Utilities.refPt
        

    def _register_pts(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            Utilities.start_coord = (x, y)
            Utilities.button_down = True
        elif event == cv.EVENT_LBUTTONUP:
            Utilities.button_down = False
            if (Utilities.start_coord[0] - x)**2 + (Utilities.start_coord[1] - y)**2 < 10: 
                print("Warning: line ingnored")
                return
            Utilities.refPt.append([Utilities.start_coord,(x, y)])
            cv.line(Utilities.mask, Utilities.start_coord, (x, y), (0, 255, 0), thickness=2)

        if Utilities.button_down:
            img_cpy = Utilities.mask.copy()    
            cv.line(img_cpy, Utilities.start_coord, (x, y), (0, 255, 0), thickness=2)
            cv.imshow('window',img_cpy)
            
    def _compose(rad,dx,dy):
        M = np.zeros((2,3),dtype='float')
        M[0,0] = np.cos(rad)   
        M[0,1] = -np.sin(rad)    
        M[1,0] = np.sin(rad) 
        M[1,1] = np.cos(rad)
        M[0,2] = dx
        M[1,2] = dy

        return M

    def _draw_circle(event,x,y,flags,param):
        if event == cv.EVENT_RBUTTONDOWN:
            Utilities.button_down = True
        elif event == cv.EVENT_MOUSEWHEEL:
            if flags > 0:
                Utilities.radius += 1
            else:
                Utilities.radius = max(1,Utilities.radius-1)
        elif event == cv.EVENT_RBUTTONUP:
            Utilities.button_down = False

        if Utilities.button_down :
            if Utilities.mode == True:
                cv.circle(Utilities.mask,(x,y),Utilities.radius,(255),-1)
            else:
                cv.circle(Utilities.mask,(x,y),Utilities.radius,(0),-1)

    def _click_to_rotate(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            Utilities.button_down = True
            Utilities.start_coord = np.array([x,y])
        elif event == cv.EVENT_LBUTTONUP:
            Utilities.button_down = False
            Utilities.angle += Utilities.angle_change
            Utilities.angle_change = 0

        if Utilities.button_down:
            v1 = (Utilities.start_coord - Utilities.mid_coord)
            v2 = np.array([x,y]) - Utilities.mid_coord
            v1 = v1/(np.linalg.norm(v1)+1e-9)
            v2 = v2/(np.linalg.norm(v2)+1e-9)
            if(v1[0] > v2[0]):
                Utilities.angle_change = np.rad2deg(np.arccos(np.sum(v1*v2)))
            else:
                Utilities.angle_change = -np.rad2deg(np.arccos(np.sum(v1*v2)))

class Segmenter:
    def __init__(self,cap, ROI,SIZE_X,SIZE_Y,feature_function):
        self.clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,random_state=64)
        self.cap = cap
        self.size = (SIZE_X,SIZE_Y)
        self.func = feature_function
        self.test_frame = None
        self.ROI = ROI

    def fit(self):
        x_train, y_train = self._generate_samples()
        x_train = self.func(x_train)
        y_train = y_train.reshape(-1)
        self.clf.fit(x_train,y_train)

    def predict(self,img):
        img = cv.medianBlur(img,7)
        img = np.expand_dims(img, axis=0)
        features = self.func(img)
        return self.clf.predict(features).reshape(self.size)

    def get_model(self):
        self.fit()
        return self.clf

    def _generate_samples(self):
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        ROI = self.ROI
        train_image = []
        training_label = []
        while(1):
            frame_number = np.random.randint(total_frames)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            assert ret is not None, "failed to read a new frame"

            frame = frame[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
            frame = cv.resize(frame,self.size,interpolation=cv.INTER_NEAREST)
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            cv.imshow("current frame",frame_gray)
            mask = Utilities.manual_segmentation(frame_gray)>0

            if not mask.any():
                self.test_frame = frame
                break
            else:
                train_image.append(cv.medianBlur(frame,7))
                training_label.append(mask)

        return np.array(train_image),np.array(training_label)

    def _compute_feature(self,frame):
        features = self.func(frame)

    def _generate_mask(self,frame):
        labels = Utilities.manual_segmentation(frame,reset=False)


class SnaptoCursor(object):
    MAX_POINTS = 3

    def __init__(self, ax, x, y):
        import matplotlib.widgets as widgets
        
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.x = x
        self.y = y
        self.pos = []
        self.x_loc,self.y_loc = 0,0
        self.value = 0
        #self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        #self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes:
             return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        indx = max(0,indx-1)
        self.x_loc, self.y_loc = self.x[indx], self.y[indx]
        self.ly.set_xdata(self.x_loc)
        #self.marker.set_data([self.x_loc],[self.y_loc])
        #self.txt.set_text('x=%1.2f, y=%1.2f' % (self.x_loc, self.y_loc))
        #self.txt.set_position((self.x_loc,self.y_loc))
        self.ax.figure.canvas.draw_idle()

    def push(self,ref):
        if len(self.pos) >= SnaptoCursor.MAX_POINTS:
            ext1,ext2 = self.pos.pop(0)
            ext1.remove()
            ext2.remove()
            self.ax.figure.canvas.draw_idle()
        self.pos.append(ref)

    def onclick(self,event):
        if not event.inaxes: return
        x,y = self.x_loc,self.y_loc
        ref1,=self.ax.plot(x, y, 'ro')
        ref2 = self.ax.text(x, y, f'Frequency: {x:.3f} Hz', fontsize=35,
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        self.push((ref1,ref2))

        
class Interactive_frequencyplot(object):
    def __init__(self, unit_time,imagesize=(224,224)):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(16, 20))
        self.image = None
        self.image_region = None
        self.allowed_frequency = None
        self.unit_time = unit_time
        self.size = imagesize
        self.artist = []
        plt.subplots_adjust(bottom=0.25)
        
    def update(self,val):
        index = np.where(self.allowed_frequency == val)[0][0]
        self.ly.set_ydata(index)
        step = len(self.allowed_frequency)
        selected_freqbin = self.image[index::step]
        self.image_region = selected_freqbin
        self.selected_freq.set_data(self.image_region)
        self.fig.canvas.draw_idle()

    def zoom(self,val):
        x1,x2 = int(val[0]),int(val[1])
        self.selected_freq.set_data(self.image_region[:,x1:x2])

        x_loc = np.linspace(0,self.image_region.shape[1],5,dtype=np.int32)
        x_ticks=[]
        start_time,end_time = x1*self.unit_time, x2*self.unit_time
        for i in np.linspace(start_time,end_time,5,dtype=np.int32):
            x_ticks.append(f"{i:.1f}")
        self.axs[1].set_xticks(x_loc)
        self.axs[1].set_xticklabels(x_ticks)
        self.fig.canvas.draw_idle()

    def get_slices_wavelet(self,data, scales):
        import pywt

        ret = []
        for i in range(data.shape[1]):
            coef, freqs=pywt.cwt(data[:,i],scales,'gaus2')
            ret.append(coef)  
        img = np.concatenate(ret,axis=0)
        img /= np.max(abs(img))
        img = np.where(img<0,img/4,img)

        return img,freqs/self.unit_time

    def get_mean_wavelet(self,data_mean, scales):
        import pywt
        coef, freqs=pywt.cwt(data_mean,scales,'gaus2')
        coef /= np.max(abs(coef))
        coef = np.where(coef<0,coef/4,coef)

        return coef

    # def plot_wavelet(self, data, mean, freq_scale, yticks):
    #     self.image, self.allowed_frequency = self.get_slices_wavelet(data, freq_scale)
    #     self.image_region = self.image
    #     full_img = self.get_mean_wavelet(mean, freq_scale)
    #     self.axs[0].contourf(full_img, levels=7, cmap="turbo")
    #     self.selected_freq = self.axs[1].imshow(self.image, cmap="turbo", interpolation='nearest', aspect='auto')
        
    #     slider_freq = widgets.FloatSlider(
    #         min=self.allowed_frequency[-1], max=self.allowed_frequency[0],
    #         step=(self.allowed_frequency[0] - self.allowed_frequency[-1]) / 100,
    #         value=self.allowed_frequency[0] * 0.1,
    #         description='Frequency (Hz)'
    #     )

    #     slider_time = widgets.FloatRangeSlider(
    #         min=0, max=self.image.shape[1], step=1, value=(0, self.image.shape[1]),
    #         description='Time (s)'
    #     )

    #     widgets.interact(self.update, freq=slider_freq)
    #     widgets.interact(self.zoom, time=slider_time)

    #     x_loc, y_loc = [], []
    #     x_ticks, y_ticks = [], []

    #     for i in np.linspace(0, self.allowed_frequency.shape[0]-1, 4, dtype=np.int32):
    #         y_ticks.append(f"{self.allowed_frequency[i]:.3f}")
    #         y_loc.append(i)

    #     end_time = self.unit_time * self.image.shape[1]
    #     x_loc = np.linspace(0, self.image.shape[1], 10, dtype=np.int32)

    #     for i in np.linspace(0, end_time, 10, dtype=np.int32):
    #         x_ticks.append(f"{i:.0f}")

    #     self.axs[0].set_yticks(y_loc)
    #     self.axs[0].set_xticks(x_loc)
    #     self.axs[0].set_yticklabels(y_ticks)
    #     self.axs[0].set_xticklabels(x_ticks)
    #     self.axs[0].set_ylabel("Frequency (Hz)")
    #     self.axs[0].set_xlabel("Time (s)")
    #     self.axs[1].set_ylabel("Location (cm)")
    #     self.axs[1].set_xlabel("Time (s)")

    #     y_loc = np.linspace(0, self.image.shape[0], len(yticks))
    #     self.axs[1].set_yticks(y_loc)
    #     self.axs[1].set_yticklabels(yticks)

    #     plt.colorbar(self.selected_freq, ax=self.axs[0])
    #     plt.show()


    # def plot_wavelet(self,data,mean,freq_scale, yticks):
    #     self.image,self.allowed_frequency = self.get_slices_wavelet(data,freq_scale)
    #     self.image_region = self.image
    #     full_img = self.get_mean_wavelet(mean,freq_scale)
    #     self.axs[0].contourf(full_img,levels=7,cmap="turbo")
    #     self.selected_freq = self.axs[1].imshow(self.image,cmap="turbo",interpolation='nearest',aspect='auto')
    #     slider_ax = plt.axes([0.20, 0.15, 0.60, 0.03])
    #     self.ly = self.axs[0].axhline(color='crimson', alpha=1, linewidth=2, linestyle='--')

    #     sfreq = Slider(
    #         slider_ax, "Frequency", self.allowed_frequency[-1], self.allowed_frequency[0],
    #         valinit=0.1, valstep=self.allowed_frequency,
    #         color="green"
    #     )
    #     timeslider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    #     timeslider = RangeSlider(
    #         timeslider_ax, "time", 0, self.image.shape[1],valstep=1
    #         )
    #     sfreq.on_changed(self.update)
    #     timeslider.on_changed(self.zoom)

    #     x_loc,y_loc=[],[]
    #     x_ticks,y_ticks=[],[]
    #     for i in np.linspace(0,self.allowed_frequency.shape[0]-1,4,dtype=np.int32):
    #         y_ticks.append(f"{self.allowed_frequency[i]:.3f}")
    #         y_loc.append(i)
    #     end_time = self.unit_time * self.image.shape[1]

    #     x_loc = np.linspace(0,self.image.shape[1],10,dtype=np.int32)
    #     for i in np.linspace(0,end_time,10,dtype=np.int32):
    #         x_ticks.append(f"{i:.0f}")

    #     self.axs[0].set_yticks(y_loc)
    #     self.axs[0].set_xticks(x_loc)
    #     self.axs[0].set_yticklabels(y_ticks)
    #     self.axs[0].set_xticklabels(x_ticks)
    #     self.axs[0].set_ylabel("Frequency(Hz)")
    #     self.axs[0].set_xlabel("Time(s)")
    #     self.axs[1].set_ylabel("Location(cm)")
    #     self.axs[1].set_xlabel("Time(s)")

    #     y_loc = np.linspace(0,self.image.shape[0],len(yticks))
    #     self.axs[1].set_yticks(y_loc);self.axs[1].set_yticklabels(yticks)

    #     plt.colorbar(self.selected_freq, ax=self.axs[0])
    #     plt.show()

    def plot_wavelet(self, data, mean, freq_scale, yticks):
        self.image, self.allowed_frequency = self.get_slices_wavelet(data, freq_scale)
        self.image_region = self.image
        full_img = self.get_mean_wavelet(mean, freq_scale)
        self.axs[0].contourf(full_img, levels=7, cmap="turbo")
        self.selected_freq = self.axs[1].imshow(self.image, cmap="turbo", interpolation='nearest', aspect='auto')
        slider_ax = plt.axes([0.20, 0.15, 0.60, 0.03])
        self.ly = self.axs[0].axhline(color='crimson', alpha=1, linewidth=2, linestyle='--')
    
        sfreq = Slider(
            slider_ax, "Frequency", self.allowed_frequency[-1], self.allowed_frequency[0],
            valinit=0.1, valstep=self.allowed_frequency,
            color="green"
        )
        timeslider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
        timeslider = RangeSlider(
            timeslider_ax, "time", 0, self.image.shape[1], valstep=1
        )
        sfreq.on_changed(self.update)
        timeslider.on_changed(self.zoom)
    
        x_loc, y_loc = [], []
        x_ticks, y_ticks = [], []
        for i in np.linspace(0, self.allowed_frequency.shape[0] - 1, 4, dtype=np.int32):
            y_ticks.append(f"{self.allowed_frequency[i]:.3f}")
            y_loc.append(i)
        end_time = self.unit_time * self.image.shape[1]
    
        x_loc = np.linspace(0, self.image.shape[1], 10, dtype=np.int32)
        for i in np.linspace(0, end_time, 10, dtype=np.int32):
            x_ticks.append(f"{i:.0f}")
    
        self.axs[0].set_yticks(y_loc)
        self.axs[0].set_xticks(x_loc)
        self.axs[0].set_yticklabels(y_ticks, fontsize=15)
        self.axs[0].set_xticklabels(x_ticks, fontsize=15)
        self.axs[0].set_ylabel("Frequency(Hz)", fontsize=15)
        self.axs[0].set_xlabel("Time(s)", fontsize=15)
        self.axs[1].set_ylabel("Location(cm)", fontsize=15)
        self.axs[1].set_xlabel("Time(s)", fontsize=15)

        self.axs[1].set_xticks(x_loc)
        self.axs[1].set_xticklabels(x_ticks, fontsize=15)  # Set font size for x-tick labels of the second map
        
        y_loc = np.linspace(0, self.image.shape[0], len(yticks))
        self.axs[1].set_yticks(y_loc)
        self.axs[1].set_yticklabels(yticks, fontsize=15)
        plt.colorbar(self.selected_freq, ax=self.axs[0]).ax.tick_params(labelsize=15)
        plt.subplots_adjust(hspace=0.4)
        plt.show()


    # def plot_stmap(self,data,yticks):
    #     self.image_region = data.T
    #     self.axs[0].imshow(data.T,cmap="turbo",interpolation='bilinear',aspect='auto')
    #     self.selected_freq = self.axs[1].imshow(self.image_region,cmap="turbo",interpolation='bilinear',aspect='auto')
    #     timeslider_ax = plt.axes([0.25, 0.1, 0.60, 0.03])
    #     timeslider = RangeSlider(
    #         timeslider_ax, "time", 0, data.shape[0],valstep=1
    #         )
    #     timeslider.on_changed(self.zoom)

    #     timeticks = np.linspace(0,data.shape[0],10)
    #     ticklabels = ["{:.0f}".format(i) for i in timeticks*self.unit_time]

    #     y_loc = np.linspace(0,data.shape[1],len(yticks))
    #     self.axs[0].set_xticks(timeticks);self.axs[0].set_xticklabels(ticklabels)
    #     self.axs[0].set_yticks(y_loc);self.axs[0].set_yticklabels(yticks)
    #     self.axs[1].set_yticks(y_loc);self.axs[1].set_yticklabels(yticks)
    #     self.axs[0].set_xlabel("time(s)")
    #     self.axs[0].set_ylabel("location(cm)")
    #     self.axs[1].set_xlabel("time(s)")
    #     self.axs[1].set_ylabel("location(cm)")
    #     plt.colorbar(self.selected_freq, ax=self.axs[0])
    #     plt.show()

    def plot_stmap(self, data, yticks):
        self.image_region = data.T
        self.axs[0].imshow(data.T, cmap="turbo", interpolation='bilinear', aspect='auto')
        self.selected_freq = self.axs[1].imshow(self.image_region, cmap="turbo", interpolation='bilinear', aspect='auto')
        timeslider_ax = plt.axes([0.25, 0.1, 0.60, 0.03])
        timeslider = RangeSlider(
            timeslider_ax, "time", 0, data.shape[0], valstep=1
        )
        timeslider.on_changed(self.zoom)
    
        timeticks = np.linspace(0, data.shape[0], 10)
        ticklabels = ["{:.0f}".format(i) for i in timeticks * self.unit_time]
    
        y_loc = np.linspace(0, data.shape[1], len(yticks))
        self.axs[0].set_xticks(timeticks)
        self.axs[0].set_xticklabels(ticklabels, fontsize=15)  # Increase fontsize here
        self.axs[0].set_yticks(y_loc)
        self.axs[0].set_yticklabels(yticks, fontsize=15)
        self.axs[1].set_yticks(y_loc)
        self.axs[1].set_yticklabels(yticks)
        self.axs[0].set_xlabel("Time(s)", fontsize=15)  # Increase fontsize here
        self.axs[0].set_ylabel("Location(cm)", fontsize=15)  # Increase fontsize here
        self.axs[1].set_xlabel("Time(s)", fontsize=15)  # Increase fontsize here
        self.axs[1].set_ylabel("Location(cm)", fontsize=15)  # Increase fontsize here
        cbar = plt.colorbar(self.selected_freq, ax=self.axs[0])
        cbar.ax.tick_params(labelsize=15)  # Increase colorbar tick label size here
        # Adjust the layout to add space between the plots
        plt.subplots_adjust(hspace=0.4)
        plt.show()
    
def features_gen(img):
    x_train = preprocess_input(img)
    features = VGG_filters.predict(x_train)
    return features.reshape(-1,features.shape[-1])


#test
if __name__ == "__main__":
    fig, ax = plt.subplots()
    x = np.linspace(0,10,100)
    y = np.sin(x)
    cursor = SnaptoCursor(ax, x, y)
    plt.connect('motion_notify_event', cursor.mouse_move)        
    plt.connect('button_press_event', cursor.onclick)
    ax.plot(x, y)
    plt.show()

    