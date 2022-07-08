"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
#from typing import Tuple


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """
    
    #raise NotImplementedError
    dif  = np.sqrt(pow((p0[0] - p1[0]),2) + pow((p0[1] - p1[1]),2))
    return dif

def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """


   
    height, width = image.shape[:2]
    corners = []
    corners = [(0,0), (0,height-1), (width-1, 0),(width-1, height-1)]
    #raise NotImplementedError
    return corners




def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and/or corner finding and/or convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    
    #raise NotImplementedError
    #if template!=None:
    #    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #    edges = cv2.Canny(templateGray, 10, 100)
    #    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,20,param1=50,param2=10,minRadius=5,maxRadius=0)

    # get dimension
    h, w, _ = 0,0,0
    shape = np.shape(image)
    if len(shape) == 3:
        h,w,_ = shape
    else:
        h,w = shape
    #y = cv2.getGaussianKernel(h, 500)
    #x = cv2.getGaussianKernel(w, 500)
    #gKernel = y+x.T
    #mask = 255*gKernel
    #masl = mask/np.linalg.norm(gKernel)
    #if _!=0:
    #    for i in range(_):
    #        image[:,:,i] = image[:,:,i]*mask
    #else:
    #    image = image*mask
    # preset image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #getcorner
    harris = cv2.cornerHarris(gray, 8, 7, 0.05)
    harrisDilated = cv2.dilate(harris, None)
    cordc = np.where(harrisDilated >= .1 * harrisDilated.max())
    c = list(zip(*cordc[::-1]))
    c = np.asarray(c)
    c = np.float32(c)
    #kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    temp, pts, means = cv2.kmeans(c, 4, None, criteria, 50,cv2.KMEANS_PP_CENTERS)
    center = np.int16(means)
    center = sorted(center, key=lambda x: x[0])
    # print np.array(center)
    center = np.array(center)
    # get 4 corners
    L = sorted(center[:2], key=lambda x: x[1])
    R = sorted(center[2:], key=lambda x: x[1])

    return tuple(L[0]), tuple(L[1]), tuple(R[0]), tuple(R[1])


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """

    #raise NotImplementedError
    
    cv2.line(image, markers[0], markers[1], (0,0,255), thickness=thickness)
    cv2.line(image, markers[1], markers[3], (0,0,255), thickness=thickness)
    cv2.line(image, markers[2], markers[3], (0,0,255), thickness=thickness)
    cv2.line(image, markers[0], markers[2], (0,0,255), thickness=thickness)
    
    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    out_image = imageB
    shape = np.shape(out_image)
    h,w,_ = 0,0,0
    if len(shape) == 3:
        h,w,_ = shape
    else:
        h,w = shape
        c =1
        #set xy
    x,y = np.indices((h,w))
    stack = np.array([y.ravel(),x.ravel(),np.ones_like(y.ravel())])
    backWrap = np.dot(np.linalg.inv(homography), stack)
    #print(backWrap)
    map1,map2 = backWrap[:-1] / backWrap[-1]
    map1,map2 = map1.reshape((h, w)), map2.reshape((h, w))
    out_image = cv2.remap(imageA, map1=np.float32(map1),map2=np.float32(map2),interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    
    
    #raise NotImplementedError
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """

    #raise NotImplementedError
    temp = []
    for (a, b), (c,d) in zip(srcPoints, dstPoints):
        A = [-a,-b,-1,0,0,0,a*c, b*c, c]
        B = [0,0,0,-a,-b,-1,a*d, b*d, d]
        temp.append(A)
        temp.append(B)
    _,__,output = np.linalg.svd(np.array(temp))
    shape = np.shape(output)
    homo = output[shape[0]-1,:]
    index = output[shape[0]-1, shape[1]-1]
    homograph = homo/index
    homograph = homograph.reshape(3, 3)
    return homograph


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    # TODO
    #raise NotImplementedError
    
    while video.isOpened():
        flag, frame = video.read()
        if flag:
            yield frame
        else:
            break
        
    
    # Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)


    def filter(self, img, filter, padding=(0,0)):
        
        #raise NotImplementedError
        h, w = image.shape
        fh, fw = filter.shape
        img = cv2.copyMakeBorder(image, fh//2, fh//2, fh//2, fh//2, cv2.BORDER_CONSTANT)
        h2, w2 = img.shape
        output = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                temp = img[i : i + fh, j : j + fh]
                temp = np.multiply(temp, filter)
                output[i][j] = temp.sum()
        return output

    def gradients(self, image_bw):

        #raise NotImplementedError
        sobelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        Ix = filter(image_bw, sobelX)
        Iy = filter(image_bw, sobelY)
        
        #print(ix)
        #print(iy)
        return Ix, Iy

    def get_gaussian(self, ksize, sigma):

        #raise NotImplementedError
        print(ksize)
        k = cv2.getGaussianKernel(ksize, sigma)
        kernel = np.outer(k, k.transpose())    
        
        return kernel

    
    def second_moments(self, image_bw, ksize=7, sigma=10):

        sx2, sy2, sxsy = None, None, None
        Ix, Iy = self.gradients(image_bw)

        #raise NotImplementedError
        k = self.get_gaussian(ksize, sigma)
        # print("k")
        if (ksize == 1):
            sx2 = pow(ix, 2)
            sy2 = pow(iy, 2)
            sxsy = ix * iy
        else:     
            sx1 = pow(ix, 2)
            sy1 = pow(iy, 2)
            sxsy1 = ix * iy
            sx2 = my_filter2D(sx1, kernel, bias = 0)
            # print(sx1)
            # print(sx2)
            sy2 = filter(sy1, k)
            sxsy = filter(sxsy1, k)
        
        return sx2, sy2, sxsy

    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):

       #raise NotImplementedError
        sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
        h, w = sx2.shape
        R = np.zeros((h, w))
        for i in range (m):
            for j in range (n):
                flag = np.array([[sx2[i][j], sxsy[i][j]], [sxsy[i][j], sy2[i][j]]])
                R[i][j] = np.linalg.det(flag) - alpha * ((np.trace(flag)) ** 2)
        #print(R)
        return R

    def pool2d(self, A, kernel_size, stride, padding, pool_mode='max'):
        '''
        2D Pooling
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        '''
        # Padding
        A = np.pad(A, padding, mode='constant')

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = np.lib.stride_tricks.as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                    stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
            

    def nms_maxpool_numpy(self, R: np.ndarray, k, ksize):
        """Pooling function that takes in an array input
        Args:
            R (np.ndarray): Harris Response Map
            k (int): the number of corners that are to be detected with highest probability
            ksize (int): pooling size
        Return:
            x: x indices of the corners
            y: y indices of the corners
        """
    
        #raise NotImplementedError
        #Suppress any point below the median
        median = np.median(R)
        mask1 = R < median
        R[mask1] = 0
        #maxpool 
        data_max = pool2d(R, k,ksize, 0)
        mask = ((R == data_max) & (data_max != 0))
        R_local_pts = np.ma.masked_where(np.ma.getmask(mask), R)
        return R_local_pts
        
    def harris_corner(self,image_bw, k=100):
        """Harris Corner Detection Function that takes in an image and detects the most likely k corner points.
        Args:
            image_bw (np.array): black and white image
            k (int): top k number of corners with highest probability to be detected by Harris Corner
        RReturn:
            x: x indices of the top k corners
            y: y indices of the top k corners
        """   
        #raise NotImplementedError
        ix, iy = self.gradients(image_bw)   
        sx2, sy2, sxsy = self.second_moments(ix, iy)
        corners = self.harris_response_map(sx2, sy2, sxsy, alpha = 0.05)
        R_local_pts = self.nms_maxpool_numpy(corners, neighborhood_size = 7)
        m, n = np.shape(R_local_pts)
        x = []
        y = []
        c = []
        for i in range(m):
            for j in range(n):
                idx = i
                idy = j
                conf = R_local_pts[i][j]
                x.append(idx)
                y.append(idy)
                c.append(conf)
        
        m, n = np.shape(image_bw)[:2]
    
        xmatrix = np.reshape(x, (m, n))
        xmatrixnew = xmatrix[k: m-k, k: n-k]
        
        ymatrixnew = np.reshape(y, (m, n))[k: m-k, k: n-k]
        y = xmatrixnew.flatten()
        cmatrix = np.reshape(c, (m, n))
        x = ymatrixnew.flatten()
        cmatrixnew = cmatrix[k: m-k, k: n-k]
        c = cmatrixnew.flatten()

        cSorted = c.flatten()

        sortedidx = np.argsort(cSorted)

        sortedidx = sortedidx[: : -1][0 : n_pts]
        x = x[sortedidx]
        y = y[sortedidx]
        

        return x, y




    def calculate_num_ransac_iterations(
            self,prob_success: float, sample_size: int, ind_prob_correct: float):

        num_samples = None

        p = prob_success
        s = sample_size
        e = 1 - ind_prob_correct

        num_samples = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        print('Num of iterations', int(num_samples))

        return int(round(num_samples))




    def ransac_homography_matrix(self, matches_a: np.ndarray, matches_b: np.ndarray):

        p = 0.999
        s = 8
        sample_size_iter = 8
        e = 0.5
        threshold = 1
        numi = self.calculate_num_ransac_iterations(p, s, e)

        org_matches_a = matches_a
        org_matches_b = matches_b
        print('matches', org_matches_a.shape, org_matches_b.shape)
        matches_a = np.hstack([matches_a, np.ones([matches_a.shape[0], 1])])
        matches_b = np.hstack([matches_b, np.ones([matches_b.shape[0], 1])])
        in_list = []
        in_sum = 0
        best_in_sum = -99
        inliers = []
        final_inliers = []

        y = Image_Mosaic().get_homography_parameters(org_matches_b, org_matches_a)

        best_F = np.full_like(y, 1)
        choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
        print('s',org_matches_b[choice].shape,matches_b[choice].shape)
        best_inliers = np.dot(matches_a[choice], best_F) - matches_b[choice]
        print('inliers shape',best_inliers.shape,best_inliers)

        count = 0
        for i in range(min(numi, 20000)):
            
            choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
            match1, match2 = matches_a[choice], matches_b[choice]


            F = Image_Mosaic().get_homography_parameters(match2, match1)

            count += 1
            inliers = np.dot(matches_a[choice], F)- matches_b[choice]

            inliers = inliers[np.where(abs(inliers) <= threshold)]

            in_sum = abs(inliers.sum())
            best_in_sum = max(in_sum, best_in_sum)
            best_inliers = best_inliers if in_sum < best_in_sum else inliers

            if abs(in_sum) >= best_in_sum:
                # helper to debug
                # print('insum', in_sum)
                pass

            best_F = best_F if abs(in_sum) < abs(best_in_sum) else F


        for j in range(matches_a.shape[0]):
            final_liers = np.dot(matches_a[j], best_F) - matches_b[j]
            final_inliers.append(abs(final_liers) < threshold)

        final_inliers = np.stack(final_inliers)

        inliers_a = org_matches_a[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]
        inliers_b = org_matches_b[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]

        print('best F', best_F.shape, inliers_a.shape, inliers_b.shape, best_F, inliers_a, inliers_b)

        return best_F, inliers_a, inliers_b






class Image_Mosaic(object):

    def __int__(self):
        pass
    
    def image_warp_inv(self, im_src, im_dst, homography):
        #raise NotImplementedError
        h, w = np.shape(im_src)
        for  i in range(w):
            for j in range(h):
                p1 = np.asarray([[i],[j],[1.0]])
                p2 = np.matmul(homography, p1).squeeze()
                p2 = Point(int(float(p2[0])/p2[2]), int(float(p2[1])/p2[2]))
                if p2.x >= 0 and p2.y>=0 and p2.y<h and p2.x < w:
                    im_dst[p2.y, p2.x] = im[j, i,:]
                    
        return im_dst
                
        

    def output_mosaic(self, img_src, img_warped):
        
        #raise NotImplementedError

        
        return img_warped

    def get_homography_parameters(self, points2, points1):
        """
        leverage your previous implementation of 
        find_four_point_transform() for this part.
        """
        #raise NotImplementedError
        return find_four_point_transform(points2, points1)
            


                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   



