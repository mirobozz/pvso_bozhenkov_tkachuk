from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()


# while cv2.waitKey() != ord('q'):
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     image = cv2.resize(image,(240,240))
    

#     cv2.imshow("test", image)
#     cv2.waitKey()   

my_list = []

for i in range(4):
    #get data and pass them from camera to img
    cam.get_image(img)
    image = img.get_image_data_numpy()
    cv2.imshow("test", image)

    image = cv2.resize(image, (240, 240))


    #cv2.waitKey()
    #get raw data from camera
    #for Python2.x function returns string
    #for Python3.x function returns bytes
    data_raw = img.get_image_data_raw()

    #transform data to list
    data = list(data_raw)
    my_list.append(image)

    #print image data and metadata
    print('Image number: ' + str(i))
    print('Image width (pixels):  ' + str(img.width))
    print('Image height (pixels): ' + str(img.height))
    print('First 10 pixels: ' + str(data[:10]))
    print('\n')

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print(len(my_list))

my_image_1 = cv2.hconcat(my_list[0:2])
my_image_2 = cv2.hconcat(my_list[2:4])

my_image = cv2.vconcat([my_image_1, my_image_2])

print(my_image.shape)

my_image = my_image[:,:,:3]

#======2=======

my_image[0:240, 240:480] = cv2.cvtColor(
                           cv2.cvtColor(my_image[0:240, 240:480],
                           cv2.COLOR_RGB2GRAY), 
                           cv2.COLOR_GRAY2BGR)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float32)

my_image[0:240, 240:480] = cv2.filter2D(my_image[0:240, 240:480], cv2.CV_64F, sobel_x)
my_image[0:240, 240:480] = cv2.filter2D(my_image[0:240, 240:480], cv2.CV_64F, sobel_y)

#======3=======

rotated_image = my_image[240:480, 0:240].copy()
rotated_image = np.zeros(rotated_image.shape, dtype=rotated_image.dtype)    
for i in range(rotated_image.shape[0]):
    for j in range(rotated_image.shape[1]):
        rotated_image[j, rotated_image.shape[0]-1-i] = my_image[240:480, 0:240][i, j]

my_image[240:480, 0:240] = rotated_image

#======4=======

my_image[240:480, 240:480, 0:2] = 0



cv2.imshow("test", my_image)
cv2.waitKey(0)


print('Done.')