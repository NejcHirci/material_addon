# https://pypi.org/project/cykooz.heif/
# Modified by Nejc Hirci from https://shuangz.com/projects/materialgan-sa20/

import sys
import numpy as np
from matplotlib import pyplot as plt
import shutil
from pupil_apriltags import Detector
import cv2
import os
import pickle
import glob
import obj_points
from PIL import Image
sys.path.insert(1, 'materialGAN/src/')
from util import *

np.set_printoptions(precision=4, suppress=True)

def rotate_needed(img):
    s = 8
    img = img.resize((round(img.width/s), round(img.height/s)), Image.LANCZOS)
    img = np.array(img.convert('L'))
    detector = Detector()
    result = detector.detect(img)
    c = result[0].corners
    c1,c2,c3,c4 = np.split(c,4, axis=0)
    c1 = c1[0];c2 = c2[0];c3 = c3[0];c4 = c4[0]
    if abs((c1[0]+c2[0])-(c3[0]+c4[0])) < abs((c1[1]+c2[1])-(c3[1]+c4[1])):
        if (c1[0]+c4[0]) < (c2[0]+c3[0]):
            return 0
        else:
            return 180
    else:
        if (c1[0]+c2[0]) < (c3[0]+c4[0]):
            return -90
        else:
            return 90

def marker_detection(img, in_dir, idx, flag=False):
    ##
    img_width = img.width if img.width > img.height else img.height
    s = max(1, img_width // 1000)
    img_resize = img.resize((round(img.width/s), round(img.height/s)), Image.LANCZOS)

    img_gray = img_resize.convert('L')
    img_gray = np.array(img_gray)
    detector = Detector()
    result = detector.detect(img_gray)

    tag_id_list = []
    center_list = []
    corners_list = []
    for marker in result:
        tag_id_list.append(marker.tag_id)
        center_list.append(marker.center.astype('float32') *s)
        corners_list.append(marker.corners.astype('float32') *s)

    if flag:
        plt.imshow(np.array(img))
        for center in center_list:
            plt.plot(center[0], center[1], 'k.')
        for corners in corners_list:
            plt.plot(corners[0,0], corners[0,1], 'r.')
            plt.plot(corners[1,0], corners[1,1], 'g.')
            plt.plot(corners[2,0], corners[2,1], 'b.')
            plt.plot(corners[3,0], corners[3,1], 'y.')
        plt.title(idx)
        # plt.show()
        plt.savefig(os.path.join(in_dir, 'detect_%02d.jpg' % idx))
        plt.close()
        # exit()

    imagePoints = np.vstack(corners_list)

    print('%d points are detected in image %d' % (imagePoints.shape[0], idx))

    return imagePoints, tag_id_list

def preprocess(in_dir, tmp_dir):
    print(os.path.join(in_dir, '*.*'))
    dir_list = sorted(glob.glob(os.path.join(in_dir, '*.*')))
    print(dir_list)

    if len(dir_list) == 0:
        print('No inputs given!')
        exit()

    for idx, dir in enumerate(dir_list):
        img = Image.open(dir)
        rot = rotate_needed(img)
        img = img.rotate(rot, expand=True)
        img.save(os.path.join(tmp_dir, 'orig_%02d.png' % idx), 'JPEG')

def calibrate(imgs, in_dir, objectPoints, flag):
    N = len(imgs)
    W = imgs[0].width
    H = imgs[0].height

    objectPoints_list = []
    imagePoints_list = []
    for idx in range(N):
        print('Calibrate image: %d' % idx)
        img = imgs[idx]
        imagePoints_this, tag_list = marker_detection(img, in_dir, idx, flag)
        imagePoints_list.append(imagePoints_this)
        objectPoints_this = []
        for tag in tag_list:
            objectPoints_this.append(objectPoints[tag*4:(tag+1)*4, :])
        objectPoints_this = np.vstack(objectPoints_this)
        objectPoints_list.append(objectPoints_this)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                            objectPoints_list, imagePoints_list,
                            (W,H), None, None)

    return mtx, dist, list(rvecs), list(tvecs), objectPoints_list, imagePoints_list

def reorder(arr, list1, list2, list3, list4, list5):
    def move_row_to_end(mat, id):
        mat = np.append(mat, np.reshape(mat[id,:], (-1,3)), axis=0)
        mat = np.delete(mat, id, axis=0)
        return mat

    def move_element_to_end(lis, id):
        lis.append(lis[id])
        del lis[id]
        return lis


    id = np.argmax( arr[:,0]+arr[:,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmax(-arr[:-1,0]-arr[:-1,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmax(-arr[:-2,0]+arr[:-2,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmax( arr[:-3,0]-arr[:-3,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmax( arr[:-4,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmin( arr[:-5,1])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmin( arr[:-6,0])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    id = np.argmax( arr[:-7,0])
    arr = move_row_to_end(arr, id)
    list1 = move_element_to_end(list1, id)
    list2 = move_element_to_end(list2, id)
    list3 = move_element_to_end(list3, id)
    list4 = move_element_to_end(list4, id)
    list5 = move_element_to_end(list5, id)

    return arr, list1, list2, list3, list4, list5

def process(in_dir, tmp_dir):

    image_size = 17.4

    ##
    objectPoints = obj_points.initObjPoints(image_size) # letter

    ##
    dir_list = sorted(glob.glob(os.path.join(tmp_dir, 'orig_*.png')))

    imgs = []
    for idx, dir in enumerate(dir_list):
        imgs.append(Image.open(dir))
    N = len(imgs)

    mtx, dist, rvecs, tvecs, _, _ = calibrate(imgs, tmp_dir, objectPoints, False)

    imgs2 = []
    for idx in range(N):
        print('Undistort image: %d' % idx)
        img = np.array(imgs[idx])
        #img = cv2.undistort(img, mtx, dist)
        imgs2.append(Image.fromarray(img))

    mtx, dist, rvecs, tvecs, objectPoints_list, imagePoints_list = \
            calibrate(imgs2, tmp_dir, objectPoints, True)

    cameraPos = []
    for idx in range(N):
        rmat, _ = cv2.Rodrigues(rvecs[idx])
        rmat_inv = np.linalg.inv(rmat)
        tvec = -tvecs[idx]
        cameraPos.append(np.matmul(rmat_inv,tvec))

    cameraPos = np.hstack(cameraPos).transpose()

    cameraPos, imgs2, objectPoints_list, imagePoints_list, rvecs, tvecs = \
        reorder(cameraPos, imgs2, objectPoints_list, imagePoints_list, rvecs, tvecs)

    ##
    xx, yy = np.meshgrid(np.linspace(-image_size/2,image_size/2,10),
                         np.linspace(-image_size/2,image_size/2,10))
    z = np.zeros_like(xx)

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2)
    plt3d.quiver(0,0,0,4,0,0)
    plt3d.quiver(0,0,0,0,4,0)
    plt3d.quiver(0,0,0,0,0,4)
    # plt3d.plot(np.array([0,1]),np.array([0,0]),np.array([0,0]))
    plt3d.set_xlim3d(-8, 8)
    plt3d.set_ylim3d(-8, 8)
    plt3d.set_zlim3d(0, 20)
    plt3d.scatter(cameraPos[:,0], cameraPos[:,1], cameraPos[:,2], color='green')
    # plt.show()
    plt.savefig(os.path.join(tmp_dir, 'camera_pos.jpg'))
    plt.close()
    # exit()

    f = open(os.path.join(tmp_dir,'store.pkl'), 'wb')
    pickle.dump([imgs2, objectPoints_list, rvecs, tvecs, mtx, dist, cameraPos, image_size], f)
    f.close()

def rectify(in_dir, tmp_dir, h=0.5):
    f = open(os.path.join(tmp_dir,'store.pkl'), 'rb')
    imgs2, objectPoints_list, rvecs, tvecs, mtx, dist, cameraPos, image_size = pickle.load(f)
    f.close()

    res = 1600
    crop = 1024
    N = len(imgs2)

    for idx in range(N):
        print('Rectifying image: %d' % idx)

        objectPoints = objectPoints_list[idx]
        warpPoints = objectPoints[:,:2].copy()
        warpPoints = (warpPoints/image_size + 0.5) * res
        warpPoints[:,1] = res - warpPoints[:,1]

        markerPoints = objectPoints.copy()
        markerPoints[:,2] = -h
        imagePoints, _ = cv2.projectPoints(
            markerPoints, rvecs[idx], tvecs[idx], mtx, dist)
        imagePoints = np.reshape(imagePoints, (imagePoints.shape[0],2))

        HomoMat, _ = cv2.findHomography(imagePoints, warpPoints)
        img_in = np.array(imgs2[idx])

        # plt.imshow(img_in)
        # plt.plot(imagePoints[:,0],imagePoints[:,1], 'r.')
        # plt.show()

        img_out = cv2.warpPerspective(img_in, HomoMat, (res,res))
        img_out_PIL = Image.fromarray(img_out)
        img_out_PIL.save(os.path.join(tmp_dir, 'marker_%02d.png' % idx))
        img_out_PIL_crop = img_out_PIL.crop(((res-crop)/2, (res-crop)/2, (res+crop)/2, (res+crop)/2))
        img_out_PIL_crop.save(os.path.join(tmp_dir, '%02d.png' % idx))


    cameraPos[:,2] += h
    # cameraPos /= 6
    # exit()
    np.savetxt(os.path.join(tmp_dir, 'camera_pos.txt'), cameraPos, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(tmp_dir, 'image_size.txt'), np.array([image_size*crop/res]), delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(tmp_dir, 'light_power.txt'), np.array([1500,1500,1500]).reshape(1,3), delimiter=',', fmt='%.4f')

def copy_to_folder(in_dir, out_dir, num_inputs):
    mat = os.path.split(in_dir)[-1]
    gyCreateFolder(out_dir)

    shutil.copyfile(os.path.join(in_dir, 'image_size.txt'), os.path.join(out_dir, 'image_size.txt'))
    shutil.copyfile(os.path.join(in_dir, 'light_power.txt'), os.path.join(out_dir, 'light_power.txt'))
    shutil.copyfile(os.path.join(in_dir, 'camera_pos.txt'), os.path.join(out_dir, 'camera_pos.txt'))
    shutil.copyfile(os.path.join(in_dir, 'camera_pos.txt'), os.path.join(out_dir, 'light_pos.txt'))
    rendered_all = None
    for i in range(num_inputs):
        shutil.copyfile(os.path.join(in_dir, '%02d.png' % i), os.path.join(out_dir, '%02d.png' % i))
        rendered_all = gyConcatPIL_h(rendered_all, Image.open(os.path.join(out_dir, '%02d.png' % i)).resize((256,256)))
    rendered_all.save(os.path.join(out_dir, 'rendered.png'))

    # Empty tmp dir
    shutil.rmtree(in_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--out_dir', required=True)

    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    tmp_dir = in_dir + '/tmp/'

    gyCreateFolder(tmp_dir)

    fn = in_dir
    print('pre-processing ', fn)
    dir_list = sorted(glob.glob(os.path.join(in_dir, '*.*')))
    num_inputs = len(dir_list)
    preprocess(fn, tmp_dir)
    print('processing ', fn)
    process(fn, tmp_dir)
    print('rectifying ', fn)
    rectify(fn, tmp_dir, h=0.5)
    print('post-processing ', fn)
    copy_to_folder(tmp_dir, out_dir, num_inputs)
