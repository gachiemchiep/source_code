"""https://raw.githubusercontent.com/clovaai/deep-text-recognition-benchmark/master/create_lmdb_dataset.py"""

import cv2
import numpy as np
import os
import glob
from random import shuffle
import lmdb
import sys
sys.path.append("/datadrive/workspace/others/202203.toppan_ocr_2/sample_submit/src/PaddleOCR")
from ppocr.data.imaug.rec_img_aug import *
from multiprocessing import Pool
from tqdm import tqdm

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createTrainValList(in_dirs, val_per=0.2, fold_cnt=1, isTrain=True):
    
    image_paths = []
    labels = []
    
    for in_dir in in_dirs:
        
        image_paths_all = glob.glob("{}/*.jpg".format(in_dir))
        txt_paths = []

        val_cnt = int(len(image_paths_all) * val_per)
        start_idx = val_cnt * (fold_cnt - 1) 

        if isTrain:
            shuffle(image_paths_all)

        for image_path in image_paths_all:
            txt_path = image_path.replace(".jpg", ".txt")

            with open(txt_path, "r") as fid:
                line = fid.readline().rstrip()

            if len(line) <= 1:
                continue
                
            image_paths.append(image_path)
            labels.append(line.rstrip())
                
    return image_paths, labels

def read_label(img_path):
    txt_path = img_path.replace(".jpg", ".txt")
    with open(txt_path, "r") as fid:
        line = fid.readline().rstrip()
        
    if len(line) <= 1:
        return ""
    else:
        return line

def createTrainValList2(in_dirs, val_per=0.2, fold_cnt=1, isTrain=False):
    """
    multiple thread
    """
    image_paths = []
    
    for in_dir in in_dirs:
        image_paths_all = glob.glob("{}/*.jpg".format(in_dir))

        val_cnt = int(len(image_paths_all) * val_per)
        start_idx = (fold_cnt - 1) * val_cnt
        end_idx = fold_cnt * val_cnt
                
        if end_idx > len(image_paths_all):
            end_idx = len(image_paths_all)

        if isTrain:
            image_paths_ = image_paths_all[:start_idx] + image_paths_all[end_idx:]
        else:
            image_paths_ = image_paths_all[start_idx:end_idx]
            
        image_paths += image_paths_

            
    with Pool(40) as pool:
        image_labels = list(tqdm(pool.imap(read_label, image_paths), total=len(image_paths)))
            
    image_paths_ret = []
    labels_ret = []
    for idx in range(len(image_labels)):
        if not image_labels[idx]:
            continue
        else:
            image_paths_ret.append(image_paths[idx])
            labels_ret.append(image_labels[idx])
            
                
    return image_paths_ret, labels_ret

def createDataset(img_paths, labels, out_path, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(out_path, exist_ok=True)
    env = lmdb.open(out_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    nSamples = len(img_paths)
    idxs = list(range(nSamples))
        
    for idx in idxs:
        img_path = img_paths[idx]
        label = labels[idx]
        if not os.path.exists(img_path):
            print('%s does not exist' % img_path)
            continue
            
        with open(img_path, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    # print('%s is not a valid image' % imagePath)
                    continue
            except:
                # print('error occured', i)
                with open(out_path + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(idx))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 20000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

# https://stackoverflow.com/questions/17967320/python-opencv-convert-image-to-byte-string
def image_to_bts(frame):
    '''
    :param frame: WxHx3 ndarray
    '''
    _, bts = cv2.imencode('.webp', frame)
    bts = bts.tostring()
    return bts

def bts_to_img(bts):
    '''
    :param bts: results from image_to_bts
    '''
    buff = np.fromstring(bts, np.uint8)
    buff = buff.reshape(1, -1)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return img
    
def aug_1_img(img_path):
    
    img = cv2.imread(img_path, 1)
    new_img = warp_2(img, ang=10, use_tia=True, prob=0.6)
    new_img_bts = image_to_bts(new_img)
    
    return new_img_bts

def aug_1_img_2(img_path, label, idx, output_dir="/datadrive/workspace/others/202203.toppan_ocr_2/dataset/train_r90_aug"):
    
    img = cv2.imread(img_path, 1)
    new_img = warp_2(img, ang=10, use_tia=True, prob=0.95)
    
    img_name = os.path.basename(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    cv2.imwrite("{}/{}_{:07d}.jpg".format(output_dir, img_name_no_ext, idx), new_img)
    with open("{}/{}_{:07d}.txt".format(output_dir, img_name_no_ext, idx), "w") as fid:
        fid.write("{}\n".format(label))

def warp_2(img, ang, use_tia=True, prob=0.4):
    """
    warp : remove crop
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img


    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

    if config.perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    if config.crop:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = get_crop(new_img)

    if config.blur:
        if random.random() <= prob:
            new_img = blur(new_img)
    if config.color:
        if random.random() <= prob:
            new_img = cvtColor(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= prob:
            new_img = add_gasuss_noise(new_img)
    if config.reverse:
        if random.random() <= prob:
            new_img = 255 - new_img
    return new_img

def genImgAug(img_paths, labels, out_path, checkValid=False, augCnt=50):
    
    os.makedirs(out_path, exist_ok=True)


    nSamples = len(img_paths)
    idxs = list(range(nSamples))
        
    # check valid
    if checkValid:
        img_paths_checked = []
        labels_checked = []
        for img_path, label in zip(img_paths, labels):
            if not os.path.exists(img_path):
                continue
                
            with open(img_path, "rb") as fid:
                img_bts = fid.read()
                
            if not checkImageIsValid(imageBin):
                # print('%s is not a valid image' % img_path)
                continue
                
            img_paths_checked.append(img_path)
            labels_checked.append(label)
            
    else:
        img_paths_checked = img_paths
        labels_checked = labels
    
    # multple thread
    img_paths_ = img_paths_checked * augCnt
    labels_ = labels_checked * augCnt
    idxs_ = range(len(labels_))
    out_paths_ = [out_path] * len(labels_)
    
    for img_path, label, idx in zip(img_paths_, labels_, idxs_):
        aug_1_img_2(img_path, label, idx, out_path) 

def createDatasetWithAug(img_paths, labels, out_path, checkValid=False, augCnt=50):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(out_path, exist_ok=True)
    env = lmdb.open(out_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    nSamples = len(img_paths)
    idxs = list(range(nSamples))
        
    # check valid
    if checkValid:
        img_paths_checked = []
        labels_checked = []
        for img_path, label in zip(img_paths, labels):
            if not os.path.exists(img_path):
                continue
                
            with open(img_path, "rb") as fid:
                img_bts = fid.read()
                
            if not checkImageIsValid(imageBin):
                # print('%s is not a valid image' % img_path)
                continue
                
            img_paths_checked.append(img_path)
            labels_checked.append(label)
            
    else:
        img_paths_checked = img_paths
        labels_checked = labels
    
    # multple thread
    img_paths_ = img_paths_checked * augCnt
    labels_ = labels_checked * augCnt
    idxs_ = range(len(labels_))
    
    # each batch contains 5000 samples
    batch_size = 1000
    batch_cnt = int(len(labels_) / batch_size) + 1
    sample_cnt = 0
    for batch_idx in range(batch_cnt):
        print("Process : {}/{}".format(batch_idx+1, batch_cnt))
        start_idx = batch_idx*batch_size
        if batch_idx == (batch_cnt - 1):
            end_idx = len(labels_)

        else:
            end_idx = (batch_idx+1)*batch_size

            
        img_paths_batch = img_paths_[start_idx:end_idx]
        labels_batch = labels_[start_idx:end_idx]
        
        with Pool(10) as pool:
            aug_imgs = list(tqdm(pool.imap(aug_1_img, img_paths_batch), total=len(img_paths_batch)))
        
        for aug_img, label in zip(aug_imgs, labels_batch):
            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = aug_img
            cache[labelKey] = label.encode()

#             output_dir = "/datadrive/workspace/others/202203.toppan_ocr_2/dataset/aaa/"
#             aug_img_cv2 = bts_to_img(aug_img)
#             cv2.imwrite("{}/{}.jpg".format(output_dir, cnt), aug_img_cv2)
#             with open("{}/{}.txt".format(output_dir, cnt), "w") as fid:
#                 fid.write("{}\n".format(label))
            
            if cnt % 20000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
            
        del aug_imgs
        
        
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

    
def createTrainValListKaggle(in_dir="/datadrive/workspace/others/202203.toppan_ocr/dataset/kuzushiji-recognition", val_per=0.2, fold_cnt=1):
    
    image_paths_ = []
    labels_ = []
    
    with open("{}/lines_label_train.txt".format(in_dir), "r") as fid:
        lines = fid.readlines()
        
    for line in lines:
        image_name, label = line.rstrip().split(" ")
        image_path = "{}/lines_img_train/{}".format(in_dir, image_name)

        if not os.path.exists(image_path):
            continue

        image_paths_.append(image_path)
        labels_.append(label)

    val_cnt = int(len(image_paths_) * val_per)
    start_idx = val_cnt * (fold_cnt - 1) 

    image_paths = image_paths_[0:start_idx] + image_paths_[start_idx+val_cnt:]
    labels = labels_[0:start_idx] + labels_[start_idx+val_cnt:]

    return image_paths, labels