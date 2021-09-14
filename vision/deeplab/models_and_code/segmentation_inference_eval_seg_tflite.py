import numpy as np  
import sys
import os 
import cv2
import tensorflow as tf
#from PIL import Image

test_dir = '../deeplabv3_reorg/ADE20K/validation/images'
val_label_dir = '../deeplabv3_reorg/ADE20K/validation//annotations'
#net_file = './checkpoints/post_train_quant/first500.tflite'
net_file = './checkpoints/float/freeze.tflite'

# Consider the model input is float or uint8
is_float_input = True # input type is float or uint8
is_input_0_255 = False # input is [0,255] or [-1,1]

num_classes = 32

interpreter = tf.lite.Interpreter(model_path=net_file)
interpreter.allocate_tensors()

mean_subtraction_value = 127.5

# This function takes the prediction and label of a single image, returns intersection and union areas for each class
# To compute over many images do:
# for i in range(Nimages):
#   (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
# IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab>0)

    # Compute area intersection:
    intersection = imPred * (imPred==imLab)
    (area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred,_) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab,_) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    
    return (area_intersection, area_union)

# This function takes the prediction and label of a single image, returns pixel-wise accuracy
# To compute over many images do:
# for i = range(Nimages):
#   (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = pixelAccuracy(imPred[i], imLab[i])
# mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
def pixelAccuracy(imPred, imLab):
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab>0)
    pixel_correct = np.sum((imPred==imLab)*(imLab>0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return (pixel_accuracy, pixel_correct, pixel_labeled)


def segment(imgfile, isquant=False):
    # open image
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not is_input_0_255:
        img = np.float32(img)
        img = img - mean_subtraction_value
        img = img / mean_subtraction_value   
    
    if is_float_input:    
        img = np.float32(img)
    else:
        img = np.uint8(img)
    
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index    = input_details[0]['index']
    argmax_index   = output_details[0]['index']
   
    interpreter.set_tensor(input_index, img[np.newaxis, ...])
    interpreter.invoke()
    labels = interpreter.get_tensor(argmax_index)
  
    return np.uint8(labels)
    

def generate_result(test_dir, max_iter=None, isquant=False):

    results     = []
    image_names = []
    
    if not max_iter:
        max_iter = len(os.listdir(test_dir))
     
    for i, im_name in enumerate(os.listdir(test_dir)):
        if i % 100 == 0:
            print(i)
        if i == max_iter:
            break
        image_names.append(im_name)
        out = segment(os.path.join(test_dir, im_name), isquant)
        results.append(np.copy(out))
        
    return image_names, results


def compute_miou(image_names, results):

    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    iou         = np.zeros(num_classes)
    
    for i in range(len(image_names)):
        imgName = image_names[i]
        
        gtLabelName, _ = imgName.split('.')
        gtLabelPath    = os.path.join(val_label_dir, gtLabelName+'.png')
        gtLabel        = cv2.imread(gtLabelPath)
        gtLabel        = cv2.cvtColor(gtLabel, cv2.COLOR_BGR2RGB)
        gtLabel        = np.asarray(gtLabel)        
        gtLabel_       = np.copy(gtLabel)
        gtLabel_[gtLabel_ >= 32] = 0
        
        predLabel = results[i]
        intersection, union = intersectionAndUnion(predLabel, gtLabel_[:, :, 0], num_classes)

        total_inter += intersection
        total_union += union
    
    for c in range(num_classes):
        if total_union[c] > 0:
            iou[c] = total_inter[c] / total_union[c]
        else:
            iou[c] = np.nan
            
    return np.nanmean(iou)


if __name__ == '__main__': 
    print(net_file)
    image_names, results = generate_result(test_dir=test_dir)
    mIuO  = compute_miou(image_names, results)

    print('mIoU is', mIuO)


