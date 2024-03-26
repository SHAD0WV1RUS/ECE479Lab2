import cv2
import picamera
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tflite_runtime.interpreter as tflite
import sys
from PIL import Image
import time

def capture_image():
    # Instrctor note: this can be directly taken from the PiCamera documentation
    # Create the in-memory stream
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 24
        time.sleep(2)
        image = np.empty((240, 320, 3), dtype=np.uint8)
        camera.capture(image, 'rgb')
        return image
    
def detect_and_crop(mtcnn, image):
    detection = mtcnn.detect_faces(image)[0]
    bounding_box = detection['box']
    bounding_box[0] = int(bounding_box[0] - 0.1 * bounding_box[2])
    bounding_box[1] = int(bounding_box[1] - 0.1 * bounding_box[3])
    bounding_box[2] = int(bounding_box[2] * 1.2)
    bounding_box[3] = int(bounding_box[3] * 1.2)
    return (image[bounding_box[0]:bounding_box[0]+bounding_box[3],bounding_box[1]:bounding_box[1]+bounding_box[2]], bounding_box)

def show_bounding_box(image, bounding_box, title="Image with Bounding Box"):
    x1, y1, w, h = bounding_box
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    ax.set_title(title)
    plt.show()
    return

def show_four_images_with_bounding_boxes(images, bounding_boxes):
    fig, ax = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            ax[i][j].imshow(images[2*i + j])
            x1, y1, w, h = bounding_boxes[2*i + j]
            ax[i][j].add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
            ax[i][j].set_title(f"Face #{2*i+j+1}:")
    plt.tight_layout()
    plt.show()
    return

def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret
                             
def run_model(model, face):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], np.expand_dims(pre_process(face), 0))
    model.invoke()
    return model.get_tensor(output_details[0]['index'])
        

def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a-b)))



mtcnn = MTCNN()
tfl_file = "./tflite_models/facial_recognition_model.tflite"
interpreter = tflite.Interpreter(model_path=tfl_file)
interpreter.allocate_tensors()
if '--compare' in sys.argv:
    cropped_images = [0,0,0,0]
    images = [0,0,0,0]
    bounding_boxes = [0,0,0,0]
    index = len(sys.argv)
    if "--imgs" in sys.argv:
        index = sys.argv.index("--imgs") + 1
    for i in range(4):
        if(index != len(sys.argv)):
            images[i] = np.asarray(Image.open(sys.argv[index]))[:,:,:3]
            index += 1
        else:
            input("Take picture? ")
            images[i] = capture_image()
            print("Picture taken")
        cropped_images[i], bounding_boxes[i] = detect_and_crop(mtcnn, images[i])
        show_bounding_box(images[i], bounding_boxes[i], f"Face {i+1}")
    output_features = np.empty((4,512))
    for i in range(4):
        output_features[i] = np.squeeze(run_model(interpreter, cropped_images[i]))
    for i in range(3):
        for j in range(i+1, 4):
            print(f"Euclidean Distance between Face {i+1} and {j+1}: {euclidean_distance(output_features[i], output_features[j])}")
    show_four_images_with_bounding_boxes(images, bounding_boxes)
    np.save("output_face_features.npy", output_features)
    print("Output features saved in output_face_features.npy")
    for i in range(4):
        Image.fromarray(images[i]).save(f"img_{i+1}.png")
        print(f"Original Image {i} saved in img_{i+1}.png")
else:
    input("Take picture?")
    image = capture_image()
    print("Picture taken")
    cropped_image, bounding_box = detect_and_crop(mtcnn, image)
    show_bounding_box(image, bounding_box)
    output_features = np.squeeze(run_model(interpreter, cropped_image))
    np.save("output_face_features.npy", output_features)
    print("Output features saved in output_face_features.npy")
    Image.fromarray(image).save("img.png")
    print("Original Image saved in img.png")

