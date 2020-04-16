import imutils
import dlib
import cv2

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.fhog_object_detector("/home/vsmart/Works/trafcamanprengine/data/model/car_175x75.svm")

# Video capture source
cap = cv2.VideoCapture("/home/vsmart/Works/Dlib_SVM/testmodle/Oto_in_020719_13h.mp4")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

win = dlib.image_window()
writer = None

while True:

    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = imutils.resize(image, width=800)

    rects = detector(image)

    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        cv2.rectangle(image,(d.left(), d.top()),(d.right(),d.bottom()),(0,0,255),2)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("/home/vsmart/Works/Dlib_SVM/testmodle/8.avi", fourcc, 30,
            (image.shape[1], image.shape[0]), True)

    writer.write(image)

    win.clear_overlay()
    win.set_image(image)
    win.add_overlay(rects)
