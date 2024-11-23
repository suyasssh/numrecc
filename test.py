import os, keras
import numpy as np
import cv2


model = keras.models.load_model("model.keras")


for i in range(10):
    try:
        if not os.path.isfile(f"test/{i}.png"):
            print(f"File not found: {i}.png")
            continue

        img = cv2.imread(f"test/{i}.png")[:,:,0]
        img = np.invert(np.array([img]))
        pred = model.predict(img)
        print("Expected: %d found Digit: %d" % (i, np.argmax(pred)))
    except Exception as e:
        print(e)
