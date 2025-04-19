from ultralytics import YOLO
import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

model = YOLO("yolo11n.pt")  # pretrained YOLO11n model


name=model.names
while True:
    ret, frame = cam.read()
    res=model(frame,conf=.5)
    # Process results list
    for result in res:
        boxeses = result.boxes  # Boxes object for bounding box outputs
        boxes=boxeses.xyxy
        clss=boxeses.cls
        conf=boxeses.conf
        # print("clss",clss)
        for box,cls,conf in zip(boxes,clss,conf):
            x1,y1=int(box[0]),int(box[1])
            x2,y2=int(box[2]),int(box[3])
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
            frame = cv2.putText(frame, f"{name[int(cls)]}", (x1,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()