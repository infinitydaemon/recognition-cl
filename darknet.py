import cv2
import numpy as np
import pyopencl as cl

# Initialize OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Load and compile the OpenCL kernel code
kernel_code = """
__kernel void imageRecognition(__global uchar* input_image, int width, int height, __global float* output_data) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = y * width + x;

    // Access pixel values in the input image
    uchar b = input_image[3 * index];
    uchar g = input_image[3 * index + 1];
    uchar r = input_image[3 * index + 2];

    // Convert RGB to grayscale
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;

    // Store the grayscale value in the output data
    output_data[index] = gray;
}
"""
program = cl.Program(context, kernel_code).build()

def process_frame(frame):
    # Convert frame to RGB for compatibility with YOLO
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load the RGB image data into a numpy array
    image_data = np.array(rgb_frame, dtype=np.uint8)

    # Create OpenCL buffers for input and output data
    image_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image_data)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=image_data.nbytes // 3)

    # Set kernel arguments
    program.imageRecognition.set_args(image_buffer, rgb_frame.shape[1], rgb_frame.shape[0], output_buffer)

    # Execute the kernel
    event = cl.enqueue_nd_range_kernel(queue, program.imageRecognition, rgb_frame.shape[:2][::-1], None)

    # Wait for the kernel execution to finish
    event.wait()

    # Read the output data back to the CPU
    output_data = np.empty_like(image_data[:, :, 0], dtype=np.float32)
    cl.enqueue_copy(queue, output_data, output_buffer)

    # Normalize the grayscale values to the range [0, 1]
    output_data /= 255.0

    # Return the processed frame and the output data
    return rgb_frame, output_data

def main():
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Load class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Set up webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Process frame and perform object detection
        processed_frame, output_data = process_frame(frame)

        # Perform object detection with YOLO
        blob = cv2.dnn.blobFromImage(processed_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        # Parse YOLO outputs and draw bounding boxes on the frame
        class_ids = []
        confidences = []
        boxes = []
        height, width, _ = processed_frame.shape

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Perform non-maximum suppression to eliminate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            # Draw bounding box and label
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"{label}: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow("Image Recognition", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
