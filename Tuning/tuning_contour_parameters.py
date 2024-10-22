import cv2
import numpy as np

# Define a range of values for each parameter
gaussian_kernel_sizes = [(3, 3), (5, 5), (7, 7)]  # Possible GaussianBlur kernel sizes
dilation_kernel_sizes = [(3, 3), (5, 5), (7, 7)]  # Possible dilation kernel sizes
erosion_kernel_sizes = [(3, 3), (5, 5), (7, 7)]  # Possible erosion kernel sizes
canny_thresholds = [(50, 150), (100, 200), (150, 250)]  # Pairs of low and high thresholds for Canny
dilation_iterations = [1, 2, 3]  # Number of iterations for dilation
erosion_iterations = [1, 2, 3]  # Number of iterations for erosion

# Function to ensure all images have the same dimensions and type (convert grayscale to BGR if necessary)
def ensure_same_dimensions_and_type(img, target_size, target_type):
    # Resize image to target size
    resized_img = cv2.resize(img, target_size)
    
    # If the image is grayscale and the target is color, convert it to BGR
    if len(resized_img.shape) == 2 and target_type == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    
    # If the image is already BGR, no need to convert
    return resized_img

# Function to evaluate parameter settings and return intermediate images for the table
def evaluate_parameters(gray, blur_kernel, dilation_kernel, erosion_kernel, canny_thresh, dilate_iter, erode_iter):
    # Convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    # Dilation to strengthen the edges
    edges_dilated = cv2.dilate(edges, np.ones(dilation_kernel, np.uint8), iterations=dilate_iter)

    # Erosion to remove small unwanted edges
    edges_eroded = cv2.erode(edges_dilated, np.ones(erosion_kernel, np.uint8), iterations=erode_iter)

    return blurred, edges, edges_dilated, edges_eroded

# Image to process (replace 'path_to_image.jpg' with your actual image path)
frame = cv2.imread("example.jpg")
height, width = frame.shape[:2]  # Get the size of the original frame
roi_size = 1100
frame = frame[height//2 - roi_size//2 : height//2 + roi_size//2,
                        width//2 - roi_size//2 : width//2 + roi_size//2]

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"./results/GRAY.jpg", gray)

target_size = (roi_size,roi_size)    # Set the target size for consistency

# Perform grid search and store the results
results = []
row_images = []
target_type = frame.shape[-1]  # Target type (3 for color, 1 for grayscale)
idx = 0

for blur_kernel in gaussian_kernel_sizes:
    for canny_thresh in canny_thresholds:
        for dilate_iter in dilation_iterations:
            for erode_iter in erosion_iterations:
                for erode_kernel in erosion_kernel_sizes:
                    for dila_kernel in dilation_kernel_sizes:
                        
                        # Evaluate the current set of parameters and get intermediate images
                        blurred, edges, edges_dilated, edges_eroded = evaluate_parameters(
                            gray, blur_kernel, dila_kernel, erode_kernel, canny_thresh, dilate_iter, erode_iter
                        )

                        # Ensure all images have the same size and type
                        # gray = ensure_same_dimensions_and_type(gray, target_size, target_type)
                        # a_gray = cv2.putText(
                        #     gray.copy(),
                        #     f"Gray",
                        #     (10, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        # )

                        blurred = ensure_same_dimensions_and_type(blurred, target_size, target_type)
                        a_blurred = cv2.putText(
                            blurred.copy(),
                            f"Blur: {blur_kernel}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        )

                        edges = ensure_same_dimensions_and_type(edges, target_size, target_type)
                        a_edges = cv2.putText(
                            edges.copy(),
                            f"Canny: {canny_thresh}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        )

                        edges_dilated = ensure_same_dimensions_and_type(edges_dilated, target_size, target_type)
                        a_edges_dilated = cv2.putText(
                            edges_dilated.copy(),
                            f"Dilation_iter: {dilate_iter}", # dilation Kernel: {dila_kernel}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        )
                        a_edges_dilated = cv2.putText(
                            a_edges_dilated.copy(),
                            f"Dilation_kernel: {dila_kernel}", # dilation Kernel: {dila_kernel}",
                            (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8, cv2.LINE_AA
                        )


                        edges_eroded = ensure_same_dimensions_and_type(edges_eroded, target_size, target_type)
                        a_edges_eroded = cv2.putText(
                            edges_eroded.copy(),
                            f"Erosion_iter: {erode_iter}", #erosion kernel: {erode_kernel}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        )
                        a_edges_eroded = cv2.putText(
                            a_edges_eroded.copy(),
                            f"Erosion_kernel: {erode_kernel}", #erosion kernel: {erode_kernel}",
                            (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8, cv2.LINE_AA
                        )

                        # Stack intermediate results horizontally (as one row)
                        result_row = cv2.hconcat([a_blurred, a_edges, a_edges_dilated, a_edges_eroded])
                        
                        # # Annotate the row with the parameters
                        # annotated_row = cv2.putText(
                        #     result_row.copy(),
                        #     f"Blur: {blur_kernel}, Canny: {canny_thresh}, Dilation: {dilate_iter}, Erosion: {erode_iter}",
                        #     (500, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10, cv2.LINE_AA
                        # )
                        
                        # for idx, result in enumerate(results):
                        idx += 1
                        cv2.imwrite(f"./results/result_{idx}.jpg", result_row)#annotated_row)
                        # cv2.imwrite(f"result_{idx}.jpg", annotated_row)
                        # row_images.append(annotated_row)
                        
                        # exit();
