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
fidx =1

max_score = 0
max_result = None


for blur_kernel in gaussian_kernel_sizes:
    for canny_thresh in canny_thresholds:
        for erode_kernel in erosion_kernel_sizes:
            for dila_kernel in dilation_kernel_sizes:
                for erode_iter in erosion_iterations:
                    for dilate_iter in dilation_iterations:
                        
                        # Evaluate the current set of parameters and get intermediate images
                        blurred, edges, edges_dilated, edges_eroded = evaluate_parameters(
                            gray, blur_kernel, dila_kernel, erode_kernel, canny_thresh, dilate_iter, erode_iter
                        )

                        # SCORE ALL PHOTOS:
                    #  ========================

                        contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        min_contour_area = 3000

                        if(len(contours) > 0):
                        # for contour in contours:
                            contour = max(contours, key=cv2.contourArea)

                            if len(contour) > 5:
                                area = cv2.contourArea(contour)
                                perimeter = cv2.arcLength(contour, True)
                                # aspect_ratio = calculate_aspect_ratio(contour)
                                
                                # if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                                #     return 0  # Not an ellipse-like shape
                                
                                ellipse = cv2.fitEllipse(contour)
                                # fit_error = calculate_ellipse_fit_error(ellipse, contour)
                                
                                # Circularity score (closer to 1 is more circular)
                                # circularity = (4 * np.pi * area) / (perimeter ** 2)
                                
                                # score = (area_weight * area) - (fit_error_weight * fit_error) + (circularity_weight * circularity)
                                score = area
                                
                                if(score > max_score):
                                    max_score = score
                                    tmp_result = frame.copy()
                                    cv2.ellipse(tmp_result,  ellipse, (0, 255, 0), 2)
                                    # max_result = tmp_result
                                    # max_blurred = blurred
                                    # max_edges = edges
                                    # max_edges_dilated = edges_dilated
                                    # max_edges_eroded = edges_eroded



                                #     PRINT ALL PHOTOS:
                                # ========================
                                #     Ensure all images have the same size and type
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
                                    max_result = cv2.hconcat([ a_blurred, a_edges, a_edges_dilated, a_edges_eroded, tmp_result])


cv2.imwrite(f"max_result.jpg", max_result)
# while True:
#     cv2.imshow("best result", max_result)

#     if cv2.waitKey(1) == ord('q'):
#         break

                        # # Annotate the row with the parameters
                        # annotated_row = cv2.putText(
                        #     result_row.copy(),
                        #     f"Blur: {blur_kernel}, Canny: {canny_thresh}, Dilation: {dilate_iter}, Erosion: {erode_iter}",
                        #     (500, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10, cv2.LINE_AA
                        # )
                        
                        # for idx, result in enumerate(results):
                        # idx += 1
                        # # cv2.imwrite(f"./results/result_{idx}.jpg", result_row)#annotated_row)
                        # # cv2.imwrite(f"result_{idx}.jpg", annotated_row)
                        # # row_images.append(annotated_row)
                        
                        # # exit()
                        
                        # comparing_edges_eroded = cv2.putText(
                        #     edges_eroded.copy(),
                        #     f"ID: {idx}",
                        #     (10, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA
                        # )
                        # row_images.append(comparing_edges_eroded)
                        
                        # # Create rows of images (e.g., 4 rows in the final grid)
                        # if len(row_images) == 9:
                        #     results.append(cv2.hconcat(row_images))
                        #     row_images = []

                        # if len(results) == 9:
                        #     final_result = cv2.vconcat(results)
                        #     cv2.imwrite(f"./results/final_result_batch_{fidx}.jpg", final_result)
                        #     fidx += 1
                        #     results = []