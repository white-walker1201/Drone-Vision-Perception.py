import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

print("Upload your image (Roundabout or Thermal Pool)...")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path)

# Resize for faster and more stable processing
img = cv2.resize(img, (800, 600))

# ==========================================
# Task 1: Precision Landing Pad Finder
# ==========================================
def task1_landing_pad_finder(img):
    output = img.copy()

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Use a stronger Gaussian Blur to ignore cars, road lines, and trees
    # This helps focus on the large, distinct round shape of the center
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 3. Strict Circle Detection
    # param2 is increased to 45 to force a much "cleaner" circle match
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,      # High distance to avoid multiple locks on the same area
        param1=100,
        param2=45,        # Higher = more perfect the circle must be
        minRadius=50,     # Ignore small circular noise like car wheels
        maxRadius=180     # Prevents grabbing the entire landscape
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        # Grab the most confident circle (the first one)
        x, y, r = circles[0]

        # Draw the target lock in GREEN
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)

        # Red Crosshair for precision targeting
        L = 25
        cv2.line(output, (x - L, y), (x + L, y), (0, 0, 255), 3)
        cv2.line(output, (x, y - L), (x, y + L), (0, 0, 255), 3)

        print(f"TARGET LOCKED: Center coordinates ({x}, {y}) with radius {r}")
    else:
        print("Searching for landing pad... No match found with current parameters.")

    return output


# ==========================================
# Task 2: Horizon Leveler (Rotation + Slicing)
# ==========================================
def task2_level_horizon(img, tilt_angle_deg):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, -tilt_angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    # NumPy slicing center crop to remove black edges after rotation
    crop_factor = 0.8
    new_w, new_h = int(w * crop_factor), int(h * crop_factor)
    start_x, start_y = (w - new_w) // 2, (h - new_h) // 2

    cropped = rotated[start_y:start_y+new_h, start_x:start_x+new_w]
    return rotated, cropped


# ==========================================
# Task 3: Obstacle Alert System (HSV Masking)
# ==========================================
def task3_detect_red_danger(img, threshold=0.10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect red color (spans two ranges in HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    total_pixels = img.shape[0] * img.shape[1]
    red_pixels = np.count_nonzero(mask)
    danger_pct = red_pixels / total_pixels

    annotated = img.copy()
    if danger_pct > threshold:
        cv2.putText(annotated, "DANGER: OBSTACLE DETECTED!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(annotated, f"SAFE: {danger_pct*100:.2f}% Red", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated, mask, danger_pct


# ==========================================
# Task 4: Night Vision Booster (Clipping)
# ==========================================
def task4_boost_brightness(img, value=60):
    # Use int16 to prevent wrapping/overflow, then clip to 0-255
    temp = img.astype(np.int16) + value
    bright = np.clip(temp, 0, 255).astype(np.uint8)
    return bright


# ==========================================
# Task 5: Motion Blur Simulator (Kernel)
# ==========================================
def task5_motion_blur(img):
    # 5x5 horizontal motion blur kernel
    kernel = np.zeros((5, 5), dtype=np.float32)
    kernel[2, :] = 0.2
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


# ==========================================
# RUN AND DISPLAY TASKS
# ==========================================

print("--- TASK 1: LANDING PAD FINDER ---")
pad_result = task1_landing_pad_finder(img)
cv2_imshow(pad_result)

print("--- TASK 2: HORIZON LEVELER ---")
_, horizon_result = task2_level_horizon(img, 15)
cv2_imshow(horizon_result)

print("--- TASK 3: OBSTACLE ALERT ---")
danger_result, _, _ = task3_detect_red_danger(img)
cv2_imshow(danger_result)

print("--- TASK 4: NIGHT VISION ---")
bright_result = task4_boost_brightness(img)
cv2_imshow(bright_result)

print("--- TASK 5: MOTION BLUR ---")
blur_result = task5_motion_blur(img)
cv2_imshow(blur_result)


# ==========================================
# SAVE AND DOWNLOAD
# ==========================================
cv2.imwrite("final_task1.jpg", pad_result)
cv2.imwrite("final_task2.jpg", horizon_result)
cv2.imwrite("final_task3.jpg", danger_result)
cv2.imwrite("final_task4.jpg", bright_result)
cv2.imwrite("final_task5.jpg", blur_result)

print("Processing complete. Downloading final images...")
for f in ["final_task1.jpg", "final_task2.jpg", "final_task3.jpg", "final_task4.jpg", "final_task5.jpg"]:
    files.download(f)
