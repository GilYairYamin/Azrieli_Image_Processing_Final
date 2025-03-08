import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(mat_like, cmap="rgb"):
    plt.figure()
    if cmap != "rgb":
        plt.imshow(mat_like, cmap=cmap)
    else:
        plt.imshow(mat_like)
    plt.show()
    plt.close()


dil_kernel_size = 15
dil_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (dil_kernel_size, dil_kernel_size)
)

orange = np.array([123, 85, 66], dtype=np.int64)
light_blue = np.array([60, 76, 101], dtype=np.int64)
gray = np.array([60, 70, 70], dtype=np.int64)
green = np.array([93, 111, 70], dtype=np.int64)
color_values = [orange, light_blue, gray, green]
color_names = ["red", "blue", "gray", "green"]


def clean_image(gray, display=False):
    gray = cv2.medianBlur(gray, 31)  # blur
    if display:
        display_image(gray, "gray")

    gray = cv2.equalizeHist(gray)  # equalize
    if display:
        display_image(gray, "gray")

    gray = 255 - gray  # inverse
    if display:
        display_image(gray, "gray")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(13, 13))
    gray = clahe.apply(gray)  # clahe
    if display:
        display_image(gray, "gray")

    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)  # threshold
    if display:
        display_image(gray, "gray")

    gray = cv2.dilate(gray, dil_kernel, iterations=3)  # dilate
    if display:
        display_image(gray, "gray")

    gray = cv2.erode(gray, dil_kernel, iterations=8)  # erode
    if display:
        display_image(gray, "gray")

    gray = cv2.dilate(gray, dil_kernel, iterations=6)  # dilate
    if display:
        display_image(gray, "gray")

    return gray


def get_circles_from_gray(gray, color=None):
    gray = clean_image(gray)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,  # down sample size
        minDist=400,  # minimum distance between detected circles
        param1=9,
        param2=13,
        minRadius=50,
        maxRadius=100,
    )

    if circles is None:
        print("no matches")
        return []

    color_img_cpy = np.copy(color)
    circles = np.uint(np.around(circles))
    circles = circles[0, :]

    if color is None:
        return circles

    for circle in circles:
        x, y, r = circle[0], circle[1], circle[2]
        cv2.circle(color_img_cpy, (x, y), r, (255, 255, 255), 15)

    return circles


def merge_circle_arr(res, add):
    for circle_new in add:
        x_new, y_new, r_new = circle_new[0], circle_new[1], circle_new[2]
        for old_circle_dict in res:
            x_old = old_circle_dict["x"]
            y_old = old_circle_dict["y"]
            r_old = old_circle_dict["radius"]

            p1 = np.array([x_new, y_new], dtype=np.int64)
            p2 = np.array([x_old, y_old], dtype=np.int64)
            values = p1 - p2
            distance = np.linalg.norm(values)
            if distance < 100:
                x_new = (x_new + x_old) // 2
                y_new = (y_new + y_old) // 2
                r_new = np.max(np.array([r_new, r_old]))

                old_circle_dict["x"] = x_new
                old_circle_dict["y"] = y_new
                old_circle_dict["radius"] = r_new
                r_new = -1
                break

        if r_new > 0:
            circle_dict = {}
            circle_dict["x"] = x_new
            circle_dict["y"] = y_new
            circle_dict["radius"] = r_new
            res.append(circle_dict)


def get_colored_circles(color, display=False):
    r, g, b = cv2.split(color)

    r_circles = get_circles_from_gray(r, color)
    g_circles = get_circles_from_gray(g, color)
    b_circles = get_circles_from_gray(b, color)

    result = []
    merge_circle_arr(result, r_circles)
    merge_circle_arr(result, g_circles)
    merge_circle_arr(result, b_circles)

    if not display:
        return result

    img_copy = np.copy(color)
    for circle in result:
        x, y, r = circle["x"], circle["y"], circle["radius"]
        cv2.circle(img_copy, (x, y), r, (255, 255, 255), 15)

    display_image(img_copy)
    return result


def get_color_name(color):
    color_int = np.array(color, dtype=np.int64)
    min_distance = np.linalg.norm(color_int - color[0])
    min_color_idx = 0

    for i in range(len(color_values)):
        distance = np.linalg.norm(color - color_values[i])
        if distance < min_distance:
            min_color_idx = i
            min_distance = distance

    return color_names[min_color_idx]


def detect_colored_circles(img_path):
    rnd_img = cv2.imread(img_path)
    color_rgb = cv2.cvtColor(rnd_img, cv2.COLOR_BGR2RGB)
    circles = get_colored_circles(color_rgb)

    for circle in circles:
        x, y, r = circle["x"], circle["y"], circle["radius"]
        mask = np.zeros(color_rgb.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r // 2, 255, thickness=-1)
        mean_color = cv2.mean(color_rgb, mask=mask)
        mean_color = np.int64(np.around(mean_color))[:3]
        circle["color"] = get_color_name(mean_color)

    return circles


def track_circles_over_time(img_paths):
    circles_from_images = []
    for img_path in img_paths:
        circles_from_images.append(detect_colored_circles(img_path))

    result = []
    for idx in range(len(circles_from_images)):
        circles = circles_from_images[idx]
        circle_id = len(result)
        for circle in circles:
            x_new = circle["x"]
            y_new = circle["y"]

            for res in result:
                x_old = res["x"]
                y_old = res["y"]

                p1 = np.array([x_new, y_new], dtype=np.int64)
                p2 = np.array([x_old, y_old], dtype=np.int64)
                values = p1 - p2
                distance = np.linalg.norm(values)
                if distance < 100:
                    circle_id = res["circle_id"]
                    break

            new_res = {}
            new_res["image_id"] = idx + 1
            new_res["circle_id"] = circle_id
            new_res["x"] = circle["x"]
            new_res["y"] = circle["y"]
            new_res["radius"] = circle["radius"]
            new_res["color"] = circle["color"]
            result.append(new_res)

    return result


current_folder = os.getcwd()
root_data_folder = os.path.join(current_folder, "local_data")
random_frames_folder = os.path.join(root_data_folder, "random_frames")
sequence_1_folder = os.path.join(root_data_folder, "sequence_1")
sequence_2_folder = os.path.join(root_data_folder, "sequence_2")
sequence_3_folder = os.path.join(root_data_folder, "sequence_3")
sequence_4_folder = os.path.join(root_data_folder, "sequence_4")

img_lst = os.listdir(sequence_4_folder)
img_paths = [os.path.join(sequence_1_folder, img_fn) for img_fn in img_lst]
print(track_circles_over_time(img_paths))
