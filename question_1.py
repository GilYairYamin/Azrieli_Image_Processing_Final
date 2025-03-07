import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def display_image(mat_like, cmap="rgb"):
    plt.figure()
    if cmap != "rgb":
        plt.imshow(mat_like, cmap=cmap)
    else:
        plt.imshow(mat_like)
    plt.show()
    plt.close()


kernel_size = 15
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
)


def erode(img, amount=1):
    for _ in range(amount):
        img = cv2.erode(img, kernel)
    return img


def dilate(img, amount=1):
    for _ in range(amount):
        img = cv2.dilate(img, kernel)
    return img


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

    gray = dilate(gray, 2)  # dilate
    if display:
        display_image(gray, "gray")

    gray = erode(gray, 7)  # erode
    if display:
        display_image(gray, "gray")

    gray = dilate(gray, 7)  # dilate
    if display:
        display_image(gray, "gray")

    return gray
