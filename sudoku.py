import cv2
import numpy as np
import operator
from tensorflow.keras.models import load_model
import tensorflow as tf


# Tim vi tri chua dien
def find_empty(bo):
    for i in range(9):
        for j in range(9):
            if bo[i][j] == 0:
                return (i, j)
    return None

# Kiem tra hang, cot, va luoi 3x3 da co num chua
def valid(bo, num, pos): 
    # Kiem tra hang
    for i in range(9):
        if bo[pos[0]][i] == num:
            return False
    
    # Kiem tra cot
    for i in range(9):
        if bo[i][pos[1]] == num:
            return False

    # Kiem tra luoi 3x3
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num:
                return False
    return True

def solve(bo):
    # Tim vi tri trong
    find = find_empty(bo)
    if not find:
        return True # Neu da giai het thi thoi
    else:
        row, col = find
    
    # Thu het cac truong hop (1 -> 9)
    for i in range(1, 10):
        # Neu ok thi:
        if valid(bo, i, (row, col)):

            # Thu gan 
            bo[row][col] = i

            # Neu sudoku dung va duoc giai het thi dung
            if solve(bo):
                return True 

            # Gan lai 0 de thu gia tri khac   
            bo[row][col] = 0
    return False

# Tien xu ly anh
def pre_process_image(img):

    # GausianBlur - lam min
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    # Threshold - phan nguong
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate - phep gian no, lam cho object to hon
    kernel = np.ones((3, 3), dtype='int8')
    proc = cv2.dilate(proc, kernel)
    return proc

# Timm ra 4 goc cua luoi sudoku
def find_corners_of_largest_polygon(img):
    # Tim contour lon nhat
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    
    # Tim 4 goc
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

# Cat 4 goc 
def crop_and_warp(img, crop_rect, flag=0):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Tim canh dai nhat
    side = max([
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(top_left - bottom_left),
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_left - top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Get ma tran bien doi va warp anh
    m = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))
    return warp

# Lay toa do bbox cua tung (top left & bottom right)
def infer_grid(img):
    squares = []
    side = img.shape[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares

# Cat cac so tu toa do
def cut_from_rect(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

# Xu ly scale ve feature 28x28 va dat vao giua 
def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]

    def centre_pad(length):

        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)

    return cv2.resize(img, (size, size))

# Tim so
def find_largest_feature(inp_img, scan_tl=None, scan_br=None):

    img = inp_img.copy()
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)
 
    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]
    
    # Fill vung mau trang sang mau xam (Fill vung trong tam)
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    # Fill toan bo vung mau trang sang mau xam (Fill vien)
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)
    
    # Full lai object sang mau trang
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0
    
    # Kiem tra va tim ra bbox cua object
    for x in range(width):
        for y in range(height):

            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)

            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return np.array(bbox, dtype='float32')

# Tao feature so 0
def feature_0():
    xx, yy = np.mgrid[:28, :28]
    circle = (xx - 14) ** 2 + (yy - 14) ** 2
    return np.logical_and(circle < (121 + 30), circle > (121 - 30))

# Cat so & tao feature
def extract_digit(img, rect, size):

    digit = cut_from_rect(img, rect)
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    bbox = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 12)
    else:
        return feature_0()

# Lay toan bo so luu vao mang de xu ly
def get_digits(img, squares, size):
    digits = []
    img = pre_process_image(img.copy())
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits

def pre_parse(img, flag=0):
    original = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    processed = pre_process_image(original)
    corners = find_corners_of_largest_polygon(processed)
    cropped = crop_and_warp(original, corners, flag)
    return cropped

def parse_grid(cropped, flag=0):

    squares = infer_grid(cropped)
    digits = get_digits(cropped, squares, 28)
    return digits

# Nhan dien chu so
def predict_digits(model, digits):
    dic = {}
    char = "0123456789"
    for i, c in enumerate(char):
        dic[i] = c
    sudoku = []
    row = []
    for i, dig in enumerate(digits):
        
        img = dig.reshape(-1, 28, 28, 1)
        pred = np.argmax(model(img))
        character = dic[pred]

        row.append(int(character))
        if ((i + 1) % 9 == 0):
            sudoku.append(row)
            row = []
    return np.array(sudoku)

# Ve so len
def draw_text(img, sudoku, sudoku_tosolvee):
    side = img.shape[0] / 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = (side / 35)
    color = (0, 0, 255)
    thickness = 2
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                cv2.putText(img, str(sudoku_tosolvee[i][j]),
                            (int(j * side + side / 5), int((i + 1) * side - side / 5)), font, fontScale, color,
                            thickness, cv2.LINE_AA)
    cv2.imshow('Solved', img)
    cv2.waitKey(0)

def main(img):
    pre = pre_parse(img)
    digits = parse_grid(pre, flag=0)
    model = load_model('digit_model.h5', custom_objects={'CNN': tf.keras.Sequential})
    sudoku = predict_digits(model, digits)
    sudoku_tosolvee = sudoku.copy()
    print(sudoku)
    solve(sudoku_tosolvee)
    draw_text(pre, sudoku, sudoku_tosolvee)

# thu
img = cv2.imread('images/sudoku_example.jpg')
main(img)