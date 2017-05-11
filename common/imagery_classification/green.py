
def is_green_type(img):
    h, w, c = img.shape
    mins = [img[:, :, i].min() for i in range(c)]
    maxs = [img[:, :, i].max() for i in range(c)]
    b1 = maxs[1] > maxs[0] * 10 and maxs[1] > maxs[2] * 10
    b2 = mins[0] == 0 and mins[2] == 0
    return b1 and b2
