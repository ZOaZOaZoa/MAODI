import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

import math

# image_file = "c:\\work\\miaodi\\n1.jpg"
# image_file = "c:\\work\\miaodi\\n4.jpg" # bad
image_file = ".\\jpg\\all_n.jpg"


images=[".\\jpg\\n1.jpg",
        ".\\jpg\\n3.jpg",
        ".\\jpg\\n4.jpg",
        ".\\jpg\\n5.jpg",
        ".\\jpg\\n6.jpg",
        ".\\jpg\\n7.jpg",
        ".\\jpg\\n8.jpg",
        ".\\jpg\\n9.jpg"]

dict_file = ".\\jpg\\dict__.jpg"

dict=['А','В','С','Е','Н','К','М','О','Р','С','Т','У','Х','0','1','2','3','4','5','6','7','8','9']
dict_="АВСЕНКМОРСТУХ0123456789"
def nothing(x):
    pass

def dekartToComplex(list):
    clist = []
    for i in range(len(list) - 1):
        dx = list[i + 1][0][0] - list[i][0][0]
        dy = list[i + 1][0][1] - list[i][0][1]
        clist.append(complex(dx, dy))

    clist.append(complex(list[0][0][0] - list[len(list) - 1][0][0], list[0][0][1] - list[len(list) - 1][0][1]))
    return clist

def multiplyCVectors(a,b):
    res = complex(a.real*b.real + a.imag*b.imag,a.imag*b.real - a.real*b.imag)
    return res

def normCVec(a):
    res = (a.real * a.real + a.imag * a.imag)
    return math.sqrt(res)

def normCVector(a):
    res = 0
    cnt = len(a)
    for j in range(cnt):
        res = res + math.sqrt(a[j].real * a[j].real + a[j].imag * a[j].imag)

    return res

def VKF(clist1,clist2):
    res = []
    cnt = len(clist1)
    for j in range(cnt):
        r = 0
        k = j
        for i in range(cnt):
            if (k >= cnt):
                k = 0
            r = r + (multiplyCVectors(clist1[i], clist2[k]))
            k += 1
        res.append(r)
    return res


def equalization(vec, p, x0, y0):
    eqvec = [complex(0,0)] * p
    eq_len = normCVector(vec) / p
    print(eq_len)
    pind = 0
    vec_ost = vec[0]
    vec_isp = complex(0,0)
    j = 0
    while pind < p:
        vlen = normCVec(vec_ost)
        if vlen > eq_len:
            eqvec[pind] = vec_ost * eq_len / vlen
            vec_ost = vec_ost - eqvec[pind]
            print('f: ' + str(pind) + '   ' + str(eqvec[pind]))
            pind += 1
        else:
            s = normCVec(vec_ost)

            for t in range(j + 1, len(vec)):
                s0 = s

                s = s + normCVec(vec[t])


                if (s > eq_len):
                    vec_isp_len = eq_len - s0
                    vec_isp = vec[t] * vec_isp_len / normCVec(vec[t])

                    vecs = complex(0,0)
                    for tt in range(j+1, t - 1):
                        vecs = vecs + vec[tt]


                    eqvec[pind]= vec_ost + vec_isp + vecs
                    vec_ost = vec[t] - vec_isp

                    print(str(pind) + '   ' + str(eqvec[pind]))

                    pind += 1
                    j = t
                    break

            if pind >= p - 1:


                eqvec[pind] = -sum(eqvec)

                print(str(pind) + '   ' + str(eqvec[pind]))
                pind += 1
                break
        if pind >= p  :
            break

    return eqvec

def showImage(imgname, image):
    cv.imshow(imgname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

imgs = []
shps = []
for i in range(len(images)):
    img = cv.imread(images[i])
    shp = img.shape
    imgs.append(img)




dict_img = cv.imread(dict_file)

dict_hsv = cv.blur(dict_img, (2, 2))

dict_hsv = cv.cvtColor(dict_hsv, cv.COLOR_BGR2HSV)

dict_mask = cv.inRange(dict_hsv, (0, 0, 0), (255, 255, 141))


cv.imshow('dict2', dict_mask)


img = cv.imread(image_file)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

wnd = cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
wnd2 = cv.namedWindow('mask', cv.WINDOW_NORMAL)
wnd3 = cv.namedWindow('contours', cv.WINDOW_NORMAL)
wnd_dict = cv.namedWindow('dict', cv.WINDOW_NORMAL)

cv.createTrackbar('minH','image', 0, 255, nothing)
cv.createTrackbar('minS','image', 0, 255, nothing)
cv.createTrackbar('minV','image', 0, 255, nothing)

cv.createTrackbar('maxH','image', 0, 255, nothing)
cv.createTrackbar('maxS','image', 0, 255, nothing)
cv.createTrackbar('maxV','image', 0, 255, nothing)

cv.createTrackbar('blur','image', 0, 10, nothing)

cv.createTrackbar('erode_delate','image', 0, 10, nothing)


cv.waitKey(0)



dict_contours, dict_hierarchy = cv.findContours(dict_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

ca = []
for i in range(len(dict_contours)):
    ca_val = cv.contourArea(dict_contours[i])
    if ca_val > 100:
        x, y, w, h = cv.boundingRect(dict_contours[i])
        px = -1
        py = -1
        pw = 0
        ph = 0
        pt = dict_hierarchy[0][i][3]
        if (pt >= 0):
            px, py, pw, ph = cv.boundingRect(dict_contours[pt])
        ca.append({'ind': i,'area':ca_val,'contour':dict_contours[i], 'w':w,'h':h,'x':x,'y':y, 'hierarchy':dict_hierarchy[0][i], 'pw':pw,'ph':ph, 'px':px, 'py':py, 'pt':pt})

res = sorted(ca, key=lambda c: c['area'], reverse=True)

ca_filt = []
for i in range(len(res)):
    cnt = dict_contours[res[i]['ind']]
    ph = res[i]['ph']
    px = res[i]['px']
    w = res[i]['w']
    h = res[i]['h']
    rh = 0
    if (px > 0) and (ph > 0):
        rh = res[i]['h'] / ph

    if (((rh >= 0.3) and (rh <= 0.59))or((rh >= 0.59) and (rh <= 0.8))) and (w < h):
        ca_filt.append(res[i])

for i in range(len(ca_filt)):
    cnt = dict_contours[ca_filt[i]['ind']]
    h = ca_filt[i]['h']
    print(str(i) + ' ' + str(len(cnt)))
    # if (i == 15) or (i ==16):
    img_cnt = cv.drawContours(dict_img, [cnt], 0, (0, 0, 255), 1)
    cv.putText(dict_img,str(i),(ca_filt[i]['x'],ca_filt[i]['y']), cv.FONT_HERSHEY_COMPLEX,0.3,(0,255,0),1,cv.LINE_AA)
    # img_cnt = cv.drawContours(img_res, [contours[ca_filt[i]['pt']]], 0, (255, 0, 0), 3)
cv.imshow('dict', dict_img)


cnt = dict_contours[ca_filt[0]['ind']]
# print(cnt)
cnt_i = []
x0 = cnt[0][0][0]
y0 = cnt[0][0][1]
# for i in range(len(cnt) - 1):
#     dx = cnt[i + 1][0][0] - cnt[i][0][0]
#     dy = cnt[i + 1][0][1] - cnt[i][0][1]
#     cnt_i.append(complex(dx,dy))
#
# cnt_i.append(complex(cnt[0][0][0] - cnt[len(cnt) - 1][0][0],cnt[0][0][1] - cnt[len(cnt) - 1][0][1]))
cnt_i = dekartToComplex(cnt)
print(cnt_i[0])
print(cnt_i[1])
print(cnt_i[2])
print(cnt_i[len(cnt_i) - 1])
print(cnt[0][0])
print(cnt[1][0])
print(cnt[2][0])
print(cnt[len(cnt) - 1][0])

print(sum(cnt_i))
print(len(cnt))
print(len(cnt_i))

cnt_i = [complex(0,1), complex(1, 0), complex(0,-1),complex(-1, 0)]

cnt1 = dict_contours[ca_filt[0]['ind']]
cnt2 = dict_contours[ca_filt[3]['ind']]

print(len(cnt1))
print(len(cnt2))

cnt_i1 = dekartToComplex(cnt1)
cnt_i2 = dekartToComplex(cnt2)

eq1 = cnt_i1
eq2 = cnt_i2

eq1 = equalization(cnt_i1, 100,0,0)
eq2 = equalization(cnt_i2, 100,0,0)

vkf_seq = VKF(eq1, eq2)
norm1 = normCVector(eq1)
norm2 = normCVector(eq2)

vkf_seq_norm = []
for i in range(len(vkf_seq)):
    vkf_seq_norm.append(vkf_seq[i] / (norm1 * norm2))
# print(vkf_seq_norm)

m = 0
for i in range(len(vkf_seq)):
    maxT = abs(vkf_seq_norm[i])
    if maxT > m:
        m = maxT

print(m)




cnt = dekartToComplex(dict_contours[ca_filt[0]['ind']])

plt.figure()
x0 = cnt[0].real
y0 = cnt[0].imag
cvec = complex(0,0)
for i in range(len(cnt)):
    cvec = cvec + cnt[i]
    plt.scatter(cvec.real, cvec.imag)
    # plt.scatter(cnt[i + 1].real + cnt[i].real, cnt[i + 1].imag + cnt[i].imag)
    # plt.scatter(dict_contours[ca_filt[0]['ind']][i][0][0], dict_contours[ca_filt[0]['ind']][i][0][1])

x0 = dict_contours[ca_filt[0]['ind']][0][0][0]
y0 = dict_contours[ca_filt[0]['ind']][0][0][1]

# cnt=[complex(2,0), complex(2,-2),complex(0,-1), complex(-2,-2),complex(-2,2),complex(0,1),complex(-1,1),complex(1,1)]
cnt=[complex(200,0), complex(0,200),complex(-200,0), complex(0,-200)]
print(sum(cnt))
print(normCVector(cnt))
eq_vec = equalization(cnt, 40, x0, y0)

print(normCVector(eq_vec))
print(len(eq_vec))
print(sum(eq_vec))
print(sum(cnt))


vkf_cnt = VKF(cnt, cnt)


plt.figure()
for i in range(len(vkf_cnt)):
    plt.scatter(vkf_cnt[i].real, vkf_cnt[i].imag)
plt.show()


# sys.exit(1)


while(True):

    minH=cv.getTrackbarPos('minH', 'image')
    minS=cv.getTrackbarPos('minS', 'image')
    minV=cv.getTrackbarPos('minV', 'image')

    maxH=cv.getTrackbarPos('maxH', 'image')
    maxS=cv.getTrackbarPos('maxS', 'image')
    maxV=cv.getTrackbarPos('maxV', 'image')

    blr = cv.getTrackbarPos('blur', 'image')
    er_dil = cv.getTrackbarPos('erode_delate', 'image')

    if (blr > 0):
        img_hsv = cv.blur(img, (blr, blr))
        img_hsv = cv.cvtColor(img_hsv, cv.COLOR_BGR2HSV)
    else:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(img_hsv, (minH,minS,minV), (maxH,maxS,maxV))

    if (er_dil > 0):
        cv.morphologyEx(mask,cv.MORPH_OPEN,(er_dil,er_dil),mask,iterations=4)

    cv.imshow('mask', mask)

    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif chr(key) == 'c':

        print('processing contours')

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        cv.imshow('contours', img)


        ca = []
        for i in range(len(contours)):
            ca_val = cv.contourArea(contours[i])
            if ca_val > 100:
                x, y, w, h = cv.boundingRect(contours[i])
                px = -1
                py = -1
                pw = 0
                ph = 0
                pt = hierarchy[0][i][3]
                if (pt >= 0):
                    px, py, pw, ph = cv.boundingRect(contours[pt])

                ca.append({'ind': i,'area':ca_val,'contour':contours[i], 'w':w,'h':h,'x':x,'y':y, 'hierarchy':hierarchy[0][i], 'pw':pw,'ph':ph, 'px':px, 'py':py, 'pt':pt})

        res = sorted(ca, key=lambda c: c['area'], reverse=True)

        ca_filt = []
        for i in range(len(res)):

            cnt = contours[res[i]['ind']]
            ph = res[i]['ph']
            px = res[i]['px']
            w = res[i]['w']
            h = res[i]['h']
            rh = 0
            if (px > 0) and (ph > 0):
                rh = res[i]['h'] / ph

            if (((rh >= 0.3) and (rh <= 0.59))or((rh >= 0.59) and (rh <= 0.8))) and (w < h):
                ca_filt.append(res[i])

        img_res = cv.imread(image_file)
        for i in range(len(ca_filt)):
            cnt = contours[ca_filt[i]['ind']]
            h = ca_filt[i]['h']

            img_cnt = cv.drawContours(img_res, [cnt], 0, (0, 0, 255), 3)

        cv.imshow('contours', img_res)
