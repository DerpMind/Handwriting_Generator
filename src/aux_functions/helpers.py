from PIL import Image, ImageFont, ImageDraw
import numpy as np
from pdb import set_trace


FONT_PATH = '../data/fonts/'


def rgb2gray(pxl):
    return np.dot(pxl[...,:3], [0.299, 0.587, 0.114])


def get_corners(pxl):
    corners={}
    bools = np.where(pxl==255, True, False)
    corners['top']=np.where(bools == False)[0].min()
    corners['left']= np.where(bools==False)[1].min()
    corners['bottom'] = np.where(bools==False)[0].max()
    corners['right'] = np.where(bools==False)[1].max()
    return corners


def crop_check(x1,x2,y1,y2):
	x_shift, y_shift = 0, 0
	if y2 >= 127:
		y_shift = y2-127
	if x2 >= 127:
		x_shift = x2-127
	# if x1 == 0:
	# 	x_shift = -10
	# if y1 == 0:
	# 	y_shift = 10
	return x_shift, y_shift


def downsize_check(x1,x2,y1,y2):
	height = x2 - x1
	width = y2 - y1
	if height >= 120:
		if height >= 140:
			downsize_factor = 0.3
		else:
			downsize_factor = 0.5
		return downsize_factor
	elif width >= 90:
		if width >= 120:
			downsize_factor = 0.3
		else:
			downsize_factor = 0.5
		return downsize_factor
	else:
		return 1


def center_check(x1,x2,y1,y2):
    center_x = (x2+x1)/2.0
    center_y = (y2+y1)/2.0
    radius = np.sqrt( (center_x - 64)**2 + (center_y - 64)**2 )
    if radius >= 10:
        x_shift = 64 - center_x#if x_shift is negative, then must be shifted down.
        y_shift = 64.0 - center_y #if y_shift is negative, then must be shifted to the left
        return x_shift, y_shift


def dilate_check(x1,x2,y1,y2):
    area = np.abs((x2 - x1) * (y2 - y1))
    if area < 4900:
        height = x2 - x1
        width = y2 - y1
        dims = [height, width]
        dim = min(dims)
        font_size_factor = int(94.0/dim) #assuming ideal pxl size of font is 94 by 94
        return font_size_factor
    else:
    	return 1


def ttf_to_png(ttf, letter, downsize_factor=1, dilate_factor=1, \
	x_shift=0, y_shift=0,pxl_shape = (128,128),top=5, left=0 ):
	fnt = ImageFont.truetype(ttf, int(90*downsize_factor*dilate_factor))
	img = Image.new('RGB', pxl_shape, color = 'white')
	d = ImageDraw.Draw(img)
	d.text((top+y_shift,left+x_shift), letter, font=fnt, fill=(0,0,0))
	return img

def png_to_coordinates(img):
	img = np.asarray(img)
	img = rgb2gray(img)
	corners = get_corners(img)
	x1 = corners['top']
	x2 = corners['bottom']
	y1 = corners['left']
	y2 = corners['right']
	return x1,x2,y1,y2


def downsize_parameters(ttf,letter):
	downsize_factor = 1
	while True:
		img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, \
			 pxl_shape = (500,500),top=100,left=100)
		x1,x2,y1,y2 = png_to_coordinates(img)
		downsize_factor = downsize_check(x1,x2,y1,y2)
		if downsize_factor == 1:
			return downsize_factor
		else:
			return downsize_factor

def decrop_parameters(ttf, letter, downsize_factor, pxl_shape = (500,500),top=100,left=100):
	x_shift, y_shift = 0,0
	while True:
		img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
		downsize_factor=downsize_factor)



def dilate_parameters(ttf,letter,downsize_factor, x_shift, y_shift):
	dilate_factor = 1
	while True:
		img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
			downsize_factor=downsize_factor, dilate_factor=dilate_factor)
		x1,x2,y1,y2 = png_to_coordinates(img)
		dilate_factor = dilate_check(x1,x2,y1,y2)
		if dilate_factor == 1:
			break
		else:
			return dilate_factor
	return dilate_factor


def center_parameters(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift):
	while True:
		img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
			downsize_factor=downsize_factor, dilate_factor=dilate_factor)
		x1,x2,y1,y2 = png_to_coordinates(img)
		new_x_shift, new_y_shift = center_check(x1,x2,y1,y2)
		if (new_x_shift, new_y_shift) == (0,0):
			break
		else:
			x_shift = new_x_shift + x_shift
			y_shift = new_y_shift + y_shift
			return x_shift, y_shift
	return x_shift, y_shift


def processed_img_constructor(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift):
	img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, dilate_factor=dilate_factor,\
		x_shift=x_shift, y_shift=y_shift)
	font_name = ttf.split('/')[-1].split('.')[0]
	img_path = f'../data/preprocess_examples/{font_name}_{letter}.png'
	img.save(img_path)
	pxls = np.asarray(img)
	return pxls

