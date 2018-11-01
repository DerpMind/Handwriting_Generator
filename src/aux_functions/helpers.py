from PIL import Image, ImageFont, ImageDraw
import numpy as np
from pdb import set_trace


FONT_PATH = '../data/fonts/'


def rgb2gray(pxl):
    return np.dot(pxl[...,:3], [0.299, 0.587, 0.114])


def get_corners(pxl):
    bools = np.where(pxl==255, True, False)
    # set_trace()
    return {'top': np.where(bools == False)[0].min(),
    'left':np.where(bools==False)[1].min(),
    'bottom': np.where(bools==False)[0].max(),
    'right': np.where(bools==False)[1].max()}


def crop_check(x1,x2,y1,y2):
	x_shift, y_shift = 0, 0
	if y2 >= 227:
		y_shift = 220-y2
	if x2 >= 227:
		x_shift = 220-x2
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
    x_shift, y_shift = 0,0
    if radius >= 20:
        x_shift = 64.0 - center_x#if x_shift is negative, then must be shifted down.
        y_shift = 64.0 - center_y #if y_shift is negative, then must be shifted to the left
        return x_shift, y_shift
    return x_shift, y_shift


def dilate_check(x1,x2,y1,y2):
    area = np.abs((x2 - x1) * (y2 - y1))
    if area < 4500:
        height = x2 - x1
        width = y2 - y1
        dims = [height, width]
        dim = min(dims)
        font_size_factor = min(2,int(94.0/dim)) #assuming ideal pxl size of font is 94 by 94
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
	return corners['top'], corners['bottom'], corners['left'], corners['right']

def downsize_parameters(ttf,letter):
	downsize_factor = 1
	img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, \
		 pxl_shape = (500,500),top=100,left=100)
	x1,x2,y1,y2 = png_to_coordinates(img)
	downsize_factor = downsize_check(x1,x2,y1,y2)
	return downsize_factor


def decrop_parameters(ttf, letter, downsize_factor, dilate_factor=1):
	x_shift, y_shift = 0,0
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
	downsize_factor=downsize_factor,dilate_factor=dilate_factor,
	 pxl_shape = (500,500),top=100,left=100)
	x1,x2,y1,y2 = png_to_coordinates(img)
	x_shift, y_shift = crop_check(x1,x2,y1,y2)
	return x_shift, y_shift



def dilate_parameters(ttf,letter,downsize_factor, x_shift, y_shift):
	dilate_factor = 1
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
		downsize_factor=downsize_factor, dilate_factor=dilate_factor)
	x1,x2,y1,y2 = png_to_coordinates(img)
	dilate_factor = dilate_check(x1,x2,y1,y2)
	return dilate_factor


def center_parameters(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift):
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
		downsize_factor=downsize_factor, dilate_factor=dilate_factor)
	x1,x2,y1,y2 = png_to_coordinates(img)
	new_x_shift, new_y_shift = center_check(x1,x2,y1,y2)
	x_shift = new_x_shift + x_shift
	y_shift = new_y_shift + y_shift
	return x_shift, y_shift


def processed_img_constructor(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift):
	img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, dilate_factor=dilate_factor,\
		x_shift=x_shift, y_shift=y_shift)
	font_name = ttf.split('/')[-1].split('.')[0]
	img_path = f'../data/{letter}/{font_name}.png'
	img.save(img_path)
	pxls = np.asarray(img)
	return pxls

