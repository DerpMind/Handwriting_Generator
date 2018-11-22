from PIL import Image, ImageFont, ImageDraw
import numpy as np
from pdb import set_trace


FONT_PATH = '../../data/'


def rgb2gray(pxl):
	'''Converts RBG image to gray scale.'''
	return np.dot(pxl[...,:3], [0.299, 0.587, 0.114])


def get_corners(pxl):
	'''Returns the x and y coordinates of the
	corners of the pixelized character'''
	bools = np.where(pxl==255, True, False)
	# set_trace()
	return {'top': np.where(bools == False)[0].min(),
	'left':np.where(bools==False)[1].min(),
	'bottom': np.where(bools==False)[0].max(),
	'right': np.where(bools==False)[1].max()}


def crop_check(x1,x2,y1,y2):
	'''This shifts the pixelized character
	based on the position of the corners.'''
	x_shift, y_shift = 0, 0
	if y2 >= 227:
		y_shift = 220-y2
	if x2 >= 227:
		x_shift = 220-x2
	return x_shift, y_shift


def downsize_check(x1,x2,y1,y2):
	'''Checks to see whether character
	needs to be downsized based on 
	the height and width of the character'''
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
	'''Checks to see whether the character
	is centered based on a radius threshold. Radius
	is determined by measuring Euclidean distance.'''
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
	'''Checks to see whether character is too 
	small based on height and width and area of 
	the character'''
	height = x2 - x1
	width = y2 - y1
	area = np.abs(height * width)
	if area < 4500:
		dims = [height, width]
		dim = min(dims)
		font_size_factor = min(2,int(94.0/dim)) #assuming ideal pxl size of font is 94 by 94
		return font_size_factor
	else:
		return 1


def ttf_to_png(ttf, letter, downsize_factor=1, dilate_factor=1, \
	x_shift=0, y_shift=0,pxl_shape = (128,128),top=5, left=0 ):
	'''This uses ttf to generate image that has a letter
	drawn on the image with the provided font from ttf file.'''
	fnt = ImageFont.truetype(ttf, int(90*downsize_factor*dilate_factor))
	img = Image.new('RGB', pxl_shape, color = 'white')
	d = ImageDraw.Draw(img)
	d.text((top+y_shift,left+x_shift), letter, font=fnt, fill=(0,0,0))
	return img


def png_to_coordinates(img):
	'''Returns x1(top),x2(bottom),y1(left),y2(right) corners
	of the image once its converted into a numpy array with 
	a depth of 1 due to grayscaling.'''
	img = np.asarray(img)
	img = rgb2gray(img)
	corners = get_corners(img)
	return corners['top'], corners['bottom'], corners['left'], corners['right']

def downsize_parameters(ttf,letter):
	'''Extracts downsize factor that will be used as a parameter value
	in processed_img_constructor function.'''
	downsize_factor = 1
	img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, \
		 pxl_shape = (500,500),top=100,left=100)
	x1,x2,y1,y2 = png_to_coordinates(img)
	downsize_factor = downsize_check(x1,x2,y1,y2)
	return downsize_factor


def decrop_parameters(ttf, letter, downsize_factor, dilate_factor=1):
	'''Extracts the amount x and y need to be shifted in order to
	alleviate the cutting off of characters. This will then be used 
	as a parameter value in processed_img_constructor function.'''
	x_shift, y_shift = 0,0
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
	downsize_factor=downsize_factor,dilate_factor=dilate_factor,
	 pxl_shape = (500,500),top=100,left=100)
	x1,x2,y1,y2 = png_to_coordinates(img)
	x_shift, y_shift = crop_check(x1,x2,y1,y2)
	return x_shift, y_shift


def dilate_parameters(ttf,letter,downsize_factor, x_shift, y_shift):
	'''Extracts dilation parameter that will then be used as an argument
	in processed_img_constructor function'''
	dilate_factor = 1
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
		downsize_factor=downsize_factor, dilate_factor=dilate_factor)
	x1,x2,y1,y2 = png_to_coordinates(img)
	dilate_factor = dilate_check(x1,x2,y1,y2)
	return dilate_factor


def center_parameters(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift):
	'''Extracts the amount x and y need to be shifted in order to center
	the image. The return values will then be used as arguments in the 
	processed_img_constructor function'''
	img = ttf_to_png(ttf, letter, x_shift=x_shift,y_shift=y_shift, \
		downsize_factor=downsize_factor, dilate_factor=dilate_factor)
	x1,x2,y1,y2 = png_to_coordinates(img)
	new_x_shift, new_y_shift = center_check(x1,x2,y1,y2)
	x_shift = new_x_shift + x_shift
	y_shift = new_y_shift + y_shift
	return x_shift, y_shift


def processed_img_constructor(ttf, letter, downsize_factor, dilate_factor,x_shift, y_shift, dir_path):
	'''Once dilation, downsizing, and shifting parameters
	are extracted, call this function to generate an image 
	that will minimize the imperfections, thus normalizing the images'''
	img = ttf_to_png(ttf, letter, downsize_factor=downsize_factor, dilate_factor=dilate_factor,\
		x_shift=x_shift, y_shift=y_shift)
	font_name = ttf.split('/')[-1].split('.')[0]
	img_path = dir_path + f'letters/{letter}/{font_name}.png'
	img.save(img_path)
	pxls = np.asarray(img)
	return pxls

