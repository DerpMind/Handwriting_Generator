from PIL import Image, ImageFont, ImageDraw
import numpy as np


FONT_PATH = '../data/fonts/'


def img_to_pxl(img_path):
	img = Image.open(img_path)
	pxl = np.array(img)
	return pxl


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


def downsize_check(x1,x2,y1,y2):
	if x2 == 127 or x1 == 0:
		font_size_factor = 0.6
		return font_size_factor
	elif y2 == 127 or y1 == 0:
		font_size_factor = 0.8
		return font_size_factor
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
        old_length = x2 - x1
        font_size_factor = 94.0/old_length
        return font_size_factor
    else:
    	return 1


def preprocess_parameterization_part_one(sample_path, letter,downsize_factor=1, dilate_factor=1):
	font_name = sample_path.split('/')[-1].split('.')[0]
	font_name = font_name[:len(font_name)-2]
	ttf_file = f'{FONT_PATH}{font_name}.ttf'
	fnt = ImageFont.truetype(ttf_file, int(90*downsize_factor*dilate_factor))
	img = Image.new('RGB', (128,128), color = 'white')
	d = ImageDraw.Draw(img)
	d.text((5,0), letter, font=fnt, fill=(0,0,0))
	temp = font_name.split('/')[-1].split('.')[0]
	png_path = f'../data/preprocess_examples/{temp}.png'
	img.save(png_path)
	return png_path

def preprocess_parameterization_part_two(sample_path, letter,downsize_factor, dilate_factor,x_shift = 0, y_shift = 0):
	font_name = sample_path.split('/')[-1].split('.')[0]
	# font_name = font_name[:len(font_name)-2]
	ttf_file = f'{FONT_PATH}{font_name}.ttf'
	fnt = ImageFont.truetype(ttf_file, int(90*downsize_factor*dilate_factor))
	img = Image.new('RGB', (128,128), color = 'white')
	d = ImageDraw.Draw(img)
	d.text((5 + x_shift,0+y_shift), letter, font=fnt, fill=(0,0,0))
	temp = font_name.split('/')[-1].split('.')[0]
	png_path = f'../data/preprocess_examples/{temp}.png'
	img.save(png_path)
	return png_path