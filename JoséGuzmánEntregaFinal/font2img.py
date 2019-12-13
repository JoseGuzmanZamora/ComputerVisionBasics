"""
Generate a synthetic dataset from system fonts

Usage: python font2img.py

Comptuter Vision
Author: Christian Medina Armas
Version: 1.1

"""

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
import numpy as np
import cv2 as cv 
import os
import glob
import ntpath

def transform_perspective(img_src, dataset_path, char, file_name, keep_size=True):
         """ Generate a set of 8 perespective images form an undistorted sample.
        
         Args:
            img_src (PIL.Image): Base image
            dataset_path (string): Path to destination directory that contains one sub directory per char
            char (char): Char cointained in the image
            file_name (string): Base image filename
            keep_size (bool): Enable resizing to keep base image dimentions 

            - Outputs 9 images: Base + 8 transformed perspective images.
            - Base image prefix is 0_0
        """
         width, height = img_src.size
         for i in range(-1,2):
                m_x = i/10
                for j in range(-1,2):
                        img =  img_src.copy()
                        m_y = j/10
                        xshift = abs(m_x) * width
                        yshift = abs(m_y) * height
                        new_width = width + int(round(xshift))
                        new_height = height + int(round(yshift))
                        mat = (1, m_y, -xshift if m_x>=0 else 0 , -m_x, 1, 0 if m_y <=0 else -yshift)
                        img = ImageOps.invert(img)
                        img = img.transform((new_width, new_height), Image.AFFINE, mat, Image.BICUBIC)
                        img = ImageOps.invert(img)
                        if keep_size:
                                img = img.resize((width,height))
                        file_path = os.path.join(dataset_path, char, str(i)+'_'+str(j)+file_name)
                        img.save(file_path)

def font2img(font_path, dataset_path, chars_to_generate, use_perspective=False):
   """ Convert a list of chars to images using a system font
   
   Args:
      font_path (string): Path to system font
      dataset_path (string): Path to destination directory that contains one sub directory per char
      chars_to_generate (list): List of chars to convert
      use_perspective (bool): Enable the use of perspective to enhance dataset. Increases output images x 9

   """
   fontSize = 25
   imgSize = (25,30)
   position = (7,0)

   font = ImageFont.truetype(font_path, fontSize)

   font_file = ntpath.basename(font_path)
   font_file = font_file.rsplit('.')
   font_file = font_file[0]
   file_name = font_file + '.jpg'
   for char in chars_to_generate:
      
      image = Image.new('RGB', imgSize, (255,255,255))
      draw = ImageDraw.Draw(image)
      draw.text(position, char, (0,0,0), font=font)

      if use_perspective:
         transform_perspective(image, dataset_path, char, file_name, keep_size=True)
      else:


         file_path = os.path.join(dataset_path, char, file_name)
         image.save(file_path)


def make_dst_dir(dst_directory, char_list):
   """ Build a structore of directories from a list of chars
   
   Args:
       dst_directory (string): Path of main directory.
       char_list (list): List of subdirectories names 

   """   
   if not os.path.exists(dst_directory):
      os.makedirs(dst_directory)

   for char in char_list:
      label_path = os.path.join(dst_directory, char)
      if not os.path.exists(label_path):
         os.makedirs(label_path)


def generate_from_font(path=None,font_file=None,perspective=False):
   """ Generate an image dataset from system fonts

   Args:
      path_to_system_fonts (str): Path to fonts in system
      font_list_file (str): Path to file that contains fonts to be used.

   """
   digits_list = [str(i) for i in range(0,10)]
   dst_directory = os.path.join(os.getcwd(),'digits_synthetic')

   if (path != None):
      if not os.path.exists(path):
         print('{0} not found'.format(path))
         return -1
      else:
         found_fonts = glob.glob(os.path.join(path,'**','*.ttf'),recursive=True)
         make_dst_dir(dst_directory, digits_list)
         for sys_font in tqdm(found_fonts, desc='Generating images', ascii=False):
            font2img(sys_font, dst_directory, digits_list,use_perspective=perspective)

   elif(font_file != None):
      make_dst_dir(dst_directory, digits_list)
      font2img(font_file, dst_directory, digits_list, use_perspective=perspective)


if __name__ == '__main__':
   import argparse
   import textwrap

   parser = argparse.ArgumentParser(prog='Generate digit images from True Type fonts',
   formatter_class=argparse.RawDescriptionHelpFormatter,
   description=textwrap.dedent('''
         Useful paths for system fonts:
            - Windows: "C:\Windows\Fonts\\"
            - Ubuntu: "/usr/share/fonts/truetype/"
            
            Example run on Windows:
            python font2img.py --f "C:\Windows\Fonts\Tahoma\tahoma.ttf" -p
            '''))

   parser.add_argument('-p', default=False, action='store_true', help='Enable perspective')
   group = parser.add_mutually_exclusive_group()
   group.add_argument('--d', type=str, default=None, help='Path to system .ttf fonts')
   group.add_argument('--f', type=str, default=None, help='Path to single .ttf font file')
   args = parser.parse_args()

   generate_from_font(path=args.d,font_file=args.f,perspective=args.p)