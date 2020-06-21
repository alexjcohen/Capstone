from PIL import Image
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='File to get min/max sizes for image resizing during training')
parser.add_argument('--image-dir', default='Data/train2017/', help='Path to image directory')
parser.add_argument('--output-dir', default=None, help='Output smallest/largest images')


if __name__ == '__main__':

    args = parser.parse_args()

    dir_path = args.image_dir

    files = os.listdir(dir_path)

    files = [os.path.join(os.getcwd(), dir_path, i) for i in files]

    min_x = 1E10
    min_y = 1E10
    max_x = -1
    max_y = -1

    min_x_file, min_y_file, max_x_file, max_y_file = '', '', '', ''

    print('\nGetting file sizes')

    for file in tqdm(files):
        size = Image.open(file).size
        if size[0] < min_x:
            min_x = size[0]
            min_x_file = file

        if size[0] > max_x:
            max_x = size[0]
            max_x_file = file

        if size[1] < min_y:
            min_y = size[1]
            min_y_file = file

        if size[1] > max_y:
            max_y = size[1]
            max_y_file = file

    if args.output_dir:
        print('\nSaving images')
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        Image.open(min_x_file).save(os.path.join(args.output_dir, 'min_x.jpg'))
        Image.open(max_x_file).save(os.path.join(args.output_dir, 'max_x.jpg'))
        Image.open(min_y_file).save(os.path.join(args.output_dir, 'min_y.jpg'))
        Image.open(max_y_file).save(os.path.join(args.output_dir, 'max_y.jpg'))
        print('Saved!')

    print(f'\nThe minimum width is {min_x} for image at {min_x_file}')
    print(f'The minimum height is {min_y} for image at {min_y_file}')
    print(f'The maximum width is {max_x} for image at {max_x_file}')
    print(f'The maximum height is {max_y} for image at {max_y_file}\n')
