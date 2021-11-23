from filtering import compute_graph
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np


RES = 0.6

def main():
    im = mplimg.imread('test.png')[None,...]
    im = (im*255).astype(np.uint8)
    gpu = 0
    csv_path = ''
    write_csv = True if csv_path != '' else None
    rec_img, labelmat = compute_graph(im, im.shape[1], im.shape[2], 
        RES, write_csv, im.shape[0], csv_path, gpu, True)

    plt.imshow(rec_img)

if __name__ == '__main__':
    main()
