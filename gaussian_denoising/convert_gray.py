import cv2
import os
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('trainsets', 'DIV2K_train', '*.png'))
    files_source.sort()
    # process data
    for f in files_source:
        print(f)
        Img = cv2.imread(f, 0)
        cv2.imwrite(os.path.join('trainsets', 'DIV2K_train_gray', f[-13:-4]+'.png'), Img)


if __name__ == "__main__":
    main()
