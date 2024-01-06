import matplotlib.pyplot as plt
def visualize_images(img_batch):
    _, axarr = plt.subplots(16,8)
    row,col=0,0
    for count, img in enumerate(img_batch):
        axarr[row][col].imshow(img.permute(1,2,0))
        if count == 7:
            break
        col+=1
    plt.show()