#call load data, spit data, transform data
from src.load_data import ProcessData
from src.visualizer import visualize_images
import matplotlib.pyplot as plt
def retrieve_data(data_path, batch_size):
    loader = ProcessData()
    images = loader.load_data(data_path)
    train_data, test_data = loader.split_data(images)
    
    r_mean_a, g_mean_a, b_mean_a, r_std_a, g_std_a, b_std_a = loader.calc_mean_and_std(train_data)
    
    f, axarr = plt.subplots(2,2)
    print(train_data[0][0])
    axarr[0,0].imshow(train_data[0][0])
    axarr[0,1].imshow(train_data[1][0])
    axarr[1,0].imshow(train_data[2][0])
    axarr[1,1].imshow(train_data[3][0])
    plt.show()
    mean = [r_mean_a,g_mean_a,b_mean_a]
    std = [r_std_a, g_std_a, b_std_a]
    train_transform, test_transform = loader.apply_transformations(mean,std)
    
    train_data.dataset.transform = train_transform
    test_data.dataset.transform = test_transform

    
    train_dataloader, test_dataloader = loader.create_dataloader(train_data, test_data, batch_size = batch_size )
    return train_dataloader, test_dataloader