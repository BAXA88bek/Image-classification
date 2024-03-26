from matplotlib import pyplot as plt
import random, os
import seaborn as sns


def visualize(data, img_num, rows, class_names, saqlash_uchun_papka = None, data_nomi = None):
    os.makedirs(saqlash_uchun_papka, exist_ok = True)
    predictions = []
    images = []
    
    indeks = [random.randint(0, len(data) - 1) for _ in range(img_num)]
    plt.figure(figsize=[20,10])
    for idx, indek in enumerate(indeks):
        image, gt = data[indek]
        plt.subplot(rows, img_num // rows, idx + 1)
        plt.imshow((image * 255).cpu().permute(1,2,0).numpy().astype("uint8"), cmap = "gray")
        plt.axis("off")
        if class_names: plt.title(f"GT -> {class_names[gt]}")
        else: plt.title(f"GT -> {gt}")
        
        plt.savefig(f"{saqlash_uchun_papka}/{data_nomi}.png")

        


def visualized(res, plot_uchun_papka = None, data_nomi = None):
            
    sns.set_style("whitegrid")

    plt.figure(figsize=(10,5))  
    plt.title("Train and Validation Accuracy score")
    plt.xlabel("Epochs")
    plt.ylabel("Train and Validation score")
    plt.plot(res["tr_acc_sc"], label = "Train accuracy", color="blue", linewidth=2)
    plt.plot(res["val_acc_sc"], label = "Validation accuracy", color="orange",linewidth=2)
    plt.legend()

    plt.figure(figsize=(10,5))
    plt.title("Loss Accuracy score")
    plt.xlabel("Epochs")
    plt.ylabel("Loss score")
    plt.plot(res["val_epoc_loss"], label = "Loss Accuracy", color="brown", linewidth=1)
    plt.legend()  
    plt.show() 
    
    plt.savefig(f"{plot_uchun_papka}/{data_nomi}.png")

# visualize(result) 