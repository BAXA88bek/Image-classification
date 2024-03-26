import torch
import random
import matplotlib.pyplot as plt

def inference(data, model, device, img_num, cls_names=None, inference_uchun_papka = None, data_nomi = None): #inference
    predictions = []
    images = []
    
    for idx, w in enumerate(data):
        if idx == 10: break
        img, label = w
        img, label = img.to(device), label.to(device)
        pred_cls = torch.argmax(model(img), dim=1)
        predictions.append(pred_cls)
        images.append(img)
        
    indexs = [random.randint(0, len(images)-1) for _ in range(img_num)]
    
    num_cols = 5  
    num_rows = (img_num + num_cols - 1) // num_cols  
    
    plt.figure(figsize=(15, 10))
    
    for idx, index in enumerate(indexs):
        img = images[index][0]
        plt.subplot(num_rows, num_cols, idx + 1)  
        plt.imshow((img*255).cpu().permute(2, 1, 0).numpy().astype("uint8"))
        plt.axis("off")
        plt.title(f"Pred -> {cls_names[predictions[index][0]]}")
        
        plt.savefig(f"{inference_uchun_papka}/{data_nomi}.png")
    
    plt.tight_layout()  
    plt.show()


