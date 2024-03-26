import argparse
from data import get_dls
from torchvision import transforms as T
from utils import visualize, visualized
from train import train, train_setup
import torch, timm, os
from inference import inference

def run(args):
    
    tfs = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    
    print("Dataset yuklanmoqda...\n")
    tr_dl, val_dl, ts_dl, classes = get_dls(root = args.data_yulagi, transformations = tfs, bs = 16)
    print("Dataloader lar yaratib olindi!\n")
    # print(len(tr_dl)); print(len(val_dl)); print(len(ts_dl)); 
    # print(classes)
    print(f"Visualization bajarilmoqda...\n")
    data_nomi = {tr_dl:"train", val_dl:"validation", ts_dl:"test"}
    for data, nomi in data_nomi.items():
        visualize(data = data.dataset, img_num = args.img_num, rows = args.rows, class_names = list(classes.keys()), saqlash_uchun_papka = args.saqlash_uchun_papka, data_nomi = nomi)
    print(f"Visualization rasmlarni {args.saqlash_uchun_papka} papkasidan tekshirib ko'rishingiz mumkin.\n")
    
    Model = timm.create_model(model_name = args.model_nomi, pretrained = True, num_classes = 5)
    Model, epochs, loss_fn, optimazer = train_setup(Model, device = args.device)

    result = train(model = Model, tr_dl = tr_dl, val_dl = val_dl, loss_fc = loss_fn, epochs = epochs, optimazer = optimazer, device = args.device, save_prefix = args.data_nomi, save_dir = args.save_dir)
    print("Train jarayoni yakunlandi!\n")
    visualized(res = result, plot_uchun_papka = args.plot_uchun_papka, data_nomi = args.data_nomi)
    
    print("Inference jarayoni boshlanmoqda...!\n")
    
    model_path = f"{args.save_dir}/{args.data_nomi}_best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
    else:
        Model.load_state_dict(torch.load(model_path))
        Model.eval()
    #Model.load_state_dict(torch.load(f"{args.save_dir}/{args.data_nomi}_best_model.pth")) 
    #m.load_state_dict(torch.load(f"{args.save_dir}/{args.data_nomi}_best_model.pth"))
    #Model.eval()
        inference(data = ts_dl, model = Model.to(args.device), device=args.device, img_num=15, cls_names=list(classes.keys()), inference_uchun_papka = args.save_dir, data_nomi = args.data_nomi) 
        print(f"Inference rasmlarni {args.inference_uchun_papka} papkasidan tekshirib ko'rishingiz mumkin.\n")
        print("Inference jarayoni yakunlandi!\n")
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Cloud classification Project Arguments")
    
    parser.add_argument("-dy", "--data_yulagi", type = str, default = "data/cloud_classification", help = "Data turgan yulak")
    parser.add_argument("-sp", "--saqlash_uchun_papka", type = str, default = "vis_images", help = "Visulization rasmlarni saqlash uchun yulak")
    parser.add_argument("-ip", "--inference_uchun_papka", type = str, default = "Inference_images", help = "Inference rasmlarni saqlash uchun yulak")
    parser.add_argument("-pp", "--plot_uchun_papka", type = str, default = "plot_images", help = "Plot qilingan rasmlarni saqlash uchun yulak")
    parser.add_argument("-sd", "--save_dir", type = str, default = "saved_models", help = "Train qilingan modelni saqlash uchun papka turgan yo'lak")
    #parser.add_argument("-sd", "--save_dir", type = str, default = "saved_models", help = "Train qilingan modelni saqlash uchun papka turgan yo'lak")
    parser.add_argument("-r", "--rows", type = int, default = 5, help = "rasmlar qatori")
    parser.add_argument("-in", "--img_num", type = int, default = 25, help = "rasmlar soni")
    parser.add_argument("-mn", "--model_nomi", type = str, default = "resnet18", help = "AI Model nomi")
    parser.add_argument("-d", "--device", type = str, default = "cuda:0", help = "GPU nomi")
    parser.add_argument("-dn", "--data_nomi", type = str, default = "clouds", help = "Data nomi")
    parser.add_argument("-my", "--model_yulagi", type = str, default = "saved_models/clouds_best_model.pth", help = "Train qilingan model  yo`li")
    
    
    args = parser.parse_args()
    
    run(args)
    

    