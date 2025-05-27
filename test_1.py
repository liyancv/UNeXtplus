# 有GPU和CPU推理时间，能够正确读取模型文件
import argparse
import os
from glob import glob
import torch.autograd.profiler as profiler
import cv2
import torch
import yaml
import time
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from thop import profile, clever_format
from tqdm.auto import tqdm
from albumentations import Resize
from utils import AverageMeter
from dataset import Dataset
from utils import maybe_to_torch, to_cuda
from model import UNet, NestedUNet, UNeXt, UNeXtPlus, UNeXt_trans, UNeXtPlus_test
from transunet import VisionTransformer
from transunet import get_r50_b16_config
import torch.backends.cudnn as cudnn

    
def compute_gpu_inference_time(model, input_tensor, iteration=100):
    """Compute GPU inference time for a given model and input.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_tensor (torch.Tensor): Input tensor for inference.
        iteration (int, optional): Number of iterations for inference. Defaults to 100.

    Returns:
        float: Average inference time in milliseconds.
    """
    model.eval()
    input_tensor = input_tensor.to(device='cuda')
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iteration):
            model(input_tensor)
        stop.record()
        stop.synchronize()
        elapsed_time = start.elapsed_time(stop) / iteration
        torch.cuda.empty_cache()
    return elapsed_time

def compute_cpu_inference_time(model, input_tensor, iteration=100):
    """Compute CPU inference time for a given model and input.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_tensor (torch.Tensor): Input tensor for inference.
        iteration (int, optional): Number of iterations for inference. Defaults to 100.

    Returns:
        float: Average inference time in milliseconds.
    """
    model.eval()
    input_tensor = input_tensor.cuda()  # Move input tensor to GPU
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iteration):
            model(input_tensor)
        elapsed_time = (time.time() - start_time) * 1000 / iteration
    return elapsed_time


    
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return iou, dice
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ISIC2018', help='dataset name')
    parser.add_argument('--method', default=5, type=int)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNeXt_tcb')
    parser.add_argument('--name', default=None, 
                        help='model name: (default: arch)')
    # parser.add_argument('--test_images', default='/juelin/UNeXt-code/ISIC2018')
    # parser.add_argument('--test_masks', default='/juelin/UNeXt-code/ISIC2018')
    parser.add_argument('--test_images', default='/juelin/UNeXt-code/DDTI')
    parser.add_argument('--test_masks', default='/juelin/UNeXt-code/DDTI')
    # parser.add_argument('--test_images', default='/juelin/UNeXt-code/BUSI_data')
    # parser.add_argument('--test_masks', default='/juelin/UNeXt-code/BUSI_data')
    parser.add_argument('--images', default='images')
    parser.add_argument('--masks', default='masks')
    parser.add_argument('--yml_path', default='/juelin/UNeXt-code/models')
    parser.add_argument('--model_path', default='/juelin/UNeXt-code/models/{dataset}/{name}/model.pth')
    args = parser.parse_args([])
    
    if args.name is None:
        timestamp = time.strftime("%Y%m%d")
        args.name = f"{args.arch}"
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()

    yml_path = args.yml_path + f'/{args.dataset}/{args.name}/config.yml'
    with open(yml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True
    
    model_dict = {
        0: UNet(),
        1: NestedUNet(),
        2: VisionTransformer(get_r50_b16_config(), img_size=224, num_classes=1),
        3: UNeXt(),
        4: UNeXtPlus(),
        5: UNeXtPlus_test(),
    }
    model = model_dict[args.method]
    # model_path = args.model_path + f'/{args.dataset}/{args.name}/model.pth'
    model_path = args.model_path.format(dataset=args.dataset, name=args.name)
    model.load_state_dict(torch.load(model_path))
    model.cuda()  # 将模型转移到 GPU 上


    # 首先定义你的模型
    model = UNeXtPlus_test()

    # 加载旧的权重
    state_dict = torch.load('old_model_weights.pth')

    # 手动初始化新添加的层
    for name, param in model.named_parameters():
        if name not in state_dict:
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    # 更新模型的权重
    model.load_state_dict(state_dict, strict=False)

    # 保存新的模型权重
    torch.save(model.state_dict(), 'updated_model_weights.pth')

    print('-' * 20)

    img_ids = glob(os.path.join(args.test_images, args.images, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    img_ids = sorted(img_ids)
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    print(val_img_ids)

    
    model.eval()
    print(f"Number of parameters: {count_parameters(model):,}")

    val_transform = Compose([
        Resize(config['img_size'], config['img_size']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(args.test_images, args.images),
        mask_dir=os.path.join(args.test_masks, args.masks),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    total_inference_time = 0

    with torch.no_grad():
        for inputs, target, meta in tqdm(val_loader, total=len(val_loader)):
            inputs = maybe_to_torch(inputs).cuda()
            target = maybe_to_torch(target).cuda()
            # inputs = maybe_to_torch(inputs).cpu()
            # target = maybe_to_torch(target).cpu()


            start_time = time.time()

            output = model(inputs)
            gpu_inference_time = compute_gpu_inference_time(model, inputs)  
            # 使用 GPU 推理时间函数
            cpu_inference_time = compute_cpu_inference_time(model, inputs)
            # 使用 CPU 推理时间函数
            
            flops, params = profile(model, inputs=(inputs,))
            flops, params = clever_format([flops, params], "%.3f")
            print(f"Number of FLOPs: {flops}")

            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, inputs.size(0))
            dice_avg_meter.update(dice, inputs.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            print('Inference Time per Iteration: %.4f s/it' % inference_time)

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Inference time: %.4f' % inference_time)
    print('gpu_inference_time: %.4f' % gpu_inference_time)
    print('cpu_inference_time: %.4f' % cpu_inference_time)
    print('Total Inference Time: %.4f seconds' % total_inference_time)

if __name__ == '__main__':
    main()
