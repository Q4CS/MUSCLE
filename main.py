import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from losses.losses import get_loss
from networks.net_factory_cls import net_factory
from utils import calculate_binary_classification_metric, save_result_to_csv, calculate_multi_classification_metric

from datasets.CheXpert.CheXpert_dataset import CheXpert_Dataset
from datasets.APTOS.APTOS_dataset import APTOS_DataSets
from datasets.Chaoyang.Chaoyang_dataset import Chaoyang_DataSets
from datasets.ISIC_2018_Task_3.ISIC_2018_Task_3_dataset import ISIC2018Task3_DataSets
from datasets.Kvasir.KvasirV2_dataset import KvasirV2_DataSets


evidences_model_list = ['ResNet_CHW_SAFS_TMSL', 'VanillaNet_CHW_SAFS_TMSL', 'SwinTransformer_CHW_SAFS_TMSL']


def parse_option():
    """
    参数设置
    :return:
    """
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='CheXpert',
                        choices=['ISIC_2018_Task_3',
                                 'APTOS',
                                 'CheXpert', 
                                 'KvasirV2',
                                 'Chaoyang',
                                 ],
                        help='dataset name')
    parser.add_argument('--CheXpert_type', type=str, default='Edema',
                        help='CheXpert type')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    # model
    parser.add_argument('--model', type=str, default='ResNet_CHW_SAFS_TMSL', help='model name')
    parser.add_argument('--model_deep', type=int, default=50,
                        help='model deep')
    parser.add_argument('--pretrained', type=int, default=2,
                        choices=[0, 1, 2],
                        help='pretrained model')
    parser.add_argument('--ckpt_path', type=str, default=r'../CheXpert/Edema/ResNet_best_model.pth',
                        help='ckpt path')
    parser.add_argument('--need_replace_module', type=int, default=0,
                        choices=[0, 1],
                        help='whether to replace module when loading ckpt')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')

    # train
    parser.add_argument('--exp_name', type=str,
                        default='cls', help='experiment name')
    parser.add_argument('--need_layer_name_list', type=list, default=['layer1', 'layer2', 'layer3', 'layer4'])
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--deterministic', type=int,  default=1,  # 0, 1
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--seed', type=int,  default=1234,
                        help='random seed')
    args = parser.parse_args()

    return args


def split_list(full_list, ratio=None, shuffle=True):
    if ratio is None:
        ratio = [0.8, 0.2]

    if len(ratio) == 1:
        ratio.append(0.0)

    assert sum(ratio) <= 1, 'The ratio sum must be less than 1'

    sub_list_num = len(ratio)
    list_num = len(full_list)

    if shuffle:
        random.shuffle(full_list)

    sub_lists = []

    count = 0
    for i in range(sub_list_num):
        elem_num = round(list_num * ratio[i])
        if (i + 1) == sub_list_num:
            sub_lists.append(full_list[count:])
        else:
            sub_lists.append(full_list[count:(count + elem_num)])
        count = count + elem_num

    return sub_lists


def set_loader(args):

    dataset = args.dataset
    patch_size = args.patch_size
    batch_size = args.batch_size

    if dataset == 'ISIC_2018_Task_3':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 7

        train_dataset = ISIC2018Task3_DataSets(split='train', img_size=patch_size)
        val_dataset = ISIC2018Task3_DataSets(split='val', img_size=patch_size)
        test_dataset = ISIC2018Task3_DataSets(split='test', img_size=patch_size)
    elif dataset == 'APTOS':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 5

        train_dataset = APTOS_DataSets(split='train', img_size=patch_size)
        val_dataset = APTOS_DataSets(split='val', img_size=patch_size)
        test_dataset = APTOS_DataSets(split='test', img_size=patch_size)
    elif dataset == 'KvasirV2':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 8

        train_dataset = KvasirV2_DataSets(split='train', img_size=patch_size)
        val_dataset = KvasirV2_DataSets(split='val', img_size=patch_size)
        test_dataset = KvasirV2_DataSets(split='test', img_size=patch_size)
    elif dataset == 'Chaoyang':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 4

        train_dataset = Chaoyang_DataSets(split='train', img_size=patch_size)
        val_dataset = Chaoyang_DataSets(split='val', img_size=patch_size)
        test_dataset = Chaoyang_DataSets(split='test', img_size=patch_size)
    elif dataset == 'CheXpert':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 2

        train_dataset = CheXpert_Dataset(split='train', type_name=args.CheXpert_type, image_size=patch_size)
        val_dataset = CheXpert_Dataset(split='val', type_name=args.CheXpert_type, image_size=patch_size)
        test_dataset = CheXpert_Dataset(split='test', type_name=args.CheXpert_type, image_size=patch_size)
    else:
        raise ValueError(batch_size)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    if args.has_val_set:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def set_model(args):
    net = net_factory(args)
    if args.model in evidences_model_list:
        criterion = get_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
    return net, criterion


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []  # predicted class
    gts = []  # ground truth
    preds_logits = []  # predicted logits
    loss_list = []  # loss per iteration
    lr_list = []  # lr per iteration
    images_name_list = []  # images name

    loop = tqdm(train_loader, total=len(train_loader))
    for i_batch, sampled_batch in enumerate(loop):
        if 'cls_label' in sampled_batch:
            label_name = 'cls_label'
        else:
            label_name = 'label'

        image_batch, label_batch = sampled_batch['image'], sampled_batch[label_name]
        image_name_batch = sampled_batch['image_name']
        images_name_list.extend(image_name_batch)

        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        if args.model in evidences_model_list:
            evidences, evidence_a, out_ = model(image_batch)
            loss = get_loss(evidences, evidence_a, label_batch, epoch, args.num_classes, args.annealing_step,
                            1, device)
            outputs = evidence_a
        else:
            outputs = model(image_batch)
            # outputs_soft = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, label_batch)

        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = args.base_lr * (1.0 - ((epoch - 1) * len(train_loader) + i_batch) / args.max_iterations) ** 0.9
        lr_list.append(lr_)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        outputs = outputs.cpu().detach()
        preds_logits.append(outputs)
        outputs = torch.argmax(outputs, dim=1).numpy()
        label_batch = label_batch.cpu().detach().numpy()
        preds.extend(outputs)
        gts.extend(label_batch)

        loop.set_description(f'train, epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item())

    if args.num_classes == 2:
        metric_dict = calculate_binary_classification_metric(preds, gts)
    else:
        metric_dict = calculate_multi_classification_metric(preds, gts, args.num_classes, average='macro')

    preds_logits_cat = torch.cat(preds_logits, dim=0)
    preds_softmax = torch.nn.functional.softmax(preds_logits_cat.detach(), dim=1).numpy()

    metric_dict['predicted_softmax'] = preds_softmax
    metric_dict['predicted_class'] = preds
    metric_dict['ground_truth'] = gts
    metric_dict['loss_average'] = np.mean(loss_list)
    metric_dict['lr_average'] = np.mean(lr_list)
    metric_dict['images_name'] = images_name_list

    return metric_dict


def validate(val_loader, model, criterion, epoch, args):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []  # predicted class
    gts = []  # ground truth
    preds_logits = []  # predicted logits
    loss_list = []  # loss per iteration
    images_name_list = []  # images name

    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader))
        for i_batch, sampled_batch in enumerate(loop):
            if 'cls_label' in sampled_batch:
                label_name = 'cls_label'
            else:
                label_name = 'label'
            image_batch, label_batch = sampled_batch['image'], sampled_batch[label_name]
            image_name_batch = sampled_batch['image_name']
            images_name_list.append(image_name_batch)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if args.model in evidences_model_list:
                evidences, evidence_a, out_ = model(image_batch)
                loss = get_loss(evidences, evidence_a, label_batch, epoch, args.num_classes, args.annealing_step,
                                1, device)
                outputs = evidence_a
            else:
                outputs = model(image_batch)
                # outputs_soft = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, label_batch)

            loss_list.append(loss.item())

            outputs = outputs.cpu().detach()
            preds_logits.append(outputs)
            outputs = torch.argmax(outputs, dim=1).numpy()
            label_batch = label_batch.cpu().detach().numpy()
            preds.extend(outputs)
            gts.extend(label_batch)

            loop.set_description(f'val, epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())

        if args.num_classes == 2:
            metric_dict = calculate_binary_classification_metric(preds, gts)
        else:
            metric_dict = calculate_multi_classification_metric(preds, gts, args.num_classes, average='macro')

        preds_logits_cat = torch.cat(preds_logits, dim=0)
        preds_softmax = torch.nn.functional.softmax(preds_logits_cat.detach(), dim=1).numpy()

        metric_dict['predicted_softmax'] = preds_softmax
        metric_dict['predicted_class'] = preds
        metric_dict['ground_truth'] = gts
        metric_dict['loss_average'] = np.mean(loss_list)
        metric_dict['images_name'] = images_name_list

        return metric_dict


def main():

    args = parse_option()

    if not args.deterministic:
        print('Not deterministic!')
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        print('Deterministic!')
        cudnn.benchmark = False
        cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if (args.pretrained == 1) or (args.pretrained == '1'):
        pretrained_str = '_pretrained'
    elif (args.pretrained == 2) or (args.pretrained == '2'):
        pretrained_str = '_pretrained2'
    else:
        pretrained_str = ''
    
    base_lr_ = str(args.base_lr).split('.')[-1]
    args.snapshot_path = f'../save_dir/{args.dataset}/{args.model}_{args.model_deep}{pretrained_str}/{args.exp_name}_{args.patch_size[0]}x{args.patch_size[1]}_{args.batch_size}bs_{args.epochs}eps_{base_lr_}lr'
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    if os.path.exists(args.snapshot_path + '/code'):
        shutil.rmtree(args.snapshot_path + '/code')
    shutil.copytree('.', args.snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    if not os.path.exists(os.path.join(args.snapshot_path, 'result_csv')):
        os.makedirs(os.path.join(args.snapshot_path, 'result_csv'))
    logging.basicConfig(filename=args.snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # build data loader
    train_loader, val_loader, test_loader = set_loader(args)

    args.max_iterations = len(train_loader) * args.epochs
    logging.info(str(args))

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    # tensorboardX
    writer = SummaryWriter(args.snapshot_path + '/log')
    logging.info(f'{len(train_loader)} iterations per epoch')

    # training routine
    best_performance = 0.0
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        time1 = time.time()
        train_metric_dict = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()
        train_epoch_time = (time2 - time1)
        logging.info(f'train: epoch {epoch}, total time {train_epoch_time:.2f}, lr {train_metric_dict["lr_average"]},'
                     f'loss {train_metric_dict["loss_average"]}, accuracy {train_metric_dict["accuracy"]}, '
                     f'sensitivity {train_metric_dict["sensitivity"]}, specificity {train_metric_dict["specificity"]}, '
                     f'precision {train_metric_dict["precision"]}, f1_score {train_metric_dict["f1_score"]}, '
                     f'confusion_matrix {train_metric_dict["confusion_matrix"]}')

        if args.num_classes > 2:
            logging.info(f'cohen_kappa_score {train_metric_dict["cohen_kappa_score"]}')
            writer.add_scalar('train_info/train_cohen_kappa_score', train_metric_dict['cohen_kappa_score'], epoch)

        # train tensorboard logger
        writer.add_scalar('train_info/lr', train_metric_dict['lr_average'], epoch)
        writer.add_scalar('train_info/train_loss', train_metric_dict['loss_average'], epoch)
        writer.add_scalar('train_info/train_accuracy', train_metric_dict['accuracy'], epoch)
        writer.add_scalar('train_info/train_sensitivity', train_metric_dict['sensitivity'], epoch)
        writer.add_scalar('train_info/train_specificity', train_metric_dict['specificity'], epoch)
        writer.add_scalar('train_info/train_precision', train_metric_dict['precision'], epoch)
        writer.add_scalar('train_info/train_f1_score', train_metric_dict['f1_score'], epoch)

        # save to csv
        train_result_save_path = os.path.join(args.snapshot_path, 'result_csv',
                                              f'train_epoch_{epoch}_acc_{train_metric_dict["accuracy"]}.csv')
        save_result_to_csv(train_metric_dict, train_result_save_path)

        # evaluation
        time1 = time.time()
        val_metric_dict = validate(val_loader, model, criterion, epoch, args)
        time2 = time.time()
        val_epoch_time = (time2 - time1)
        logging.info(f'val: epoch {epoch}, total time {val_epoch_time:.2f}, '
                     f'loss {val_metric_dict["loss_average"]}, accuracy {val_metric_dict["accuracy"]}, '
                     f'sensitivity {val_metric_dict["sensitivity"]}, specificity {val_metric_dict["specificity"]}, '
                     f'precision {val_metric_dict["precision"]}, f1_score {val_metric_dict["f1_score"]}, '
                     f'confusion_matrix {val_metric_dict["confusion_matrix"]}')

        if args.num_classes > 2:
            logging.info(f'cohen_kappa_score {val_metric_dict["cohen_kappa_score"]}')
            writer.add_scalar('val_info/val_cohen_kappa_score', val_metric_dict['cohen_kappa_score'], epoch)

        # evaluation tensorboard logger
        writer.add_scalar('val_info/val_loss', val_metric_dict['loss_average'], epoch)
        writer.add_scalar('val_info/val_accuracy', val_metric_dict['accuracy'], epoch)
        writer.add_scalar('val_info/val_sensitivity', val_metric_dict['sensitivity'], epoch)
        writer.add_scalar('val_info/val_specificity', val_metric_dict['specificity'], epoch)
        writer.add_scalar('val_info/val_precision', val_metric_dict['precision'], epoch)
        writer.add_scalar('val_info/val_f1_score', val_metric_dict['f1_score'], epoch)

        # save to csv
        val_result_save_path = os.path.join(args.snapshot_path, 'result_csv',
                                            f'val_epoch_{epoch}_acc_{val_metric_dict["accuracy"]}.csv')
        save_result_to_csv(val_metric_dict, val_result_save_path)

        # save val best model
        val_accuracy = val_metric_dict['accuracy']
        if val_accuracy > best_performance:
            best_performance = val_accuracy
            save_mode_path = os.path.join(args.snapshot_path, f'epoch_{epoch}_acc_{round(best_performance, 4)}.pth')
            save_best = os.path.join(args.snapshot_path, f'{args.model}_best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)
            logging.info(f'save model to {save_mode_path}')

        if epoch % 50 == 0:
            save_mode_path = os.path.join(args.snapshot_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f'save model to {save_mode_path}')

    logging.info(f'best accuracy: {best_performance:.4f}')

    # test
    if args.has_val_set:
        model = net_factory(args)
        ckpt = torch.load(os.path.join(args.snapshot_path, f'{args.model}_best_model.pth'))  # load model

        if (args.need_replace_module == 1) or (args.need_replace_module == '1'):
            new_state_dict = {}
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
        else:
            new_state_dict = ckpt

        model.load_state_dict(new_state_dict)

        epoch = 0
        time1 = time.time()
        test_metric_dict = validate(test_loader, model, criterion, epoch, args)
        time2 = time.time()
        test_time = (time2 - time1)
        logging.info(f'test: total time {test_time:.2f}, '
                     f'loss {test_metric_dict["loss_average"]}, accuracy {test_metric_dict["accuracy"]}, '
                     f'sensitivity {test_metric_dict["sensitivity"]}, specificity {test_metric_dict["specificity"]}, '
                     f'precision {test_metric_dict["precision"]}, f1_score {test_metric_dict["f1_score"]}, '
                     f'confusion_matrix {test_metric_dict["confusion_matrix"]}')

        if args.num_classes > 2:
            logging.info(f'cohen_kappa_score {test_metric_dict["cohen_kappa_score"]}')
            writer.add_scalar('test_info/test_cohen_kappa_score', test_metric_dict['cohen_kappa_score'], epoch)

        # evaluation tensorboard logger
        writer.add_scalar('test_info/test_loss', test_metric_dict['loss_average'], epoch)
        writer.add_scalar('test_info/test_accuracy', test_metric_dict['accuracy'], epoch)
        writer.add_scalar('test_info/test_sensitivity', test_metric_dict['sensitivity'], epoch)
        writer.add_scalar('test_info/test_specificity', test_metric_dict['specificity'], epoch)
        writer.add_scalar('test_info/test_precision', test_metric_dict['precision'], epoch)
        writer.add_scalar('test_info/test_f1_score', test_metric_dict['f1_score'], epoch)

        # save to csv
        test_result_save_path = os.path.join(args.snapshot_path, 'result_csv',
                                             f'test_epoch_{epoch}_acc_{test_metric_dict["accuracy"]}.csv')
        save_result_to_csv(test_metric_dict, test_result_save_path)

    writer.close()
    return 'Training Finished!'


if __name__ == "__main__":
    main()
