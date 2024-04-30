import os
import glob
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset_tool import PointCloudDataset, PointCloudDatasetTest
from torch.utils.data import DataLoader
import util
import metrics
import error_analysis

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def predict(net, testloader, device):
    with torch.no_grad():
        net.eval()
        pred_list = []
        for batch in testloader:
            input_coord = batch
            input_coord = input_coord.to(device)
            pred = net(input_coord)
            pred_list.append(pred)
        return pred_list

def eval(net, testloader, device, criterion=None):
    running_val_loss = 0.0
    with torch.no_grad():
        net.eval()
        pred_list = []
        gt_list = []
        lprot_str, ax_load_str, boundary_str = [], [], []
        batch_ind = 0
        for batch in testloader:
            input_coord, gt_rc, lprot_batch, ax_load_batch, boundary_batch = batch['pc_local'], batch['rc'], batch['lprot'], batch['ax_load'], batch['boundary']
            lprot_str.extend(lprot_batch)
            ax_load_str.extend(ax_load_batch)
            boundary_str.extend(boundary_batch)
            input_coord = input_coord.to(device)
            gt_rc = gt_rc.to(device)
            pred = net(input_coord)
            if criterion is not None:
                loss_val = criterion(pred, gt_rc)
                running_val_loss += loss_val.item()
            pred_list.append(pred)
            gt_list.append(gt_rc)
            batch_ind+=1
        if criterion is not None:
            return pred_list, gt_list, running_val_loss/len(testloader), lprot_str, ax_load_str, boundary_str
        else:
            return pred_list, gt_list, lprot_str, ax_load_str, boundary_str


def train(net, trainloader, valloader, num_epochs, save_path, device, optimizer, criterion, scheduler=None):
    net.train()
    train_loss = []
    val_loss = []
    best_error = np.inf
    for epoch in range(num_epochs):
        running_loss = 0.0
        for bind, data in enumerate(trainloader):   
            input_coord, gt_rc = data['pc_local'], data['rc']
            input_coord = input_coord.to(device)
            gt_rc = gt_rc.to(device)
            #input_coord = input_coord.view(input_coord.size(0), -1)
            optimizer.zero_grad()
            output = net(input_coord)
            loss = criterion(output, gt_rc)
            #loss += 0.001*feature_transform_reguliarzer(features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #scheduler.step()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        #eval
        pred_val, gt_val, loss_v, _, _, _ = eval(net, valloader, device, criterion)
        val_loss.append(loss_v)
        rc_pred_val = np.vstack([(pred_val[i]).detach().cpu().numpy() for i in range(len(pred_val))])
        rc_gt_val = np.vstack([(gt_val[i]).detach().cpu().numpy() for i in range(len(gt_val))])
        mape_mean_val, mape_std_val, avg_percentage_err_val = metrics.mape(rc_gt_val, rc_pred_val)
        if epoch % 5 == 0:
            print('Epoch {} of {}, Train Loss: {:.6f}'.format(epoch+1, num_epochs, loss))
        if best_error > mape_mean_val + mape_std_val:
            best_error = mape_mean_val + mape_std_val
            print('Saving network parameters - Epoch {}.'.format(epoch+1))
            checkpoint = { 
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),}
                #'lr_sched': scheduler}
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
    return train_loss, val_loss


def training_loop(opts):
    # Initialize
    batch_size = opts.batch_size
    num_epochs = opts.epoch
    lr = opts.lr
    loss = opts.loss
    device = get_device()

    # Create the output folder
    folder_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = opts.model_type + '_' + opts.segment + '_' + opts.lprot + ("" if opts.norm is None else '_' + opts.norm) + '_' +folder_time
    current_path = os.getcwd()
    save_path = os.path.join(current_path, 'results', model_name)
    print('Saving results in ', save_path)
    os.makedirs(save_path, exist_ok=True)

    #Load data
    trainset = PointCloudDataset(opts.path_data, opts.segment, opts.norm, opts.lprot, split='training')
    valset = PointCloudDataset(opts.path_data, opts.segment, opts.norm, opts.lprot, split='validation')
    testset = PointCloudDataset(opts.path_data, opts.segment, opts.norm, opts.lprot, split='test')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Construct the network and data loader
    net = util.construct_class_by_name(**opts.model_kwargs).train().to(device)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("# trainable parameters: ", total_params)
    criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam(
    #         net.parameters(),
    #         lr=lr,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=1e-4
    #     )
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    optimizer = optim.Adam(net.parameters(), lr=lr) 
    
    #Train the network
    training_loss, validation_loss = train(net, trainloader, valloader, num_epochs, save_path, device, optimizer, criterion)#, scheduler)
    #Plot the training and validation losses
    reserve_capacity_bins = np.array(opts.reserve_capacity_bins)

    plt.figure()
    plt.plot(training_loss,label= 'Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'model_loss.png'))

    #Test the network and save the result dictionary
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['model'])
    pred_test, gt_test, lprot_test, ax_load_test, boundary_test = eval(net, testloader, device)
    rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
    rc_gt_test = np.vstack([(gt_test[i]).detach().cpu().numpy() for i in range(len(gt_test))])
    error_analysis.construct_result_dictionary(save_path, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins)

def evaluate(opts):
    error_dict_path = glob.glob(os.path.join(opts.path_model, "*.npy"))
    if not bool(error_dict_path):
        #Load data
        print('Constructing the error dictionary')
        batch_size = opts.batch_size
        device = get_device()
        reserve_capacity_bins = np.array(opts.reserve_capacity_bins)
        testset = PointCloudDataset(opts.path_data, opts.segment, opts.norm, opts.lprot, split='test')
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        net = util.construct_class_by_name(**opts.model_kwargs).to(device)
        checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
        net.load_state_dict(checkpoint['model'])
        pred_test, gt_test, lprot_test, ax_load_test, boundary_test = eval(net, testloader, device)
        rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
        rc_gt_test = np.vstack([(gt_test[i]).detach().cpu().numpy() for i in range(len(gt_test))])
        error_analysis.construct_result_dictionary(opts.path_model, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins)
    print('Loading the error dictionary')
    eval_dict = np.load(os.path.join(opts.path_model, 'error_dict.npy'), allow_pickle=True).item()
    error_analysis.plot_pred_vs_gt(eval_dict, opts.path_model, category=0)
    error_analysis.plot_pred_vs_gt(eval_dict, opts.path_model, category=1)
    error_analysis.plot_pred_vs_gt(eval_dict, opts.path_model, category=2)
    error_analysis.err_per_reserve_capacity(eval_dict)
    
def test(opts):    
    #Load data
    torch.manual_seed(37)
    batch_size = opts.batch_size
    testset = PointCloudDatasetTest(opts.path_data, opts.segment, opts.norm, opts.lprot)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    device = get_device()

    # Construct the network
    net = util.construct_class_by_name(**opts.model_kwargs).to(device)
    checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
    net.load_state_dict(checkpoint['model'])
    pred_test = predict(net, testloader, device)
    rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
    print(rc_pred_final_test.shape)
    print(rc_pred_final_test)
    os.makedirs(os.path.join(opts.path_model, 'predictions'), exist_ok=True)
    np.save(os.path.join(opts.path_model, 'predictions', 'results_' + testset.data_name + '_model_' + opts.model_type +'.npy'), rc_pred_final_test)