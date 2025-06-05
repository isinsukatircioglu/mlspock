import os
import shutil
import glob
import json
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
from dataset_tool import PointCloudDataset, PointCloudDatasetTest
from model.pointnet_utils import feature_transform_reguliarzer
from torch.utils.data import DataLoader
import util
import metrics
import error_analysis

def set_seeds(seed: int=42):
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

class AsymmetricMSE(torch.nn.Module):
    def __init__(self, overprediction_penalty=2.0, underprediction_penalty=1.0):
        super(AsymmetricMSE, self).__init__()
        self.overprediction_penalty = overprediction_penalty
        self.underprediction_penalty = underprediction_penalty

    def forward(self, predictions, targets):
        diff = predictions - targets
        loss = torch.where(
            diff > 0,
            self.overprediction_penalty * diff ** 2,
            self.underprediction_penalty * diff ** 2
        )
        return loss.mean()

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def save_args(options, result_folder):
    with open(os.path.join(result_folder, 'options.json'), 'w') as file:
        json.dump(options, file)

def predict(net, testloader, device, classify=None, lratio_known=False, boundary_known=False, lprotocol_known=False, dimension_known=False):
    target_names = {}
    if classify is not None:
        target_names['lprot'] = {'Symmetric': 0, 'Monotonic': 1, 'Collapse_consistent': 2}
        target_names['lratio'] = {'0.1': 0, '0.2':1, '0.3':2, '0.4': 3, '0.5':4}
        target_names['boundary'] = {'Fixed':0, 'Flexible':1}
        if classify == 'lprot':
            tag = 'lprot'
        elif classify == 'lratio':
            tag = 'ax_load'
        elif classify == 'boundary':
            tag = 'boundary'
        elif classify == 'all':
            class_pred, gt_class = [[],[],[]], [[],[],[]]
    with torch.no_grad():
        net.eval()
        pred_list = []
        gt_list = []
        lprot_str, ax_load_str, boundary_str = [], [], []
        for batch in testloader:
            input_coord, lratio_input, boundary_input, lprot_input, column_size, knn_idx = batch['pc_local'], batch['lratio_label'], batch['boundary_label'], batch['lprot_label'], batch['column_size'], batch['knn_idx']
            input_coord = input_coord.to(device)
            lratio_input = lratio_input.to(device)
            boundary_input = boundary_input.to(device)
            lprot_input = lprot_input.to(device)
            column_size = column_size.to(device)
            knn_idx = knn_idx.to(device)
            #read the lratio one-hot vector
            if lratio_known:
                lratio_vec = lratio_input
            else:
                lratio_vec = None
            #read the boundary one-hot vector
            if boundary_known:
                boundary_vec = boundary_input
            else:
                boundary_vec = None
            if lprotocol_known:
                lprotocol_vec = lprot_input
            else:
                lprotocol_vec = None
            if dimension_known:
                dimension_vec = column_size
            else:
                dimension_vec = None
            pred = net(input_coord, knn_idx, lratio_vec, boundary_vec, lprotocol_vec, dimension_vec)
            if isinstance(pred, tuple):
                if classify != 'all':
                    class_pred.extend(torch.argmax(pred[1], dim=-1).cpu().tolist())
                    for s_ind, s in enumerate(batch[tag]):
                        gt_class.append(target_names[classify][s])
                else:
                    class_pred[0].extend(torch.argmax(pred[1][:, 0:3], dim=-1).cpu().tolist())
                    class_pred[1].extend(torch.argmax(pred[1][:, 3:5], dim=-1).cpu().tolist())
                    class_pred[2].extend(torch.argmax(pred[1][:, -5:], dim=-1).cpu().tolist())
                    for s_ind, s in enumerate(batch['lprot']):
                        gt_class[0].append(target_names['lprot'][batch['lprot'][s_ind]])
                        gt_class[1].append(target_names['boundary'][batch['boundary'][s_ind]])
                        gt_class[2].append(target_names['lratio'][batch['ax_load'][s_ind]])
                pred = pred[0]
            pred_list.append(pred)
            if 'rc' in batch.keys():
                lprot_batch, ax_load_batch, boundary_batch =  batch['lprot'], batch['ax_load'], batch['boundary']
                gt = batch['rc'].to(device)
                gt_list.append(gt)
                lprot_str.extend(lprot_batch)
                ax_load_str.extend(ax_load_batch)
                boundary_str.extend(boundary_batch)
        if 'rc' in batch.keys():
            return pred_list, gt_list, lprot_str, ax_load_str, boundary_str
        return pred_list

def eval(net, testloader, device, criterion=None, classify=None, lratio_known=False, boundary_known=False, lprotocol_known=False, dimension_known=False):
    target_names = {}
    running_val_loss = 0.0
    tag = ''
    pred_list, gt_list = [],[]
    lprot_str, ax_load_str, boundary_str = [], [], []
    class_pred, gt_class = [],[]
    if classify is not None:
        target_names['lprot'] = {'Symmetric': 0, 'Monotonic': 1, 'Collapse_consistent': 2}
        target_names['lratio'] = {'0.1': 0, '0.2':1, '0.3':2, '0.4': 3, '0.5':4}
        target_names['boundary'] = {'Fixed':0, 'Flexible':1}
        if classify == 'lprot':
            tag = 'lprot'
        elif classify == 'lratio':
            tag = 'ax_load'
        elif classify == 'boundary':
            tag = 'boundary'
        elif classify == 'all':
            class_pred, gt_class = [[],[],[]], [[],[],[]]
    with torch.no_grad():
        net.eval()
        batch_ind = 0
        score = 0
        for batch in testloader:
            input_coord, gt_rc, lprot_batch, ax_load_batch, boundary_batch, lratio_input, boundary_input, lprot_input, direction_target, column_size, knn_idx = batch['pc_local'], batch['rc'], batch['lprot'], batch['ax_load'], batch['boundary'], batch['lratio_label'], batch['boundary_label'], batch['lprot_label'], batch['direction_label'], batch['column_size'], batch['knn_idx']
            lprot_str.extend(lprot_batch)
            ax_load_str.extend(ax_load_batch)
            boundary_str.extend(boundary_batch)
            input_coord = input_coord.to(device)
            gt_rc = gt_rc.to(device)
            lratio_input = lratio_input.to(device)
            boundary_input = boundary_input.to(device)
            lprot_input = lprot_input.to(device)
            column_size = column_size.to(device)
            knn_idx = knn_idx.to(device)
            #read the lratio one-hot vector
            if lratio_known:
                lratio_vec = lratio_input
            else:
                lratio_vec = None
            #read the boundary one-hot vector
            if boundary_known:
                boundary_vec = boundary_input
            else:
                boundary_vec = None
            if lprotocol_known:
                lprotocol_vec = lprot_input
            else:
                lprotocol_vec = None
            if dimension_known:
                dimension_vec = column_size
            else:
                dimension_vec = None
            pred = net(input_coord, knn_idx, lratio_vec, boundary_vec, lprotocol_vec, dimension_vec)
            if isinstance(pred, tuple):
                if classify != 'all':
                    class_pred.extend(torch.argmax(pred[1], dim=-1).cpu().tolist())
                    for s_ind, s in enumerate(batch[tag]):
                        gt_class.append(target_names[classify][s])
                else:
                    class_pred[0].extend(torch.argmax(pred[1][:, 0:3], dim=-1).cpu().tolist())
                    class_pred[1].extend(torch.argmax(pred[1][:, 3:5], dim=-1).cpu().tolist())
                    class_pred[2].extend(torch.argmax(pred[1][:, -5:], dim=-1).cpu().tolist())
                    for s_ind, s in enumerate(batch['lprot']):
                        gt_class[0].append(target_names['lprot'][batch['lprot'][s_ind]])
                        gt_class[1].append(target_names['boundary'][batch['boundary'][s_ind]])
                        gt_class[2].append(target_names['lratio'][batch['ax_load'][s_ind]])
                pred = pred[0]
            if criterion is not None:
                loss_val = criterion(pred, gt_rc)
                running_val_loss += loss_val.item()
            pred_list.append(pred)
            gt_list.append(gt_rc)
            batch_ind+=1
        
        if len(class_pred) > 0:
            if classify != 'all':
                # Count the number of correct predictions
                correct_predictions = sum(p == g for p, g in zip(class_pred, gt_class))
                # Calculate the accuracy
                accuracy = (correct_predictions / len(class_pred)) * 100
                print('Classification score: ', accuracy)
            else:
                # Count the number of correct predictions
                correct_predictions_lprot = sum(p == g for p, g in zip(class_pred[0], gt_class[0]))
                correct_predictions_boundary = sum(p == g for p, g in zip(class_pred[1], gt_class[1]))
                correct_predictions_lratio = sum(p == g for p, g in zip(class_pred[2], gt_class[2]))
                # Calculate the accuracy
                accuracy_lprot = (correct_predictions_lprot / len(class_pred[0])) * 100
                accuracy_boundary = (correct_predictions_boundary / len(class_pred[1])) * 100
                accuracy_lratio = (correct_predictions_lratio / len(class_pred[2])) * 100
                print(f"Classification score lprot: {accuracy_lprot}, boundary: {accuracy_boundary}, lratio: {accuracy_lratio}")
        if criterion is not None:
            return pred_list, gt_list, running_val_loss/len(testloader), lprot_str, ax_load_str, boundary_str
        else:
            return pred_list, gt_list, lprot_str, ax_load_str, boundary_str


def train(net, trainloader, valloader, num_epochs, save_path, device, optimizer, epoch_init, best_error, criterion, criterion_class=None, classify=None, lratio_known=False, boundary_known=False, lprotocol_known=False, dimension_known=False, scheduler=None, folder_time=None):
    net.train()
    train_loss = []
    val_loss = []
    for epoch in range(epoch_init, num_epochs+1):
        running_loss = 0.0
        for bind, data in enumerate(trainloader):   
            input_coord, gt_rc, lprot_target, lratio_input, boundary_input, direction_target, column_size, knn_idx = data['pc_local'], data['rc'], data['lprot_label'], data['lratio_label'], data['boundary_label'], data['direction_label'], data['column_size'], data['knn_idx']
            input_coord = input_coord.to(device)
            gt_rc = gt_rc.to(device)
            lprot_target = lprot_target.to(device)
            direction_target = direction_target.to(device)
            lratio_input = lratio_input.to(device)
            boundary_input = boundary_input.to(device)
            column_size = column_size.to(device)
            knn_idx = knn_idx.to(device)
            optimizer.zero_grad()
            #read the lratio one-hot vector
            if lratio_known:
                lratio_vec = lratio_input
            else:
                lratio_vec = None
            #read the boundary one-hot vector
            if boundary_known:
                boundary_vec = boundary_input
            else:
                boundary_vec = None
            #read the loading protocol one-hot vector
            if lprotocol_known:
                lprotocol_vec = lprot_target
            else:
                lprotocol_vec = None
            if dimension_known:
                dimension_vec = column_size
            else:
                dimension_vec = None
            if criterion_class is not None:
                target_class = None
                if classify == 'lprot':
                    target_class = lprot_target
                elif classify == 'boundary':
                    target_class = boundary_input
                elif classify == 'lratio':
                    target_class = lratio_input
                elif classify == 'all':
                    target_class = torch.cat((lprot_target, boundary_input, lratio_input), dim=-1)
                output, output_class = net(input_coord, knn_idx, lratio_vec, boundary_vec, lprotocol_vec, dimension_vec)
                loss_rc = criterion(output, gt_rc)
                if classify == 'all':
                    loss_lprot = criterion_class(output_class[:, :3], target_class[:, :3])
                    loss_boundary = criterion_class(output_class[:, 3:5], target_class[:, 3:5])
                    loss_lratio = criterion_class(output_class[:, -5:], target_class[:, -5:])
                    loss_class = loss_lprot + loss_boundary + loss_lratio
                else:
                    loss_class = criterion_class(output_class, target_class)
                loss = loss_rc + 0.001*loss_class #0.01
            else:
                output = net(input_coord, knn_idx, lratio_vec, boundary_vec, lprotocol_vec, dimension_vec)
                if isinstance(output, tuple):
                    features = output[1]
                    output = output[0]
                    loss_rc = criterion(output, gt_rc)
                    loss = loss_rc + 0.001*feature_transform_reguliarzer(features)
                else:
                    loss_rc = criterion(output, gt_rc)
                    loss = loss_rc
            #loss += 0.001*feature_transform_reguliarzer(features)
            loss.backward()
            optimizer.step()
            running_loss += loss_rc.item()
        if scheduler is not None:
            scheduler.step()
            print('lr: ' , scheduler.get_last_lr())
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        #Evaluate the current model
        pred_val, gt_val, loss_v, _, _, _ = eval(net, valloader, device, criterion, classify, lratio_known, boundary_known, lprotocol_known, dimension_known)
        val_loss.append(loss_v)
        rc_pred_val = np.vstack([(pred_val[i]).detach().cpu().numpy() for i in range(len(pred_val))])
        rc_gt_val = np.vstack([(gt_val[i]).detach().cpu().numpy() for i in range(len(gt_val))])
        mape_mean_val, mape_std_val, avg_percentage_err_val = metrics.mape(rc_gt_val, rc_pred_val)
        if epoch % 5 == 0:
            print('[{}] Epoch {} of {}, Train Loss: {:.6f}'.format(folder_time, epoch, num_epochs, loss))
        if best_error > mape_mean_val + mape_std_val:
            best_error = mape_mean_val + mape_std_val
            print('Saving network parameters for the best model - Epoch {}.'.format(epoch))
            if scheduler is not None:
                checkpoint = { 
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler.state_dict(),
                'best_error': best_error}
            else:
                checkpoint = { 
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_error': best_error}
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
        if scheduler is not None:
            checkpoint = { 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler.state_dict(),
            'best_error': best_error}
        else:
            checkpoint = { 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_error': best_error}
        torch.save(checkpoint, os.path.join(save_path, 'last_model.pth'))
    return train_loss, val_loss

def training_loop(opts):
    # Initialize
    set_seeds(opts.seed)
    batch_size = opts.batch_size
    num_epochs = opts.epoch
    lr = opts.lr
    loss = opts.loss
    lratio_known = opts.lratio_known
    boundary_known = opts.boundary_known
    lprotocol_known = opts.lprotocol_known
    dimension_known = opts.dimension_known
    aux_classify = opts.classify
    device = get_device()
    epoch_start = 1
    best_error = np.inf
    # Create the output folder
    folder_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') #not used anymore
    extra_inp = "3d" + "".join(['_'+name for name, known in zip(['lratio', 'boundary', 'lprot', 'colsize'], [opts['lratio_known'], opts['boundary_known'], opts['lprotocol_known'],opts['dimension_known']]) if known])
    aug_inp = "_".join([name for name, known in zip(['rotate180', 'mirror', 'delta', 'delta3d'], [opts['rotate180'], opts['mirror'], opts['delta'], opts['delta3d']]) if known])
    classify_opt = f"_classify[{opts.classify}_{opts.split}]_" if opts.classify is not None else f""
    lratio_select = f"_lratio[{opts.lratio}]" if opts.lratio is not None else ""
    #model_name = opts.model_type + '_' + opts.segment + '_' + opts.lprot + ("" if opts.norm is None else '_' + opts.norm) + '_' +folder_time
    model_name = f"{opts.model_type}_{opts.model_size}_segment[{opts.segment}]_{opts.norm}[{opts.scaling_mode}_scale]_input[{extra_inp}]_lprot[{opts.lprot}]{lratio_select}_batch_{opts.batch_size}_lr_{opts.lr}_optimizer_{opts.optimizer}{classify_opt}{aug_inp}_seed{opts.seed}_v{opts.version}".replace("__", "_")
    current_path = os.getcwd()
    save_path = os.path.join(current_path, 'experiments', model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    #Start logging
    log_id = 1
    if os.path.exists(os.path.join(save_path, 'best_model.pth')):
        print('Renaming the best model from the last training session.')
        while os.path.exists(os.path.join(save_path, 'best_model' + str(log_id) + '.pth')):
            log_id += 1
        os.rename(os.path.join(save_path, 'best_model.pth'), os.path.join(save_path, 'best_model' + str(log_id) + '.pth'))
    util.Logger(file_name=os.path.join(save_path, 'log.txt'), file_mode='a', should_flush=True)
    print('Saving results in ', save_path)
    print('Training started on ', folder_time)
    save_args(opts, save_path)
    #Load data
    trainset = PointCloudDataset(opts, split='training')
    valset = PointCloudDataset(opts, split='validation')
    testset = PointCloudDataset(opts, split='test')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=opts.num_worker, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=opts.num_worker, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=opts.num_worker, pin_memory=True)
    # Construct the network and data loader
    net = util.construct_class_by_name(**opts.model_kwargs).train().to(device)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('-----------------------')
    print("# trainable parameters: ", total_params)
    print('#training set size: ', len(trainset))
    print('-----------------------')
    if opts.delta:
        point_dim = 6
    else:
        point_dim = 3
    dummy_input = torch.rand(1, trainset.npoints, point_dim).to(device)
    dummy_knn_ind = torch.randint(1, 100, (1, trainset.npoints, trainset.num_neigh)).to(device)
    input_data_dummy = []
    input_data_dummy.append(dummy_input)
    input_data_dummy.append(dummy_knn_ind)
    dummy_lratio=None
    if lratio_known:
        dummy_lratio=((torch.nn.functional.one_hot(torch.ones(1).long(), num_classes=5))[0]).float().to(device).view(1,-1)
        input_data_dummy.append(dummy_lratio)
    dummy_boundary=None
    if boundary_known:
        dummy_boundary=((torch.nn.functional.one_hot(torch.ones(1).long(), num_classes=2))[0]).float().to(device).view(1,-1)
        input_data_dummy.append(dummy_boundary)
    dummy_lprotocol=None
    if lprotocol_known:
        dummy_lprotocol=((torch.nn.functional.one_hot(torch.ones(1).long(), num_classes=3))[0]).float().to(device).view(1,-1)
        input_data_dummy.append(dummy_lprotocol)
    dummy_dimension=None
    if dimension_known:
        dummy_dimension=((torch.nn.functional.one_hot(torch.ones(1).long(), num_classes=4))[0]).float().to(device).view(1,-1)
        input_data_dummy.append(dummy_dimension)
    model_stats = summary(net,  input_data=input_data_dummy, batch_dim = 0, verbose=False)
    summary_str = str(model_stats)
    with open(os.path.join(save_path , "model_architecture.txt"), "w") as text_file:
        text_file.write(summary_str)
    if opts.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif opts.loss == 'AsymmetricMSE':
        criterion = AsymmetricMSE(overprediction_penalty=2.0, underprediction_penalty=1.0)
    if opts.classify is not None:
        criterion_class = torch.nn.CrossEntropyLoss()
    else:
        criterion_class = None
    
    #Set optimizer
    if opts.optimizer == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=lr)
    elif opts.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    
    #Set scheduler
    scheduler = None
    if opts.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    elif opts.scheduler == 'CosineAnnealingLR':
        scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif opts.scheduler == 'MultiStepLR':
        scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    #Resume training
    if opts.resume and os.path.exists(os.path.join(opts.path_model, 'best_model.pth')):
        checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        epoch_start = checkpoint['epoch'] + 1
        best_error = checkpoint['best_error']
        print('Loading model and optimizer state dict from ', opts.path_model, ' - resume training from epoch ', epoch_start)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['lr_sched'])

    #Train the network
    training_loss, validation_loss = train(net, trainloader, valloader, num_epochs, save_path, device, optimizer, epoch_start, best_error, criterion, criterion_class, aux_classify, lratio_known, boundary_known, lprotocol_known, dimension_known, scheduler, folder_time)
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
    checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0]
    net.load_state_dict(checkpoint['model'])
    pred_test, gt_test, lprot_test, ax_load_test, boundary_test = eval(net, testloader, device, classify=opts.classify, lratio_known=lratio_known, boundary_known=boundary_known, lprotocol_known=lprotocol_known, dimension_known=dimension_known)
    rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
    rc_gt_test = np.vstack([(gt_test[i]).detach().cpu().numpy() for i in range(len(gt_test))])
    error_analysis.construct_result_dictionary(save_path, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins)

def evaluate(opts):
    npy_files = [f for f in os.listdir(opts.path_model) if f.endswith('.npy')]
    error_dict_path = None
    if len(npy_files) > 0:
        error_dict_path = os.path.join(opts.path_model, npy_files[0])
    #error_dict_path = glob.glob(os.path.join(opts.path_model, "*.npy"))
    lratio_known = opts.lratio_known
    boundary_known = opts.boundary_known
    lprotocol_known = opts.lprotocol_known
    dimension_known = opts.dimension_known
    if not bool(error_dict_path):
        #Load data
        print('Constructing the error dictionary')
        batch_size = opts.batch_size
        device = get_device()
        reserve_capacity_bins = np.array(opts.reserve_capacity_bins)
        testset = PointCloudDataset(opts, split='test')
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=opts.num_worker, pin_memory=True)
        net = util.construct_class_by_name(**opts.model_kwargs).to(device)
        checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        net.load_state_dict(checkpoint['model'])
        pred_test, gt_test, lprot_test, ax_load_test, boundary_test = eval(net, testloader, device, classify=opts.classify, lratio_known=lratio_known, boundary_known=boundary_known, lprotocol_known=lprotocol_known, dimension_known=dimension_known)
        rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
        rc_gt_test = np.vstack([(gt_test[i]).detach().cpu().numpy() for i in range(len(gt_test))])
        error_analysis.construct_result_dictionary(opts.path_model, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins)
    print('Loading the error dictionary') 
    eval_dict = np.load(os.path.join(opts.path_model, 'error_dict.npy'), allow_pickle=True).item()
    current_path = os.getcwd()
    plot_path = os.path.join(opts.path_model, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    for ct in range(0, 3):
        error_analysis.plot_pred_vs_gt(eval_dict, plot_path, category=ct)
        error_analysis.plot_mape_vs_rc(eval_dict, plot_path, category=ct)
    error_analysis.err_per_reserve_capacity(eval_dict)

    
def test(opts):    
    #Load data
    torch.manual_seed(37)
    batch_size = opts.batch_size
    test_cases = os.listdir(opts.path_data)
    reserve_capacity_bins = np.array(opts.reserve_capacity_bins)
    lratio_known = opts.lratio_known
    boundary_known = opts.boundary_known
    lprotocol_known = opts.lprotocol_known
    dimension_known = opts.dimension_known
    classify = opts.classify
    for tc in test_cases:
        #Create the folder where the predictions and error_dict.npy will be saved
        test_folder = os.path.join(opts.path_data, tc)
        model_name = opts.path_model.split('/')[-1]
        pred_folder = os.path.join(test_folder, 'predictions_'+model_name)
        if os.path.exists(pred_folder):
            shutil.rmtree(pred_folder)
        os.makedirs(pred_folder, exist_ok=True)
        #Read the test data
        testset = PointCloudDatasetTest(test_folder, opts.segment, opts.norm, opts.lprot, opts.gt)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=opts.num_worker, pin_memory=True)
        device = get_device()
        # Construct the network
        net = util.construct_class_by_name(**opts.model_kwargs).to(device)
        checkpoint = torch.load(os.path.join(opts.path_model, 'best_model.pth'))
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        net.load_state_dict(checkpoint['model'])
        pred_test = predict(net, testloader, device, classify, lratio_known, boundary_known, lprotocol_known, dimension_known)
        if isinstance(pred_test, tuple):
            rc_pred_final_test = np.vstack([(pred_test[0][0][i]).detach().cpu().numpy() for i in range(len(pred_test[0][0]))])
            rc_gt_test = np.vstack([(pred_test[1][0][i]).detach().cpu().numpy() for i in range(len(pred_test[1][0]))])
            lprot_test, ax_load_test, boundary_test = pred_test[2],  pred_test[3], pred_test[4]
            error_analysis.construct_result_dictionary(pred_folder, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins)
            eval_dict = np.load(os.path.join(pred_folder, 'error_dict.npy'), allow_pickle=True).item()
            print('test case: ', tc)
            print('RC+ overall mean error: ',  eval_dict['error']['best_config']['rc+']['mean'], ' std: ', eval_dict['error']['best_config']['rc+']['std'])
            print('RC- overall mean error: ',  eval_dict['error']['best_config']['rc-']['mean'], ' std: ', eval_dict['error']['best_config']['rc-']['std'])
            print('RC overall mean error: ',  eval_dict['error']['best_config']['rc']['mean'], ' std: ', eval_dict['error']['best_config']['rc']['std'])
        else:
            rc_pred_final_test = np.vstack([(pred_test[i]).detach().cpu().numpy() for i in range(len(pred_test))])
        np.save(os.path.join(pred_folder, 'results_' + testset.data_name + '_model_' + opts.model_type +'.npy'), rc_pred_final_test)
       