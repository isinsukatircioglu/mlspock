import os
import click
import util
from util import EasyDict
import training_loop

@click.command()
#Model related arguments
@click.option('--path_model',  help='Path to the pretrained model', type=str, default=None)
@click.option('--model_type', help='Model to train/test', type=click.Choice(['pnet', 'pnet2', 'ptr', 'resnet']), default='pnet2', show_default=True)
@click.option('--mode', help='Train or evaluate a model', type=click.Choice(['train', 'eval', 'test']), default='eval', show_default=True)
@click.option('--batch_size', help='Batch size', type=click.IntRange(min=1),  default=128)
@click.option('--epoch', help='Number of training epochs', type=click.IntRange(min=1),  default=200)
@click.option('--lr', help='Learning rate', type=click.FloatRange(min=0),  default=1e-4)
@click.option('--loss', help='Training losses', type=click.Choice(['MSE']), default='MSE', show_default=True)

#Dataset related arguments
@click.option('--path_data',  help='Path to the dataset', type=str, default='/mydata/mlspock/shared/column_dataset/column_ssc.hdf5', show_default=True)
@click.option('--segment', help='Part of the column used for training/evaluation', type=click.Choice(['bottom', 'top', 'top_bottom']), default='bottom', show_default=True)
@click.option('--lprot', help='Loading protocol(s) used for training', type=click.Choice(['Symmetric', 'Collapse_consistent', 'Monotonic', 'all']), default='all', show_default=True)
@click.option('--norm', help='Type of data normalization', type=click.Choice(['snorm', 'cnorm', 'mnorm', 'bnorm']), default=None, show_default=True)

def main(**kwargs):
    #Initialize general config
    opts = EasyDict(kwargs)
    opts['reserve_capacity_bins'] = list(range(30, 106, 5))
    # Initialize model specific config
    if opts.model_type == 'ptr':
        opts.model_kwargs = EasyDict(class_name='model.pointtransformer_utils.PointTransformerCls', npoints=441, nblocks=4, nneighbor=16, n_c=2, d_points=3, transformer_dim=512)        
    elif opts.model_type == 'pnet2':
        opts.model_kwargs = EasyDict(class_name='model.pointnet2_reg.PointNet2Cls', num_class=2, normal_channel=False)        
    elif opts.model_type == 'pnet':
        opts.model_kwargs = EasyDict(class_name='model.pointnet_reg.PointNetCls')        
    # Execute training/evaluation loop
    if opts.mode == 'train':
        print('Training model: ', opts.model_type)
        training_loop.training_loop(opts)
    elif opts.mode == 'eval':
        print('Evaluating model: ', opts.model_type)
        training_loop.evaluate(opts)
    elif opts.mode == 'test':
        print('Testing model: ', opts.model_type)
        training_loop.test(opts)

if __name__ == "__main__":
    main()
