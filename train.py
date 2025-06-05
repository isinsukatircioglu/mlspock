import os
import click
import util
from util import EasyDict
from types import SimpleNamespace
import training_loop

@click.command()
#Model arguments
@click.option('--path_model',  help='Path to the pretrained model', type=str, default=None)
@click.option('--model_type', help='Model to train/test', type=click.Choice(['pnet', 'pnet2', 'ptr', 'ptr_classify', 'detr3d']), default='pnet2', show_default=True)
@click.option('--mode', help='Train or evaluate a model', type=click.Choice(['train', 'eval', 'test']), default='eval', show_default=True)
@click.option('--batch_size', help='Batch size', type=click.IntRange(min=1),  default=32)
@click.option('--num_worker', help='Number of workers', type=click.IntRange(min=0),  default=8)
@click.option('--model_size', help='Model size', type=click.Choice(['small', 'large']),  default='large')
@click.option('--lratio_known', is_flag=True, show_default=True, default=False, help="Lratio is provided as input in the form of one-hot encoding.")
@click.option('--boundary_known', is_flag=True, show_default=True, default=False, help="Boundary is provided as input in the form of one-hot encoding.")
@click.option('--lprotocol_known', is_flag=True, show_default=True, default=False, help="Loading protocol is provided as input in the form of one-hot encoding.")
@click.option('--dimension_known', is_flag=True, show_default=True, default=False, help="Dimension is provided as input in the form of one-hot encoding.")

#Dataset arguments
@click.option('--path_data',  help='Path to the dataset', type=str, default='./data/mlspock_column_pc3d', show_default=True)
@click.option('--segment', help='Part of the column used for training/evaluation', type=click.Choice(['bottom', 'top', 'top_bottom']), default='bottom', show_default=True)
@click.option('--lprot', help='Loading protocol(s) used for training', type=click.Choice(['Symmetric', 'Collapse_consistent', 'Monotonic', 'Collapse_consistent+Monotonic', 'all']), default='all', show_default=True)
@click.option('--lratio', help='Loading ratio used for training', type=click.Choice(['0.1', '0.2', '0.3', '0.4', '0.5']), default=None, show_default=True)
@click.option('--norm', help='Type of data normalization', type=click.Choice(['snorm', 'cnorm', 'mnorm', 'bnorm']), default=None, show_default=True)
@click.option('--scaling_mode', help='Whether to use the same or a different scaling factor for each column', type=click.Choice(['same', 'diff']), default='diff', show_default=True)
@click.option('--rotate180', is_flag=True, show_default=True, default=False, help="Augment the dataset with columns that are rotated by 180 degrees.")
@click.option('--mirror', is_flag=True, show_default=True, default=False, help="Augment the dataset with columns that are mirrored wrt yz plane.")
@click.option('--delta', is_flag=True, show_default=True, default=False, help="Input is given as [x, y, z, deltax, deltay, deltaz]")
@click.option('--delta3d', is_flag=True, show_default=True, default=False, help="Input is given as [deltax, deltay, deltaz]")

#Optimization arguments
@click.option('--lr', help='Learning rate', type=click.FloatRange(min=0),  default=0.0005) #1e-4
@click.option('--optimizer', help='Optimizer', type=click.Choice(['AdamW', 'Adam']), default='AdamW', show_default=True)
@click.option('--loss', help='Training losses', type=click.Choice(['MSE', 'AsymmetricMSE']), default='MSE', show_default=True)
@click.option('--scheduler', help='Scheduler for learning rate', type=click.Choice(['StepLR', 'MultiStepLR', 'CosineAnnealingLR']), default=None, show_default=True)
@click.option('--epoch', help='Number of training epochs', type=click.IntRange(min=1),  default=300)
@click.option('--seed', help='Set seed for torch operations', type=click.IntRange(min=1),  default=42)
@click.option('--version', help='Experiment ID', type=click.IntRange(min=1),  default=1)
@click.option('--resume', is_flag=True, show_default=True, default=False, help="Resume training")

#Auxiliary task
@click.option('--classify', help='Auxiliary classification task for predicting deformation-related characteristics.', type=click.Choice(['lprot', 'boundary', 'lratio', 'all']), default=None, show_default=True)
@click.option('--split', help='Early/late split of the classification branch from the regression branch', type=click.Choice(['early', 'late']), default='early', show_default=True)

#Test options
@click.option('--gt', is_flag=True, help="Evaluate test data using ground truth.")

def main(**kwargs):
    #Initialize general config
    opts = EasyDict(kwargs)
    opts['reserve_capacity_bins'] = list(range(30, 106, 5))
    opts['radius'] = 100
    if opts.norm is not None:
        if opts.norm == 'bnorm':
            opts.radius = 0.2
        else:
            if opts.delta3d:
                opts.radius = 0.1
            else:
                opts.radius = 0.2
    # Initialize model specific config
    print(opts.radius)
    if opts.model_type == 'ptr':
        if opts.delta:
            opts.model_kwargs = EasyDict(class_name='model.pointtransformer_reg_delta.PointTransformerCls', num_class=2, d_points=6, transformer_dim=64, num_neigh=8, model_size=opts.model_size, radius=opts.radius, lratio_known=opts.lratio_known, boundary_known=opts.boundary_known, lprotocol_known=opts.lprotocol_known, dimension_known=opts.dimension_known,  normal_channel=opts.delta)        
        else:
            opts.model_kwargs = EasyDict(class_name='model.pointtransformer_reg.PointTransformerCls', num_class=2, d_points=3, transformer_dim=64, num_neigh=8, model_size=opts.model_size, radius=opts.radius, lratio_known=opts.lratio_known, boundary_known=opts.boundary_known, lprotocol_known=opts.lprotocol_known,  dimension_known=opts.dimension_known, normal_channel=opts.delta)        
    elif opts.model_type == 'pnet2':
        opts.model_kwargs = EasyDict(class_name='model.pointnet2_reg.PointNet2Cls', num_rc=2, num_class=5, classify=opts.classify, model_size=opts.model_size, radius=opts.radius, lratio_known=opts.lratio_known, boundary_known=opts.boundary_known, lprotocol_known=opts.lprotocol_known, dimension_known=opts.dimension_known, normal_channel=opts.delta)        
    elif opts.model_type == 'pnet':
        opts.model_kwargs = EasyDict(class_name='model.pointnet_reg.PointNetCls')  
    elif opts.model_type == 'ptr_classify':
        num_class_aux = 1
        if opts.classify == 'lprot':
            num_class_aux = 3 
        elif opts.classify == 'boundary':
            num_class_aux = 2
        elif opts.classify == 'lratio':
            num_class_aux = 5 
        elif opts.classify == 'all':
            num_class_aux = 10
        opts.model_kwargs = EasyDict(class_name='model.pointtransformer_reg_classify.PointTransformerCls', num_rc=2, num_class=num_class_aux, d_points=3, transformer_dim=64, num_neigh=8, model_size=opts.model_size, radius=opts.radius, lratio_known=opts.lratio_known, boundary_known=opts.boundary_known, lprotocol_known=opts.lprotocol_known, dimension_known=opts.dimension_known, normal_channel=opts.delta)        
    # Execute training/evaluation loop
    if opts.mode == 'train':
        training_loop.training_loop(opts)
    elif opts.mode == 'eval':
        training_loop.evaluate(opts)
    elif opts.mode == 'test':
        training_loop.test(opts)

if __name__ == "__main__":
    main()
