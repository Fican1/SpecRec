import argparse  
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  
from utils.print_args import print_args  
import random
import numpy as np





if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    
    # Basic Configs
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status: 1-training, 0-testing')
    parser.add_argument('--model_id', type=str, default='ETTm2_96_96', help='model id')
    parser.add_argument('--model', type=str, default='FreRWKV', help='model name, options: [PaiFilter, TexFilter]')

    # Data Loading
    parser.add_argument('--data', type=str, default='ETTm2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./all_six_datasets/ETT-small', help='root path')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m] or detailed like 15min')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints path')
    parser.add_argument('--results', type=str, default='./results/', help='results path')


    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Start label length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4 dataset subset')
    parser.add_argument('--inverse', action='store_true', default=False, help='Whether to invert the output data')


    parser.add_argument('--top_k', type=int, default=5, help='Top-k selection for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='Number of convolutional kernels in Inception module')
    parser.add_argument('--enc_in', type=int, default=1, help='Encoder input dimension')
    parser.add_argument('--dec_in', type=int, default=1, help='Decoder input dimension')
    parser.add_argument('--c_out', type=int, default=1, help='Output dimension')
    parser.add_argument('--d_model', type=int, default=128, help='Model hidden layer dimension')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of multi-head attention heads')
    parser.add_argument('--e_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='Feed-forward network dimension')
    parser.add_argument('--moving_avg', type=int, default=25, help='Moving average window size')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='Whether to use distillation in the encoder')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature encoding method: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', action='store_true', help='Whether to output encoder attention')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='Channel independence setting (1: dependent, 0: independent)')


    parser.add_argument('--embed_size', default=128, type=int, help='Embedding layer dimension')
    parser.add_argument('--hidden_size', default=256, type=int, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading threads')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiment repetitions')
    parser.add_argument('--train_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--des', type=str, default='test', help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Whether to use mixed precision training')


    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='List of multi-GPU device IDs')


    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='List of projector hidden layer dimensions')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='Number of projector hidden layers')
    parser.add_argument('--seg_len', type=int, default=48, help='Time series segmentation length')

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='Masking rate')
    parser.add_argument('--reconstruction_type', type=str, default="imputation", help='Reconstruction type')

    # FreLoss
    parser.add_argument('--rec_lambda', type=float, default=0.1, help='Weight of the reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=0.9, help='Weight of the auxiliary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='损失函数')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')

    parser.add_argument('--add_noise', action='store_true', help='add noise')


    args = parser.parse_args()
    
   
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

   
    print('Args in experiment:')
    print_args(args)

   
    Exp = Exp_Long_Term_Forecast
    
  
    if args.is_training == 1:
        for ii in range(args.itr):
            setting = '{}_{}_{}_sl{}_pl{}_embed{}_hidden{}_bs{}_lr{}_drop{}_rec{}_auxi{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.pred_len,
            args.embed_size, args.hidden_size, args.batch_size, args.learning_rate,
            args.dropout, args.rec_lambda,args.auxi_lambda,ii)
            
            exp = Exp(args)  
            
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_embed{}_hidden{}_bs{}_lr{}_drop{}_rec{}_auxi{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.pred_len,
            args.embed_size, args.hidden_size, args.batch_size, args.learning_rate,
            args.dropout, args.rec_lambda,args.auxi_lambda,ii)
        
        exp = Exp(args)  
        
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)


        torch.cuda.empty_cache()