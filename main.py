import json
import os
from argparse import ArgumentParser
import copy

import torch
from torch.cuda import is_available as cuda_available
import yaml

from data import Data
from pretrain import trainer as PreTrainer, generative_losses, contrastive_losses
import modules
from downstream import predictor as DownPredictor, task
import utils

from modules.model.LLMEnhancer import LLMEnhancer


def main():
    def parse_args():
        """
        Parse command line arguments and set up CUDA device.
        
        Returns:
            tuple: (parsed_args, device_string)
            - parsed_args contains config file name and cuda device index
            - device_string is 'cuda:0' or 'cpu' depending on availability
        """
        parser = ArgumentParser()
        parser.add_argument('-c', '--config', help='path of the config file to use', type=str, required=True)
        parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=0)
        args = parser.parse_args()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        device = f'cuda:0' if cuda_available() else 'cpu'
        return args, device

    def load_data(data_entry):
        """
        Load and initialize dataset based on config entry.
        
        Args:
            data_entry (dict): Dataset configuration containing name and road_type
            
        Returns:
            Data: Initialized data object with loaded statistics
        """
        data = Data(data_entry['name'], data_entry.get('road_type', 'road_network'))
        data.load_stat()
        return data

    def create_model(model_entry, data, pretrain):
        """
        Create model instance based on configuration.
        
        Args:
            model_entry (dict): Model configuration containing name and parameters
            data (Data): Dataset object for loading metadata
            pretrain (bool): Whether the model is pretrained
            
        Returns:
            nn.Module: Initialized model instance
            
        Raises:
            NotImplementedError: If model name is not recognized
        """
        # Add global declarations at the start of the function
        global vocab_size, dist_path, hidden_size, num_roads, num_class
        
        # Prepare sampler
        sampler = create_preprocessor(model_entry.get('preprocessor', {'name': 'pass'}))
        
        # Prepare model config
        model_config = model_entry.get('config', {})
        if "pre_embed" in model_config:
            model_config["pre_embed"] = data.load_meta(model_config.get("pre_embed"), 0)[0]
            model_config["pre_embed_update"] = model_config.get("pre_embed_update", True)

        # Create model based on name
        model_name = model_entry['name']
        if model_name == 'ia':
            from modules.model.induced_att import InducedAttEncoder
            return InducedAttEncoder(sampler=sampler, **model_config)
        elif model_name == 'llama_encoder':
            from modules.model.llama import LlamaEncoder
            return LlamaEncoder(sampler=sampler, **model_config)
        elif model_name == 'llama_encoder2':
            from modules.model.llama import LlamaEncoder2
            return LlamaEncoder2(sampler=sampler, **model_config)
        elif model_name == 'llama_decoder':
            from modules.model.llama import LlamaDecoder
            return LlamaDecoder(**model_config)
        elif model_name == 'transformer_encoder':
            from modules.model.transformer import TransformerEncoder
            return TransformerEncoder(sampler=sampler, **model_config)
        elif model_name == 'transformer_decoder':
            from modules.model.transformer import TransformerDecoder
            return TransformerDecoder(**model_config)
        elif model_name == 'transformer_denoiser':
            from modules.model.transformer import TransformerDenoiser
            return TransformerDenoiser(**model_config)
        elif model_name == 'dualpos_transformer':
            from modules.model.transformer import DualPosTransformer
            return DualPosTransformer(sampler=sampler, **model_config)
        elif model_name == 'mlm_transformer':
            from modules.model.transformer import MLMTransformer
            return MLMTransformer(sampler=sampler, **model_config)
        elif model_name == 'cde':
            from modules.model.ode import CDEEncoder
            return CDEEncoder(sampler=sampler, **model_config)
        elif model_name == 'coa':
            from modules.model.ode import CoeffAttEncoder
            return CoeffAttEncoder(sampler=sampler, **model_config)
        elif model_name == 'stode':
            from modules.model.ode import STODEEncoder
            return STODEEncoder(sampler=sampler, **model_config)
        elif model_name == 'trajode_decoder':
            from modules.model.ode import TrajODEDecoder
            return TrajODEDecoder(**model_config)
        elif model_name == 'rnn_encoder':
            from modules.model.rnn import RnnEncoder
            return RnnEncoder(sampler=sampler, num_embed=num_roads, **model_config)
        elif model_name == 'rnn_decoder':
            from modules.model.rnn import RnnDecoder
            return RnnDecoder(num_roads=num_roads, **model_config)
        elif model_name == 'gmvsae_encoder':
            from modules.model.gmvsae import GMVSAEEncoder
            return GMVSAEEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'gmvsae_decoder':
            from modules.model.gmvsae import GMVSAEDecoder
            return GMVSAEDecoder(num_embed=num_roads, **model_config)
        elif model_name == 'bert':
            from modules.model.start import BERTEncoder
            return BERTEncoder(sampler=sampler, vocab_size=num_roads, **model_config)
        elif model_name == 'trajectory2vec_encoder':
            from modules.model.trajectory2vec import Trajectory2VecEncoder
            return Trajectory2VecEncoder(sampler=sampler, **model_config)
        elif model_name == 'trajectory2vec_decoder':
            from modules.model.trajectory2vec import Trajectory2VecDecoder
            return Trajectory2VecDecoder(sampler=sampler, **model_config)
        elif model_name == 'trajsim_embedding':
            from modules.model.trajectorysim import TrajSimEmbed
            model = TrajSimEmbed(meta_dir=data.meta_dir, **model_config, pretrain=pretrain)
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 'trajsim_encoder':
            from modules.model.trajectorysim import TrajSimEncoder
            return TrajSimEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'trajsim_decoder':
            from modules.model.trajectorysim import TrajSimDecoder
            model = TrajSimDecoder(**model_config)
            hidden_size = model.hidden_size
            return model
        elif model_name == 't2vecEmbedding':
            from modules.model.t2vec import t2vecEmbedding
            model = t2vecEmbedding(meta_dir=data.meta_dir, **model_config, pretrain=pretrain)
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 't2vecEncoder':
            from modules.model.t2vec import t2vecEncoder
            return t2vecEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 't2vecDecoder':
            from modules.model.t2vec import t2vecDecoder
            model = t2vecDecoder(**model_config)
            hidden_size = model.hidden_size
            return model
        elif model_name == 'traj2vec_encoder':
            from modules.model.trembr import Traj2VecEncoder
            return Traj2VecEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'traj2vec_decoder':
            from modules.model.trembr import Traj2VecDecoder
            return Traj2VecDecoder(**model_config)
        elif model_name == 'cae_encoder':
            from modules.model.cnn import CNNEncoder
            return CNNEncoder(sampler=sampler, **model_config)
        elif model_name == 'cae_decoder':
            from modules.model.cnn import CNNDecoder
            return CNNDecoder(**model_config)
        elif model_name == 'geoconstrains_skipgram':
            from modules.model.word2vec import GeoConstrainSkipGramEncoder
            return GeoConstrainSkipGramEncoder(sampler=sampler, **model_config)
        elif model_name == 'dual_view_encoder':
            from modules.model.dual_view import DualViewEncoder
            return DualViewEncoder(sampler=sampler, num_users=num_class, **model_config)
        elif model_name == 'robustDAAEncoder':
            from modules.model.robustDAA import RobustDAA_Encoder
            return RobustDAA_Encoder(sampler=sampler, **model_config)
        elif model_name == 'robustDAADecoder':
            from modules.model.robustDAA import RobustDAA_Decoder
            return RobustDAA_Decoder(**model_config)
        elif model_name == 'robustDAA_attention':
            from modules.model.robustDAA import RobustDAA_Attention
            return RobustDAA_Attention(**model_config)
        elif model_name == 'maerrcdvit':
            from modules.model.light_path import MAERRCD
            return MAERRCD(sampler=sampler, num_roads=num_roads, **model_config)
        else:
            raise NotImplementedError(f'No model called "{model_name}".')

    def create_preprocessor(preprocessor_entry):
        """
        Create data augmentation sampler based on configuration.
        
        Args:
            preprocessor_entry (dict): Preprocessor configuration containing name and parameters
            
        Returns:
            Sampler: Initialized augmentation sampler
            
        Raises:
            NotImplementedError: If augmentation name is not recognized
        """
        preprocessor_name = preprocessor_entry['name']
        preprocessor_config = preprocessor_entry.get('config', {})
        
        if preprocessor_name == 'pass':
            return modules.preprocessor.PassSampler()
        elif preprocessor_name == 'khop':
            return modules.preprocessor.KHopSampler(**preprocessor_config)
        elif preprocessor_name == 'index':
            return modules.preprocessor.IndexSampler(**preprocessor_config)
        elif preprocessor_name == 'pool':
            return modules.preprocessor.PoolSampler(**preprocessor_config)
        elif preprocessor_name == 'Trajectory2VecSampler':
            return modules.preprocessor.Trajectory2VecSampler(**preprocessor_config)
        elif preprocessor_name == 'random':
            return modules.preprocessor.RandomViewSampler(**preprocessor_config)
        else:
            raise NotImplementedError(f'No preprocessor called "{preprocessor_name}".')

    def create_loss_functions(loss_entries, models, device):
        """
        Create loss functions based on configuration entries.
        
        Args:
            loss_entries (dict or list): Loss function configurations
            models (list): List of model instances
            
        Returns:
            list or object: Single loss function or list of loss functions
            
        Raises:
            NotImplementedError: If loss function name is not recognized
        """
        global num_roads, hidden_size, vocab_size, dist_path

        # Handle single loss entry case
        if isinstance(loss_entries, dict):
            loss_entries = [loss_entries]
            single_loss = True
        else:
            single_loss = False

        loss_funcs = []
        for loss_entry in loss_entries:
            loss_name = loss_entry['name']
            loss_param = loss_entry.get('config', {})
            
            if loss_name == 'infonce':
                loss_funcs.append(contrastive_losses.InfoNCE(**loss_param))
            elif loss_name == 'mec':
                loss_funcs.append(contrastive_losses.MEC(**loss_param,
                                                       teachers=(copy.deepcopy(model) for model in models)))
            elif loss_name == 'ddpm':
                loss_funcs.append(generative_losses.DDPM(**loss_param))
            elif loss_name == 'autoreg':
                loss_funcs.append(generative_losses.AutoRegressive(**loss_param))
            elif loss_name == 'mlm':
                loss_funcs.append(generative_losses.MLM(**loss_param))
            elif loss_name == 'gmvsae':
                loss_funcs.append(generative_losses.GMVSAE(**loss_param))
            elif loss_name == 'simclr':
                loss_funcs.append(contrastive_losses.SimCLR(**loss_param))
            elif loss_name == 'trajectory2vec':
                loss_funcs.append(generative_losses.Trajectory2Vec(**loss_param))
            elif loss_name == 'trajsim':
                loss_funcs.append(generative_losses.TrajectorySim(device=device, 
                                                                hidden_size=hidden_size,
                                                                vocab_size=vocab_size,
                                                                knn_vocabs_path=dist_path, 
                                                                **loss_param))
            elif loss_name == 't2vec':
                loss_funcs.append(generative_losses.t2vec(device=device,
                                                        hidden_size=hidden_size,
                                                        vocab_size=vocab_size,
                                                        knn_vocabs_path=dist_path,
                                                        **loss_param))
            elif loss_name == 'trembr':
                loss_funcs.append(generative_losses.Trembr(num_roads=num_roads, **loss_param))
            elif loss_name == 'cae':
                loss_funcs.append(generative_losses.ConvolutionalAutoRegressive(**loss_param))
            elif loss_name == 'geoconstrains_word2vec':
                loss_funcs.append(contrastive_losses.GeoConstrainWord2Vec(**loss_param))
            elif loss_name == 'robustDAA':
                loss_funcs.append(generative_losses.RobustDAA(**loss_param))
            elif loss_name == 'trajode':
                loss_funcs.append(generative_losses.TrajODE(**loss_param))
            elif loss_name == 'maerr':
                loss_funcs.append(generative_losses.MAERR(**loss_param))
            elif loss_name == 'llama':
                loss_funcs.append(generative_losses.Llama(**loss_param))
            else:
                raise NotImplementedError(f'No loss function called "{loss_name}".')

        return loss_funcs[0] if single_loss else loss_funcs

    def create_pretrainer(pretrainer_entry, data, models, loss_func, device, datetime_key, num_entry, repeat_i):
        """
        Create pretraining trainer based on configuration.
        
        Args:
            pretrainer_entry (dict): Trainer configuration
            data (Data): Dataset object
            models (list): List of model instances
            loss_func: Loss function(s)
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            Trainer: Initialized trainer instance
            
        Raises:
            NotImplementedError: If trainer name is not recognized
        """
        pretrainer_name = pretrainer_entry['name']
        pretrainer_config = pretrainer_entry.get('config', {})
        
        # Common parameters for all trainers
        common_params = {
            "data": data,
            "models": models,
            "loss_func": loss_func,
            "device": device,
            "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'
        }
        
        if pretrainer_name == 'contrastive':
            return PreTrainer.ContrastiveTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'generative':
            return PreTrainer.GenerativeTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'generativeiteration':
            return PreTrainer.GenerativeIterationTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'momentum':
            return PreTrainer.MomentumTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'multiple':
            return PreTrainer.MultiTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'ADMM':
            return PreTrainer.ADMMTrainer(**common_params, **pretrainer_config)
        else:
            raise NotImplementedError(f'No trainer called "{pretrainer_name}".')

    def setup_pretraining(entry, models, data, device, datetime_key, num_entry, repeat_i):
        """
        Set up and execute model pretraining based on configuration.
        
        Args:
            entry (dict): Full experiment configuration
            models (list): List of model instances
            data (Data): Dataset object
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            tuple: (trainer, models)
            - trainer is the pretraining trainer instance
            - models are the pretrained model instances
        """
        if 'pretrain' not in entry:
            pre_trainer = PreTrainer.NoneTrainer(models=models, data=data, device=device)
            pre_trainer.save_models()
            print('Skip pretraining.')
            return pre_trainer, models

        pretrain_entry = entry['pretrain']
        loss_func = create_loss_functions(pretrain_entry['loss'], models, device)
        pre_trainer = create_pretrainer(pretrain_entry['trainer'], data, models, loss_func, device, 
                                        datetime_key, num_entry, repeat_i)

        # Handle training or loading
        if pretrain_entry.get('load', False):
            if pretrain_entry.get('load_epoch', None):
                pre_trainer.load_models(epoch=int(pretrain_entry['load_epoch']))
            else:
                pre_trainer.load_models()
        else:
            pre_trainer.train(pretrain_entry.get('resume', -1))

        return pre_trainer, pre_trainer.get_models()

    def run_downstream_tasks(entry, pre_trainer, models, data, device, datetime_key, num_entry, repeat_i):
        """
        Execute downstream tasks after pretraining.
        
        Args:
            entry (dict): Full experiment configuration
            pre_trainer: Pretraining trainer instance
            models (list): List of pretrained models
            data (Data): Dataset object
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
        """
        if 'downstream' not in entry:
            print('Finishing program without performing downstream tasks.')
            return

        for down_i, down_entry in enumerate(entry['downstream']):
            print(f'\n....{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat, '
                  f'{down_i+1}/{len(entry["downstream"])} downstream task ....\n')

            if down_i > 0:
                pre_trainer.load_models()
                models = pre_trainer.get_models()

            down_trainer = setup_downstream_task(down_entry, models, data, device, pre_trainer.BASE_KEY,
                                                 datetime_key, num_entry, repeat_i, data.data_info['num_road'])

            if down_entry.get('load', False):
                down_trainer.load_models()
            else:
                down_trainer.train()
            down_trainer.eval(down_entry['eval_set'])

    def setup_downstream_task(down_entry, models, data, device, base_key, datetime_key, num_entry, repeat_i, num_roads):
        """
        Set up downstream task trainer based on configuration.
        
        Args:
            down_entry (dict): Downstream task configuration
            models (list): List of pretrained models
            data (Data): Dataset object
            device (str): Device to run training on
            base_key (str): Base key for model loading
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            Trainer: Initialized downstream task trainer
            
        Raises:
            NotImplementedError: If task name is not recognized
        """
        # Select models and calculate embedding size
        down_models = [models[i] for i in down_entry['select_models']]
        # 加入llama增强器
        sampler = modules.preprocessor.PassSampler()
        enhancer = LLMEnhancer(10, "meta-llama/Llama-3.2-1B-Instruct", sampler=sampler, useLoRA=False)
        down_models.append(enhancer)

        down_embed_size = sum([model.output_size for model in down_models])
        
        # Get task configuration
        down_task = down_entry['task']
        down_config = down_entry.get('config', {})
        predictor_entry = down_entry.get('predictor', {})
        predictor_config = predictor_entry.get('config', {})
        
        # Common parameters for all tasks
        common_params = {
            "data": data,
            "models": down_models,
            "device": device,
            "base_name": base_key,
            "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'
        }
        
        # Create appropriate predictor and trainer based on task
        if down_task == 'classification':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=data.data_info['num_class'],
                **predictor_config
            )
            return task.Classification(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'destination':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=num_roads,
                **predictor_config
            )
            return task.Destination(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'search':
            predictor = DownPredictor.NonePredictor()
            return task.Search(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'tte':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=1,
                **predictor_config
            )
            return task.TTE(predictor=predictor, **common_params, **down_config)
        
        else:
            raise NotImplementedError(f'No downstream task called "{down_task}".')

    # Main execution flow
    args, device = parse_args()
    datetime_key = utils.get_datetime_key()
    print('Datetime key', datetime_key)
    torch.autograd.set_detect_anomaly(True)

    # Load config file
    if args.config.endswith('.json'):
        with open(args.config, 'r') as fp:
            config = json.load(fp)
    elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
        import yaml
        with open(args.config, 'r') as fp:
            config = yaml.safe_load(fp)
    else:
        raise ValueError(f"Config file must be .json, .yaml or .yml, got {args.config}")

    import yaml
    for num_entry, entry in enumerate(config):
        print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')
        
        # Load dataset
        data = load_data(entry['data'])
        
        # Save config
        conf_save_dir = os.path.join(data.base_path, 'config')
        utils.create_if_noexists(conf_save_dir)
        with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.yaml'), 'w') as fp:
            yaml.dump(entry, fp)

        # Run experiments
        num_repeat = entry.get('repeat', 1)
        for repeat_i in range(num_repeat):
            print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')
            global num_roads, num_class
            num_roads = data.data_info['num_road']
            num_class = data.data_info['num_class']
            
            # Create models
            models = [create_model(model_entry, data, 'pretrain' in entry) 
                      for model_entry in entry['models']]
            
            # Handle pretraining
            pre_trainer, models = setup_pretraining(entry, models, data, device, datetime_key, 
                                                    num_entry, repeat_i)
            
            # Run downstream tasks
            run_downstream_tasks(entry, pre_trainer, models, data, device, datetime_key, 
                                 num_entry, repeat_i)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    main()