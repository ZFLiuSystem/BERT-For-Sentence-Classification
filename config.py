import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # file address
        file_path = 'your address'
        parser.add_argument('--train_file', default=file_path + 'data/train.txt', type=str)
        parser.add_argument('--valid_file', default=file_path + 'data/dev.txt', type=str)
        parser.add_argument('--test_file', default=file_path + 'data/test.txt', type=str)
        parser.add_argument('--label_file', default=file_path + 'data/labels.txt', type=str)
        parser.add_argument('--vocab_file', default=file_path + 'data/vocab.txt', type=str)
        parser.add_argument('--checkpoint', default=file_path + 'checkpoint/best_bert_cls.pth')
        # loader
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--eval_batch_size', default=32, type=int)
        parser.add_argument('--shuffle', default=False, type=bool)
        # model parameters
        parser.add_argument('--pretrained_weights',
                            default=file_path + 'bert-base-chinese', type=str)
        parser.add_argument('--num_class', default=10, type=int)
        parser.add_argument('--max_length', default=40, type=int)
        # hyperparameters
        parser.add_argument('--device', default='cuda:0', type=str)
        parser.add_argument('--epoch', default=30, type=int)
        parser.add_argument('--learning_rate', default=0.00005, type=float)
        parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
        parser.add_argument('--max_grad_norm', default=1.0, type=float)
        # trained model
        parser.add_argument('--save_model', default='', type=str)
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
