import torch
import torch.nn as nn
from config import Args
from process_word import get_set, TextClassificationDataset
from transformers import (get_linear_schedule_with_warmup,
                          AutoModelForSequenceClassification,
                          AutoConfig
                          )
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
    def __init__(self, args, train_loader, valid_loader, test_loader, return_losses=False):
        self.args = args
        self.device = torch.device("cuda:0")
        self.model = self.load_model()

        model_parameters = self.load_parameters()
        self.optimizer = AdamW(params=model_parameters, lr=args.learning_rate, eps=1e-9)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=500,
                                                         num_training_steps=len(train_loader) * args.epoch)
        self.loss = nn.CrossEntropyLoss()
        self.return_losses = return_losses

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        pass

    def load_model(self):
        config = AutoConfig.from_pretrained(self.args.pretrained_weights, num_labels=self.args.num_class)
        bert_classification = AutoModelForSequenceClassification.from_pretrained(self.args.pretrained_weights,
                                                                                 config=config)
        return bert_classification

    def load_parameters(self):
        model_parameters = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'Layer.weight']
        optimized_grouped_parameters = [
            {'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)], 'weight_decay':.0}
        ]
        return optimized_grouped_parameters

    @staticmethod
    def save_checkpoint(state, checkpoint_path):
        torch.save(state, checkpoint_path)
        pass

    @staticmethod
    def load_checkpoint(model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
        pass

    def train(self):
        self.model.train()
        self.model.to(self.device)
        best_valid_accuracy = 0.0
        train_losses = []
        valid_losses = []
        for epoch in range(1, self.args.epoch + 1):
            batch_loss = .0
            train_predicts = []
            train_labels = []
            print(80 * '-')
            print('Epoch[{}/{}]'.format(self.args.epoch, epoch))
            start_time = time.time()
            # Before the training cycle begins, use optimizer.zero_grad() to clear the gradients of the model.
            self.optimizer.zero_grad()
            for i, (samples, labels) in enumerate(tqdm(self.train_loader)):
                input_ids = samples['input_ids'].to(self.device)
                attention_masks = samples['attention_mask'].to(self.device)
                labels = labels.to(self.device)

                # Forward propagation computes model output.
                outputs = self.model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                assert labels.max() < logits.size(1)  # 确保labels中的最大值小于类别数
                assert labels.min() >= 0
                # Loss function value.
                loss = self.loss(logits, labels)
                batch_loss += loss.item()
                if self.args.gradient_accumulation_steps > 0:
                    loss = loss / self.args.gradient_accumulation_steps
                # Calculate the gradient, but do not update the model weights at this time.
                loss.backward()

                ''' The number of small batches currently processed is an integer multiple 
                    of the cumulative number of steps. '''
                if (i + 1) % self.args.gradient_accumulation_steps == 0 or len(self.train_loader) == (i + 1):
                    # Crop the accumulated gradient.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # Update model weights using cumulative gradients.
                    self.optimizer.step()
                    # Clear the gradients of the model in preparation for the next round of gradient accumulation.
                    self.optimizer.zero_grad()

                train_pred = torch.argmax(logits, dim=-1)
                train_predicts.extend(train_pred.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Update learning rate.
            self.scheduler.step()

            end_time = time.time()
            train_span = end_time - start_time
            train_accuracy = self.get_metrics(train_labels, train_predicts)
            train_losses.append(batch_loss)
            valid_span, valid_accuracy, best_valid_accuracy = self.eval_(valid_losses, best_valid_accuracy, epoch)
            train_prompt = f"Train Time: {train_span}secs, Train Loss: {train_losses[epoch-1]},\n" \
                           + f"Train Accuracy: {train_accuracy[0]}, Train F1_micro: {train_accuracy[1]}, "\
                           + f"Train F1_macro: {train_accuracy[2]}."
            print(train_prompt)
            valid_prompt = f"Valid Time: {valid_span}secs, Valid Loss: {valid_losses[epoch - 1]},\n" \
                           + f"Valid Accuracy: {valid_accuracy[0]}, Valid F1_micro: {valid_accuracy[1]}, " \
                           + f"Valid F1_macro: {valid_accuracy[2]}."
            print(valid_prompt)
            print(80 * '-')

        if self.return_losses:
            return {'Train Losses': train_losses, 'Valid Losses': valid_losses}
        pass

    def eval_(self, valid_losses: list, best_valid_accuracy, epoch):
        batch_loss = .0
        valid_labels = []
        valid_predicts = []
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (samples, labels) in enumerate(tqdm(self.valid_loader)):
                input_ids = samples['input_ids'].to(self.device)
                attention_masks = samples['attention_mask'].to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                loss = self.loss(logits, labels)
                batch_loss += loss.item()

                valid_pred = torch.argmax(logits, dim=1)
                valid_predicts.extend(valid_pred.cpu().numpy())
                valid_labels.extend((labels.cpu().numpy()))
        end_time = time.time()
        valid_span = end_time - start_time
        valid_accuracy = self.get_metrics(valid_labels, valid_predicts)
        valid_losses.append(batch_loss)
        if valid_accuracy[0] > best_valid_accuracy:
            checkpoint = {
                'epoch': epoch,
                'loss': batch_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            best_valid_accuracy = valid_accuracy[0]
            self.save_checkpoint(checkpoint, self.args.checkpoint)
        return valid_span, valid_accuracy, best_valid_accuracy
        pass

    def test_(self, id2label=None, convert_ids2labels: bool = False):
        # model_structure & model_weights, optimizer, epoch, minimal_loss
        model, optimizer, epoch, loss = self.load_checkpoint(self.model, self.optimizer, self.args.checkpoint)
        print(model)
        model.eval()
        model.to(self.device)
        test_labels = []
        test_predicts = []
        batch_loss = .0
        start_time = time.time()
        with torch.no_grad():
            for i, (samples, labels) in enumerate(tqdm(self.test_loader)):
                input_ids = samples['input_ids'].to(self.device)
                attention_mask = samples['attention_mask'].to(self.device)
                labels = labels.to(self.device)

                predicts = model(input_ids, attention_mask=attention_mask)
                logits = predicts.logits
                loss = self.loss(logits, labels)
                batch_loss += loss.item()

                logits = torch.argmax(logits, dim=1)
                test_predicts.extend(logits.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        end_time = time.time()
        test_span = end_time - start_time
        test_accuracy = self.get_metrics(test_labels, test_predicts)
        if convert_ids2labels and id2label is not None:
            length = len(test_labels)
            pred_labels = [id2label[i] for i in test_labels]
            pred_results = [id2label[i] for i in test_predicts]
            print(f"There are {length} tested samples totally.")
            for i, (label, pred) in enumerate(zip(pred_labels, pred_results)):
                print(90 * '-')
                if i + 1 == 1:
                    prompts = 'The tested {}st sample: Actual Label: {}, Predicted Label: {}.'
                elif i + 1 == 2:
                    prompts = 'The tested {}nd sample: Actual Label: {}, Predicted Label: {}.'
                elif i + 1 == 3:
                    prompts = 'The tested {}rd sample: Actual Label: {}, Predicted Label: {}.'
                else:
                    prompts = 'The tested {}th sample: Actual Label: {}, Predicted Label: {}.'
                if label == pred:
                    print('This judgement is correct.')
                else:
                    print('This prediction exists some errors.')
                print(prompts.format(i+1, label, pred))
                print(90 * '-')
        return test_span, test_accuracy
        pass

    @staticmethod
    def get_metrics(labels: list, predicts: list):
        accuracy = accuracy_score(labels, predicts)
        f1_micro = f1_score(labels, predicts, average='micro')
        f1_macro = f1_score(labels, predicts, average='macro')
        return accuracy, f1_micro, f1_macro
        pass


def main():
    parser_instance = Args()
    args = parser_instance.get_parser()

    label2id = {}
    id2label = {}
    with open(args.label_file, 'r', encoding='utf-8') as file:
        labels = file.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    train_samples, train_labels = get_set(args.train_file, args.pretrained_weights)
    train_set = TextClassificationDataset(samples=train_samples, labels=train_labels)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0)

    valid_samples, valid_labels = get_set(args.valid_file, args.pretrained_weights)
    valid_set = TextClassificationDataset(samples=valid_samples, labels=valid_labels)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.eval_batch_size,
                              num_workers=0, shuffle=args.shuffle)
    test_samples, test_labels = get_set(args.test_file, args.pretrained_weights)
    test_set = TextClassificationDataset(samples=test_samples, labels=test_labels)
    test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size,
                             num_workers=0, shuffle=args.shuffle)
    trainer = Trainer(args=args, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)

    trainer.train()

    test_span, test_accuracy = trainer.test_(id2label=id2label, convert_ids2labels=True)
    prompts = f"Test Time: {test_span}secs,\n" \
              + f"Test Accuracy: {test_accuracy[0]}, Test F1_micro: {test_accuracy[1]}, " \
              + f"Test F1_macro: {test_accuracy[2]}."

    print(prompts)
    pass


if __name__ == '__main__':
    main()
