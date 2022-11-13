import torch
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sys

import argparse

from preprocess.tools.paths import train_data_file, validation_data_file, ckpt_dir


def parse():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser()

    # (Required arguments) Model name and version name
    # Valid model names: reranker, dpr, colbert
    # Version name is used as a directory name to save log and checkpoints
    parser.add_argument('--model_name', type=str, required=True) # 'reranker', 'dpr', 'colbert'
    parser.add_argument('--version_name', type=str, required=True)  # ex. 0419_1_lr1e-6...

    # Path to train data file and validation data file
    parser.add_argument('--train_data_file', type=str, default='default') # Write path to the file
    parser.add_argument('--validation_data_file', type=str, default='default')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=24)

    # Number of training and validation steps
    parser.add_argument('--n_train_steps', type=int, default=200000)
    parser.add_argument('--n_validation_steps', type=int, default=1000)

    # Learning rate and learning rate scheduling
    # lr: learning rate, not_lr_schedule: use this to not use linear learning rate scheduler
    # warm_up_steps: linear warm up steps for learning rate, linear_decay_steps: learning rate will be 0 at this point
    parser.add_argument('--lr', type=float, default=3e-7)
    parser.add_argument('--not_lr_schedule', default=False, action='store_true')
    parser.add_argument('--warm_up_steps', type=int, default=20000)
    parser.add_argument('--linear_decay_steps', type=int, default=200000)

    # Number of steps before each validation
    parser.add_argument('--measure_steps', type=int, default=1000)

    # Number of checkpoints to save
    parser.add_argument('--n_checkpoints', type=int, default=0)

    # Only for reranker: collate_fn2 truncates queries at size 64 and npr is the negative-to-positive data ratio(npr >= 1)
    parser.add_argument('--collate_fn2', default=False, action='store_true')
    parser.add_argument('--npr', type=int, default=1)

    # Only for ColBERT (N_q: query token length, vector_size: vector size for each token)
    parser.add_argument('--N_q', type=int, default=32) # Only for ColBERT (query token length)
    parser.add_argument('--vector_size', type=int, default=128) # Only for ColBERT (vector size)

    args = parser.parse_args()

    assert args.model_name in ['reranker', 'dpr', 'colbert']
    return args



#################### Tools

def linear_learning_rate_scheduler(optimizer, steps, target, warm_up, decay):
    """
    Change the learning rate of the optimizer using a linear learning rate schedule
    :param optimizer: optimizer that are being used
    :param steps: current number of steps
    :param target: maximum learning rate
    :param warm_up: number of warm up steps
    :param decay: number of steps at which the learning rate will be 0
    :return: modified optimizer
    """
    if steps < warm_up:
        running_lr = target * steps / warm_up
    else:
        running_lr = target * (decay - steps) / (decay - warm_up)

    for g in optimizer.param_groups:
        g['lr'] = running_lr
    return optimizer, running_lr


def main():
    args = parse()

    # If not specified, use default train and validation data files
    if args.train_data_file != 'default':
        train_data_file = args.train_data_file
    if args.validation_data_file != 'default':
        validation_data_file = args.validation_data_file
    
    # Get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Depending on the model_name given as argument, make a model instance
    if args.model_name == 'reranker':
        from models.reranker import Reranker, get_dataloader
        model = Reranker(device=device)
        dataloader_args = {'npr': args.npr, 'use_collate_fn2': args.collate_fn2}
    elif args.model_name == 'dpr':
        from models.dpr import DPR
        from models.dpr import get_dataloader
        model = DPR(B=args.batch_size, device=device)
        dataloader_args = {'mode': 'in-batch train'}
    elif args.model_name == 'colbert':
        from models.colbert import ColBERT, get_dataloader, tokenizer
        model = ColBERT(tokenizer=tokenizer, B=args.batch_size, vector_size=args.vector_size, device=device)
        dataloader_args = {'N_q': args.N_q, 'mode': 'in-batch train'}
    else:
        print("Invalid argument: --model (not in ['reranker', 'dpr', 'colbert'])")
        sys.exit(1)

    # Send model to the device
    model = model.to(device)

    # Make a train dataloader
    train_dataloader = get_dataloader(
        data_file=train_data_file,
        n_instances=args.batch_size * args.n_train_steps,
        batch_size=args.batch_size,
        shuffle=True,
        args=dataloader_args
    )

    # Make a validation dataloader
    validation_dataloader = get_dataloader(
        data_file=validation_data_file,
        n_instances=args.batch_size * args.n_validation_steps,
        batch_size=args.batch_size,
        shuffle=False,
        args=dataloader_args
    )

    # Using a Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Checkpoint directory
    ckpt_folder = os.path.join(ckpt_dir, args.model_name, args.version_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_name = os.path.join(ckpt_folder, 'steps=%d_loss-interval=%.4f.pth')
    ckpt_info = [[None, float('inf')] for i in range(10)]

    # Log directory
    log_dir = os.path.join('log', args.model_name, args.version_name)
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Variables for early_stopping
    min_val_losses = [] # save max 10
    early_stop_count = 0

    running_loss = 0.0
    running_accuracy = 0.0
    running_lr = args.lr

    pbar = tqdm(enumerate(train_dataloader), desc='train', total=len(train_dataloader))
    for i, data in pbar:

        # Adjust learning rate if linear scheduling is used
        if not args.not_lr_schedule:
            optimizer, running_lr = linear_learning_rate_scheduler(optimizer, i + 1, args.lr, args.warm_up_steps,
                                                                   args.linear_decay_steps)

        # Calculate the output of the model
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss(outputs, data['labels'].to(device) if 'labels' in data.keys() else None)

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Calculate and save the loss and accuracy
        running_loss += loss.item()
        running_accuracy += model.accuracy(
            outputs=outputs,
            labels=data['labels'] if 'labels' in data else None
        )

        # Run validation at each 'measure_steps'
        # Record train and validation measurements and save checkpoints
        if (i + 1) % args.measure_steps == 0:

            # Average running loss and accuracy in the current interval on training data
            avg_loss = running_loss / args.measure_steps
            avg_acc = running_accuracy / args.measure_steps

            running_loss = 0.0
            running_accuracy = 0.0

            # Perform validation
            model.train(False)
            val_loss = 0
            val_acc = 0
            for val_data in tqdm(validation_dataloader,
                                 desc='validation',
                                 leave=False,
                                 total=len(validation_dataloader)):

                optimizer.zero_grad()
                outputs = model(val_data)
                loss = model.loss(outputs, data['labels'].to(device) if 'labels' in val_data.keys() else None)

                val_loss += loss.item()
                val_acc += model.accuracy(
                    outputs=outputs,
                    labels=val_data['labels'] if 'labels' in val_data else None
                )

            # Calculate the average validation loss and accuracy
            val_loss /= len(validation_dataloader)
            val_acc /= len(validation_dataloader)

            # Save 'args.n_checkpoints' checkpoints with minimum validation losses
            # Remove previous ones that don't have a minimal loss
            ckpt_info.sort(key=lambda x: x[1], reverse=True)
            if val_loss < ckpt_info[0][1]:
                if ckpt_info[0][0] != None:
                    os.remove(ckpt_info[0][0])
                save_name = ckpt_name % (i + 1, val_loss)
                torch.save(model, save_name)
                ckpt_info[0] = [save_name, val_loss]

            # Early stop

            min_val_losses.append(val_loss)
            min_val_losses.sort()
            min_val_losses = min_val_losses[:10]
            if val_loss > sum(min_val_losses) / len(min_val_losses) * 1.1:
                early_stop_count += 1
                if early_stop_count == 5:
                    break
            else:
                early_stop_count = 0


            # Write in tensorboard log
            writer.add_scalar('loss', avg_loss, i+1)
            writer.add_scalar('acc', avg_acc, i+1)
            writer.add_scalar('lr', running_lr, i+1)
            writer.add_scalar('val_loss', val_loss, i+1)
            writer.add_scalar('val_acc', val_acc, i+1)
            writer.flush()

            # Show intermediate results in tqdm
            postfix = {
                'loss': '%.4f' % avg_loss,
                'val_loss': '%.4f' % val_loss,
                'acc': '%.4f' % avg_acc,
                'val_acc': '%.4f' % val_acc,
                'lr': '%.2e' % running_lr
            }
            pbar.set_postfix(postfix)

            model.train(True)

    # Save the last checkpoint
    save_name = ckpt_name % (i + 1, avg_loss)
    torch.save(model, save_name)

    writer.close()

    # Print out all saved checkpoints
    print('Saved checkpoints:')
    for save_name, val_loss in sorted(ckpt_info):
        print('\t%s' % save_name)


if __name__ == '__main__':
    main()
