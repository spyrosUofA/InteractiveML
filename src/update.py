#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
import autograd_hacks
import copy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True) #self.args.local_bs
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def dp_sgd(self, model, global_round, clipping_norm=1, noise_mag=1.0):
        #################
        ## ALGORITHM 1 ##
        #################

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch (1...E)
        for iter in range(self.args.local_ep):
            batch_loss = []

            # batch data (x,y) (1...m)
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                print(labels)
                print(labels[[[0]]])

                print(labels.shape)

                # Compute loss
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, [labels])

                print(loss)

                # clip, average, and perturb gradients
                hi = 0
                sum_clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

                # clip individual gradient g_i for i in {1...m}
                for i in range(self.args.local_bs):  #loss.size(0)
                    hi += 1
                    print(hi)

                    # backward pass: compute gradient g_i
                    loss[i].backward(retain_graph=True)

                    # L2 norm of g_i
                    total_norm = 0
                    for param in model.parameters():
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    print(total_norm)

                    #model.zero_grad()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    for name, param in model.named_parameters():



                        #print(name)
                        #print(param.grad.size())
                        #print(param.grad.sum())
                        #print(loss[i])


                        sum_clipped_grads[name] += param.grad / loss.size(0)

                        #param *= min(1, clipping_norm / sum(param ** 2))
                        #per_sample_grad = param.grad.detach().clone()
                        #clip_grad_norm_(per_sample_grad, max_norm=clipping_norm)
                        #param.accumulated_grads.append(per_sample_grad)

                # Update
                optimizer.step()


                exit()



                # Aggregate gradients (Temp)
                for param in model.parameters():
                    param.grad = torch.stack(param.accumulated_grads, dim=0)

                # Add noise
                for param in model.parameters():
                    param += torch.normal(mean=0, std=noise_mag)

                # theta_{i+1}
                optimizer.step()


                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                # go to next epoch....
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def dp_sgdV2(self, model, global_round, clipping_norm=1, noise_mag=1.0):
        #################
        ## ALGORITHM 1 ##
        #################

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch (1...E)
        for iter in range(self.args.local_ep):

            batch_loss = []
            #sum_clipped_grads = {param in model.named_parameters()}
            sum_clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

            # batch data (x,y) (1...m)
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)
                print("jskfa")
                print(images)

                print("jskfa")
                print(images[[[0]]])

                print(labels.shape)

                # gradient g_i for i in {1...m}
                for i in range(self.args.local_bs):  # loss.size(0)

                    # Compute loss i
                    model.zero_grad()
                    log_probs = model(torch.FloatTensor(images[[[i]]]))
                    print(log_probs)
                    loss = self.criterion(log_probs, torch.FloatTensor(labels[i]))

                    # backward pass: compute gradient g_i
                    loss.backward(retain_graph=True)

                    # L2 norm of g_i
                    total_norm = 0
                    for param in model.parameters():
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    print(total_norm)

                    # Clipped g_i
                    for param in model.parameters():
                        sum_clipped_grads += param * min(1, clipping_norm / total_norm)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

            # Add noise to clipped parameters
            for param in model.parameters():
                sum_clipped_grads += torch.normal(mean=0, std=noise_mag)

            # take average
            sum_clipped_grads / self.args.local_bs

            # Update: theta_{i+1}
            optimizer.step() # do this manually

            # Append loss, go to next epoch...
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def dp_sgdV3(self, model, global_round, clipping_norm=15., noise_mag=1.0):
        #################
        ## ALGORITHM 1 ##
        #################

        # Set mode to train model
        model.train()
        epoch_loss = []


        model_dummy = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch (1...E)
        for iter in range(self.args.local_ep):

            batch_loss = []

            print("here")

            # for each batch data (1...B)
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                print(batch_idx)

                model.zero_grad()
                # add hooks
                autograd_hacks.add_hooks(model)

                # Forward pass, compute loss, backwards pass
                log_probs = model(torch.FloatTensor(images))
                loss = self.criterion(log_probs, labels)
                loss.backward(retain_graph=True)

                # Per-sample gradients g_i
                autograd_hacks.compute_grad1(model) # PROBLEM 2ND iteraton

                #print(autograd_hacks.is_supported(model))
                #exit()

                autograd_hacks.disable_hooks()

                print("images.shape")
                print(len(labels))

                # Calculate L2 norm for each g_i
                g_norms = torch.zeros(labels.shape)
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        print(param.grad1.data.shape)
                        g_norms += param.grad1.data.norm(2, dim=(1,2)) ** 2
                    else:
                        g_norms += param.grad1.data.norm(2, dim=1) ** 2
                g_norms.sqrt

                # Clipping factor =  min(1, S / norm(g_i))
                clip_factor = torch.clamp(clipping_norm * g_norms ** -0.5, max=1)

                # Clip gradients
                for param in model.parameters():
                    for i in range(len(labels)):
                        param.grad1[i] = param.grad1[i] * clip_factor[i] * 2.0

                # Noisy batch update
                for param in model.parameters():
                    # take average
                    param.grad = param.grad1.mean(dim=0)

                    # add noise
                    param.grad.add_(torch.randn(param.size()) * noise_mag)

                    # update weights
                    param.data -= self.args.lr * param.grad

                # revert model back to old format... bizarre
                model_dummy.load_state_dict(model.state_dict())
                model = copy.deepcopy(model_dummy)

                # Record loss, reset gradients
                batch_loss.append(loss.item())


            # Append loss, go to next epoch...
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
