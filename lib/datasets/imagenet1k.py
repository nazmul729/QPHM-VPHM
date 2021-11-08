import torch
import torchvision
from torchvision import datasets, transforms


def imagenet1k(args, distributed=False):
    
    train_dirs = args.train_dirs
    val_dirs = args.val_dirs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    color_jitter = args.color_jitter
    print('==> Preparing data..')
    if args.datasetType == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])
        
        #trainset = pytorch_dataset_class(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        if args.datasetType == 'cifar10':
            pytorch_dataset_class = torchvision.datasets.CIFAR10
            num_classes = 10
        elif args.datasetType == 'cifar100':
            pytorch_dataset_class = torchvision.datasets.CIFAR100
            num_classes = 100
        else:
            raise ValueError('Unrecognized dataset name...')
        
        trainset = pytorch_dataset_class(root='/scratch/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        testset = pytorch_dataset_class(root='/scratch/data', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.val_batch_size, shuffle=False, drop_last=True)
        
 
    return train_loader, val_loader
