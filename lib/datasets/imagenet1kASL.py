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
            transforms.Resize(200, 200),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
            ])

        transform_test = transforms.Compose([
            transforms.Resize(200, 200),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.224)),
            
        ])
    
        if args.datasetType == 'asl':
            #pytorch_dataset_class = torchvision.datasets.CIFAR10
            num_classes = 28
        elif args.datasetType == 'cifar100':
            #pytorch_dataset_class = torchvision.datasets.CIFAR100
            num_classes = 100
        else:
            raise ValueError('Unrecognized dataset name...')
        
        trainset = datasets.ImageFolder(train_dirs, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        
        testset = datasets.ImageFolder(val_dirs, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.val_batch_size)
        
 
    return train_loader, val_loader
