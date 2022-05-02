# Runs a series of checks to confirm that the operating system is stable and GPUs are available
# NB! Should only be ran when training
import os
import torch
from tqdm import tqdm
import time


def systems_configurations(configurations):
    print("Running systems configurations, stand by: ")
    for functions in tqdm(configurations):
        functions()
    print("Configurations completed")
    return


def check_for_cuda():
    # Checks if cuda is available
    # Checks for the number of GPUs
    # Checks for the type of GPU
    cuda = False
    cpu = False
    print("----- Checking for GPU on your system -----")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("GPU found:")
        print("----------------------------------------------------------")
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number of CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(
            0).total_memory/1e9)
        print("----------------------------------------------------------")
        cuda = True
        setting_device(cuda, cpu)
    else:
        print("No GPU available")
        print("This means your computer\'s CPU will be utilized. Consider finding a GPU as this is not recommended.")
        promptActivated = True
        while promptActivated:
            try:
                answer = input(
                    "Do you wish to continue training on your CPU? (yes/no): ")
                if answer == "yes":
                    cpu = True
                    setting_device(cuda, cpu)
                    promptActivated = False

                elif answer == "no":
                    print("Terminating session")
                    promptActivated = False
                    break
            except ValueError:
                print("Expected yes or no, but got {}".format(answer))
    return


def setting_device(cuda=False, cpu=False):
    if cuda == True:
        torch.device('cuda')
        print("Device set to CUDA")
    if cpu == True:
        torch.device('cpu')
        print("Device set to CPU")
    return


configurations = [check_for_cuda]

# systems_configurations(configurations)
