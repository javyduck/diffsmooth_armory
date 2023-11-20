import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop
from typing import Optional
from art.estimators.classification import PyTorchClassifier
from .improved_diffusion import cifar_ddpm

def get_diffsmooth_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = DiffsmoothModel(model_kwargs, wrapper_kwargs, weights_path)
    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
    
class DiffsmoothModel(nn.Module):
    def __init__(self, model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None):
        super(DiffsmoothModel, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.denoising_sigma = model_kwargs.get('denoising_sigma', 0.12)
        self.local_sigma = model_kwargs.get('local_sigma', 0.12)
        
        # Number of noisy samples per input
        self.num_samples = model_kwargs.get('num_samples', 20)
        self.local_vote = model_kwargs.get('local_vote', 5)
        
        # Set path
        dir_name, _ = os.path.split(weights_path)
        denoising_ckpt = os.path.join(dir_name, 'cifar10_uncond_50M_500K.pt')
        self.denoiser = cifar_ddpm(self.denoising_sigma, denoising_ckpt, device = self.device)
        self.denoiser.eval()
        # Load Base model from checkpoint
        model_ckpt = os.path.join(dir_name, f'vit_cifar10_sigma_{self.local_sigma}')
        self.model = ViTForImageClassification.from_pretrained(model_ckpt).to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        size = self.processor.size["height"]
        self.transforms = Compose([
            Resize(size),
            CenterCrop(size),
            lambda x: self.add_local_gaussian_noise(x, self.local_sigma),
            Normalize(mean=image_mean, std=image_std)
        ])
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        ## the attack batch size should be 1
        assert x.shape[0] == 1, "The attack batch size should be 1"
        
        ## Gaussian aug for randomized smoothing ##
        x = x.repeat_interleave(repeats=self.num_samples, dim=0)
        
        ## purification ##
        if self.denoising_sigma:
            ## add gaussian noise ##
            x += torch.randn_like(x) * self.denoising_sigma
            x = self.denoiser(x)
            
        # local smoothing (soft version)
        x = self.transforms(x.repeat_interleave(repeats=self.local_vote, dim=0))
        x = torch.softmax(self.model(x).logits, dim=1)
        # should be the majority of hard prediction actually
        x = x.mean(dim=0, keepdims=True)
        return self.confidences_to_log_softmax(x)

    def add_local_gaussian_noise(self, img, sigma):
        noisy_img = img + torch.randn_like(img) * sigma
        return noisy_img.clamp(0, 1)
    
    def confidences_to_log_softmax(self, confidences: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        # Clamp the confidences to avoid log(0) and log(1)
        confidences = torch.clamp(confidences, epsilon, 1-epsilon)

        # Convert probabilities to logits
        logits = torch.log(confidences) - torch.log1p(-confidences)

        # Normalize logits for numerical stability
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values

        # Apply log_softmax
        log_softmax_values = F.log_softmax(logits, dim=-1)

        return log_softmax_values