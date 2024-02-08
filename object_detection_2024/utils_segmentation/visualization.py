"""
File containing functions for visualizing images, masks, bounding boxes, and performance metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_images(images, targets=None, metrics=None, ncols=None):
    """
    Display images. If targets is not None, it also shows masks and bounding boxes 
    for kit (red) and membrane (green). If metrics are not None, then it also shows scores and IoU values.
    
    Args:
        images (single or list of torch.Tensors (3, H, W)):
        targets (single or list of dict containg masks torch.Tensor (2, H, W), and boxes torch.Tensor (2, 4)):
        metrics (single or list of dict containing scores and IoUs)
    """
    
    # Catch single example case and bring it to tuple format
    if type(images) not in [tuple, list]:
        images = [images]
        if targets is not None:
            targets = [targets]

    n_imgs = len(images)
    
    # To regulate image overlap in plot
    alpha = 0.7
    beta = 1 - alpha
    
    if targets is not None:
        # Colors
        kit_color = [255, 0, 0]
        membrane_color = [0, 255, 0]
        # To write text
        if metrics is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = .5
            thickness = 2

    # Build the right grid for plotting multiple images
    if not ncols:
        factors = [i for i in range(1, n_imgs + 1) if n_imgs % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else n_imgs // 4 + 1
    nrows = int(n_imgs / ncols) + int(n_imgs % ncols)
    images = [images[i] if n_imgs > i else None for i in range(nrows * ncols)]
    
    if targets is not None:
        targets = [targets[i] if n_imgs > i else None for i in range(nrows * ncols)]
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if n_imgs != 1:
        ax = ax.flatten()[:len(images)]
    else:
        ax = [ax]
    
    for i in range(n_imgs):
        
        img = images[i]
        
        # Read masks and boxes
        if targets is not None:
            masks = targets[i]['masks']
            boxes = targets[i]['boxes']
            
            if type(masks) is np.ndarray:
                masks = torch.Tensor(masks)
            if type(boxes) is np.ndarray:
                boxes = torch.Tensor(boxes)
        
        # Read scores and IoU
        if metrics is not None:
            scores = np.round(metrics[i]['scores'], 3)
            iou_kit = np.round((metrics[i]['iou_masks'][0], metrics[i]['iou_boxes'][0]), 3)
            iou_membrane = np.round((metrics[i]['iou_masks'][1], metrics[i]['iou_boxes'][1]), 3)
    
        # Convert img to right format for cv2 (reverse channels from BGR (cv2) to RGB (plt))
        img_n = (255*img).to(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1]

        if targets is not None:
            
            # Reformat kit mask (red), and membrane mask (green)
            mask_kit = (masks[0].unsqueeze(-1).repeat(1, 1, 3).numpy() * kit_color).astype(np.uint8)
            mask_membrane = (masks[1].unsqueeze(-1).repeat(1, 1, 3).numpy() * membrane_color).astype(np.uint8)
            # Sum to get one single mask
            mask_sum = mask_kit + mask_membrane
            # Get boxes coordinates
            x1_k, y1_k, x2_k, y2_k = boxes[0].int().tolist()  # kit
            x1_m, y1_m, x2_m, y2_m = boxes[1].int().tolist()  # membrane
            # Add mask
            img_n = cv2.addWeighted(src1=img_n, alpha=alpha, src2=mask_sum, beta=beta, gamma=0)
            # Add kit bbox (red)
            img_n = cv2.rectangle(img_n, (x1_k, y1_k), (x2_k, y2_k), color=kit_color, thickness=2)
            # Add membrane bbox (green)
            img_n = cv2.rectangle(img_n, (x1_m, y1_m), (x2_m, y2_m), color=membrane_color, thickness=2)
        
            # Add scores and IoUs
            if metrics is not None:
                img_n = cv2.putText(img_n, f'K | Score: {scores[0]} | IoU (m, b): {iou_kit}', (0, 15), 
                                    font, font_scale, kit_color, thickness)  # Kit
                img_n = cv2.putText(img_n, f'M | Score: {scores[1]}| IoU (m, b): {iou_membrane}', (0, 35), 
                                    font, font_scale, membrane_color, thickness)  # Membrane

        # Plot
        ax[i].imshow(img_n)
        ax[i].axis('off')
            
    plt.tight_layout()
    plt.show()


def plot_metrics(metrics):
    """
    Plot metrics as coming out from the training app.
    """

    plt.figure(figsize=(20, 10))
    styles = ['-or', '-og', '--or', '--og']

    average = 0
    for i, col in enumerate(metrics.columns[2:]):
        plt.plot(metrics[col], styles[i], label=col)
        average += metrics[col]

    plt.plot(average/4, 'k-', label='IoU average')

    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.yticks(np.arange(0, 1, .05))
    plt.legend()
    plt.grid()
    plt.show()
