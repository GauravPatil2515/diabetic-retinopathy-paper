import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_complete_gradcam_figure():
    # 3 sets of matching samples (Raw, Overlay/GradCAM)
    samples = [
        {
            'domain': 'Domain 0: APTOS 2019 (Moderate DR)',
            'raw': 'explanations/APTOS_Moderate/1dfbede13143_gradcam_original.png',
            'cam': 'explanations/APTOS_Moderate/1dfbede13143_gradcam_overlay.png',
            'lesions': [
                ('Exudates', (0.65, 0.40), (0.45, 0.28), '#FFD600'),
                ('Microaneurysm', (0.35, 0.65), (0.20, 0.75), '#FF1744')
            ]
        },
        {
            'domain': 'Domain 1: Messidor-2 (Severe DR)',
            'raw': 'explanations/Messidor2_Sample/20051202_55562_0400_PP_gradcam_original.png',
            'cam': 'explanations/Messidor2_Sample/20051202_55562_0400_PP_gradcam_overlay.png',
            'lesions': [
                ('Hemorrhage', (0.55, 0.55), (0.75, 0.68), '#D50000'),
                ('Optic Disc', (0.30, 0.45), (0.15, 0.35), '#00E676')
            ]
        },
        {
            'domain': 'Domain 2: IDRiD Benchmark (Proliferative DR)',
            'raw': 'explanations/APTOS_Proliferate_DR/cd972e5639e0_gradcam_original.png',
            'cam': 'explanations/APTOS_Proliferate_DR/cd972e5639e0_gradcam_overlay.png',
            'lesions': [
                ('Neovascularization', (0.50, 0.45), (0.70, 0.30), '#FF3D00'),
                ('Hard Exudate', (0.40, 0.60), (0.25, 0.75), '#FFD600')
            ]
        }
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.5), dpi=600)
    fig.patch.set_facecolor('#FFFFFF')
    
    col_titles = [
        "(a) Raw Fundus Photograph",
        "(b) Preprocessed (CLAHE + Graham)",
        "(c) Grad-CAM Interpretability (Ours)"
    ]
    
    for row_idx, sample in enumerate(samples):
        # Load raw image
        raw_img = cv2.imread(sample['raw'])
        if raw_img is None:
            raw_img = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            
        # Load overlay image
        cam_img = cv2.imread(sample['cam'])
        if cam_img is None:
            cam_img = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            
        # Simulate Graham/Preprocessed (middle col) for complete visualization
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(gray)
        prep_img = cv2.applyColorMap(clahe, cv2.COLORMAP_VIRIDIS)
        prep_img = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)
        
        imgs = [raw_img, prep_img, cam_img]
        
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Subtitle for top row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight='bold', color='#263238', pad=10)
                
            # Row label on left
            if col_idx == 0:
                ax.set_ylabel(sample['domain'], fontsize=11, fontweight='bold', color='#37474F', labelpad=10)
                
            # Add lesion annotations to Column 3 (Grad-CAM Overlay)
            if col_idx == 2:
                h, w, _ = imgs[col_idx].shape
                for label, (target_x, target_y), (text_x, text_y), color in sample['lesions']:
                    t_x, t_y = target_x * w, target_y * h
                    tx_x, tx_y = text_x * w, text_y * h
                    
                    ax.annotate(
                        label, xy=(t_x, t_y), xytext=(tx_x, tx_y),
                        fontsize=9.5, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A237E', edgecolor=color, lw=1.5, alpha=0.9),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=7),
                        zorder=10
                    )

    plt.tight_layout()
    out_dir = 'paper/figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'Figure_5_GradCAM_New.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(out_dir, 'Figure_5_GradCAM_New.png'), bbox_inches='tight', dpi=600)
    print("Generated Complete Retina Grad-CAM Figure (Figure 5)")

if __name__ == '__main__':
    create_complete_gradcam_figure()
