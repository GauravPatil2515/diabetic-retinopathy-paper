import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_preprocessing_fig():
    fig, ax = plt.subplots(figsize=(14, 2.8), dpi=600)
    ax.axis('off')
    
    # Palette
    bg_color = '#F8F9FA'
    stroke_color = '#CFD8DC'
    
    steps = [
        ("Raw Fundus", "Input Image", "#37474F", "#ECEFF1", "1"),
        ("Crop ROI", "Remove Borders", "#00695C", "#E0F2F1", "2"),
        ("CLAHE", "Local Contrast", "#0277BD", "#E1F5FE", "3"),
        ("Graham Norm", "Local Gaussian", "#6A1B9A", "#F3E5F5", "4"),
        ("Augmentation", "Flips & Jitter", "#E65100", "#FFF3E0", "5"),
        ("Tensor Input", "256×256 / 384×384", "#263238", "#CFD8DC", "6")
    ]
    
    start_x = 0.2
    block_w = 1.8
    block_h = 1.4
    spacing = 0.45
    y_pos = 0.7
    
    for i, (title, sub, stroke, fill, step_num) in enumerate(steps):
        # Card Box
        box = patches.FancyBboxPatch(
            (start_x, y_pos), block_w, block_h,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            linewidth=1.4, edgecolor=stroke, facecolor=fill, zorder=2
        )
        ax.add_patch(box)
        
        # Step Badge Circle
        badge_circle = plt.Circle((start_x + 0.25, y_pos + block_h - 0.25), 0.16,
                                  color=stroke, zorder=3)
        ax.add_artist(badge_circle)
        ax.text(start_x + 0.25, y_pos + block_h - 0.25, step_num,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='#FFFFFF', fontfamily='sans-serif', zorder=4)
        
        # Title & Subtitle
        ax.text(start_x + block_w/2, y_pos + 0.65, title,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=stroke, fontfamily='sans-serif', zorder=3)
        ax.text(start_x + block_w/2, y_pos + 0.3, sub,
                ha='center', va='center', fontsize=8.5,
                color='#546E7A', fontfamily='sans-serif', zorder=3)
        
        # Connection Arrow
        if i < len(steps) - 1:
            arrow_start = start_x + block_w + 0.05
            arrow_end = arrow_start + spacing - 0.1
            ax.annotate('', xy=(arrow_end, y_pos + block_h/2), xytext=(arrow_start, y_pos + block_h/2),
                        arrowprops=dict(facecolor='#78909C', edgecolor='#78909C',
                                        width=1.8, headwidth=6, headlength=7, shrink=0),
                        zorder=2)
            
        start_x += block_w + spacing
        
    ax.set_xlim(0, start_x - spacing + 0.2)
    ax.set_ylim(0.3, 2.4)
    
    out_dir = 'paper/figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Figure_1_Preprocessing_New.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(out_dir, 'Figure_1_Preprocessing_New.png'), bbox_inches='tight', dpi=600)
    print("Redrawn Figure 1: Preprocessing (Segmented Cards & Badges)")

if __name__ == '__main__':
    create_preprocessing_fig()
