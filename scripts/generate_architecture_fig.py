import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_fig():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=600)
    ax.axis('off')
    
    # Palette: Professional, muted, high-contrast
    palette = {
        'input': {'fill': '#ECEFF1', 'stroke': '#37474F', 'text': '#263238'},
        'cnn': {'fill': '#FFF3E0', 'stroke': '#E65100', 'text': '#BF360C'},
        'vit': {'fill': '#E3F2FD', 'stroke': '#1565C0', 'text': '#0D47A1'},
        'dsbn': {'fill': '#E0F2F1', 'stroke': '#00695C', 'text': '#004D40'},
        'fusion': {'fill': '#F3E5F5', 'stroke': '#6A1B9A', 'text': '#4A148C'},
        'output': {'fill': '#ECEFF1', 'stroke': '#263238', 'text': '#263238'},
        'badge': {'fill': '#FFFFFF', 'stroke': '#78909C', 'text': '#37474F'}
    }
    
    # 1. Background Containers (Grouping)
    containers = [
        (2.8, 4.3, 6.4, 2.0, 'CNN Feature Branch', '#FFF8E1', '#FFE082'),
        (2.8, 1.7, 6.4, 2.0, 'Vision Transformer Branch', '#E8EAF6', '#C5CAE9'),
        (9.6, 2.5, 2.8, 3.2, 'Feature Fusion', '#F3E5F5', '#E1BEE7'),
        (12.8, 2.5, 3.8, 3.2, 'Classification & Calibration', '#ECEFF1', '#CFD8DC')
    ]
    for cx, cy, cw, ch, clabel, cfill, cstroke in containers:
        c_box = patches.FancyBboxPatch(
            (cx, cy), cw, ch, boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=1.2, edgecolor=cstroke, facecolor=cfill, alpha=0.5, zorder=1
        )
        ax.add_patch(c_box)
        ax.text(cx + 0.15, cy + ch - 0.25, clabel, fontsize=10, fontweight='bold',
                color='#455A64', ha='left', va='top', fontfamily='sans-serif', zorder=2)

    # Helper: Uniform Box
    def draw_node(x, y, w, h, title, subtitle, style_key):
        st = palette[style_key]
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.5, edgecolor=st['stroke'], facecolor=st['fill'], zorder=3
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.12 if subtitle else 0), title,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=st['text'], fontfamily='sans-serif', zorder=4)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.18, subtitle,
                    ha='center', va='center', fontsize=8.5,
                    color='#546E7A', fontfamily='sans-serif', zorder=4)

    # Helper: Dimension Pill Badge
    def draw_badge(x, y, text):
        badge = patches.FancyBboxPatch(
            (x - 0.45, y - 0.15), 0.9, 0.3, boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.0, edgecolor='#90A4AE', facecolor='#FFFFFF', zorder=5
        )
        ax.add_patch(badge)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold',
                color='#37474F', fontfamily='sans-serif', zorder=6)

    # Helper: Thicker, Uniform Arrow
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#37474F', edgecolor='#37474F',
                                    width=1.8, headwidth=7, headlength=8, shrink=0),
                    zorder=3)

    # Node Placement
    # Input
    draw_node(0.2, 3.4, 2.0, 1.0, "Fundus Image", "256×256×3", "input")

    # Top Path (CNN)
    draw_arrow(2.2, 3.9, 3.2, 5.0)
    draw_node(3.2, 4.6, 2.4, 0.9, "EfficientNet-B3", "Local Features", "cnn")
    draw_arrow(5.6, 5.05, 6.6, 5.05)
    draw_badge(6.1, 5.05, "1536-d")
    draw_node(6.6, 4.6, 2.2, 0.9, "DSBN Module", "Domain 0,1,2", "dsbn")
    draw_arrow(8.8, 5.05, 10.0, 4.3)
    draw_badge(9.4, 4.65, "1536-d")

    # Bottom Path (ViT)
    draw_arrow(2.2, 3.9, 3.2, 2.8)
    draw_node(3.2, 2.2, 2.4, 0.9, "DINOv2 ViT-S", "Global Context", "vit")
    draw_arrow(5.6, 2.65, 6.6, 2.65)
    draw_badge(6.1, 2.65, "384-d")
    draw_node(6.6, 2.2, 2.2, 0.9, "DSBN Module", "Domain 0,1,2", "dsbn")
    draw_arrow(8.8, 2.65, 10.0, 3.5)
    draw_badge(9.4, 3.05, "384-d")

    # Fusion & Output
    draw_node(10.0, 3.5, 2.0, 1.0, "Concatenation", "1920-d Vector", "fusion")
    draw_arrow(12.0, 4.0, 13.2, 4.0)
    draw_badge(12.6, 4.0, "1920-d")
    
    draw_node(13.2, 4.6, 2.8, 0.85, "MLP Classifier", "Ordinal Regression", "output")
    draw_arrow(14.6, 4.6, 14.6, 3.75)
    draw_node(13.2, 2.9, 2.8, 0.85, "Nelder-Mead", "Boundary Calibration", "output")

    ax.set_xlim(-0.2, 17.0)
    ax.set_ylim(1.2, 6.8)
    
    out_dir = 'paper/figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Figure_2_Architecture_New.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(out_dir, 'Figure_2_Architecture_New.png'), bbox_inches='tight', dpi=600)
    print("Redrawn Figure 2: Architecture (Professional Quality)")

if __name__ == '__main__':
    create_architecture_fig()
