#generates:
# - Figure 5 (Appendix)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

# palette
c_u = "#FF8300"
c_p = "#785EF0"
c_d = "#DC267F"
c_dp = "#648FFF"

fontsize = 21
matplotlib.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": fontsize,
    "pgf.preamble": rf"""
        \usepackage{{xcolor}}
        \usepackage{{mathptmx}}
        \definecolor{{ibmyellow}}{{HTML}}{{FFB000}}
        \definecolor{{ibmorange}}{{HTML}}{{FE6100}}
        \definecolor{{ibmpurple}}{{HTML}}{{785EF0}}
        \definecolor{{ibmblue}}{{HTML}}{{648FFF}}
        \definecolor{{ibmred}}{{HTML}}{{DC267F}}
        % Define macros to ensure they work in text and math mode
        \newcommand{{\U}}{{\textcolor{{ibmorange}}{{U}}}}
        \newcommand{{\Pp}}{{\textcolor{{ibmpurple}}{{P}}}}
        \newcommand{{\D}}{{\textcolor{{ibmred}}{{D}}}}
    """
})

words = ["He", "built", "many", "houses", "in", "the", "village",
         "But", "he", "was", "sleeping", "in", "a", "bad", "hut",
         "The", "village", "people", "talked", "about", "him", "every", "day"]

# copy-pasted
boundaries = [6.5, 14.5] 
u = [11.56, 12.37, 6.72, 3.20, 2.27, 1.16, 3.97, 13.50, 5.78, 3.38, 15.06, 2.91, 2.34, 9.44, 9.75, 3.72, 13.0, 13.69, 14.31, 1.19, 5.69, 9.00, 1.16]
d = [np.nan]*7 + [8.06, 3.02, 2.56, 16.00, 1.88, 1.95, 9.31, 5.53, 3.88, 5.31, 2.90, 9.00, 1.41, 1.29, 7.56, 0.75]
p = [11.50, 12.69, 7.88, 1.73, 2.63, 1.35, 2.73, 12.44, 6.69, 3.36, 9.94, 2.73, 2.33, 9.13, 6.31, 2.91, 8.06, 5.44, 11.37, 1.71, 5.5, 7.47, 1.06]
dp = [np.nan]*7 + [5.53, 2.41, 2.91, 12.00, 1.98, 1.87, 10.56, 1.68, 2.78, 4.16, 3.67, 6.12, 0.53, 1.91, 7.47, 0.83]
x = np.arange(len(words))

fig, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(25, 5), sharex=True, 
                                     gridspec_kw={'height_ratios': [0.5, 2.5]})

fig.subplots_adjust(hspace=0.00)
ax_img.axis('off')
ax_img.set_ylim(0, 1)

def add_img(ax, path, x_pos, target_h=400):
    try:
        img = plt.imread(path)
        zoom = target_h / img.shape[0]
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x_pos, 0.0),
            frameon=False,
            box_alignment=(0.59, 0.0),
            clip_on=False      
        )
        #ab.set_in_layout(False) 
        ax.add_artist(ab)
    except:
        pass

add_img(ax_img, "example/ex_1.png", 2.98)
add_img(ax_img, "example/ex_2.png", 10.5)
add_img(ax_img, "example/ex_3.png", 18.59)

ax_plot.plot(x, u, color=c_u, label=r"{\U} (no context)", marker='o', markersize=4)
ax_plot.plot(x, p, color=c_p, label=r"{\Pp} (no context)", marker='^', markersize=4)
ax_plot.plot(x, d, color=c_d, label=r"{\D}", marker='s', markersize=4)
ax_plot.plot(x, dp, color=c_dp, label=r"{\Pp}+{\D}", marker='x', markersize=4)


for b in boundaries:
    ax_plot.axvline(x=b, color='black', linestyle='--', alpha=0.3)


ax_plot.set_xticks(x)
ax_plot.set_xticklabels([fr"\textbf{{{w}}}" for w in words], rotation=0)
ax_plot.set_ylabel(r"Surprisal (bits)")
ax_plot.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=4, 
    frameon=False, 
    fontsize=23,
    handletextpad=0.1,
    columnspacing=2.0
)


fig.subplots_adjust(bottom=0.2, top=0.85, hspace=0.0)

ax_plot.set_ylim(0, 18)
ax_plot.grid(axis='y', linestyle=':', alpha=0.5)


plt.savefig("latex/imgs/pics/BLOOM_surprisal_ex.pdf", bbox_inches='tight')
