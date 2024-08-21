#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#--------------------------------------------------#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#
###################################################################################################################
###################################################################################################################
# Imports
import random
#--------------------------------------------------#
import scipy 
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.stats import linregress
#--------------------------------------------------#
import warnings
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc 
from matplotlib import cm
#--------------------------------------------------#
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#--------------------------------------------------#
from AP_funcs import cart_prod, cart_dual_prod_re
from N00_Data_Preprocessing import beautiful_print




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      `7MMF'  `7MMF'`7MM"""YMM        db      MMP""MM""YMM `7MMM.     ,MMF'      db      `7MM"""Mq.  .M"""bgd                                         #
#        MM      MM    MM    `7       ;MM:     P'   MM   `7   MMMb    dPMM       ;MM:       MM   `MM.,MI    "Y                                         #
#        MM      MM    MM   d        ,V^MM.         MM        M YM   ,M MM      ,V^MM.      MM   ,M9 `MMb.                                             #
#        MMmmmmmmMM    MMmmMM       ,M  `MM         MM        M  Mb  M' MM     ,M  `MM      MMmmdM9    `YMMNq.                                         #
#        MM      MM    MM   Y  ,    AbmmmqMA        MM        M  YM.P'  MM     AbmmmqMA     MM       .     `MM                                         #
#        MM      MM    MM     ,M   A'     VML       MM        M  `YM'   MM    A'     VML    MM       Mb     dM                                         #
#      .JMML.  .JMML..JMMmmmmMMM .AMA.   .AMMA.   .JMML.    .JML. `'  .JMML..AMA.   .AMMA..JMML.     P"Ybmmd"                                          #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
def categorical_heatmap_ZX(data_book, 
                           num_x, 
                           num_y, 
                           value_min,
                           value_max,
                           c_ticks, 
                           c_ticklabels,
                           plot_title,
                           x_label,
                           y_label,
                           cmap,
                           font_size,
                           result_folder,
                           file_name,
                           ):
    #--------------------------------------------------#                       
    # data_book = [x_idx, y_idx, z_val,]
    random.shuffle(data_book)
    #--------------------------------------------------#
    font = {'family' : "DejaVu Sans"}
    plt.rc('font', **font)
    #--------------------------------------------------#
    df_data = pd.DataFrame(data_book, columns =['x_idx', 'y_idx', 'value'])

    df_data_mat = df_data.reset_index().pivot(index='y_idx', columns='x_idx', values='value')
    df_data_mat.fillna(0, inplace=True)
    df_data_mat = df_data_mat.reindex(range(0, num_y), axis=0, fill_value=0)
    df_data_mat = df_data_mat.reindex(range(0, num_x), axis=1, fill_value=0)

    #--------------------------------------------------#
    fig, ax = plt.subplots(figsize=(9,7))
    extent  = [np.arange(0, num_x, 1).min(), np.arange(0, num_x, 1).max(), np.arange(0, num_y, 1).min(), np.arange(0, num_y, 1).max(), ]
    cax     = ax.imshow(df_data_mat, 
                        cmap = cmap,
                        vmin = value_min,
                        vmax = value_max, 
                        extent = extent,
                        origin ='lower',
                        )
    #--------------------------------------------------#
    cbar = fig.colorbar(cax, shrink = 0.7)
    cbar.set_ticks(c_ticks)
    cbar.set_ticklabels(c_ticklabels)
    cbar.ax.tick_params(labelsize = font_size)
    #--------------------------------------------------#
    ax.set_title(plot_title, fontsize = font_size + 2)
    ax.set_xlabel(x_label,   fontsize = font_size)
    ax.set_ylabel(y_label,   fontsize = font_size)
    ax.xaxis.set_ticks(np.arange(0, num_x, 10))
    ax.xaxis.set_tick_params(rotation=45)  # or use ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_ticks(np.arange(0, num_y, 10))
    ax.xaxis.set_ticklabels(np.arange(0, num_x, 10))
    ax.yaxis.set_ticklabels(np.arange(0, num_y, 10))
    #plt.show()
    #--------------------------------------------------#
    fig.savefig(result_folder / file_name , dpi=1000 )
    mpl.rcParams.update(mpl.rcParamsDefault)
    return








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
#                                      ,og.  ,                                                                                                         #
#                  ,gM""bg            "  `6o"                 `7MM"""YMM `7MMF'MMP""MM""YMM MMP""MM""YMM `7MMF'`7MN.   `7MF'  .g8"""bgd   .M"""bgd     #
#                  8MI  ,8                                     MM    `7   MM  P'   MM   `7 P'   MM   `7   MM    MMN.    M  .dP'     `M  ,MI    "Y      #
#  `7M'   `MF'      WMp,"           `7M'   `MF'                MM   d     MM       MM           MM        MM    M YMb   M  dM'       `  `MMb.          #
#    VA   ,V       ,gPMN.  jM"'       VA   ,V                  MM""MM     MM       MM           MM        MM    M  `MN. M  MM             `YMMNq.      #
#     VA ,V       ,M.  YMp.M'          VA ,V                   MM   Y     MM       MM           MM        MM    M   `MM.M  MM.    `7MMF'.     `MM      #
#      VVV        8Mp   ,MMp            VVV                    MM         MM       MM           MM        MM    M     YMM  `Mb.     MM  Mb     dM      #
#      ,V         `YMbmm'``MMm.         ,V                   .JMML.     .JMML.   .JMML.       .JMML.    .JMML..JML.    YM    `"bmmmdPY  P"Ybmmd"       #
#     ,V                               ,V                                                                                                              #
#  OOb"                             OOb"                                                                                                               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 

from mpl_toolkits.axes_grid1 import make_axes_locatable
def reg_scatter_distn_plot(y_pred,
                           y_real,
                           fig_size       = (10,8),
                           marker_size    = 35,
                           fit_line_color = "brown",
                           distn_color_1  = "gold",
                           distn_color_2  = "lightpink",
                           title          = "Predictions VS. Acutual Values",
                           plot_title     = "Predictions VS. Acutual Values",
                           x_label        = "Actual Values",
                           y_label        = "Predictions",
                           cmap           = None,
                           cbaxes         = (0.425, 0.055, 0.525, 0.015),
                           font_size      = 18,
                           result_folder  = Path("./"),
                           file_name      = Path("file_name"),
                           ): #For checking predictions fittings.
    #--------------------------------------------------# 
    warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
    # Get r_value
    # _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)
    # Get a DataFrame
    pred_vs_real_df = pd.DataFrame(np.ones(len(y_pred)))
    pred_vs_real_df["real"] = y_real
    pred_vs_real_df["pred"] = y_pred
    pred_vs_real_df.drop(columns=0, inplace=True)
    pred_vs_real_df.head()
    #--------------------------------------------------#
    # Main plot settings
    font = {'family' : "DejaVu Sans"}
    plt.rc('font', **font)
    mpl.rc('font', family='serif', serif="DejaVu Sans")

    sns.set_theme(style="darkgrid")
    sns.set_style({'font.family': 'DejaVu Sans'})

    #--------------------------------------------------#
    # 
    y_interval = max(np.concatenate((y_pred, y_real),axis=0)) - min(np.concatenate((y_pred, y_real), axis=0))
    x_y_range = (min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)

    #--------------------------------------------------#

    fig, axes = plt.subplots(1, 2, figsize=(8,6))

    g = sns.jointplot(
                       x            = "real"            , 
                       y            = "pred"            , 
                       data         = pred_vs_real_df   ,
                       kind         = "reg"             ,
                       ci           = 99                ,
                       truncate     = False             ,
                       xlim         = x_y_range         , 
                       ylim         = x_y_range         ,
                       color        = fit_line_color    ,
                       scatter_kws  = {"s": 5}          ,
                       height       = fig_size[1] - 1   ,
                       space        = 0.1               ,
                       marginal_kws = {'color': distn_color_1},  )

    
    sns.histplot(x = y_real, color = distn_color_1, alpha = 0.2, ax = g.ax_marg_x, fill = True, kde = True)
    plt.setp(g.ax_marg_x.patches, color = distn_color_1, alpha = 0.2)

    sns.histplot(y = y_pred, color = distn_color_2, alpha = 0.2, ax = g.ax_marg_y, fill = True, kde = True)
    plt.setp(g.ax_marg_y.patches, color = distn_color_2, alpha = 0.2)
    
    #--------------------------------------------------#
    # Three more different ways of changing the dist plot color.

    #plt.setp(g.ax_marg_y.patches, color="r")

    #g.ax_marg_y.hist(y_pred, color = "r", alpha = .7, orientation = "horizontal")

    #for patch in g.ax_marg_y.patches:
        #patch.set_facecolor("r")

    #--------------------------------------------------#
    # Plot scatters with colors that reflecting densities.
    # 
    values_vstack = np.vstack([y_real, y_pred])
    gaussian_kde = stats.gaussian_kde(values_vstack)(values_vstack)
    # 
    ax1_overlay = g.ax_joint.scatter(x         =  "real"          ,
                                     y         =  "pred"          ,
                                     data      =  pred_vs_real_df ,
                                     c         =  gaussian_kde    ,
                                     s         =  marker_size     ,
                                     edgecolor =  []              ,
                                     )
    #--------------------------------------------------#
    # cbar
    divider = make_axes_locatable(g.ax_joint)
    cbar = plt.colorbar(ax1_overlay                                   ,
                        shrink       =  1.5                           ,
                        orientation  =  "horizontal"                  ,
                        aspect       =  0.05 * 1.5                    ,
                        ax           =  g.ax_joint                    ,
                        cax          =  g.ax_joint.inset_axes(cbaxes) ,
                        )

    cbar.ax.tick_params(labelsize = font_size - 9, rotation = 0)
    cbar.set_label('Scatter Density', size = font_size - 4, labelpad = -40)

    g.fig.suptitle(title, fontsize = font_size, fontweight='bold')
    #g.fig.tight_layout()
    #g.fig.subplots_adjust(top = 0.95)

    g.ax_joint.text(x_y_range[1] - 1.18 * y_interval , 
                    x_y_range[1] - 0.18 * y_interval ,
                    plot_title                      , 
                    fontsize = font_size - 1        ,
                    )

    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()

    g.ax_joint.tick_params(axis = "x", labelsize = font_size - 4)
    g.ax_joint.tick_params(axis = "y", labelsize = font_size - 4)

    g.ax_joint.set_xlabel(x_label, fontsize = font_size + 1, fontweight = 'bold')
    g.ax_joint.set_ylabel(y_label, fontsize = font_size + 1, fontweight = 'bold')
    g.ax_joint.set_aspect('equal', adjustable='box')

    #g.fig.set_figwidth(8)
    #g.fig.set_figheight(6)

    (g).savefig(result_folder / (file_name) , dpi = 1000 )

    #--------------------------------------------------#
    # clear formats and return
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close("all")
    return



y_pred = [0,1,3,4,5,6,7,8,9,9]
y_real = [0,1,3,4,5,6,7,8,9,8.9]



r_value = 0.9
ts_rho  = 0.9
epoch   = 100

'''
reg_scatter_distn_plot(y_pred,
                       y_real,
                       fig_size       = (10,8),
                       marker_size    = 35,
                       fit_line_color = "pink",
                       distn_color_1  = "gold",
                       distn_color_2  = "cyan",
                        # title         =  "Predictions vs. Actual Values\n R = " + \
                        #                         str(round(r_value,3)) + \
                        #                         ", Epoch: " + str(epoch+1) ,
                        title           =  "",
                        plot_title      =  "R = " + str(round(r_value,3)) + \
                                            "\n" + r'$\rho$' + " = " + str(round(ts_rho,3)) + \
                                                "\nEpoch: " + str(epoch+1) ,
                       x_label        = "Actual Values",
                       y_label        = "Predictions",
                       cmap           = None,
                       font_size      = 18,
                       result_folder  = Path("./"),
                       file_name      = Path("file_name"),
                       )
'''












#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
#  `7MM"""Mq. `7MMF'        .g8""8q.   MMP""MM""YMM     `YMM'   `MP' 
#    MM   `MM.  MM        .dP'    `YM. P'   MM   `7       VMb.  ,P   
#    MM   ,M9   MM        dM'      `MM      MM             `MM.M'    
#    MMmmdM9    MM        MM        MM      MM               MMb     
#    MM         MM      , MM.      ,MP      MM             ,M'`Mb.   
#    MM         MM     ,M `Mb.    ,dP'      MM            ,P   `MM.  
#  .JMML.     .JMMmmmmMMM   `"bmmd"'      .JMML.        .MM:.  .:MMa.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 



#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#   `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'       `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'  #
#     VAMAV          VAMAV          VAMAV          VAMAV          VAMAV           VAMAV          VAMAV          VAMAV          VAMAV          VAMAV    #
#      VVV            VVV            VVV            VVV            VVV             VVV            VVV            VVV            VVV            VVV     #
#       V              V              V              V              V               V              V              V              V              V      #

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                                                                                                                                                          
#      `MM.              `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.       
#        `Mb.              `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.     
# MMMMMMMMMMMMD     MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD   
#         ,M'               ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'     
#       .M'               .M'              .M'              .M'              .M'              .M'              .M'              .M'              .M'       
#                                                                                                                                                          

#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #














