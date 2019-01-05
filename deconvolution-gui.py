__author__ = 'Daniel Pelliccia- Instruments & Data Tools'

import glob
import sys
import platform
import os
from shutil import copyfile
import time
import pandas as pd
from skimage.external import tifffile as tif
from skimage.restoration import denoise_tv_chambolle

if platform.system() == 'Windows':
    import winsound

# Handle the imports of the GPU modules
GPU = False
CPU = False
try:
    import pycuda.autoinit
    from deconvolution_GPUutilities import runDeconvolutionGPU
    GPU = True
except ImportError:

    from deconvolution_CPUutilities import runDeconvolutionCPU
    CPU = True

# Handle the import of the ASTRA Toolbox module
AST = False
if GPU is True:
    try:
        import astra
        from astra_GPUutilities import iradon_astra
        AST = True
    except:
        from skimage.transform import iradon
else:
    from skimage.transform import iradon

from common_utilities import sino_centering, remove_blob_sino_wavelet

if sys.version_info[0] < 3:
    import Tkinter as Tk
    from tkFileDialog import askopenfilename, askdirectory
    from ttk import Progressbar
    import tkMessageBox as messagebox
else:
    import tkinter as Tk
    from tkinter.filedialog import askopenfilename, askdirectory
    from tkinter.ttk import Progressbar
    from tkinter import messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure

class deconvolution():
    def __init__(self):
        self.root = Tk.Tk()

        ############################################################################################
        ###  Set up of window
        ############################################################################################

        # Set the window title
        self.root.title("Deconvolution")
        # Read screen resolution
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.screen_mmwidth = self.root.winfo_screenmmwidth()
        self.screen_mmheight = self.root.winfo_screenmmheight()
        # Size and position of the window
        self.root.geometry('%dx%d+%d+%d' % (int(0.75*self.screen_width), int(0.9*self.screen_height), 0, 0) )

        # Get the background color of the root window
        self.bgcol = self.root.cget("bg")
        # if platform.system() == 'Windows':
        #     self.root.iconbitmap('.\idtlogo.ico')
        # if platform.system() == 'Linux':
        #     self.root.iconbitmap('idtlogo.ico')
        ###########################################################################################


        ############################################################################################
        ###  Title
        ############################################################################################

        self.titleLabel = Tk.Label(text="Deconvolution of OPT data", \
	          font=("Helvetica", 14), relief='flat', pady=3, padx=5)
        self.titleLabel.grid(row=0, column=0, columnspan=1, sticky='w')

        ###########################################################################################


        ############################################################################################
        ###  Top buttons
        ############################################################################################

        self.close_button = Tk.Button(self.root, text="Quit", bg = '#F45B16', command=self._quit)
        self.close_button.grid(row=0, column=7, sticky='e', padx=15, pady=5)


        # Create the open directory button
        self.opendirButton = Tk.Button(self.root, text='Load Data', bg = '#b2b2b2', command=self.openDirectory)
        self.opendirButton.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        # Create an empty label for the messages
        self.stringvar = Tk.StringVar()
        self.stringvar.set(" ")
        self.messageLab = Tk.Label(textvariable=self.stringvar, relief='flat', fg='red')
        self.messageLab.grid(row=1, column=0, sticky='w', padx=90, pady=5)

        ############################################################################################


        ############################################################################################
        ###  FIGURE 1 commands
        ############################################################################################

        # Define figure 1 area
        self.fig1, self.ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4.9, 0.68*self.screen_mmheight/25.4))
        plt.axis('off')
        # Set tight layout
        self.fig1.set_tight_layout(True)
        # Define and place the tk.DrawingArea
        self.canvas1_frame = Tk.Frame(self.root)
        self.canvas1_frame.grid(row=2,column=0, columnspan=2, rowspan=4, sticky='w',padx=3)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.canvas1_frame)
        self.toolbar1 = NavigationToolbar2TkAgg(self.canvas1, self.canvas1_frame)
        self.canvas1.get_tk_widget().pack(side=Tk.TOP)

        ############################################################################################


        ############################################################################################
        ###  FIGURE 2 commands
        ############################################################################################

        # Define figure 2 area
        self.fig2, self.ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4.6, 0.3*self.screen_mmheight/25.4))
        plt.axis('off')
        # Set tight layout
        self.fig2.set_tight_layout(True)
        # Define and place the tk.DrawingArea
        self.canvas2_frame = Tk.Frame(self.root)
        self.canvas2_frame.grid(row=2,column=2, columnspan=3, sticky='n',padx=3, pady=0)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.canvas2_frame)
        self.toolbar2 = NavigationToolbar2TkAgg(self.canvas2, self.canvas2_frame)
        self.canvas2.get_tk_widget().pack(side=Tk.TOP)

        # Create a label for the selected slice level
        self.stringvar2 = Tk.StringVar()
        self.stringvar2.set("")
        self.sliceLab = Tk.Label(textvariable=self.stringvar2, relief='flat') #, bg='white'
        self.sliceLab.grid(row=1, column=2, sticky='nw', padx=5, pady=0)

        # Create spinbox that will contain the min contract adjustment for the slice
        self.minCsliceSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.minCsliceSpinbox.grid(row=3, column=2, sticky='nw', padx=10, pady=0)
        self.minCsliceSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.minCsliceSpinbox.insert(0,0) # Insert starting point 0

        # Create spinbox that will contain the max contract adjustment for the slice
        self.maxCsliceSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.maxCsliceSpinbox.grid(row=3, column=2, sticky='nw', padx=55, pady=0)
        self.maxCsliceSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.maxCsliceSpinbox.insert(0,0) # Insert starting point 0

        # Create the adjust contrast slice button
        self.adjContSliceButton = Tk.Button(self.root, text='Adjust', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.adjustSliceContrast)
        self.adjContSliceButton.grid(row=3, column=3, sticky='nw', padx=0, pady=0)

        # Create the reset adjuctment slice button
        self.rstContSliceButton = Tk.Button(self.root, text='Reset', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.resetSliceContrast)
        self.rstContSliceButton.grid(row=3, column=3, sticky='nw', padx=40, pady=0)

        ############################################################################################


        ############################################################################################
        ###  FIGURE 3 commands
        ############################################################################################

        # Define figure 3 area
        self.fig3, self.ax3 = plt.subplots(nrows=1, ncols=1, figsize=(4.6, 0.3*self.screen_mmheight/25.4))
        plt.axis('off')
        # Set tight layout
        self.fig3.set_tight_layout(True)
        # Define and place the tk.DrawingArea
        self.canvas3_frame = Tk.Frame(self.root)
        self.canvas3_frame.grid(row=4,column=2, columnspan=3, sticky='n',padx=3)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.canvas3_frame)
        self.toolbar3 = NavigationToolbar2TkAgg(self.canvas3, self.canvas3_frame)
        self.canvas3.get_tk_widget().pack(side=Tk.TOP)

        # Create spinbox that will contain the min contract adjustment for the deconvolved slice
        self.minCdecSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.minCdecSpinbox.grid(row=5, column=2, sticky='nw', padx=10, pady=0)
        self.minCdecSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.minCdecSpinbox.insert(0,0) # Insert starting point

        # Create spinbox that will contain the max contract adjustment for the deconvolved slice
        self.maxCdecSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.maxCdecSpinbox.grid(row=5, column=2, sticky='nw', padx=55, pady=0)
        self.maxCdecSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.maxCdecSpinbox.insert(0,0) # Insert starting point

        # Create the adjust contrast deconvolved slice button
        self.adjContDecButton = Tk.Button(self.root, text='Adjust', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.adjustDecContrast)
        self.adjContDecButton.grid(row=5, column=3, sticky='nw', padx=0, pady=0)

        # Create the reset adjuctment deconvolved slice button
        self.rstContDecButton = Tk.Button(self.root, text='Reset', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.resetDecContrast)
        self.rstContDecButton.grid(row=5, column=3, sticky='nw', padx=40, pady=0)

        ############################################################################################


        ############################################################################################
        ###  FIGURE 4 commands
        ############################################################################################

        # Define figure 4 area
        self.fig4, self.ax4 = plt.subplots(nrows=1, ncols=1, figsize=(4.6, 0.3*self.screen_mmheight/25.4))
        plt.axis('off')
        # Set tight layout
        self.fig4.set_tight_layout(True)
        # Define and place the tk.DrawingArea
        self.canvas4_frame = Tk.Frame(self.root)
        self.canvas4_frame.grid(row=2,column=5, columnspan=3, sticky='n',padx=3)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.canvas4_frame)
        self.toolbar4 = NavigationToolbar2TkAgg(self.canvas4, self.canvas4_frame)
        self.canvas4.get_tk_widget().pack(side=Tk.TOP)

        # Create spinbox that will contain the min contract adjustment for the deconvolved slice
        self.minCblobSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.minCblobSpinbox.grid(row=3, column=5, sticky='nw', padx=10, pady=0)
        self.minCblobSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.minCblobSpinbox.insert(0,0) # Insert starting point

        # Create spinbox that will contain the max contract adjustment for the deconvolved slice
        self.maxCblobSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.maxCblobSpinbox.grid(row=3, column=5, sticky='nw', padx=55, pady=0)
        self.maxCblobSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.maxCblobSpinbox.insert(0,0) # Insert starting point

        # Create the adjust contrast blob-removed slice button
        self.adjContBlobButton = Tk.Button(self.root, text='Adjust', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.adjustBlobContrast)
        self.adjContBlobButton.grid(row=3, column=6, sticky='nw', padx=0, pady=0)

        # Create the reset adjuctment deconvolved slice button
        self.rstContBlobButton = Tk.Button(self.root, text='Reset', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.resetBlobContrast)
        self.rstContBlobButton.grid(row=3, column=6, sticky='nw', padx=40, pady=0)

        ############################################################################################


        ############################################################################################
        ###  FIGURE 5 commands
        ############################################################################################

        # Define figure 4 area
        self.fig5, self.ax5 = plt.subplots(nrows=1, ncols=1, figsize=(4.6, 0.3*self.screen_mmheight/25.4))
        plt.axis('off')
        # Set tight layout
        self.fig5.set_tight_layout(True)
        # Define and place the tk.DrawingArea
        self.canvas5_frame = Tk.Frame(self.root)
        self.canvas5_frame.grid(row=4,column=5, columnspan=3, sticky='n',padx=3)
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=self.canvas5_frame)
        self.toolbar5 = NavigationToolbar2TkAgg(self.canvas5, self.canvas5_frame)
        self.canvas5.get_tk_widget().pack(side=Tk.TOP)

        # Create spinbox that will contain the min contract adjustment for the deconvolved +blob-removed slice
        self.minCdecblobSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.minCdecblobSpinbox.grid(row=5, column=5, sticky='nw', padx=10, pady=0)
        self.minCdecblobSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.minCdecblobSpinbox.insert(0,0) # Insert starting point

        # Create spinbox that will contain the max contract adjustment for the deconvolved +blob-removed slice
        self.maxCdecblobSpinbox = Tk.Spinbox(self.root, width=5, from_=-1.0, to=1.0, increment=0.005)
        self.maxCdecblobSpinbox.grid(row=5, column=5, sticky='nw', padx=55, pady=0)
        self.maxCdecblobSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.maxCdecblobSpinbox.insert(0,0) # Insert starting point

        # Create the adjust contrast deconvolved +blob-removed slice button
        self.adjContDecblobButton = Tk.Button(self.root, text='Adjust', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.adjustDecblobContrast)
        self.adjContDecblobButton.grid(row=5, column=6, sticky='nw', padx=0, pady=0)

        # Create the reset adjuctment deconvolved +blob-removed  slice button
        self.rstContDecblobButton = Tk.Button(self.root, text='Reset', bg = '#b2b2b2', \
            font=("Arial", 7), pady=0, command=self.resetDecblobContrast)
        self.rstContDecblobButton.grid(row=5, column=6, sticky='nw', padx=40, pady=0)

        ############################################################################################




        ############################################################################################
        ###  Reconstruction commands
        ############################################################################################

        # Create the preview reconstruction button
        self.previewrecButton = Tk.Button(self.root, text='Preview Reconstruction', bg = '#b2b2b2', command=self.loadSino)
        self.previewrecButton.grid(row=6, column=0, sticky='w', padx=5, pady=10)

        # Create the preview progress bar
        self.progr1 = Progressbar(self.root, orient=Tk.HORIZONTAL, length=100, mode='determinate')
        self.progr1.grid(row=6, column=0, sticky='w', padx=150, pady=5)


        # Create spinbox containing the pixel to centre the sinogram
        self.cenSpinbox = Tk.Spinbox(self.root, width=5, from_=-50, to=50, increment=1)
        self.cenSpinbox.grid(row=7, column=0, sticky='w', padx=5, pady=3)
        self.cenSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.cenSpinbox.insert(0,0) # Insert starting point 0

        # Creat spinbox label
        self.cenSBlabel = Tk.Label(text=" Misalignment compensation", relief='flat',fg='black')
        self.cenSBlabel.grid(row=7, column=0, sticky='w', padx=50, pady=3)


        # Create spinbox containing the size of the reconstructed slice
        self.sizeSpinbox = Tk.Spinbox(self.root, width=5, from_=100, to=2000, increment=10)
        self.sizeSpinbox.grid(row=8, column=0, sticky='w', padx=5, pady=3)
        self.sizeSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.sizeSpinbox.insert(0,400) # Insert starting point 400

        # Create spinbox label
        self.sizeSBlabel = Tk.Label(text=" Size of the reconstructed slice", relief='flat',fg='black')
        self.sizeSBlabel.grid(row=8, column=0, sticky='w', padx=50, pady=3)

        ############################################################################################


        ############################################################################################
        ###  Preview deconvolution commands
        ############################################################################################

        # Create the preview deconvolution button
        self.previewdecButton = Tk.Button(self.root, text='Preview Deconvolution', bg = '#b2b2b2', command=self.dec_radon)
        self.previewdecButton.grid(row=6, column=2, sticky='w', padx=5, pady=10)

        # Create spinbox containing the value of the noise level
        self.noiseSpinbox = Tk.Spinbox(self.root, width=5, from_=0.005, to=0.5, increment=0.005)
        self.noiseSpinbox.grid(row=7, column=2, sticky='w', padx=5, pady=3)
        self.noiseSpinbox.delete(0,5) # delete all characters that were prep-populated
        self.noiseSpinbox.insert(0,0.05) # Insert starting point 400

        # Create spinbox label
        self.noiseSBlabel = Tk.Label(text=" Noise level", relief='flat',fg='black')
        self.noiseSBlabel.grid(row=7, column=2, sticky='w', padx=50, pady=3)

        # # Create spinbox containing the value of the denoising weight
        # self.denoiseSpinbox = Tk.Spinbox(self.root, width=5, from_=0.0, to=50.0, increment=0.1)
        # self.denoiseSpinbox.grid(row=8, column=2, sticky='w', padx=5, pady=3)
        # self.denoiseSpinbox.delete(0,5) # delete all characters that were pre-populated
        # self.denoiseSpinbox.insert(0,1.0) # Insert starting point

        # # Create spinbox label
        # self.denoiseSBlabel = Tk.Label(text=" Denoising weight (0 = no denoising)", \
        #     relief='flat',fg='black')
        # self.denoiseSBlabel.grid(row=8, column=2, sticky='w', padx=50, pady=3)


        #####################################################################################


        ############################################################################################
        ###  Commands for preview the blob correction
        ############################################################################################


        # Create the show sinogram button
        self.removeBlobSino = Tk.Button(self.root, text='Preview Remove Blobs', bg = '#b2b2b2', \
                                        command=self.remove_blob)
        self.removeBlobSino.grid(row=6, column=4, sticky='e', padx=5, pady=10)

        # Create spinbox containing the value of the sigma for the median filter
        self.sigmaSpinbox = Tk.Spinbox(self.root, width=5, from_=0, to=20, increment=1)
        self.sigmaSpinbox.grid(row=7, column=4, sticky='e', padx=5, pady=3)
        self.sigmaSpinbox.delete(0,5) # delete all characters that were pre-populated
        self.sigmaSpinbox.insert(0,5) # Insert starting value
        # Create spinbox label
        self.sigmaSBlabel = Tk.Label(text="Sigma   ", relief='flat',fg='black')
        self.sigmaSBlabel.grid(row=7, column=4, sticky='e', padx=50, pady=3)

        # # Create spinbox containing the value of the threshold for the mask
        # self.thSpinbox = Tk.Spinbox(self.root, width=5, from_=0, to=2000, increment=1)
        # self.thSpinbox.grid(row=8, column=3, sticky='e', padx=5, pady=3)
        # self.thSpinbox.delete(0,5) # delete all characters that were pre-populated
        # self.thSpinbox.insert(0,100) # Insert starting value
        # # Create spinbox label
        # self.thSBlabel = Tk.Label(text="Threshold   ", relief='flat',fg='black')
        # self.thSBlabel.grid(row=8, column=3, sticky='e', padx=50, pady=3)


        ############################################################################################
        ###  Commands for the combined preview
        ############################################################################################


        # Create the combined preview button
        self.combinedPreview = Tk.Button(self.root, text='Deconvolution + Blob rem.', bg = '#b2b2b2', \
                                        command=self.dec_blob)
        self.combinedPreview.grid(row=6, column=5, sticky='w', padx=5, pady=10)

        # # Create spinbox containing the value of the sigma for the median filter
        # self.sigmaSpinbox = Tk.Spinbox(self.root, width=5, from_=0, to=20, increment=1)
        # self.sigmaSpinbox.grid(row=7, column=3, sticky='e', padx=5, pady=3)
        # self.sigmaSpinbox.delete(0,5) # delete all characters that were pre-populated
        # self.sigmaSpinbox.insert(0,5) # Insert starting value
        # # Create spinbox label
        # self.sigmaSBlabel = Tk.Label(text="Sigma   ", relief='flat',fg='black')
        # self.sigmaSBlabel.grid(row=7, column=3, sticky='e', padx=50, pady=3)





        ############################################################################################
        ###  Process scan commands
        ############################################################################################


        # Create the run deconvolution button
        self.rundecButton = Tk.Button(self.root, text='Process scan', bg = '#6AABD8', \
            command=self.run_dec_series)
        self.rundecButton.grid(row=6, column=6, sticky='e', padx=5, pady=10)

        # Create spinbox containing the value of the bottom slice to reconstruct
        self.botSpinbox = Tk.Spinbox(self.root, width=5, from_=0, to=2000, increment=1)
        self.botSpinbox.grid(row=7, column=6, sticky='e', padx=5, pady=3)
        self.botSpinbox.delete(0,5) # delete all characters that were pre-populated
        self.botSpinbox.insert(0,0) # Insert starting point

        # Create spinbox label
        self.botSBlabel = Tk.Label(text="Lower slice   ", relief='flat',fg='black')
        self.botSBlabel.grid(row=7, column=6, sticky='e', padx=50, pady=3)

        # Create spinbox containing the value of the top slice to reconstruct
        self.topSpinbox = Tk.Spinbox(self.root, width=5, from_=0, to=2000, increment=1)
        self.topSpinbox.grid(row=8, column=6, sticky='e', padx=5, pady=3)
        self.topSpinbox.delete(0,5) # delete all characters that were pre-populated
        self.topSpinbox.insert(0,0) # Insert starting point 400

        # Create spinbox label
        self.topSBlabel = Tk.Label(text="Upper slice   ", relief='flat',fg='black')
        self.topSBlabel.grid(row=8, column=6, sticky='e', padx=50, pady=3)


        # Create the run deconvolution progress bar
        self.progr2 = Progressbar(self.root, orient=Tk.HORIZONTAL, length=80, mode='determinate')
        self.progr2.grid(row=6, column=7, sticky='e', padx=15, pady=5)

        self.cb1var =Tk.IntVar()
        self.cbutton1 = Tk.Checkbutton(self.root, text="Deconvol.      ", variable=self.cb1var)
        self.cbutton1.grid(row=7, column=7, sticky='e', padx=8, pady=0)

        self.cb2var =Tk.IntVar()
        self.cbutton2 = Tk.Checkbutton(self.root, text="Blob removal", variable=self.cb2var)
        self.cbutton2.grid(row=8, column=7, sticky='e', padx=10, pady=0)


        #####################################################################################
        # Pulldown menu
        #####################################################################################

        self.menubar = Tk.Menu(self.root)

        # create File pulldown menu, and add it to the menu bar
        self.filemenu = Tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load Data", command = self.openDirectory)
        self.filemenu.add_command(label="Quit", command = self._quit)
        self.menubar.add_cascade(label="File", menu = self.filemenu)

        # create About pulldown menu, and add it to the menu bar
        self.aboutmenu = Tk.Menu(self.menubar, tearoff=0)
        self.aboutmenu.add_command(label="About")
        self.menubar.add_cascade(label="Help", menu = self.aboutmenu)

        # display the menu
        self.root.config(menu = self.menubar)

        #######################################################################################
        #######################################################################################


        self.root.mainloop()



    def _quit(self):
    	# stops mainloop
        self.root.quit()
        # Destroy all windows, necessary on Windows to prevent Fatal Python Error
        self.root.destroy()

    def onclick(self, event):
        #self.toolbar = plt.get_current_fig_manager().toolbar
        #self.tog = self.fig1.canvas.manager.toolmanager.active_toggle
        # Select the point with double click only
        if event.dblclick:
            self.ix, self.iy = event.xdata, event.ydata

            # Draw line (flat rectangle) and delete the previous if present.
            # Update selected slice label
            try:
                self.rect
            except:
                self.rect = matplotlib.patches.Rectangle((0,int(self.iy)),self.nx,1, \
                        linewidth=1,edgecolor='r',facecolor='none')
                self.ax1.add_patch(self.rect)
                self.stringvar2.set("Slice selected "+str(int(self.iy)))
                self.root.update_idletasks()

            else:
                self.rect.remove()
                self.rect = matplotlib.patches.Rectangle((0,int(self.iy)),self.nx,1, \
                        linewidth=1,edgecolor='r',facecolor='none')
                self.ax1.add_patch(self.rect)
                self.stringvar2.set("Slice selected "+str(int(self.iy)))
                self.root.update_idletasks()

            # Append the coordinates if selected within the image area.
            if self.ix is None:
                return
            else:
                self.coords.append((self.ix, self.iy))

    def AreYouSure(self):
        self.root.update_idletasks()

        if int(self.cb1var.get()) == 1 and int(self.cb2var.get()) == 1:
            self.mbox = messagebox.askokcancel("Deconvolution and blob removal OPT scan",
                "  Parameters are set to:"   +"\n "+ \
                " Noise level (deconvolution) = "+self.noiseSpinbox.get()         +"\n "+ \
                " Sigma (blob removal) = "+self.sigmaSpinbox.get() +"\n "+ \
                " Lower slice = "+self.botSpinbox.get()          +"\n "+ \
                " Upper slice = "+self.topSpinbox.get()          +"\n "+ \
                                                                  "\n "+ \
                " Continue?")
        if int(self.cb1var.get()) == 0 and int(self.cb2var.get()) == 1:
            self.mbox = messagebox.askokcancel("Blob removal OPT scan",
                "  Parameters are set to:"   +"\n "+ \
                #" Noise level (deconvolution) = "+self.noiseSpinbox.get()         +"\n "+ \
                " Sigma (blob removal) = "+self.sigmaSpinbox.get() +"\n "+ \
                " Lower slice = "+self.botSpinbox.get()          +"\n "+ \
                " Upper slice = "+self.topSpinbox.get()          +"\n "+ \
                                                                  "\n "+ \
                " Continue?")

        if int(self.cb1var.get()) == 1 and int(self.cb2var.get()) == 0:
            self.mbox = messagebox.askokcancel("Deconvolution OPT scan",
                "  Parameters are set to:"   +"\n "+ \
                " Noise level (deconvolution) = "+self.noiseSpinbox.get()         +"\n "+ \
                #" Sigma (blob removal) = "+self.denoiseSpinbox.get() +"\n "+ \
                " Lower slice = "+self.botSpinbox.get()          +"\n "+ \
                " Upper slice = "+self.topSpinbox.get()          +"\n "+ \
                                                                  "\n "+ \
                " Continue?")

        return self.mbox

    def AlreadyExists(self):
        self.root.update_idletasks()
        self.mbox = messagebox.askokcancel("The folder already exists",
        	"The deconvolution folder already exists:"   +"\n "+ \
            "and it contains the log file. \n "+ \
            "If you continue all file will be overwritten. \n "+ \
                                                              "\n "+ \
            " Continue?")

        return self.mbox

    def adjustSliceContrast(self):

        try:
            self.slice
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = float(self.minCsliceSpinbox.get())
            self.maxC = float(self.maxCsliceSpinbox.get())

            # Display the slice with adjusted contrast
            self.ax2.imshow(self.slice, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig2.canvas.draw()

    def resetSliceContrast(self):

        try:
            self.slice
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = np.min(self.slice)
            self.maxC = np.max(self.slice)

            # Display the slice with adjusted contrast
            self.ax2.imshow(self.slice, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig2.canvas.draw()

            # Reset values of the spinboxes
            self.minCsliceSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.minCsliceSpinbox.insert(0,int(np.min(self.slice))) # Insert starting point
            self.maxCsliceSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.maxCsliceSpinbox.insert(0,int(np.max(self.slice))) # Insert starting point


    def adjustDecContrast(self):

        try:
            self.decdenslice
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = float(self.minCdecSpinbox.get())
            self.maxC = float(self.maxCdecSpinbox.get())

            # Display the slice with adjusted contrast
            self.ax3.imshow(self.decdenslice, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig3.canvas.draw()

    def resetDecContrast(self):

        try:
            self.decdenslice
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = np.min(self.decdenslice)
            self.maxC = np.max(self.decdenslice)

            # Display the slice with adjusted contrast
            self.ax3.imshow(self.decdenslice, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig3.canvas.draw()

            # Reset values of the spinboxes
            self.minCdecSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.minCdecSpinbox.insert(0,int(np.min(self.decdenslice))) # Insert starting point
            self.maxCdecSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.maxCdecSpinbox.insert(0,int(np.max(self.decdenslice))) # Insert starting point

    def adjustBlobContrast(self):

        try:
            self.slicenoBlobs
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = float(self.minCblobSpinbox.get())
            self.maxC = float(self.maxCblobSpinbox.get())

            # Display the slice with adjusted contrast
            self.ax4.imshow(self.slicenoBlobs, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig4.canvas.draw()

    def resetBlobContrast(self):

        try:
            self.slicenoBlobs
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = np.min(self.slicenoBlobs)
            self.maxC = np.max(self.slicenoBlobs)

            # Display the slice with adjusted contrast
            self.ax4.imshow(self.slicenoBlobs, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig4.canvas.draw()

            # Reset values of the spinboxes
            self.minCblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.minCblobSpinbox.insert(0,int(np.min(self.slicenoBlobs))) # Insert starting point
            self.maxCblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.maxCblobSpinbox.insert(0,int(np.max(self.slicenoBlobs))) # Insert starting point



    def adjustDecblobContrast(self):

        try:
            self.decslicem
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = float(self.minCdecblobSpinbox.get())
            self.maxC = float(self.maxCdecblobSpinbox.get())

            # Display the slice with adjusted contrast
            self.ax5.imshow(self.decslicem, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig5.canvas.draw()

    def resetDecblobContrast(self):

        try:
            self.decslicem
        except:
            return
        else:
            # Get the values from the spinbox
            self.minC = np.min(self.decslicem)
            self.maxC = np.max(self.decslicem)

            # Display the slice with adjusted contrast
            self.ax5.imshow(self.decslicem, cmap='gray_r', vmin = self.minC, vmax = self.maxC)
            self.fig5.canvas.draw()

            # Reset values of the spinboxes
            self.minCdecblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.minCdecblobSpinbox.insert(0,int(np.min(self.decslicem))) # Insert starting point
            self.maxCdecblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.maxCdecblobSpinbox.insert(0,int(np.max(self.decslicem))) # Insert starting point




    def openDirectory(self):

        #self.dir = askdirectory(initialdir = "E:\\OPT Scans\\Daniele\\02Oct2017\\wallaby7113M-kid2\\7113M-k2_red\\")
        self.dir = askdirectory(initialdir = "E:\\OPT Scans\Daniele\\30Oct2017_5822_blobcorrection\\w5822_red\\")
        self.stringvar.set(" ")
        self.stringvar.set("Loading files...")

        #Check if a valid directory is selected
        if len(self.dir) > 0:
            # Check if the log file is present
            self.log = glob.glob(self.dir+'\\*log')[0]

            if len(self.log) > 0:
                self.f = open(self.log, 'r')
                self.logtext = self.f.read().split('\n')
                self.f.close()

                # Read the pixel size
                self.pix = float(self.logtext[5].split('=')[-1])

                # Read if the scan is 360 (YES) or 180 (NO)
                self.whichrotation = self.logtext[14].split('=')[-1]
                if self.whichrotation == 'NO':
                    self.stringvar.set(" ")
                    self.stringvar.set("The scan is 180 degree only. Can't correct this scan.")
                else:
                	# Read the file list
                    self.fnames = glob.glob(self.dir+'\\*_0*tif')
                    # Load the first image in the list
                    self.r = tif.imread(self.fnames[0])
                    # Get the size of the image
                    self.nx, self.ny = self.r.shape
                    # Get the number of images
                    self.nangles = len(self.fnames)

                    # Display the image
                    self.ax1.imshow(self.r, cmap='gray_r')

                    self.fig1.canvas.draw()
                    # Activate the cursor click
                    self.coords = []
                    #self.cursor1 = Cursor(self.ax1, vertOn=False, useblit=True, color='red', linewidth=1)
                    self.cid = self.fig1.canvas.mpl_connect('button_press_event', self.onclick)

                    self.stringvar.set(" ")

                    # Update upper slice spinbox
                    self.topSpinbox.delete(0,5) # delete all characters that were pre-populated
                    self.topSpinbox.insert(0,self.nx) # Insert starting point 400
                    self.topSpinbox['to_'] = self.nx
            else:
                if platform.system() == 'Windows':
                    winsound.PlaySound("*", winsound.SND_ALIAS)
                self.stringvar.set(" ")
                self.stringvar.set("No valid log file present. Operation cancelled.")
        else:
            self.stringvar.set(" ")
            self.stringvar.set("No directory selected. Operation cancelled.")

    def runFBP(self, sinog):
        ''' Filtered Back Projection routine'''

        if int(self.cenSpinbox.get()) == 0:
            self.shift = sino_centering(sinog)
        else:
            self.shift = int(self.cenSpinbox.get())
        if AST is True:
            slic = iradon_astra(np.roll(sinog,self.shift, axis=0), \
                         theta = np.linspace(0,360, len(self.fnames)), output_size = int(self.sizeSpinbox.get()))
        else:
            slic = iradon(np.roll(sinog,self.shift, axis=0), \
                         theta = np.linspace(0,360, len(self.fnames)), output_size = int(self.sizeSpinbox.get()))

        # Update value of the spinbox
        self.cenSpinbox.delete(0,5)
        self.cenSpinbox.insert(0,self.shift)

        return slic

    def denoise(self, array):

        if float(self.denoiseSpinbox.get()) != 0.0:
            array_denoise = denoise_tv_chambolle(array, weight=float(self.denoiseSpinbox.get()))
        else:
            array_denoise = array

        return array_denoise

    def runDec(self, sinog):

        self.noise = float(self.noiseSpinbox.get())
        if 'pycuda.autoinit' in sys.modules:
            decsino = runDeconvolutionGPU(sinog, self.pix, noise_level=self.noise)
        else:
            decsino = runDeconvolutionCPU(sinog, self.pix, noise_level=self.noise)

        return decsino

    def loadSino(self):

        # Load sinogram data at the specified position, run FBP and display data

        # Check if data are loaded
        try:
            self.log

            # Check if the sinogram position has been selected
            if (len(self.coords)==0):
                self.messageLab.config(bg="white")
                winsound.PlaySound("*", winsound.SND_ALIAS)
                self.stringvar.set(" ")
                self.stringvar.set("Position not selected!")
                self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))
            else:
            	# Set the message text color as black
                self.messageLab.config(fg="black")
                # Reset the text to null
                self.stringvar.set(" ")
                self.root.update_idletasks()
                # Allocate the array for the sinogram
                self.sino = np.zeros((self.ny, self.nangles))

                # Check if the csv file containing the xy correction is present
                self.base = os.path.basename(self.fnames[0]).index('_0')
                self.bn =  os.path.basename(self.fnames[0])[:self.base+1]
                self.fol = os.path.dirname(self.fnames[0])
                try:

                    self.xy = pd.read_csv(self.fol+"/"+self.bn+"_TS.csv", skiprows=2)
                    self.xs = self.xy[" Y1"].values
                    self.ys = self.xy[" Y2"].values
                    for i,j in enumerate(self.fnames):
                        # Load image and assign it to sinogram line
                        #self.sino[:,i] = tif.imread(j)[int(self.iy),:]
                        self.sino[:,i] = np.roll(np.roll(tif.imread(j), int(self.xs[i]), axis=1), \
                                          int(self.ys[i]), axis=0)[int(self.iy),:]

                        # Update the message
                        self.stringvar.set("Preview reconstruction with xy correction "+str(int(100.0*(i+1)/len(self.fnames)))+"% complete")
                        self.progr1['value'] = int(100*(i+1)/len(self.fnames))
                        self.root.update_idletasks()
                        self.root.update()

                except:
                    for i,j in enumerate(self.fnames):
                        # Load image and assign it to sinogram line
                        self.sino[:,i] = tif.imread(j)[int(self.iy),:]

                        # Update the message
                        self.stringvar.set("Preview reconstruction "+str(int(100.0*(i+1)/len(self.fnames)))+"% complete")
                        self.progr1['value'] = int(100*(i+1)/len(self.fnames))
                        self.root.update_idletasks()
                        self.root.update()

                # Run FBP reconstruction
                self.slice = self.runFBP(self.sino)

                # Display the reconstructed slice
                self.ax2.imshow(self.slice, cmap='gray_r')
                self.fig2.canvas.draw()


                # Update spinboxes
                self.minCsliceSpinbox['from_'] = min( int(2*np.min(self.slice)), int(0.5*np.min(self.slice)))
                self.maxCsliceSpinbox['from_'] = min( int(2*np.min(self.slice)), int(0.5*np.min(self.slice)))
                self.minCsliceSpinbox['to_'] = max(int(2*np.max(self.slice)), int(0.5*np.max(self.slice)))
                self.maxCsliceSpinbox['to_'] = max(int(2*np.max(self.slice)), int(0.5*np.max(self.slice)))

                self.minCsliceSpinbox['increment'] = (np.max(self.slice)-np.min(self.slice))/255
                self.maxCsliceSpinbox['increment'] = (np.max(self.slice)-np.min(self.slice))/255

                self.minCsliceSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.minCsliceSpinbox.insert(0,int(np.min(self.slice))) # Insert starting point
                self.maxCsliceSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.maxCsliceSpinbox.insert(0,int(np.max(self.slice))) # Insert starting point



        except AttributeError:
            self.messageLab.config(bg="white")
            winsound.PlaySound("*", winsound.SND_ALIAS)
            self.stringvar.set(" ")
            self.stringvar.set("Data not loaded yet!")
            self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))


    def dec_radon(self):


        # Check if data are loaded
        try:
            self.log

        except AttributeError:
            winsound.PlaySound("*", winsound.SND_ALIAS)
            self.messageLab.config(bg="white")
            self.stringvar.set(" ")
            self.stringvar.set("Data not loaded yet!")
            self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

        else:
            # Check if the normal slice has been reconstructed. If yes, run dec_radon
            try:
                self.slice

            except:
                winsound.PlaySound("*", winsound.SND_ALIAS)
                self.messageLab.config(bg="white")
                self.stringvar.set(" ")
                self.stringvar.set("Preview reconstruction first!")
                self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

            else:

                # Run deconvolution
                self.decsino = self.runDec(self.sino)

                # Run FBP reconstruction
                self.decslice = self.runFBP(self.decsino)

                # Denoising
                self.decdenslice = self.decslice#self.denoise(self.decslice)

                # Display the reconstructed slice
                self.ax3.imshow(self.decdenslice, cmap='gray_r')
                self.fig3.canvas.draw()

                # Update spinboxes
                self.minCdecSpinbox['from_'] = min( 2*np.min(self.decdenslice), 0.5*np.min(self.decdenslice))
                self.maxCdecSpinbox['from_'] = min( 2*np.min(self.decdenslice), 0.5*np.min(self.decdenslice))
                self.minCdecSpinbox['to_'] = max(2*np.max(self.decdenslice), 0.5*np.max(self.decdenslice))
                self.maxCdecSpinbox['to_'] = max(2*np.max(self.decdenslice), 0.5*np.max(self.decdenslice))

                self.minCdecSpinbox['increment'] = (np.max(self.decdenslice)-np.min(self.decdenslice))/255
                self.maxCdecSpinbox['increment'] = (np.max(self.decdenslice)-np.min(self.decdenslice))/255

                self.minCdecSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.minCdecSpinbox.insert(0,int(np.min(self.decdenslice))) # Insert starting point
                self.maxCdecSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.maxCdecSpinbox.insert(0,int(np.max(self.decdenslice))) # Insert starting point




    def dec_series(self):

        # Copy the log file across
        copyfile(self.log, self.dec_dir+'\\'+os.path.basename(self.log))

        # create memmap
        self.sinomm = np.memmap(self.dec_dir+'sino_memmap', dtype='float32', \
                      mode='w+', shape=(self.sino.shape[0], self.sino.shape[1], self.hi-self.low))

        # Load sinogram
        for k in range(self.low, self.hi, 1):
            #print(k)

            self.newsino = np.zeros_like(self.sino)
            for i,j in enumerate(self.fnames):
                self.newsino[:,i] = tif.imread(j)[k,:]

            self.decnewsino = np.copy(self.newsino)

            if int(self.cb1var.get()) == 1:
                # Deconvolve sinogram
                self.decnewsino = self.runDec(self.decnewsino)
                #print("deconvolution")

            if int(self.cb2var.get()) == 1:
                # Remove blob
                self.decnewsino = remove_blob_sino_wavelet(self.decnewsino, sigma=int(self.sigmaSpinbox.get()))
                #print("blob removal")

            # Populate the memory map with the deconcolved sino data
            self.sinomm[:,:,k-self.low] = self.decnewsino.astype('float32')

            # Update progress bar
            self.progr2['value'] = int(100.0*(k-self.low+1)/self.sinomm.shape[2])
            self.root.update_idletasks()
            self.stringvar.set("Saving deconvolved sinograms "+str(int(100.0*(k-self.low+1)/self.sinomm.shape[2]))+"% complete" )
            self.root.update_idletasks()
            self.root.update()

        # Get the range of the deconvolved sinogram (it may be larger than 16-bit)
        self.sinommrange = np.abs(np.max(self.sinomm)-np.min(self.sinomm))
        #print("corrected sinogram range: "+str(self.sinommrange))
        # Set the minimum to zero
        self.sinomm = self.sinomm - np.min(self.sinomm)
        # Rescale to 16-bit if required
        if self.sinommrange > 65537.0:
            self.sinomm = self.sinomm/(self.sinommrange/65000)

        # Save projections (reslice the memory map)
        for k in range(self.nangles):
            self.newproj = (self.sinomm[:,k,:].T).astype('uint16')
            tif.imsave(self.dec_dir+os.path.basename(self.log)[:-4]+str(k).zfill(4)+'.tif', self.newproj)


            # Update progress bar
            self.progr2['value'] = int(100.0*(k+1)/self.nangles)
            self.root.update_idletasks()
            self.stringvar.set("Saving new projections "+str(int(100.0*(k+1)/self.nangles))+"% complete" )
            self.root.update_idletasks()
            self.root.update()

        # Delete memory map
        del self.sinomm
        os.remove(self.dec_dir+'sino_memmap')

    def run_dec_series(self):

        # Check if data are loaded
        try:
            self.log

        except AttributeError:
            winsound.PlaySound("*", winsound.SND_ALIAS)
            self.messageLab.config(bg="white")
            self.stringvar.set(" ")
            self.stringvar.set("Data not loaded yet!")
            self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

        else:
                        # Check if at least one option is selected
            if int(self.cb2var.get()) + int(self.cb1var.get()) > 0:

                # try:
                #     self.decsino
                # except:
                #     winsound.PlaySound("*", winsound.SND_ALIAS)
                #     self.messageLab.config(bg="white")
                #     self.stringvar.set(" ")
                #     self.stringvar.set("Select at least one option: Deconvol. - Blob removal")
                #     self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

                # else:
                self.go = self.AreYouSure()
                if self.go is True:
                    self.low = int(self.botSpinbox.get())
                    self.hi =  int(self.topSpinbox.get())
                    self.dec_dir = self.dir+'\\deconvolution\\'

                    if os.path.exists(self.dec_dir+'\\'+os.path.basename(self.log)) or os.path.exists(self.dec_dir+'\\'):
                        self.exist = self.AlreadyExists()
                        if self.exist is True:

                            self.dec_series()

                            self.stringvar.set(" ")
                            self.stringvar.set("Done!")

                    else:
                        os.makedirs(self.dec_dir)

                        self.dec_series()

                        self.stringvar.set(" ")
                        self.stringvar.set("Done!")

            else:
                winsound.PlaySound("*", winsound.SND_ALIAS)
                self.messageLab.config(bg="white")
                self.stringvar.set(" ")
                self.stringvar.set("Select at least one option: Deconvol. - Blob removal")
                self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))


    def remove_blob(self):

        # Check if data are loaded
        try:
            self.log

        except AttributeError:
            winsound.PlaySound("*", winsound.SND_ALIAS)
            self.messageLab.config(bg="white")
            self.stringvar.set(" ")
            self.stringvar.set("Data not loaded yet!")
            self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

        else:
            self.sinom = remove_blob_sino_wavelet(self.sino, sigma=int(self.sigmaSpinbox.get()))

            self.slicenoBlobs = self.runFBP(self.sinom)
            #self.slicenoBlobs = self.slicenoBlobs*np.mean(self.slice)

            self.ax4.imshow(self.slicenoBlobs, cmap='gray_r')
            self.fig4.canvas.draw()

            # Update spinboxes
            self.minCblobSpinbox['from_'] = min( int(2*np.min(self.slicenoBlobs)), int(0.5*np.min(self.slicenoBlobs)))
            self.maxCblobSpinbox['from_'] = min( int(2*np.min(self.slicenoBlobs)), int(0.5*np.min(self.slicenoBlobs)))
            self.minCblobSpinbox['to_'] = max(int(2*np.max(self.slicenoBlobs)), int(0.5*np.max(self.slicenoBlobs)))
            self.maxCblobSpinbox['to_'] = max(int(2*np.max(self.slicenoBlobs)), int(0.5*np.max(self.slicenoBlobs)))

            self.minCblobSpinbox['increment'] = (np.max(self.slicenoBlobs)-np.min(self.slicenoBlobs))/255
            self.maxCblobSpinbox['increment'] = (np.max(self.slicenoBlobs)-np.min(self.slicenoBlobs))/255

            self.minCblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.minCblobSpinbox.insert(0,int(np.min(self.slicenoBlobs))) # Insert starting point
            self.maxCblobSpinbox.delete(0,5) # delete all characters that were pre-populated
            self.maxCblobSpinbox.insert(0,int(np.max(self.slicenoBlobs))) # Insert starting point


    def dec_blob(self):

        # Check if data are loaded
        try:
            self.log

        except AttributeError:
            winsound.PlaySound("*", winsound.SND_ALIAS)
            self.messageLab.config(bg="white")
            self.stringvar.set(" ")
            self.stringvar.set("Data not loaded yet!")
            self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

        else:

            # Check if the normal slice has been reconstructed. If yes, run dec_radon
            try:
                self.slice

            except:
                winsound.PlaySound("*", winsound.SND_ALIAS)
                self.messageLab.config(bg="white")
                self.stringvar.set(" ")
                self.stringvar.set("Preview reconstruction first!")
                self.messageLab.after(700, lambda: self.messageLab.config(bg=self.bgcol))

            else:

                # Run deconvolution
                self.decsino = self.runDec(self.sino)

                # Run blob removal first
                self.sinom = remove_blob_sino_wavelet(self.decsino, sigma=int(self.sigmaSpinbox.get()))

                # Run FBP reconstruction
                self.decslicem = self.runFBP(self.sinom)

                #self.decslicem = self.decslicem*np.mean(self.slice)

                # Display the reconstructed slice
                self.ax5.imshow(self.decslicem, cmap='gray_r')
                self.fig5.canvas.draw()

                # Update spinboxes
                self.minCdecblobSpinbox['from_'] = min( int(2*np.min(self.decslicem)), int(0.5*np.min(self.decslicem)))
                self.maxCdecblobSpinbox['from_'] = min( int(2*np.min(self.decslicem)), int(0.5*np.min(self.decslicem)))
                self.minCdecblobSpinbox['to_'] = max(int(2*np.max(self.decslicem)), int(0.5*np.max(self.decslicem)))
                self.maxCdecblobSpinbox['to_'] = max(int(2*np.max(self.decslicem)), int(0.5*np.max(self.decslicem)))

                self.minCdecblobSpinbox['increment'] = (np.max(self.decslicem)-np.min(self.decslicem))/255
                self.maxCdecblobSpinbox['increment'] = (np.max(self.decslicem)-np.min(self.decslicem))/255

                self.minCdecblobSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.minCdecblobSpinbox.insert(0,int(np.min(self.decslicem))) # Insert starting point
                self.maxCdecblobSpinbox.delete(0,5) # delete all characters that were pre-populated
                self.maxCdecblobSpinbox.insert(0,int(np.max(self.decslicem))) # Insert starting point




dec = deconvolution()
