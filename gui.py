import wx  # pandas only works with wxPython 4.0.7
import os
from matplotlib import pyplot as plt
from rheoplots.plotting import DynamicCompression
from rheoplots.plotting import Sweep

# TODO: create an executable. pyInstaller

plottypes = [
    'Amplitude sweeps | Stress sweep',
    'Oscillatory | Frequency sweep',
    'Dynamic compression | Full',
    'Dynamic compression | Cyclic'
]


class PlotDlg(wx.Dialog):
    def __init__(self, parent, title, data_path):
        self.title = title
        super().__init__(
            parent,
            title=self.title,
            style=wx.DEFAULT_DIALOG_STYLE)

        self.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False, ' Helvetica Neue'))

        # panel = wx.Panel(self, size=(500, 500))
        self.data_path = data_path

        # Dialog sizers
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.mainPlot_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Plot configuration')

        self.txt_sizer = wx.FlexGridSizer(2, 2, 5, 5)

        self.colorSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.color1 = (30, 144, 255, 255)
        self.color2 = (255, 105, 180, 255)

        # Dialog elements
        self.ctrl_size = (50, -1)

        self.txt_nCycles = wx.StaticText(self, -1, 'Number of cycles/periods:')
        self.ctrl_nCycles = wx.TextCtrl(self, -1, '3', size=self.ctrl_size)

        self.txt_dpi = wx.StaticText(self, -1, 'Figure resolution (dpi):')
        self.ctrl_dpi = wx.TextCtrl(self, -1, '300', size=self.ctrl_size)

        self.colorButton1 = wx.Button(
            self, 1,
            'Color 1', size=(-1, -1))
        self.colorButton1.SetBackgroundColour(self.color1)

        self.colorButton2 = wx.Button(
            self, 2,
            'Color 2', size=(-1, -1))
        self.colorButton2.SetBackgroundColour(self.color2)

        self.plotButton = wx.Button(
            self, 3,
            'Plot', size=(-1, -1))

        # Variables: Dynamic oscillation - full
        self.cb_displacFit, self.cb_displacExp, self.cb_dampedFit, self.cb_absoluFit = None, None, None, None

        # Variables: Dynamic oscillation - cyclic
        self.txt_peakSize, self.txt_initStrain, self.txt_finStrain = None, None, None
        self.ctrl_peakSize, self.ctrl_initStrain, self.ctrl_finStrain = None, None, None
        self.cb_plotPeak, self.cb_plotYM = None, None

        # Events
        self.Bind(wx.EVT_BUTTON, self.OnColor1, id=1)
        self.Bind(wx.EVT_BUTTON, self.OnColor2, id=2)
        self.Bind(wx.EVT_BUTTON, self.OnPlot, id=3)

    def dynamicFull(self):
        self.cb_displacExp = wx.CheckBox(self, -1, 'Experimental height data.', (10, 10))
        self.cb_displacFit = wx.CheckBox(self, -1, 'Fitted height data.', (10, 10))
        self.cb_dampedFit = wx.CheckBox(self, -1, 'Stress vs. Strain fit: Damped sine wave.', (10, 10))
        self.cb_absoluFit = wx.CheckBox(self, -1, 'Stress vs. Strain fit: Absolute sine wave.', (10, 10))

        self.mainPlot_sizer.AddMany((
            (self.cb_displacExp, 0, wx.ALL, 10),
            (self.cb_displacFit, 0, wx.ALL, 10),
            (self.cb_dampedFit, 0, wx.ALL, 10),
            (self.cb_absoluFit, 0, wx.ALL, 10)
        ))

        self.colorButton1.SetLabel('Stress')
        self.colorButton2.SetLabel('Height')
        self.init_gui()

    def dynamicCyclic(self):
        self.txt_sizer = wx.FlexGridSizer(5, 2, 5, 5)

        self.txt_peakSize = wx.StaticText(self, -1, 'Stress peak range:')
        self.ctrl_peakSize = wx.TextCtrl(self, -1, '3', size=self.ctrl_size)
        self.txt_initStrain = wx.StaticText(self, -1, 'Initial strain linear region:')
        self.ctrl_initStrain = wx.TextCtrl(self, -1, '10', size=self.ctrl_size)
        self.txt_finStrain = wx.StaticText(self, -1, 'Final strain linear region:')
        self.ctrl_finStrain = wx.TextCtrl(self, -1, '18', size=self.ctrl_size)

        self.txt_sizer.AddMany((
            (self.txt_peakSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.ctrl_peakSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.txt_initStrain, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.ctrl_initStrain, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.txt_finStrain, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.ctrl_finStrain, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        ))

        self.cb_plotPeak = wx.CheckBox(self, -1, 'Highlight peak region.', (10, 10))
        self.cb_plotYM = wx.CheckBox(self, -1, "Show Young's Modulus linear fit.", (10, 10))

        self.mainPlot_sizer.AddMany((
            (self.cb_plotPeak, 0, wx.ALL, 10),
            (self.cb_plotYM, 0, wx.ALL, 10)
        ))

        self.colorButton1.SetLabel('Stress')
        self.colorButton2.SetLabel('Fitted curve')
        self.init_gui()

    def stressSweep(self):
        self.colorButton1.SetLabel('Storage Modulus')
        self.colorButton2.SetLabel('Loss Modulus')
        self.init_gui()

    def oscilSweep(self):
        self.colorButton1.SetLabel('Storage Modulus')
        self.colorButton2.SetLabel('Loss Modulus')
        self.init_gui()

    def init_gui(self):
        self.txt_sizer.AddMany((
            (self.txt_nCycles, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.ctrl_nCycles, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.txt_dpi, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5),
            (self.ctrl_dpi, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        ))

        self.mainPlot_sizer.Add(
            self.txt_sizer, 1,
            wx.EXPAND | wx.ALL, 15)

        self.colorSizer.Add(
            self.colorButton1, 1,
            wx.EXPAND | wx.ALL, 0)

        self.colorSizer.Add(
            self.colorButton2, 1,
            wx.EXPAND | wx.ALL, 0)

        self.mainPlot_sizer.Add(
            self.colorSizer, 0,
            wx.EXPAND | wx.ALL, 15)

        self.mainPlot_sizer.Add(
            self.plotButton, 0,
            wx.EXPAND | wx.ALL, 5)
        self.plotButton.Enable(True)

        self.main_sizer.Add(
            self.mainPlot_sizer, 1,
            wx.EXPAND | wx.ALL, 20)

        self.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self)
        self.Layout()

    def OnPlot(self, e):
        if self.title == plottypes[0]:
            print(f'Plotting {self.title}...')
            Sweep.stress(Sweep(data_path=self.data_path),
                         colorStorage=tuple(c / 255 for c in self.color1),
                         colorLoss=tuple(c / 255 for c in self.color2))
            plt.show()

        if self.title == plottypes[1]:
            print(f'Plotting {self.title}...')
            Sweep.oscilatory(Sweep(data_path=self.data_path),
                             colorStorage=tuple(c / 255 for c in self.color1),
                             colorLoss=tuple(c / 255 for c in self.color2))
            plt.show()

        if self.title == plottypes[2]:
            print(f'Plotting {self.title}...')
            data = DynamicCompression(
                data_path=self.data_path,
                cycles=int(self.ctrl_nCycles.GetValue()),
                mode='Total',
                figure_size=(34, 14)
            )
            DynamicCompression.plotTotal(
                data,
                normal=self.cb_displacFit.GetValue(),
                damped=self.cb_dampedFit.GetValue(),
                absolute=self.cb_absoluFit.GetValue(),
                plot_exp_h=self.cb_displacExp.GetValue(),
                colorax1=tuple(c / 255 for c in self.color1),
                colorax2=tuple(c / 255 for c in self.color2)
            )
            plt.show()

        if self.title == plottypes[3]:
            print(f'Plotting {self.title}...')

            data = DynamicCompression(
                data_path=self.data_path,
                cycles=int(self.ctrl_nCycles.GetValue()),
                mode='Cyclic',
                figure_size=(34, 14)
            )
            DynamicCompression.plotCyclic(
                data,
                peak_size=int(self.ctrl_peakSize.GetValue()),
                initial_strain=float(self.ctrl_initStrain.GetValue()),
                final_strain=float(self.ctrl_finStrain.GetValue()),
                plotPeak=self.cb_plotPeak.GetValue(),
                plotFit=self.cb_plotYM.GetValue(),
                colorSeries=tuple(c / 255 for c in self.color1),
                colorLinRange=tuple(c / 255 for c in self.color2)
            )
            plt.show()

    def OnColor1(self, evt):
        dlg = wx.ColourDialog(self)
        dlg.GetColourData().SetChooseFull(True)
        if dlg.ShowModal() == wx.ID_OK:
            self.color1 = dlg.GetColourData().GetColour().Get()
            print(f'Color 1 selected: {self.color1}')
        self.colorButton1.SetBackgroundColour(self.color1)
        dlg.Destroy()

    def OnColor2(self, evt):
        dlg = wx.ColourDialog(self)
        dlg.GetColourData().SetChooseFull(True)
        if dlg.ShowModal() == wx.ID_OK:
            self.color2 = dlg.GetColourData().GetColour().Get()
            print(f'Color 2 selected: {self.color2}')
        self.colorButton2.SetBackgroundColour(self.color2)
        dlg.Destroy()


class DataGui(wx.Frame):
    def __init__(self, parent):
        super().__init__(
            parent,
            title='Rheometer Plots',
            style=wx.DEFAULT_FRAME_STYLE)

        self.filename = None
        self.CreateStatusBar()
        self.SetBackgroundColour('white')
        # self.SetForegroundColour('white')
        self.SetIcon(wx.Icon('images/chart_icon.ico'))
        self.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False, ' Helvetica Neue'))

        # Creating the menubar.
        filemenu = wx.Menu()
        menuOpen = filemenu.Append(wx.ID_OPEN, "&Open", " Open a data file.")
        menuAbout = filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program.")
        menuExit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program.")

        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        # Events.
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_COMBOBOX, self.OnCombo)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Data selection variables
        self.data_path = 'No file selected.'
        self.mainData_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Data')
        self.topData_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.dirname = ''
        self.data_ctrl = None

        self.txt_plottype = wx.StaticText(self, -1, 'Plot type:')
        self.combo_plot = wx.ComboBox(
            self, -1, size=(-1, -1), choices=plottypes,
            style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_plot.Enable(False)

        self.init_gui()

    def DataSelectGui(self):
        self.topData_sizer.Add(
            self.txt_plottype,
            0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.topData_sizer.Add(
            self.combo_plot,
            0, wx.EXPAND | wx.ALL, 5)

        self.data_ctrl = wx.TextCtrl(self, size=(300, 400), style=wx.TE_MULTILINE)
        # self.control.SetBackgroundColour('black')
        # self.control.SetForegroundColour('white')

        self.mainData_sizer.Add(
            self.data_ctrl,
            1, wx.EXPAND | wx.ALL, 10)
        self.mainData_sizer.Add(
            self.topData_sizer,
            0, wx.ALL, 10)

    def init_gui(self):
        self.DataSelectGui()

        self.main_sizer.Add(
            self.mainData_sizer,
            5, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self)
        self.Layout()

    def OnAbout(self, e):
        # Create a message dialog box
        print(f'Opening "About" dialog...')
        dlg = wx.MessageDialog(self, 'By Petrus Kirsten', 'About Rheometer Plots', wx.OK)
        dlg.ShowModal()  # Shows it
        dlg.Destroy()  # finally destroy it when finished.

    def OnExit(self, e):
        print(f'Closing the frame...')
        self.Close(True)  # Close the frame.

    def OnOpen(self, e):
        """ Open a file"""
        print(f'Opening a data file...')
        dlg = wx.FileDialog(
            self,
            'Select the data',
            self.dirname, '', '*.*',
            wx.FD_OPEN | wx.FD_MULTIPLE)

        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilenames()
            self.dirname = dlg.GetDirectory()
            self.data_path = os.path.join(self.filename[0])
            file = open(self.data_path, 'r')
            self.data_ctrl.SetValue(file.read())
            file.close()

        dlg.Destroy()
        self.combo_plot.Enable(True)
        print(f'File(s) selected: {self.filename}')

    def OnCombo(self, e):
        plottype_choice = self.combo_plot.GetValue()
        print(f'Plot type selected: {plottype_choice}.')

        dlg = PlotDlg(self, plottype_choice, self.filename)
        dlg.Show()

        if plottype_choice == plottypes[0]:
            dlg.stressSweep()

        if plottype_choice == plottypes[1]:
            dlg.oscilSweep()

        if plottype_choice == plottypes[2]:
            dlg.dynamicFull()

        if plottype_choice == plottypes[3]:
            dlg.dynamicCyclic()


class MyApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = None

    def OnInit(self):
        self.frame = DataGui(None)
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    print('App not executable as main code.\n'
          'Please, run app script (app.py).')
