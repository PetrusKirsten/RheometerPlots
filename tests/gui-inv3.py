import wx  # pandas only works with wxPython 4.0.7
import os
from Plotting.DynamicCompression import DynamicCompression


# from . import spiralTMS
# import invesalius.project as prj
# from serial import SerialException
# from pubsub import pub as Publisher
# import invesalius.constants as const
# import invesalius.data.vtk_utils as vtku
# from invesalius.gui import task_navigator
# from vtkmodules.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor

class PlotDlg(wx.Dialog):
    def __init__(self, parent, title):
        super().__init__(
            parent,
            title=title,
            style=wx.DEFAULT_DIALOG_STYLE)

        # panel = wx.Panel(self, size=(500, 500))

        # Dialog configuration variables
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.mainPlot_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Plot configuration')

        self.ctrl_size = (50, -1)

        self.txt_sizer = wx.FlexGridSizer(2, 2, 10, 10)

        self.txt_npoints = wx.StaticText(self, -1, 'Number of points:')
        self.ctrl_npoints = wx.TextCtrl(self, -1, '196', size=self.ctrl_size)

        self.txt_dpi = wx.StaticText(self, -1, 'Figure resolution (dpi):')
        self.ctrl_dpi = wx.TextCtrl(self, -1, '300', size=self.ctrl_size)

        self.okButton = wx.Button(
            self, 1,
            'Plot', size=(-1, -1))

        # Dynamic oscillation - full
        self.cb_displacFit, self.cb_displacExp, self.cb_dampedFit, self.cb_absoluFit = None, None, None, None

        # # Dynamic oscillation - cyclic
        self.txt_peakSize, self.txt_initStrain, self.txt_finStrain = None, None, None
        self.ctrl_peakSize, self.ctrl_initStrain, self.ctrl_finStrain = None, None, None

    def dynamicFull(self):
        self.cb_displacExp = wx.CheckBox(self, -1, 'Experimental height data.', (10, 10))
        self.cb_displacFit = wx.CheckBox(self, -1, 'Fitted height data.', (10, 10))
        self.cb_dampedFit = wx.CheckBox(self, -1, 'Stress vs. Strain fit: Damped sine wave.', (10, 10))
        self.cb_absoluFit = wx.CheckBox(self, -1, 'Stress vs. Strain fit: Absolute sine wave.', (10, 10))

        self.mainPlot_sizer.AddMany((
            (self.cb_displacExp, 0, wx.ALL, 5),
            (self.cb_displacFit, 0, wx.ALL, 5),
            (self.cb_dampedFit, 0, wx.ALL, 5),
            (self.cb_absoluFit, 0, wx.ALL, 5)
        ))

        self.init_gui()

    def dynamicCyclic(self):
        self.txt_sizer = wx.FlexGridSizer(5, 2, 10, 10)

        self.txt_peakSize = wx.StaticText(self, -1, 'Stress peak range:')
        self.ctrl_peakSize = wx.TextCtrl(self, -1, '3', size=self.ctrl_size)
        self.txt_initStrain = wx.StaticText(self, -1, 'Initial strain linear region:')
        self.ctrl_initStrain = wx.TextCtrl(self, -1, '10', size=self.ctrl_size)
        self.txt_finStrain = wx.StaticText(self, -1, 'Final strain linear region:')
        self.ctrl_finStrain = wx.TextCtrl(self, -1, '18', size=self.ctrl_size)

        self.txt_sizer.AddMany((
            (self.txt_peakSize, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_peakSize, 0, wx.ALIGN_CENTER_VERTICAL),
            (self.txt_initStrain, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_initStrain, 0, wx.ALIGN_CENTER_VERTICAL),
            (self.txt_finStrain, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_finStrain, 0, wx.ALIGN_CENTER_VERTICAL)
        ))

        self.init_gui()

    def init_gui(self):
        self.txt_sizer.AddMany((
            (self.txt_npoints, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_npoints, 0, wx.ALIGN_CENTER_VERTICAL),
            (self.txt_dpi, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_dpi, 0, wx.ALIGN_CENTER_VERTICAL)
        ))

        self.mainPlot_sizer.Add(
            self.txt_sizer, 1,
            wx.EXPAND | wx.ALL, 10)

        self.mainPlot_sizer.Add(
            self.okButton, 0,
            wx.EXPAND | wx.ALL, 10)
        self.okButton.Enable(True)

        self.main_sizer.Add(
            self.mainPlot_sizer, 1,
            wx.EXPAND | wx.ALL, 10)

        self.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self)
        self.Layout()


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
        self.SetIcon(wx.Icon('chart_icon.ico'))
        self.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False, ' Helvetica Neue'))

        # Creating the menubar.
        filemenu = wx.Menu()
        menuOpen = filemenu.Append(wx.ID_OPEN, "&Open", " Open a file to edit")
        menuAbout = filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        menuExit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

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
        self.mainData_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Data')
        self.topData_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.dirname = ''
        self.data_ctrl = None

        self.txt_plottype = wx.StaticText(self, -1, 'Plot type:')
        self.plottypes = [
            'Dynamic compression | Total',
            'Dynamic compression | Cyclic',
            'Oscillatory sweep']
        self.combo_plot = wx.ComboBox(
            self, -1, size=(-1, -1), choices=self.plottypes,
            style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.txt_test = wx.StaticText(self, -1, '')

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
        dlg = wx.MessageDialog(self, 'By Petrus Kirsten', 'About Rheometer Plots', wx.OK)
        dlg.ShowModal()  # Shows it
        dlg.Destroy()  # finally destroy it when finished.

    def OnExit(self, e):
        self.Close(True)  # Close the frame.

    def OnOpen(self, e):
        """ Open a file"""
        dlg = wx.FileDialog(self, 'Select the data', self.dirname, '', '*.*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname, self.filename), 'r')
            self.data_ctrl.SetValue(f.read())
            f.close()
        dlg.Destroy()

    def OnCombo(self, e):
        plottype_choice = self.combo_plot.GetValue()

        dlg = PlotDlg(self, plottype_choice)
        dlg.Show()

        if plottype_choice == 'Dynamic compression | Total':
            dlg.dynamicFull()

        if plottype_choice == 'Dynamic compression | Cyclic':
            dlg.dynamicCyclic()

        if plottype_choice == 'Oscillatory sweep':
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
    app = MyApp(0)
    app.MainLoop()
