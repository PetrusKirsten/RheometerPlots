import wx  # pandas only works with wxPython 4.0.7
import os


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
            style=wx.DEFAULT_DIALOG_STYLE,
            title=title)
        panel = wx.Panel(self, size=(250, 150))

        # Plot configuration variables
        self.mainPlot_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Plot configuration')
        self.txt_sizer = wx.FlexGridSizer(2, 2, 7, 10)
        self.tot_sizer = wx.FlexGridSizer(5, 2, 7, 10)
        self.cyc_sizer = wx.FlexGridSizer(4, 2, 7, 10)

        self.ctrl_npoints = None
        self.ctrl_dpi = None

        self.txt_npoints = wx.StaticText(self, -1, 'Number of points:')
        self.txt_dpi = wx.StaticText(self, -1, 'Figure resolution (dpi):')

        # Dynamic oscillation - integral
        self.cb_displacFit = wx.CheckBox(self, -1, 'Fitted height oscillation', (10, 10))
        self.cb_displacExp = wx.CheckBox(self, -1, 'Experimental height data', (10, 10))
        self.cb_dampedFit = wx.CheckBox(self, -1, 'Damped wave from stress oscillation', (10, 10))
        self.cb_absoluFit = wx.CheckBox(self, -1, 'Absolute wave from stress oscillation', (10, 10))

        # Dynamic oscillation - cyclic
        self.txt_peakSize = wx.StaticText(self, -1, 'Peak range:')
        self.txt_initStrain = wx.StaticText(self, -1, 'Initial strain linear region:')
        self.txt_finStrain = wx.StaticText(self, -1, 'Final strain linear region:')


class DataGui(wx.Frame):
    def __init__(self, parent):
        super().__init__(
            parent,
            title='Rheometer Plots',
            style=wx.DEFAULT_FRAME_STYLE)

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
        self.plottypes = ['Dynamic compression | Total', 'Dynamic compression | Cyclic']
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
        if self.combo_plot.GetValue() == 'Dynamic compression | Total':
            PlotDlg(self, self.combo_plot.GetValue()).Show()
            # self.tot_sizer.AddMany(
            #     (
            #         (self.txt_npoints, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_npoints, 0),
            #         (self.cb_displacFit, 0, wx.ALIGN_CENTER_VERTICAL), (self.txt_test, 0),
            #         (self.cb_displacExp, 0, wx.ALIGN_CENTER_VERTICAL), (wx.StaticText(self, -1, ''), 0),
            #         (self.cb_dampedFit, 0, wx.ALIGN_CENTER_VERTICAL), (wx.StaticText(self, -1, ''), 0),
            #         (self.cb_absoluFit, 0, wx.ALIGN_CENTER_VERTICAL), (wx.StaticText(self, -1, ''), 0)
            #     )
            # )
            # self.mainPlot_sizer.Add(
            #     self.tot_sizer, 0,
            #     wx.EXPAND | wx.ALL, 20)

        if self.combo_plot.GetValue() == 'Dynamic compression | Cyclic':
            PlotDlg(self, self.combo_plot.GetValue()).Show()
            # self.cyc_sizer.AddMany(
            #     (
            #         (self.txt_npoints, 0, wx.ALIGN_CENTER_VERTICAL), (self.ctrl_npoints, 0),
            #         (self.txt_peakSize, 0, wx.ALIGN_CENTER_VERTICAL), (self.txt_test, 0),
            #         (self.txt_initStrain, 0, wx.ALIGN_CENTER_VERTICAL), (wx.StaticText(self, -1, ''), 0),
            #         (self.txt_finStrain, 0, wx.ALIGN_CENTER_VERTICAL), (wx.StaticText(self, -1, ''), 0)
            #     )
            # )
            # self.mainPlot_sizer.Add(
            #     self.cyc_sizer, 0,
            #     wx.EXPAND | wx.ALL, 20)


class MyApp(wx.App):
    def OnInit(self):
        self.dlg = DataGui(None)
        self.SetTopWindow(self.dlg)
        self.dlg.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
