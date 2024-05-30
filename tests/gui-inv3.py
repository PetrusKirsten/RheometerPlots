import wx  # pandas only works with wxPython 4.0.7
# import vtk
import sys
# import serial
# from . import emg
import numpy as np
import os


# from . import spiralTMS
# import invesalius.project as prj
# from serial import SerialException
# from pubsub import pub as Publisher
# import invesalius.constants as const
# import invesalius.data.vtk_utils as vtku
# from invesalius.gui import task_navigator
# from vtkmodules.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor


class MotorMapGui(wx.Frame):
    def __init__(self, parent):
        super().__init__(
            parent,
            title='Rheometer Plots',
            style=wx.DEFAULT_FRAME_STYLE)
        self.SetSize(self.GetSize())
        self.CreateStatusBar()
        self.SetBackgroundColour('white')
        # self.SetForegroundColour('white')
        self.SetIcon(wx.Icon('chart_icon.ico'))
        self.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False, ' Helvetica Neue'))

        filemenu = wx.Menu()
        menuOpen = filemenu.Append(wx.ID_OPEN, "&Open", " Open a file to edit")
        menuAbout = filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        menuExit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        # Events.
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Data selection variables
        self.mainData_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Data')
        self.dirname = ''
        self.data_ctrl = None

        # Plot configuration variables
        self.mainPlot_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Plot configuration')

        self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_sizer = wx.FlexGridSizer(6, 2, 7, 100)

        self.x_ctrl = None
        self.y_ctrl = None
        self.z_ctrl = None
        self.ecc_ctrl = None
        self.radius_ctrl = None
        self.pointsdist_ctrl = None
        self.generateButton = None
        self.doneButton = None
        self.x_marker = None
        self.y_marker = None

        self.txt_plottype = wx.StaticText(self, -1, 'Plot type:')

        self.x_sta = wx.StaticText(self, -1, 'X axis hotspot:')
        self.y_sta = wx.StaticText(self, -1, 'Y axis hotspot:')
        self.z_sta = wx.StaticText(self, -1, 'Z axis hotspot:')
        self.ecc_sta = wx.StaticText(self, -1, 'Ellipse eccentricity:')
        self.radius_sta = wx.StaticText(self, -1, 'Max ellipse radius [mm]:')
        self.pointsdist_sta = wx.StaticText(self, -1, 'Points distance [mm]:')

        self.init_gui()

    def DataSelectGui(self):
        self.data_ctrl = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        # self.control.SetBackgroundColour('black')
        # self.control.SetForegroundColour('white')
        self.mainData_sizer.Add(
            self.data_ctrl,
            -1, wx.EXPAND | wx.ALL, 20)

    def PlotConfigGui(self):
        self.top_sizer.Add(
            self.txt_plottype, 0,
            wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        combo_plot = wx.ComboBox(self, -1, size=(220, -1),
                                 style=wx.CB_DROPDOWN | wx.CB_READONLY)
        # combo_plot.SetBackgroundColour('black')
        # combo_plot.SetForegroundColour('white')
        self.top_sizer.Add(
            combo_plot, 0,
            wx.EXPAND | wx.LEFT, 50)

        self.mainPlot_sizer.Add(
            self.top_sizer, 0,
            wx.ALL, 10)

        ctrl_size = (70, -1)
        self.x_ctrl = wx.TextCtrl(self, -1, '155.99', size=ctrl_size)
        self.y_ctrl = wx.TextCtrl(self, -1, '112.13', size=ctrl_size)
        self.z_ctrl = wx.TextCtrl(self, -1, '149.12', size=ctrl_size)
        self.ecc_ctrl = wx.TextCtrl(self, -1, '0.75', size=ctrl_size)
        self.radius_ctrl = wx.TextCtrl(self, -1, '40', size=ctrl_size)
        self.pointsdist_ctrl = wx.TextCtrl(self, -1, '20', size=ctrl_size)

        # self.x_ctrl.SetBackgroundColour('black')
        # self.y_ctrl.SetBackgroundColour('black')
        # self.z_ctrl.SetBackgroundColour('black')
        # self.ecc_ctrl.SetBackgroundColour('black')
        # self.radius_ctrl.SetBackgroundColour('black')
        # self.pointsdist_ctrl.SetBackgroundColour('black')
        #
        # self.x_ctrl.SetForegroundColour('white')
        # self.y_ctrl.SetForegroundColour('white')
        # self.z_ctrl.SetForegroundColour('white')
        # self.ecc_ctrl.SetForegroundColour('white')
        # self.radius_ctrl.SetForegroundColour('white')
        # self.pointsdist_ctrl.SetForegroundColour('white')

        self.txt_sizer.AddMany(
            ((self.x_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.x_ctrl, 0),
             (self.y_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.y_ctrl, 0),
             (self.z_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.z_ctrl, 0),
             (self.ecc_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.ecc_ctrl, 0),
             (self.radius_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.radius_ctrl, 0),
             (self.pointsdist_sta, 0, wx.ALIGN_CENTER_VERTICAL), (self.pointsdist_ctrl, 0)))
        self.mainPlot_sizer.Add(
            self.txt_sizer, 0,
            wx.EXPAND | wx.ALL, 20)

    def init_gui(self):
        self.DataSelectGui()
        self.PlotConfigGui()

        self.main_sizer.Add(
            self.mainData_sizer, 3,
            wx.EXPAND | wx.ALL, 10)
        self.main_sizer.Add(
            self.mainPlot_sizer, 2,
            wx.ALL, 10)

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


class MyApp(wx.App):
    def OnInit(self):
        self.dlg = MotorMapGui(None)
        self.SetTopWindow(self.dlg)
        self.dlg.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
