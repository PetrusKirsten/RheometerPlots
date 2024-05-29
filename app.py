import wx
import os


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        self.dirname = ''

        # A "-1" in the size parameter instructs wxWidgets to use the default size.
        # In this case, we select 200px width and the default height.
        wx.Frame.__init__(self, parent, title=title, size=(200, 200))
        self.SetIcon(wx.Icon('Data/chart_icon.ico'))

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # self.control = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)

        self.CreateStatusBar()  # A Statusbar in the bottom of the window

        # Setting up the menu.
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

        # Robot coil trajectory variables
        self.mainTraj_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self,
            'Robotic coil trajectory')

        # Use some sizers to see layout options
        # self.sizer = wx.BoxSizer(wx.VERTICAL)
        # self.sizer.Add(self.control, 1, wx.ALIGN_CENTER)

        # Layout sizers
        # self.SetSizer(self.sizer)
        # self.SetAutoLayout(1)
        # self.sizer.Fit(self)
        self.Show()

    def init_gui(self):
        # self.EmgVisGui()
        # self.CoilTrajGui()

        self.main_sizer.Add(
            self.mainTraj_sizer, 1,
            wx.ALL, 20)

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
            # self.control.SetValue(f.read())
            f.close()
        dlg.Destroy()


def main() -> None:
    app = wx.App(False)
    frame = MainWindow(None, 'Rheometer Plots').Center()
    app.MainLoop()


if __name__ == "__main__":
    main()
