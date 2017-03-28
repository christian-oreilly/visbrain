"""Visual objects of sleep module.

This file contains and initialize visual objects (channel plot, spectrogram,
hypnogram, indicator, shortcuts)
"""
import numpy as np
import scipy.signal as scpsig

from vispy import scene
import vispy.visuals.transforms as vist

from .marker import Markers
from ...utils import array2colormap, color2vb, filt


__all__ = ["visuals"]


"""
###############################################################################
# OBJECTS
###############################################################################
Classes below are used to create visual objects, display on sevral canvas :
- ChannelPlot : plot data (one signal per channel)
- Spectrogram : on a specific channel
- Hypnogram : sleep stages (can be optional)
"""


class PrepareData(object):
    """Prepare data before plotting.

    This class group a set of signal processing tools including :
        - De-meaning
        - De-trending
        - Filtering
    """

    def __init__(self, axis=0, demean=False, detrend=False, filt=False,
                 fstart=12., fend=16., forder=3, way='lfilter',
                 filt_meth='butterworth', btype='bandpass'):
        # Axis along which to perform preparation :
        self.axis = axis
        # Demean and detrend :
        self.demean = demean
        self.detrend = detrend
        # Filtering :
        self.filt = filt
        self.fstart, self.fend = fstart, fend
        self.forder, self.filt_meth = forder, filt_meth
        self.way, self.btype = way, btype

    def __bool__(self):
        """Return if data have to be prepared."""
        return any([self.demean, self.detrend, self.filt])

    def _prepare_data(self, sf, data, time):
        """Prepare data before plotting."""
        # ============= DEMEAN =============
        if self.demean:
            mean = np.mean(data, axis=self.axis, keepdims=True)
            np.subtract(data, mean, out=data)

        # ============= DETREND =============
        if self.detrend:
            data = scpsig.detrend(data, axis=self.axis)

        # ============= FILTERING =============
        if self.filt:
            data = filt(sf, np.array([self.fstart, self.fend]), data,
                        btype=self.btype, order=self.forder, way=self.way,
                        method=self.filt_meth, axis=self.axis)

        return data

    def update(self):
        """Update object."""
        if self._fcn is not None:
            self._fcn()


class ChannelPlot(PrepareData):
    """Plot each channel."""

    def __init__(self, channels, time, color=(.2, .2, .2),
                 color_detection='red', width=1.5, method='gl', camera=None,
                 parent=None, fcn=None):
        # Initialize PrepareData :
        PrepareData.__init__(self, axis=1)

        # Variables :
        self._camera = camera
        self.rect = []
        self.width = width
        self.autoamp = False
        # Don't use self.colidx = [{...}] * len(channels)
        self.colidx = [{'color': color_detection, 'idx': np.array([
                                            ])} for _ in range(len(channels))]
        self._fcn = fcn
        self.visible = np.array([True] + [False] * (len(channels) - 1))
        self.consider = np.ones((len(channels),), dtype=bool)

        # Get color :
        self.color = color2vb(color)
        self.color_detection = color2vb(color_detection)

        # Create one line pear channel :
        pos = np.zeros((1, 3), dtype=np.float32)
        self.mesh, self.report, self.grid, self.peak = [], [], [], []
        self.loc, self.node = [], []
        for i, k in enumerate(channels):
            # ----------------------------------------------
            # Create a node parent :
            node = scene.Node(name=k+'plot')
            node.parent = parent[i].wc.scene
            self.node.append(node)

            # ----------------------------------------------
            # Create main line (for channel plot) :
            mesh = scene.visuals.Line(pos, name=k+'plot', color=self.color,
                                      method=method, parent=node)
            mesh.set_gl_state('translucent')
            self.mesh.append(mesh)
            # ----------------------------------------------
            # Create marker peaks :
            mark = Markers(pos=np.zeros((1, 3), dtype=np.float32),
                           parent=node)
            mark.set_gl_state('translucent')
            mark.visible = False
            self.peak.append(mark)
            # ----------------------------------------------
            # Report line :
            rep = scene.visuals.Line(pos, name=k+'report', method=method,
                                     color=self.color_detection, parent=node)
            rep.set_gl_state('translucent')
            self.report.append(rep)
            # ----------------------------------------------
            # Locations :
            loc = scene.visuals.Line(pos, name=k+'location', method=method,
                                     color=(.1, .1, .1, .3), parent=node,
                                     connect='segments')
            loc.set_gl_state('translucent')
            self.loc.append(loc)
            # ----------------------------------------------
            # Create a grid :
            grid = scene.visuals.GridLines(color=(.1, .1, .1, .5),
                                           scale=(30.*time[-1]/len(time), .1),
                                           parent=parent[i].wc.scene)
            grid.set_gl_state('translucent')
            self.grid.append(grid)

    def __iter__(self):
        """Iterate over visible mesh."""
        for i, k in enumerate(self.mesh):
            if self.visible[i]:
                yield i, k

    def __len__(self):
        """Return the number of channels."""
        return len(self.mesh)

    def set_data(self, sf, data, time, sl=None, ylim=None, autoamp=True):
        """Set data to channels.

        Args:
            data: np.ndarray
                Array of data of shape (n_channels, n_points)

            time: np.ndarray
                The time vector.

        Kargs:
            sl: slice, optional, (def: None)
                A slice object for the time selection of data.

            ylim: np.ndarray, optional, (def: None)
                Y-limits of each channel. Must be a (n_channels, 2) array.
        """
        if ylim is None:
            ylim = np.array([data.min(1), data.max(1)]).T

        # Manage slice :
        sl = slice(0, data.shape[1]) if sl is None else sl

        # Slice selection (of time and data) :
        timeSl = time[sl]
        self.x = (timeSl.min(), timeSl.max())
        dataSl = data[self.visible, sl]
        z = np.full_like(timeSl, .5, dtype=np.float32)

        # Prepare the data (only if needed) :
        if self:
            dataSl = self._prepare_data(sf, dataSl.copy(), timeSl)

        # Build a index vector:
        idx = np.arange(sl.start, sl.stop)

        # Set data to each plot :
        for l, (i, k) in enumerate(self):
            # ________ MAIN DATA ________
            # Select channel ;
            datchan = dataSl[l, :]

            # Concatenate time / data / z axis :
            dat = np.vstack((timeSl, datchan, z)).T

            # Set main ligne :
            k.set_data(dat, color=self.color, width=self.width)

            # ________ COLOR ________
            # Indicator line :
            if self.colidx[i]['idx'].size:
                # Find index that are both in idx and in indicator :
                inter = np.intersect1d(idx, self.colidx[i]['idx']) - sl.start
                # Build a array for connecting only consecutive segments :
                index = np.zeros((len(idx)), dtype=bool)
                index[inter] = True
                index[-1] = False
                dat[:, 2] = -2.
                # Send data to report :
                self.report[i].set_data(pos=dat, connect=index, width=4.,
                                        color=self.colidx[i]['color'])

            # ________ CAMERA ________
            # Use either auto / fixed adaptative camera :
            ycam = (datchan.min(), datchan.max()) if self.autoamp else ylim[i]

            # Get camera rectangle and set it:
            rect = (self.x[0], ycam[0], self.x[1]-self.x[0],
                    ycam[1] - ycam[0])
            self._camera[i].rect = rect
            k.update()
            self.rect.append(rect)

    def set_grid(self, time, length=30., y=1.):
        """Set grid lentgh."""
        # Get scaling factor :
        sc = (length * time[-1] / len(time), y)
        # Set to the grid :
        for k in self.grid:
            k._grid_color_fn['scale'].value = sc
            k.update()

    def set_location(self, sf, data, channel, start, end, factor=100.):
        """Set vertical lines for detections."""
        # Get data limits :
        y = (data.min(), data.max())
        # # Build pos :
        pos = np.zeros((4, 3), dtype=np.float32)
        pos[0, 0:2] = [start, y[0]]
        pos[1, 0:2] = [start, y[1]]
        pos[2, 0:2] = [end, y[0]]
        pos[3, 0:2] = [end, y[1]]
        # Set data pos :
        self.loc[channel].set_data(pos=pos, width=2.)

    def clean(self):
        """Clean all the data."""
        # Empty position :
        pos = np.zeros((1, 3), dtype=np.float32)
        for k in range(len(self)):
            # Main mesh :
            self.mesh[k].set_data(pos=pos, color='gray')
            self.mesh[k].parent = None
            # Report :
            self.report[k].set_data(pos=pos, color='gray')
            self.report[k].parent = None
            # Grid :
            self.grid[k].parent = None
            # Peak locations :
            self.peak[k].set_data(pos=pos, face_color='gray')
            self.peak[k].parent = None
            # Vertical lines :
            self.loc[k].set_data(pos=pos, color='gray')
            self.loc[k].parent = None
        self.mesh, self.report, self.grid, self.peak = [], [], [], []
        self.loc = []

    # ----------- PARENT -----------
    @property
    def parent(self):
        """Get the parent value."""
        return self.mesh[0].parent

    @parent.setter
    def parent(self, value):
        """Set parent value."""
        for i, k, in zip(value, self.mesh):
            k.parent = i.wc.scene

    # ----------- AUTOAMP -----------
    @property
    def autoamp(self):
        """Get the autoamp value."""
        return self._autoamp

    @autoamp.setter
    def autoamp(self, value):
        """Set autoamp value."""
        self._autoamp = value


class Spectrogram(PrepareData):
    """Create and manage a Spectrogram object.

    After object creation, use the set_data() method to pass new data, new
    color, new frequency / time range, new settings...
    """

    def __init__(self, camera, parent=None, fcn=None):
        # Initialize PrepareData :
        PrepareData.__init__(self, axis=0)

        # Keep camera :
        self._camera = camera
        self._rect = (0., 0., 0., 0.)
        self._fcn = fcn

        # Create a vispy image object :
        self.mesh = scene.visuals.Image(np.zeros((2, 2)), name='spectrogram',
                                        parent=parent)

    def set_data(self, sf, data, time, cmap='rainbow', nfft=30., overlap=.5,
                 fstart=.5, fend=20., contraste=.5):
        """Set data to the spectrogram.

        Use this method to change data, colormap, spectrogram settings, the
        starting and ending frequencies.

        Args:
            sf: float
                The sampling frequency.

            data: np.ndarray
                The data to use for the spectrogram. Must be a row vector.

            time: np.ndarray
                The time vector.

        Kargs:
            cmap: string, optional, (def: 'viridis')
                The matplotlib colormap to use.

            nfft: float, optional, (def: 30.)
                Number of fft points for the spectrogram (in seconds).

            overlap: float, optional, (def: .5)
                Time overlap for the spectrogram (in seconds).

            fstart: float, optional, (def: .5)
                Frequency from which the spectrogram have to start.

            fend: float, optional, (def: 20.)
                Frequency from which the spectrogram have to finish.

            contraste: float, optional, (def: .5)
                Contraste of the colormap.
        """
        # =================== CONVERSION ===================
        nperseg = int(round(nfft * sf))
        overlap = int(round(overlap * sf))

        # =================== PREPARE DATA ===================
        # Prepare data (only if needed)
        if self:
            data = self._prepare_data(sf, data.copy(), time)

        # =================== COMPUTE ===================
        # Compute the spectrogram :
        freq, t, mesh = scpsig.spectrogram(data, fs=sf, nperseg=nperseg,
                                           noverlap=overlap, window='hamming')
        mesh = 20 * np.log10(mesh)

        # =================== FREQUENCY SELECTION ===================
        # Find where freq is [fstart, fend] :
        f = [0., 0.]
        f[0] = np.abs(freq - fstart).argmin() if fstart else 0
        f[1] = np.abs(freq - fend).argmin() if fend else len(freq)
        # Build slicing and select frequency vector :
        sls = slice(f[0], f[1]+1)
        freq = freq[sls]
        self._fstart, self._fend = freq[0], freq[-1]

        # =================== TIME SELECTION ===================
        t = []
        q = 0
        for k in range(mesh.shape[1]):
            t.append(time[q:q+nperseg].mean())
            q += nperseg-overlap
        t = np.array(t)

        # =================== COLOR ===================
        # Get clim :
        clim = (contraste * mesh.min(), contraste * mesh.max())
        # Turn mesh into color array for selected frequencies:
        self.mesh.set_data(array2colormap(mesh[sls, :], cmap=cmap,
                                          clim=clim))

        # =================== TRANSFORM ===================
        # Re-scale the mesh for fitting in time / frequency :
        fact = (freq.max()-freq.min())/len(freq)
        sc = (t.max()/mesh.shape[1], fact, 1)
        tr = [t[0], freq.min(), 0]
        self.mesh.transform = vist.STTransform(scale=sc, translate=tr)
        # Update object :
        self.mesh.update()
        # Get camera rectangle :
        self.rect = (time.min(), freq.min(), time.max()-time.min(),
                     freq.max()-freq.min())
        self.freq = freq

    def clean(self):
        """Clean indicators."""
        pos = np.zeros((3, 4), dtype=np.float32)
        self.mesh.set_data(pos)
        self.mesh.parent = None
        self.mesh = None

    # ----------- RECT -----------
    @property
    def rect(self):
        """Get the rect value."""
        return self._rect

    @rect.setter
    def rect(self, value):
        """Set rect value."""
        self._rect = value
        self._camera.rect = value


class Hypnogram(object):
    """Create a hypnogram object."""

    def __init__(self, time, camera, color='darkblue', width=2., parent=None):
        # Keep camera :
        self._camera = camera
        self._rect = (0., 0., 0., 0.)
        self.rect = (time.min(), -5., time.max() - time.min(), 7.)
        self.width = width
        self.n = len(time)
        # Get color :
        self.color = {k: color2vb(color=i) for k, i in zip(color.keys(),
                                                           color.values())}
        # Create a default line :
        pos = np.array([[0, 0], [0, 100]])
        self.mesh = scene.visuals.Line(pos, name='hypnogram', method='gl',
                                       parent=parent)
        self.mesh.set_gl_state('translucent', depth_test=True)
        # Create a default marker (for edition):
        self.edit = Markers(parent=parent)
        # Create a reported marker (for detection):
        self.report = Markers(parent=parent)
        # self.mesh.set_gl_state('translucent', depth_test=True)
        self.mesh.set_gl_state('translucent', depth_test=False,
                               cull_face=True, blend=True, blend_func=(
                                          'src_alpha', 'one_minus_src_alpha'))
        # Add grid :
        self.grid = scene.visuals.GridLines(color=(.1, .1, .1, 1.),
                                            scale=(30.*time[-1]/len(time), 1.),
                                            parent=parent)
        self.grid.set_gl_state('translucent')

    def __len__(self):
        """Return the time length."""
        return self.n

    def set_data(self, sf, data, time):
        """Set data to the hypnogram.

        Args:
            sf: float
                The sampling frequency.

            data: np.ndarray
                The data to send. Must be a row vector.

            time: np.ndarray
                The time vector
        """
        # Build color array :
        color = np.zeros((len(data), 4), dtype=np.float32)
        for k, v in zip(self.color.keys(), self.color.values()):
            color[data == k, :] = v
        # Set data to the mesh :
        self.mesh.set_data(pos=np.vstack((time, -data)).T, width=self.width,
                           color=color)
        self.mesh.update()

    def set_report(self, time, index, symbol='triangle_down', y=1., size=13.,
                   color='red'):
        """Report additional markers to the hypnogram.

        Args:
            time: np.ndarray
                The time vector.

            index: np.ndarray
                Marker index (in sample unit).

            symbol: string
                The marker symbol to use (see vispy.scene.visuals.Markers.
                set_data description).

            y: float or np.ndarray
                The y axis position. If y is a float, all markers will be at
                the same level. Otherwise, use an array with same size as index

            size: float
                Marker size.

            color: tuple/string/np.ndarray
                The color to use.
        """
        # Get reduced version of time :
        timeSl = time[index]
        # Build y-position :
        y = np.full_like(timeSl, y) if isinstance(y, (int, float)) else y
        # Build data array :
        pos = np.vstack((timeSl, y)).T
        # Set data to report :
        self.report.set_data(pos=pos, face_color=color2vb(color, alpha=.9),
                             size=size, symbol=symbol, edge_width=0.)

    def set_grid(self, time, length=30., y=1.):
        """Set grid lentgh."""
        # Get scaling factor :
        sc = (length * time[-1] / len(time), y)
        # Set to the grid :
        self.grid._grid_color_fn['scale'].value = sc
        self.grid.update()

    def clean(self):
        """Clean indicators."""
        pos = np.zeros((1, 3), dtype=np.float32)
        # Mesh :
        self.mesh.set_data(pos=pos, color='gray')
        self.mesh.parent = None
        self.mesh = None
        # Edit :
        self.edit.set_data(pos=pos, face_color='gray')
        self.edit.parent = None
        self.edit = None
        # Report :
        self.report.set_data(pos=pos, face_color='gray')
        self.report.parent = None
        self.report = None
        # Grid :
        self.grid.parent = None
        self.grid = None

    # ----------- RECT -----------
    @property
    def rect(self):
        """Get the rect value."""
        return self._rect

    @rect.setter
    def rect(self, value):
        """Set rect value."""
        self._rect = value
        self._camera.rect = value


"""
###############################################################################
# INDICATORS
###############################################################################
Visual indicators can be used to help the user to see in which time window the
signal is currently plotted. Those indicators are two vertical lines displayed
on the spectrogram and hypnogram.
"""


class Indicator(object):
    """Create a visual indicator (for spectrogram and hypnogram)."""

    def __init__(self, name='indicator', alpha=.3, visible=True, parent=None):
        # Create a vispy image object :
        image = color2vb('gray', alpha=alpha)[np.newaxis, ...]
        self.mesh = scene.visuals.Image(data=image, name=name,
                                        parent=parent)
        self.mesh.visible = visible

    def set_data(self, xlim, ylim):
        """Move the visual indicator.

        Args:
            xlim: tuple
                A tuple of two float indicating where xlim start and xlim end.

            ylim: tuple
                A tuple of two floats indicating where ylim start and ylim end.
        """
        tox = (xlim[0], ylim[0], -1.)
        sc = (xlim[1]-xlim[0], ylim[1]-ylim[0], 1.)
        # Move the square
        self.mesh.transform = vist.STTransform(translate=tox, scale=sc)

    def clean(self):
        """Clean indicators."""
        self.mesh.parent = None
        self.mesh = None


"""
###############################################################################
# SHORTCUTS
###############################################################################
Shortcuts applied on each canvas.
"""


class vbShortcuts(object):
    """This class add some shortcuts to the main canvas.

    It's also use to initialize to panel of shortcuts.

    Args:
        canvas: vispy canvas
            Vispy canvas to add the shortcuts.
    """

    def __init__(self, canvas):
        """Init."""
        self.sh = [('n', 'Go to the next window'),
                   ('b', 'Go to the previous window'),
                   ('s', 'Display / hide spectrogram'),
                   ('h', 'Display / hide hypnogram'),
                   ('z', 'Enable / disable zooming'),
                   ('a', 'Scoring: set current window to Art (-1)'),
                   ('w', 'Scoring: set current window to Wake (0)'),
                   ('1', 'Scoring: set current window to N1 (1)'),
                   ('2', 'Scoring: set current window to N2 (2)'),
                   ('3', 'Scoring: set current window to N3 (3)'),
                   ('r', 'Scoring: set current window to REM (4)'),
                   ('CTRL + s', 'Save hypnogram'),
                   ('CTRL + t', 'Display shortcuts'),
                   ('CTRL + e', 'Display documentation'),
                   ('CTRL + d', 'Display / hide setting panel'),
                   ('CTRL + n', 'Take a screenshot'),
                   ('CTRL + q', 'Close Sleep graphical interface'),
                   ]

        # Add shortcuts to vbCanvas :
        @canvas.events.key_press.connect
        def on_key_press(event):
            """Executed function when a key is pressed on a keyboard over canvas.

            :event: the trigger event
            """
            if event.text == ' ':
                pass
            # ------------ SLIDER ------------
            if event.text == 'n':  # Next (slider)
                self._SlGoto.setValue(
                                self._SlGoto.value() + self._SigSlStep.value())
            if event.text == 'b':  # Before (slider)
                self._SlGoto.setValue(
                                self._SlGoto.value() - self._SigSlStep.value())
            # ------------  VISIBILITY ------------
            if event.text == 's':  # Toggle visibility on spec
                self._PanSpecViz.setChecked(not self._PanSpecViz.isChecked())
                self._fcn_specViz()

            if event.text == 'h':  # Toggle visibility on hypno
                self._PanHypViz.setChecked(not self._PanHypViz.isChecked())
                self._fcn_hypViz()

            if event.text == 'z':  # Enable zoom
                viz = self._PanTimeZoom.isChecked()
                self._PanTimeZoom.setChecked(not viz)
                self._PanHypZoom.setChecked(not viz)
                self._PanSpecZoom.setChecked(not viz)
                self._fcn_Zooming()

            if event.text == 'm':
                viz = self._slMagnify.isChecked()
                self._slMagnify.setChecked(not viz)
                self._fcn_sliderMagnify()

            # ------------ SCORING ------------
            if event.text == 'a':
                self._add_stage_on_win(-1)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())
            if event.text == 'w':
                self._add_stage_on_win(0)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())
            if event.text == '1':
                self._add_stage_on_win(1)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())
            if event.text == '2':
                self._add_stage_on_win(2)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())
            if event.text == '3':
                self._add_stage_on_win(3)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())
            if event.text == 'r':
                self._add_stage_on_win(4)
                self._SlGoto.setValue(self._SlGoto.value(
                                                 ) + self._SigSlStep.value())

        @canvas.events.mouse_release.connect
        def on_mouse_release(event):
            """Executed function when the mouse is pressed over canvas.

            This method set the transformation to the canvas to NullTransform.
            """
            # Get canvas name :
            name = canvas.title
            condition = bool(name.find('Canvas') + 1)
            if condition and not self._slMagnify.isChecked():
                # Get channel name :
                chan = name.split('Canvas_')[1]
                # Get index :
                idx = self._channels.index(chan)
                # Build transformation :
                self._chan.node[idx].transform = vist.NullTransform()

        @canvas.events.mouse_double_click.connect
        def on_mouse_double_click(event):
            """Executed function when double click mouse over canvas.

            :event: the trigger event
            """
            pass

        @canvas.events.mouse_move.connect
        def on_mouse_move(event):
            """Executed function when the mouse move over canvas.

            Magnify for all channels under cursor locations.
            """
            if self._slMagnify.isChecked():
                val = self._SlVal.value()
                step = self._SigSlStep.value()
                win = self._SigWin.value()
                tm, tM = (val*step, val*step+win)
                # tm, tM = self._time.min(), self._time.max()
                cursor = tm + ((tM - tm) * event.pos[0] / canvas.size[0])
                for i, k in self._chan:
                    self._chan.node[i].transform.center = (cursor, 0.)
                    k.update()
                tm, tM = self._time.min(), self._time.max()

        @canvas.events.mouse_press.connect
        def on_mouse_press(event):
            """Executed function when single click mouse over canvas.

            Magnigy the signal under the mouse cursor only.
            """
            # Get canvas name :
            name = canvas.title
            condition = bool(name.find('Canvas') + 1)
            if condition and not self._slMagnify.isChecked():
                # Get channel name :
                chan = name.split('Canvas_')[1]
                # Get index :
                idx = self._channels.index(chan)
                # Get cursor position :
                val = self._SlVal.value()
                step = self._SigSlStep.value()
                win = self._SigWin.value()
                tm, tM = (val*step, val*step+win)
                cursor = tm + ((tM - tm) * event.pos[0] / canvas.size[0])
                # Build transformation :
                kwargs = {'center': (cursor, 0.), 'radii': (3, 15), 'mag': 10}
                transform = vist.nonlinear.Magnify1DTransform(**kwargs)
                self._chan.node[idx].transform = transform


class visuals(vbShortcuts):
    """Create the visual objects to be added to the scene."""

    def __init__(self):
        """Init."""
        # =================== VARIABLES ===================
        sf, data, time = self._sf, self._data, self._time
        channels, hypno, cameras = self._channels, self._hypno, self._allCams
        method = self._linemeth

        # =================== CHANNELS ===================
        self._chan = ChannelPlot(channels, time, camera=cameras[0],
                                 method=method, color=self._chancolor,
                                 width=self._lw, color_detection=self._indicol,
                                 parent=self._chanCanvas,
                                 fcn=self._fcn_sliderMove)

        # =================== SPECTROGRAM ===================
        # Create a spectrogram object :
        self._spec = Spectrogram(camera=cameras[1], fcn=self._fcn_specSetData,
                                 parent=self._specCanvas.wc.scene)
        self._spec.set_data(sf, data[0, ...], time, cmap=self._defcmap)
        # Create a visual indicator for spectrogram :
        self._specInd = Indicator(name='spectro_indic', visible=True, alpha=.3,
                                  parent=self._specCanvas.wc.scene)
        self._specInd.set_data(xlim=(0, 30), ylim=(0, 20))

        # =================== HYPNOGRAM ===================
        # Create a hypnogram object :
        self._hyp = Hypnogram(time, camera=cameras[2], color=self._hypcolor,
                              width=self._lwhyp,
                              parent=self._hypCanvas.wc.scene)
        self._hyp.set_data(sf, hypno, time)
        # Create a visual indicator for hypnogram :
        self._hypInd = Indicator(name='hypno_indic', visible=True, alpha=.3,
                                 parent=self._hypCanvas.wc.scene)
        self._hypInd.set_data(xlim=(0., 30.), ylim=(-6., 2.))

        vbcanvas = self._chanCanvas + [self._specCanvas, self._hypCanvas]
        for k in vbcanvas:
            vbShortcuts.__init__(self, k.canvas)

        # Initialize popup window with shotcuts :
        self._shpopup.set_shortcuts(self.sh)