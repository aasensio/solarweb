from flask import Flask, render_template, send_file
from flask_socketio import SocketIO, emit, Namespace
import numpy as np
import json
import drms
import os
import scipy.ndimage as nd
from astropy.io import fits
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import ColumnDataSource, CustomJS, Rect, Slider, HoverTool
from bokeh.layouts import row, widgetbox, column
from bokeh.models.widgets import TextInput, RadioGroup, Button, RadioButtonGroup
from bokeh.events import ButtonClick
import enhance as neural
import hazel

from ipdb import set_trace as stop

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

c = drms.Client()

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/codes', methods=['GET'])
def codes_page():
    return render_template('codes/codes.html')

@app.route('/enhance', methods=['GET'])
def enhance_page():
    return render_template('codes/enhance.html')

@app.route('/hazel', methods=['GET'])
def hazel_page():
    return render_template('codes/hazel.html')

@app.route('/people', methods=['GET'])
def people_page():
    return render_template('people.html')

@app.route('/offline_codes', methods=['GET'])
def offline_codes_page():
    return render_template('offline/offline.html')

@app.route('/sir_offline', methods=['GET'])
def offline_sir_page():
    return render_template('offline/sir.html')

@app.route('/nicole_offline', methods=['GET'])
def offline_nicole_page():
    return render_template('offline/nicole.html')

@app.route('/hazel_offline', methods=['GET'])
def offline_hazel_page():
    return render_template('offline/hazel.html')

@app.route('/download_fits', methods=['POST'])
def download_fits():
    return send_file('tmp.fits', attachment_filename='deconvolved.fits', mimetype='application/octet-stream', as_attachment=True)

class custom_namespace(Namespace):        
    
    def on_connect(self):
        # need visibility of the global thread object
        print('Client connected')

    def on_click_download_sdo(self):
        # First some sanity checks
        d, mo, y = self.date_input.value.split('/')

        h, m = self.time_input.value.split(':')

        if (self.which == 'intensity'):
            query = 'hmi.Ic_noLimbDark_720s[{0}.{1}.{2}_{3}:{4}:00_TAI/10m]'.format(y,mo,d,h,m)
            k, s = c.query(query, key='T_REC, DATAMEAN, OBS_VR', seg='continuum')
            url = 'http://jsoc.stanford.edu{0}'.format(s.continuum[0])
        else:
            query = 'hmi.m_720s[{0}.{1}.{2}_{3}:{4}:00_TAI/10m]'.format(y,mo,d,h,m)
            k, s = c.query(query, key='T_REC, DATAMEAN, OBS_VR', seg='magnetogram')
            url = 'http://jsoc.stanford.edu{0}'.format(s.magnetogram[0])

        socketio.emit('label2', {'data': json.dumps(url)}, namespace='/test')

        socketio.emit('label', {'data': json.dumps('Downloading...')}, namespace='/test')

        socketio.sleep(0.1)

        print('Getting file {0}...'.format(url))
        data = np.nan_to_num(fits.getdata(url)).astype('float')

        self.image_original = data
        self.image_original /= np.max(self.image_original)
        self.image_original = np.flipud(np.fliplr(self.image_original))

        tmp = nd.zoom(data, 0.1)
        tmp /= np.max(tmp)
        tmp = np.flipud(np.fliplr(tmp))

        self.image = tmp

        self.dims = tmp.shape

        # tmp *= 255
        #data = tmp.astype('int')
        print('Downloaded', data.shape, np.max(data))

        self.source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})

        jscode="""
            var data = source.data;
            var start = cb_obj.start;
            var end = cb_obj.end;
            data['%s'] = [start + (end - start) / 2];
            data['%s'] = [end - start];
            source.change.emit();
            //var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
            socket.emit('modified', data);
        """

        self.p1 = figure(title='Pan and Zoom Low Resolution', x_range=(0, self.dims[0]), y_range=(0, self.dims[1]),
                tools='box_zoom,wheel_zoom,pan,reset', plot_width=400, plot_height=400)
        self.p1.image(image=[tmp], x=0, y=0, dw=self.dims[0], dh=self.dims[1], palette="Greys256")
        # p1.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

        self.p1.x_range.callback = CustomJS(
            args=dict(source=self.source), code=jscode % ('x', 'width'))
        self.p1.y_range.callback = CustomJS(
            args=dict(source=self.source), code=jscode % ('y', 'height'))

        self.p2 = figure(title='See Zoom Window Here', x_range=(0, self.dims[0]), y_range=(0, self.dims[1]),
                    tools='', plot_width=400, plot_height=400)
        self.p2.image(image=[tmp], x=0, y=0, dw=self.dims[0], dh=self.dims[1], palette="Greys256")
        rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.1,
                    line_color='black', fill_color='black')
        self.p2.add_glyph(self.source, rect)
    
        layout = row(self.p1, self.p2)

        script, div = components(layout)

        socketio.emit('update_figure', {'script': json.dumps(script), 'div': json.dumps(div)}, namespace='/test')

        socketio.emit('label2', {'data': json.dumps('')}, namespace='/test')
        socketio.emit('label', {'data': json.dumps('Ready')}, namespace='/test')

        socketio.sleep(0.1)

        self.downloaded = True

    def on_click_deconvolve(self):

        if (self.downloaded):
        
            width = 10*int(self.data['width'][0])
            height = 10*int(self.data['height'][0])
            x = int(10*self.data['x'][0] - width / 2)
            y = int(10*self.data['y'][0] - height / 2)        

            if ((width > 400) or (height > 400)):
                socketio.emit('label', {'data': json.dumps('Image too large for deconvolution')}, namespace='/test')

            else:
                self.im = self.image_original[y:y+height,x:x+width]

                socketio.emit('label', {'data': json.dumps('Deconvolving...')}, namespace='/test')
                socketio.sleep(1)

                out = neural.enhance()
                out.define_network(image=self.im, network=self.which)
                self.tmp, elapsed = out.predict()

                self.p3 = figure(title='Original', x_range=(0, width), y_range=(0, height),
                    tools='box_zoom,wheel_zoom,pan,reset', plot_width=400, plot_height=400)
                self.p3.image(image=[self.im], x=0, y=0, dw=width, dh=height, palette="Greys256")

                self.p4 = figure(title='Deconvolved', x_range=(0, width), y_range=(0, height),
                    tools='box_zoom,wheel_zoom,pan,reset', plot_width=400, plot_height=400)
                self.p4.image(image=[self.tmp], x=0, y=0, dw=width, dh=height, palette="Greys256")

                layout = row(self.p3, self.p4)

                script, div = components(layout)

                socketio.emit('update_deconvolved', {'script': json.dumps(script), 'div': json.dumps(div)}, namespace='/test')

                socketio.emit('label', {'data': json.dumps('Elapsed time: {0:.3f} s'.format(elapsed))}, namespace='/test')
                socketio.sleep(0.1)

                self.deconvolved = True
        else:
            socketio.emit('label', {'data': json.dumps('SDO image not yet downloaded')}, namespace='/test')
            socketio.sleep(0.1)
                
    def on_modified(self, data):
        self.data = data

    def on_click_type(self):
        if (self.which == 'intensity'):
            self.which = 'magnetogram'
        else:
            self.which = 'intensity'

    def on_click_download(self):

        if (self.downloaded and self.deconvolved):
            hdu1 = fits.PrimaryHDU(self.tmp)
            hdu2 = fits.ImageHDU(self.im)
            new_hdul = fits.HDUList([hdu1, hdu2])
            new_hdul.writeto('tmp.fits', overwrite=True)

            socketio.emit('download_fits_file', namespace='/test')
        else:
            socketio.emit('label', {'data': json.dumps('SDO image not yet downloaded and/or deconvolved')}, namespace='/test')
            socketio.sleep(0.1)        
        
        # os.remove('tmp.fits')

    def on_ready(self):
        self.downloaded = False
        self.deconvolved = False
        self.which = 'intensity'

        self.date_input = TextInput(value="12/03/2016", title="Date:")
        self.time_input = TextInput(value="12:00", title="Time:")
        self.radio_group = RadioGroup(labels=["Continuum", "Magnetogram"], active=0, inline=True, 
            callback=CustomJS(code="socket.emit('click_type');"))

        self.download_sdo = Button(label="Get from SDO", button_type="success")
        self.deconvolve = Button(label="Deconvolve", button_type="success", disabled=False)
        self.download = Button(label="Download", button_type="success", disabled=False)

        
        self.download_sdo.js_on_event(ButtonClick, CustomJS(code="socket.emit('click_download_sdo');"))

        self.deconvolve.js_on_event(ButtonClick, CustomJS(code="socket.emit('click_deconvolve');"))

        self.download.js_on_event(ButtonClick, CustomJS(code="socket.emit('click_download');"))

        t1 = row(self.date_input, self.time_input, self.radio_group)

        t2 = row(self.download_sdo, self.deconvolve, self.download)

        t3 = column(t1, t2)

        script, div = components(t3)


        # script, div = components(WidgetBox(self.date_input, self.time_input, self.radio_group, 
        #     self.download_sdo, self.deconvolve, self.download))

        socketio.emit('update_buttons', {'script': json.dumps(script), 'div': json.dumps(div)}, namespace='/test')


class custom_namespace_hazel(Namespace):        
    
    def on_connect(self):
        # need visibility of the global thread object
        print('Client connected to hazel')

    def callback(self, modified_variable, radio_button=False):

        if (radio_button):
            value = 'active'
        else:
            value = 'value'

        jscode = """
            var vars = variables.data;
            var data = source.data;
            var value = cb_obj.{1};

            vars['{0}'] = [value];

            console.log(vars);
                        
            socket.emit('modified', vars);

            replot_after_computed(source);     
            
        """

        return CustomJS(args=dict(source=self.source, variables=self.variables), code=jscode.format(modified_variable, value))

    def on_ready(self):

        try:
            self.model.exit()
        except:
            pass        

        self.model = hazel.Model(working_mode='synthesis')                

        topology = 'ch1'
                
        self.model.add_spectral({'Name': 'spec1', 'Wavelength': [10826, 10833, 100], 'topology': topology, 
            'LOS': [0.,0.,90.], 'boundary condition': [1,0,0,0]})
        self.model.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': 3.0, 'Line': '10830', 
            'Wavelength': [10826, 10833]})
        self.model.setup()
        self.model.atmospheres['ch1'].set_parameters([0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 1.0, 0.0], 1.0)
        self.model.synthesize()

        stokes = self.model.spectrum['spec1'].stokes
        l = self.model.spectrum['spec1'].wavelength_axis

        socketio.emit('update_stokes', {'l': json.dumps(l.tolist()), 'stI': json.dumps(stokes[0,:].tolist()), 'stQ': json.dumps(stokes[1,:].tolist()), 
            'stU': json.dumps(stokes[2,:].tolist()), 'stV': json.dumps(stokes[3,:].tolist())}, namespace='/hazel')

        
        # Define the plots

        hover = HoverTool(tooltips=[("(x,y)", "($x, $y)"),])

        self.source = ColumnDataSource({'lambda': l, 'stI': stokes[0,:], 'stQ': stokes[1,:], 'stU': stokes[2,:], 'stV': stokes[3,:]})

        self.variables = ColumnDataSource({'B': [0.], 'thB': [0.], 'phiB': [0.], 'th': [0.], 'phi': [0.], 'gamma': [90.], 'h': [3.], 'vd': [8.], 
            'tau': [1.0], 'v': [0.0], 'a': [0.0], 'mode': [0]})

        
        self.sI = figure(title='Stokes I', x_range=(10826, 10833),
                tools='box_zoom,wheel_zoom,pan,reset,hover', plot_width=300, plot_height=300)
        self.sI.line('lambda', 'stI', source=self.source)
        
        self.sQ = figure(title='Stokes Q', x_range=(10826, 10833),
                    tools='box_zoom,wheel_zoom,pan,reset,hover', plot_width=300, plot_height=300)
        self.sQ.line('lambda', 'stQ', source=self.source)

        self.sU = figure(title='Stokes U', x_range=(10826, 10833),
                tools='box_zoom,wheel_zoom,pan,reset,hover', plot_width=300, plot_height=300)
        self.sU.line('lambda', 'stU', source=self.source)

        self.sV = figure(title='Stokes V', x_range=(10826, 10833),
                tools='box_zoom,wheel_zoom,pan,reset,hover', plot_width=300, plot_height=300)
        self.sV.line('lambda', 'stV', source=self.source)

        t1 = row(self.sI, self.sQ)
        t2 = row(self.sU, self.sV)

        layout_figs = column(t1, t2)       
                    
        self.B_slider = Slider(start=0.0, end=1000, value=0, step=.1, title="B [G]", callback_policy='mouseup') #, callback=callback)
        self.thB_slider = Slider(start=0.0, end=180, value=0, step=.1, title=u'\u03B8'+ "B [deg]", callback_policy='mouseup') #, callback=callback)
        self.phiB_slider = Slider(start=0.0, end=360, value=0, step=.1, title=u'\u03C6'+ "B [deg]", callback_policy='mouseup') #, callback=callback)

        self.th_slider = Slider(start=0.0, end=180, value=0, step=.1, title=u'\u03B8'+ " [deg]", callback_policy='mouseup') #, callback=callback)
        self.phi_slider = Slider(start=0.0, end=360, value=0, step=.1, title=u'\u03C6'+ " [deg]", callback_policy='mouseup') #, callback=callback)
        self.gamma_slider = Slider(start=0.0, end=180, value=0, step=.1, title=u'\u03B3'+ ' [deg]', callback_policy='mouseup') #, callback=callback)

        self.h_slider = Slider(start=0.0, end=30, value=3, step=.01, title="h [arcsec]", callback_policy='mouseup') #, callback=callback)
        self.vdopp_slider = Slider(start=1.0, end=20, value=8, step=.01, title="vD [km/s]", callback_policy='mouseup') #, callback=callback)
        self.tau_slider = Slider(start=0.0, end=5, value=1, step=.01, title=u'\u03C4', callback_policy='mouseup') #, callback=callback)

        self.v_slider = Slider(start=-20.0, end=20, value=0, step=.01, title='v [km/s]', callback_policy='mouseup') #, callback=callback)
        self.a_slider = Slider(start=0.0, end=2.0, value=0, step=.01, title='a', callback_policy='mouseup') #, callback=callback)


        self.B_slider.callback = self.callback('B')
        self.thB_slider.callback = self.callback('thB')
        self.phiB_slider.callback = self.callback('phiB')

        self.th_slider.callback = self.callback('th')
        self.phi_slider.callback = self.callback('phi')
        self.gamma_slider.callback = self.callback('gamma')

        self.h_slider.callback = self.callback('h')        
        self.vdopp_slider.callback = self.callback('vd')
        self.tau_slider.callback = self.callback('tau')

        self.v_slider.callback = self.callback('v')
        self.a_slider.callback = self.callback('a')

        menu = [("On-disk", "ondisk"), ("Off-limb", "offlimb")]
        self.mode = RadioButtonGroup(labels=["On disk", "Off-limb"], active=0)

        self.mode.callback = self.callback('mode', radio_button=True)

        layout_widget = widgetbox(self.B_slider, self.thB_slider, self.phiB_slider, self.th_slider, self.phi_slider, 
            self.gamma_slider, self.h_slider, self.vdopp_slider, self.tau_slider, self.v_slider, self.a_slider, self.mode)

        layout = row(layout_widget, layout_figs)

        script, div = components(layout)
        
        socketio.emit('update_buttons', {'script': json.dumps(script), 'div': json.dumps(div)}, namespace='/hazel')        

    def on_modified(self, data):

        B = 1.0*data['B'][0]
        thB = data['thB'][0] * np.pi / 180.0
        phiB = data['phiB'][0] * np.pi / 180.0
        tau = 1.0*data['tau'][0]
        v = 1.0*data['v'][0]
        delta = 1.0*data['vd'][0]
        beta = 1.0
        a = 1.0*data['a'][0]
        th = 1.0*data['th'][0]
        phi = 1.0*data['phi'][0]
        gamma = 1.0*data['gamma'][0]

        Bx = B * np.sin(thB) * np.cos(phiB)
        By = B * np.sin(thB) * np.sin(phiB)
        Bz = B * np.cos(thB)

        # Option 1
        # self.model = hazel.Model(working_mode='synthesis')                
        # topology = 'ch1'
        # self.model.add_spectral({'Name': 'spec1', 'Wavelength': [10826, 10833, 100], 'topology': topology, 
        #     'LOS': [th, phi, gamma], 'boundary condition': [1,0,0,0]})
        # self.model.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': 3.0, 'Line': '10830', 
        #     'Wavelength': [10826, 10833]})
        # self.model.setup()
        
        self.model.atmospheres['ch1'].set_parameters([Bx, By, Bz, tau, v, delta, beta, a], 1.0)

        self.model.spectrum['spec1'].set_los(np.asarray([th, phi, gamma]).astype('float64'))
        
        if (data['mode'][0] == 0):
            self.model.spectrum['spec1'].set_boundary(np.asarray([1,0,0,0]).astype('float64'))
        if (data['mode'][0] == 1):
            self.model.spectrum['spec1'].set_boundary(np.asarray([0,0,0,0]).astype('float64'))

        self.model.spectrum['spec1'].set_los(np.asarray([th, phi, gamma]).astype('float64'))
        
        self.model.synthesize()

        stokes = self.model.spectrum['spec1'].stokes
        l = self.model.spectrum['spec1'].wavelength_axis

        socketio.emit('update_stokes', {'l': json.dumps(l.tolist()), 'stI': json.dumps(stokes[0,:].tolist()), 'stQ': json.dumps(stokes[1,:].tolist()), 
            'stU': json.dumps(stokes[2,:].tolist()), 'stV': json.dumps(stokes[3,:].tolist()), 'done': json.dumps([1])}, namespace='/hazel')
        
if (__name__ == '__main__'):
    socketio.on_namespace(custom_namespace('/test'))
    socketio.on_namespace(custom_namespace_hazel('/hazel'))
    socketio.run(app)

