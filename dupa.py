from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
from traits.api import Instance
from mayavi.core.ui.api import MlabSceneModel
import moviepy.editor as mpy

# @mlab.animate(delay=100)
# def anim():
#     f = mlab.gcf()
#     while 1:
#         for d in range(360):
#             print('Updating scene...')
#             mlab.view(azimuth=d)
#             yield

def make_frame(t):   
    mlab.clf()
    mlab.pipeline.volume(sfa, figure=scene.mayavi_scene)
    angle, _, _, _ = mlab.view()
    mlab.view(azimuth=angle+1)
    f = mlab.gcf()
    f.scene._lift()
    return mlab.screenshot()

scene = Instance(MlabSceneModel, ())
volume = np.random.rand(10,10,10)
sfa = mlab.pipeline.scalar_field(volume, figure=scene.mayavi_scene)    
mlab.pipeline.volume(sfa, figure=scene.mayavi_scene)
# anim()
# mlab.show()
animation = mpy.VideoClip(make_frame, duration = 5)
animation.write_gif("animation.gif", fps=10)