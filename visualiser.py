import numpy as np
from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group
import moviepy.editor as mpy
from mayavi import mlab
from tvtk.tools import visual
from mayavi.core.ui.api import MlabSceneModel, SceneEditor


class MyDialog(HasTraits):

    # @mlab.animate(delay=100)
    # def anim(self):
    #     while 1:
    #         for d in self.circle:
    #             print('Updating scene...')
    #             mlab.view(azimuth=d)
    #             yield

    def make_frame(self, t):   
        mlab.clf()
        mlab.pipeline.volume(self.sfa, figure=self.scene.mayavi_scene)
        angle, _, _, _ = mlab.view()
        mlab.view(azimuth=angle+2)
        f = mlab.gcf()
        f.scene._lift()
        return mlab.screenshot()

    def display(self):
        scene = Instance(MlabSceneModel, ())

        # self.rest[self.label>0] = 0
        sfa_data = mlab.pipeline.scalar_field(self.rest, figure=scene.mayavi_scene)
        sfa_label = mlab.pipeline.scalar_field(self.label, figure=scene.mayavi_scene)
        sfa_data.spacing = self.spacing
        sfa_label.spacing = self.spacing
        mlab.pipeline.volume(sfa_data, figure=scene.mayavi_scene)
        mlab.pipeline.volume(sfa_label, figure=scene.mayavi_scene, color=(1,1,0))

        # self.anim()
        # mlab.show()
        # animation = mpy.VideoClip(self.make_frame, duration = 18)
        # animation.write_gif("b0.gif", fps=10)

