import numpy as np
import torch
import torch.nn as nn
import math


class Deform_IK:
    def __init__(self, vismask_thrd=0.35):
        self.vismask_thrd = vismask_thrd
        self.crit = nn.MSELoss(reduction='none')

    @staticmethod
    def transform_from_euler(rotation, order='xyz'):
        #rotation = rotation / 180 * math.pi
        transform = torch.matmul(Deform_IK.transform_from_axis(rotation[..., 1], order[1]),
                                 Deform_IK.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(Deform_IK.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    def FK(self, locals, offsets, root_id, parent, root_translation):
        globals = list(torch.chunk(locals, int(locals.shape[0]), dim=0))
        jpos_res = torch.zeros_like(offsets)
        jpos_res[root_id] = offsets[root_id] + root_translation
        this_level = [root_id]
        while this_level:
            next_level = []
            for p_id in this_level:
                ch_list = np.argwhere(parent==p_id).squeeze(axis=1)
                for ch_id in ch_list:
                    globals[ch_id] = torch.matmul(globals[p_id], locals[ch_id])
                    jpos_res[ch_id] = torch.matmul(globals[p_id], offsets[ch_id][:, None]).squeeze() + jpos_res[p_id]
                next_level+=ch_list.tolist()
            this_level = next_level
        globals = torch.cat(globals, dim=0)
        return globals, jpos_res

    def run(self, locals_in, offsets, parent, root_id, vert_local, skinning, constraints, vismask, iter_time=100, lr=5e-2, w_invis=0.0):
        self.locals = locals_in
        self.offsets = offsets
        self.parent = parent
        self.root_id = root_id
        self.vert_local = vert_local
        self.skinning = skinning
        self.constraints = constraints.clone()
        self.vismask = (vismask > self.vismask_thrd).float().detach()
        self.vismask[self.vismask==0] = w_invis

        self.rotation_angles = torch.ones((len(self.locals), 3), dtype=torch.float32, device=self.locals.device) * 0.01

        # add contrain if necessary
        # self.mask = torch.ones_like(self.rotation_angles)
        # self.mask[:, 2] = 0.05
        # self.mask[:, 1] = 0.1

        #self.mask = torch.ones_like(self.rotation_angles)
        #self.mask[:, 1] = 0.2

        # self.mask = torch.ones_like(self.rotation_angles)
        # self.mask[11] = 0.0
        # self.mask[14] = 0.01
        # self.mask[16] = 0.01
        # self.mask[2] = 0.01
        # self.mask[8] = 0.0

        # self.mask = torch.ones_like(self.rotation_angles)
        # self.mask[2] = 0.0
        # self.mask[3, 1] = 0.2
        # self.mask[6] = 0.01
        # self.mask[13:16] = 0.0


        self.translation = torch.ones(3, dtype=torch.float32, device=self.locals.device) * 0.01
        self.rotation_angles.requires_grad_(True)
        self.translation.requires_grad_(True)
        self.optimizer = torch.optim.Adam([{'params': self.rotation_angles, 'lr': lr*math.pi},
                                           {'params': self.translation, 'lr': lr}],
                                          lr=0.1, betas=(0.9, 0.999), weight_decay=1e-4)

        for i in range(iter_time):
            self.optimizer.zero_grad()
            self.rotations = Deform_IK.transform_from_euler(self.rotation_angles.clone())
            #self.rotations = Deform_IK.transform_from_euler(self.rotation_angles.clone() * self.mask)
            locals = torch.matmul(self.rotations, self.locals)
            globals, jpos = self.FK(locals, self.offsets, self.root_id, self.parent, self.translation)
            vert_src_update = torch.matmul(globals, self.vert_local[:,0:3,:]) + jpos[..., None]
            vert_src_update = torch.sum(vert_src_update * self.skinning.T[:, None, :], dim=0).T
            loss = self.crit(vert_src_update, self.constraints)
            loss = (loss * self.vismask[:, None]).mean()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            #print(f"iter {i}, loss: {loss.item()}")
        return locals, globals, jpos

