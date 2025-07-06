import json
import os
import time
import pickle
import vtk
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from scipy import sparse
from vtkmodules.numpy_interface import dataset_adapter as dsa
from fimpy.solver import FIMPY
import meshio


def Bmatrix(pts, elm):
    nodeCoords = np.moveaxis(pts[elm, :], 0, -1)
    x1 = nodeCoords[0, 0, :]; x2 = nodeCoords[1, 0, :]; x3 = nodeCoords[2, 0, :]; x4 = nodeCoords[3, 0, :]
    y1 = nodeCoords[0, 1, :]; y2 = nodeCoords[1, 1, :]; y3 = nodeCoords[2, 1, :]; y4 = nodeCoords[3, 1, :]
    z1 = nodeCoords[0, 2, :]; z2 = nodeCoords[1, 2, :]; z3 = nodeCoords[2, 2, :]; z4 = nodeCoords[3, 2, :]
    x14 = x1 - x4; x34 = x3 - x4; x24 = x2 - x4
    y14 = y1 - y4; y34 = y3 - y4; y24 = y2 - y4
    z14 = z1 - z4; z34 = z3 - z4; z24 = z2 - z4
    detJ = x14 * (y34 * z24 - z34 * y24) - y14 * (x34 * z24 - z34 * x24) + z14 * (x34 * y24 - y34 * x24)
    Jinv_11 =        y34 * z24 - y24 * z34 ; Jinv_12 = -1. * (x34 * z24 - x24 * z34); Jinv_13 =        x34 * y24 - x24 * y34
    Jinv_21 = -1. * (y14 * z24 - y24 * z14); Jinv_22 =        x14 * z24 - x24 * z14 ; Jinv_23 = -1. * (x14 * y24 - x24 * y14)
    Jinv_31 =        y14 * z34 - y34 * z14 ; Jinv_32 = -1. * (x14 * z34 - x34 * z14); Jinv_33 =        x14 * y34 - x34 * y14
    Jinv = np.array([[Jinv_11, Jinv_12, Jinv_13],
                     [Jinv_21, Jinv_22, Jinv_23],
                     [Jinv_31, Jinv_32, Jinv_33]])
    B_def = np.array([[1., 0., 0., -1.],
                      [0., 0., 1., -1.],
                      [0., 1., 0., -1.]])
    B = Jinv.T @ B_def[...,:]
    return B, detJ

def localStiffnessMatrix(B, J, G):
    K = np.einsum('nji,njk,nkl->nil', B, G, B) / 6.0
    K = 1 / J[:, None, None] * K
    return K


class MyocardialMesh:
    """VTK mesh of endocardium (left, right or both)"""
    SIDE_LV = 1
    SIDE_RV = 2

    def __init__(self,
                 myo_mesh,
                 electrodes_position=None,
                 fibers=None,
                 device='cpu',
                 conductivity_params=None,
                 lead_fields_dict=None):
        if electrodes_position is not None and lead_fields_dict is not None:
            raise ValueError('Specify either electrodes_position or lead_fields_dict, not both')

        reader = vtk.vtkDataSetReader()
        reader.SetFileName(myo_mesh)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        mesh_endo = reader.GetOutput()
        self.vtk_mesh = mesh_endo
        self.device = device
        loc = vtk.vtkCellLocator()
        loc.SetDataSet(self.vtk_mesh)
        loc.BuildLocator()
        self.vtk_locator = loc
        dd = dsa.WrapDataObject(self.vtk_mesh)
        act = np.empty(dd.Points.shape[0])
        act.fill(np.inf)
        dd.PointData.append(act, "activation")
        self.xyz = dd.Points
        self.cells = dd.Cells.reshape((-1, 5))[:, 1:]

        if electrodes_position is not None:
            with open(electrodes_position, 'rb') as input_file:
                self.electrode_pos = pickle.load(input_file)
        else:
            self.electrode_pos = None

        f0_reader = vtk.vtkDataSetReader()
        f0_reader.SetFileName(fibers)
        f0_reader.ReadAllVectorsOn()
        f0_reader.ReadAllScalarsOn()
        f0_reader.Update()
        f0 = f0_reader.GetOutput()
        dd_f0 = dsa.WrapDataObject(f0)
        xyz_f0 = dd_f0.Points
        if 'fiber' in dd_f0.PointData.keys():
            fiber_directions = dd_f0.PointData['fiber']
            distances = cdist(self.xyz, xyz_f0)
            closest_indices = np.argmin(distances, axis=1)
            l = fiber_directions[closest_indices]
            assert len(closest_indices) == len(set(closest_indices))
            assert max(distances[np.arange(len(self.xyz)), closest_indices]) < 1e-3
            mesh = pv.UnstructuredGrid(mesh_endo)
            mesh["l"] = l
            mesh_cell_data = mesh.point_data_to_cell_data()
        elif 'fiber' in dd_f0.CellData.keys():
            fiber_directions = dd_f0.CellData['fiber']
            mesh_cell_data = pv.UnstructuredGrid(mesh_endo)
            mesh_cell_data["l"] = fiber_directions
            mesh_convert_data = pv.UnstructuredGrid(mesh_endo)
            mesh_convert_data["l"] = fiber_directions
            mesh_point_data = mesh_convert_data.cell_data_to_point_data()
            l_vtkDataArray = dsa.numpyTovtkDataArray(mesh_point_data["l"])
            l = dsa.vtkDataArrayToVTKArray(l_vtkDataArray)
        else:
            raise ValueError("Fibers directions should be named 'fiber'")
        l_cell_norms = np.linalg.norm(mesh_cell_data['l'], axis=1, keepdims=True)
        l_cell = mesh_cell_data['l'] / l_cell_norms
        l_cell = l_cell.astype(np.float64)
        if conductivity_params is None:
            sigma_il = 3.0
            sigma_el = 3.0
            sigma_it = 0.3
            sigma_et = 1.2
            alpha = 2.0
            beta = 800.
        else:
            sigma_il = conductivity_params['sigma_il']
            sigma_el = conductivity_params['sigma_el']
            sigma_it = conductivity_params['sigma_it']
            sigma_et = conductivity_params['sigma_et']
            alpha = conductivity_params['alpha']
            beta = conductivity_params['beta']
        I = np.eye(self.xyz.shape[1])
        sigma_mt = (sigma_et * sigma_it) / (sigma_et + sigma_it)
        sigma_ml = (sigma_el * sigma_il) / (sigma_el + sigma_il)
        Gm = sigma_mt * I + (sigma_ml - sigma_mt) * l_cell[:, :, np.newaxis] @ l_cell[:, np.newaxis, :]
        self.D = alpha**2 / beta * Gm * 100.
        l_nodes_norms = np.linalg.norm(l, axis=1, keepdims=True)
        l_nodes = l / l_nodes_norms
        l_nodes = l_nodes.astype(np.float64)
        self.Gi_nodal = sigma_it * I + (sigma_il - sigma_it) * l_nodes[:, :, np.newaxis] @ l_nodes[:, np.newaxis, :]
        self.Gi_cell = sigma_it * I + (sigma_il - sigma_it) * l_cell[:, :, np.newaxis] @ l_cell[:, np.newaxis, :]
        self.K = self.assemble_K(self.xyz, self.cells, self.Gi_cell)
        if electrodes_position is not None:
            self.aux_int_Vl = self.new_get_ecg_aux_Vl()
            self.lead_field = self.get_lead_field()
        elif lead_fields_dict is not None:
            self.aux_int_Vl = None
            self.lead_field = self.get_lead_field(lead_fields_dict)
        else:
            self.aux_int_Vl = None
            self.lead_field = None
        start_time = time.time()
        cells = dd.Cells.reshape((-1, 5))[:, 1:]
        self.fim = FIMPY.create_fim_solver(self.xyz, cells, self.D, device=device)
        print(time.time() - start_time)

    def assemble_K(self, pts, elm, Gi):
        B, J = Bmatrix(pts, elm)
        K = localStiffnessMatrix(B, J, Gi)
        N = pts.shape[0]
        Kvals = K.ravel('C')
        II = np.repeat(elm, 4, axis=1).ravel()
        JJ = np.tile(elm, 4).ravel()
        Kp = sparse.coo_matrix((Kvals, (II, JJ)), shape=(N, N)).tocsr()
        return Kp

    def find_closest_pmjs(self, pmjs):
        loc = self.vtk_locator
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        ppmjs = np.zeros_like(pmjs)
        for k in range(pmjs.shape[0]):
            loc.FindClosestPoint(pmjs[k, :], ppmjs[k, :], cellId, subId, d)
        return ppmjs

    def probe_activation(self, x0):
        x0 = self.find_closest_pmjs(x0)
        vtk_points = vtk.vtkPoints()
        for p in x0:
            vtk_points.InsertNextPoint(p)
        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(self.vtk_mesh)
        probe.SetInputData(vtk_poly)
        probe.Update()
        pout = dsa.WrapDataObject(probe.GetOutput())
        act = pout.PointData['activation']
        return act

    def activate_fim(self, x0, x0_vals, return_only_pmjs=False):
        if return_only_pmjs:
            x0_purkinje = x0.copy()
        dd = dsa.WrapDataObject(self.vtk_mesh)
        cells = dd.Cells.reshape((-1, 5))[:, 1:]
        xyz = dd.Points
        act = np.empty(xyz.shape[0])
        act.fill(np.inf)
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        x_proj = [0.0, 0.0, 0.0]
        dist = vtk.reference(0.0)
        print('computing closest nodes to PMJs')
        start_time = time.time()
        for k in range(x0.shape[0]):
            x_orig = x0[k, :]
            self.vtk_locator.FindClosestPoint(x_orig, x_proj, cellId, subId, dist)
            Gcell = np.linalg.inv(self.D[cellId, ...])
            cell_pts = cells[cellId, :]
            v = xyz[cell_pts, :] - x0[[k], :]
            new_act = x0_vals[k] + np.sqrt(np.einsum('ij,kj,ki->k', Gcell, v, v))
            act[cell_pts] = np.minimum(new_act, act[cell_pts])
        print(time.time() - start_time)
        x0 = np.isfinite(act)
        x0_vals = act[x0]
        print('solving')
        start_time = time.time()
        act = self.fim.comp_fim(x0, x0_vals)
        print(time.time() - start_time)
        if self.device == 'gpu':
            act = act.get()
        dd.PointData['activation'][:] = act
        if return_only_pmjs:
            x0_pv = pv.PolyData(x0_purkinje)
            result = x0_pv.sample(pv.UnstructuredGrid(self.vtk_mesh), tolerance=1e-6, snap_to_closest_point=True)
            assert np.sum(result['vtkValidPointMask'] == 0) == 0, 'Error while sampling to x0_purkinje'
            return result['activation']
        else:
            return act

    def save_meshio(self, fname, point_data=None, cell_data=None):
        dd = dsa.WrapDataObject(self.vtk_mesh)
        cells = dd.Polygons.reshape((-1, 4))[:, 1:]
        xyz = dd.Points
        mesh = meshio.Mesh(points=xyz, cells={'triangle': cells})
        mesh.point_data = point_data or {}
        mesh.cell_data = cell_data or {}
        mesh.write(fname)

    def save(self, fname):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(self.vtk_mesh)
        writer.SetFileName(fname)
        writer.Update()

    def save_pv(self, fname):
        save_mesh = pv.UnstructuredGrid(self.vtk_mesh)
        save_mesh.save(fname)

    def new_get_ecg(self, record_array=True):
        if self.lead_field is None:
            raise ValueError('Lead field not defined for this mesh')
        dd = dsa.WrapDataObject(self.vtk_mesh)
        u = dd.PointData['activation']
        cells = dd.Cells.reshape((-1, 5))[:, 1:]
        V0, V1 = -80, 20
        eps = 1.0
        umin, umax = np.min(u), np.max(u)
        uu = u - umin
        req_time_ini = -5.
        req_time_fin = 200.
        n_times = int(req_time_fin - req_time_ini + 1)
        electrode_names = self.electrode_pos.keys() if self.electrode_pos is not None else self.lead_field.keys()
        V_l_dict = {key: np.empty(n_times) for key in electrode_names}
        save_Vm = False
        save_gradU = False
        for n_t, req_time in enumerate(np.linspace(req_time_ini, req_time_fin, n_times)):
            Vm = V0 + (V1 - V0) / 2 * (1 + np.tanh((req_time - uu) / eps))
            if save_Vm:
                dd.PointData.append(Vm, f"Vm_{req_time:03d}")
            for electrode_name in electrode_names:
                V_l_dict[electrode_name][n_t] = np.dot(self.lead_field[electrode_name], self.K.dot(Vm))
            if save_gradU:
                dd.PointData.append(grad_u, f"U_grad_{n_t:03d}")
        if save_Vm or save_gradU:
            self.save_pv("test_vm.vtu")
        leads_names = ["E1", "E2", "E3", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        V_W = 1. / 3. * (V_l_dict["RA"] + V_l_dict["LA"] + V_l_dict["LL"])
        arrays = []
        for l_name in leads_names:
            if l_name == "E1":
                arrays.append(V_l_dict["LA"] - V_l_dict["RA"])
            elif l_name == "E2":
                arrays.append(V_l_dict["LL"] - V_l_dict["RA"])
            elif l_name == "E3":
                arrays.append(V_l_dict["LL"] - V_l_dict["LA"])
            elif l_name == "aVR":
                arrays.append(3. / 2. * (V_l_dict["RA"] - V_W))
            elif l_name == "aVL":
                arrays.append(3. / 2. * (V_l_dict["LA"] - V_W))
            elif l_name == "aVF":
                arrays.append(3. / 2. * (V_l_dict["LL"] - V_W))
            else:
                arrays.append(V_l_dict[l_name] - V_W)
        if record_array:
            ecg_pat_array = np.rec.fromarrays(arrays, names=leads_names)
        else:
            ecg_pat_array = np.asarray(arrays)
        return ecg_pat_array

    def new_get_ecg_aux_Vl(self):
        if self.electrode_pos is None:
            raise ValueError('Electrodes positions not available')
        dd = dsa.WrapDataObject(self.vtk_mesh)
        xyz = dd.Points
        aux_int_l = {}
        for electrode_name, electrode_coords in self.electrode_pos.items():
            r = xyz - np.array(electrode_coords)
            r_norm3 = r / np.linalg.norm(r, axis=1) ** 3
            Gi_nodal_T = np.transpose(self.Gi_nodal, axes=(0, 2, 1))
            aux_int_l[electrode_name] = np.sum(Gi_nodal_T * r_norm3, axis=1)
        return aux_int_l

    def get_lead_field(self, lead_fields_dict=None):
        if lead_fields_dict is not None:
            if isinstance(lead_fields_dict, str):
                with open(lead_fields_dict) as f:
                    aux_int_l = json.load(f)
            elif isinstance(lead_fields_dict, dict):
                aux_int_l = lead_fields_dict
            else:
                raise ValueError('lead_fields_dict must be a dict or a path to json')
            return {k: np.asarray(v) for k, v in aux_int_l.items()}
        if self.electrode_pos is None:
            raise ValueError('Electrodes positions not available')
        aux_int_l = {}
        for electrode_name, electrode_coords in self.electrode_pos.items():
            r = self.xyz - np.array(electrode_coords)
            aux_int_l[electrode_name] = 1 / np.linalg.norm(r, axis=1)
        return aux_int_l
