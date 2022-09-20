import pandas as pd
import numpy as np

import os
from shutil import rmtree
from itertools import product
from datetime import datetime
import glob

from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.sgems import sg

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def abs_join_path(path1, path2):
    return os.path.abspath(os.path.join(path1, path2))

def get_datetime():
    return str(datetime.now().replace(second=0, microsecond=0)).replace(' ', '_').replace(':', '.')

def get_xyz(dimens=[60, 220, 85]):
    xyz = np.array(list(product([k for k in range(dimens[2])], [j for j in range(dimens[1])], [i for i in range(dimens[0])])))
    xyz[:, [0, 2]] = xyz[:, [2, 0]]
    return xyz

    
class Generator():
    '''Put your docstring here!'''
    def __init__(self, root: str, path_to_save: str, model='spe10m2', alg='ord_kriging', prime_var=None, second_var=None, cell_size=[20, 10, 2], dimens=[60, 220, 85], exe_name_sgems='sgems-x64.exe'):
        '''Arguments:
             root: str - a folder with original data of spe 10 model 2.
             path_to_save: str - a folder to save all files for this particular 'alpha'/'n_wells' and 'size'.
             model: str - a type of open source model to consider.
             alg: str - a chosen geostatistical algorithm.
             cell_size: list - a list with 3 ints that represents a dimension of cell in model.
             dimens: lsit - a list with 3 ints that represents dimenions of the model.
             exe_name_sgems: str - the name of .exe file. May vary with regard to installed verison of SGeMS.
        '''
        self.root = root
        self.path_to_save = path_to_save
        
        self.poro_full = None
        self.permx_full = None
        self.permy_full = None
        self.permz_full = None
        self.est_results = {}
        
        model = model.lower()
        if not model in ['spe10m2']:
            raise ValuerError('Only implemented for models: [spe10m2].')
        else:
            self.model = model
            
        alg = alg.lower()
        if not alg in ['ord_kriging', 'cokriging']:
            raise ValuerError('Only implemented for algorithms: [ord_kriging, cokriging].')
        else:
            self.alg = alg
        
        if (self.alg == 'cokriging') & (not prime_var):
            raise ValueError('You chose cokriging. Choose prime variable for the algorithm.')
            
        if (self.alg == 'cokriging') & (not prime_var):
            raise ValueError('You chose cokriging. Choose secondary variable for the algorithm.')
        if self.alg == 'cokriging':
            self.prime_var = prime_var
            self.second_var = second_var
            
        self.cell_size = cell_size
        self.dimens = dimens
        self.exe_name_sgems = 'sgems-x64.exe'
        
    def PreprocessRawData(self, save_dat_files: str, change_root=True):
        '''Save raw SPE10 properties to .dat with GSLIB format.
           
           Arguments:
             save_dat_files: str - a directory to save .dat files in GSLIB format.
             change_root: bool - pass True if want to change the root folder data where gslib files will
               be saved. To utilize SGeMS root folder has to contain .dat files.
        '''
        if save_dat_files == self.root:
            raise ValueError('Saving folder matches root folder. Please, choose another one.')
        if self.model == 'spe10m2':
            poro_spe = pd.read_csv(abs_join_path(self.root,'spe_phi.dat'), dtype='float64', engine='python', sep='        \t', header=None)
            perm_spe = pd.read_csv(abs_join_path(self.root,'spe_perm.dat'), dtype='float64', engine='python', sep='        \t', header=None)

            poro = np.array([poro_spe.iloc[i,j] for i in range(187000) for j in range(6)])
            permx = np.array([perm_spe.iloc[i,j] for i in range(187000) for j in range(6)])
            permy = np.array([perm_spe.iloc[i,j] for i in range(187000, 374000) for j in range(6)])
            permz = np.array([perm_spe.iloc[i,j] for i in range(374000, 187000*3) for j in range(6)])
            
            poro = poro.reshape(-1, 1)            
            permx = np.array(permx).reshape(-1, 1)
            permy = np.array(permy).reshape(-1, 1)
            permz = np.array(permz).reshape(-1, 1)
            
            xyz = get_xyz()        
            all_data = np.hstack((xyz, poro, permx, permy, permz))
            pd.DataFrame(
                all_data,
                columns=['x', 'y', 'z', 'poro', 'permx', 'permy', 'permz']
            ).to_csv(abs_join_path(save_dat_files, 'spe10_all_orig.dat'), sep=' ')
            
            print('=> .dat files saved to {}.'.format(abs_join_path(save_dat_files, 'spe10_all_orig.dat')))
        if change_root:
            self.root = save_dat_files
            print(f'=> root folder is {self.root} now.')
        return
    
    def __call__(self, size: int, alpha=None, n_wells=None, search_ellipsoid='400 350 100  0 90 0'):
        '''Arguments:
             size: int - a number of data sets to be created with this particular alpha/n_wells.
             alpha: float in (0, 1) - a part of points to be taken. If alpha=None considered that this run is for well data.
             n_wells: int - a number of unit wells to create data. If n_wells=None considered that this run is taking random points.
               Either 'alpha' or 'n_wells' has to be equal to 'None'.
             search_ellipsoid: str - a string of 6 values that represent a searching ellipsoid. First 3 values are ranges of the ellipsoid,
               last three values are angles of the ellipsoid. Values are separated by spaces and one double-space in the middle.
        '''
        self.run_name = get_datetime()
        self._set_generation_mode(size, alpha, n_wells)
        self._upload_data()        
        self._match_coord()
        self.search_ellipsoid = search_ellipsoid
        
        # Run kriging in SGeMS for specified times on every property of SPE10 model 2
        for number in range(1, self.size+1):
            print(f'\nRun {number} out of {self.size}.')

            # Take part fo inital data.
            if self.alpha: 
                self.work_dir = abs_join_path(self.size_dir, str(alpha*100)+'%_'+str(number))
                self._take_points()
            elif self.n_wells:
                self.work_dir = abs_join_path(self.size_dir, str(n_wells)+'wells_'+str(number))
                self._take_wells()
                             
            # Save taken part in GSLIB format for SGeMS
            os.mkdir(self.work_dir)
            print('   Saving GSLIB files to: ' + self.work_dir)
            self._save_gslib_all(number)

            self.est_results = {}
            '''Iterate through properties and run estimation for every property in a current run.'''
            if self.alg == 'ord_kriging':
                for prop_name in ['poro', 'permx', 'permy', 'permz']:
                    self.property_name = prop_name       # Name of current property
                    self._set_algorithm()

                    # Can change parameters of the chosen algorithm
    #                     self._print_alg_tree()
    #                     self._change_alg_params(search_ellipsoid)

                    self._run_alg()
                    kriging_res = pd.read_csv(abs_join_path(self.path_to_save, 'results/results.grid'), skiprows=3, names=[self.property_name]).to_numpy()
                    kriging_res = np.clip(kriging_res, a_min=0, a_max=None).ravel()
                    self.est_results[self.property_name] = kriging_res
                    print(f'   For {self.property_name} in run {number} estimation is completed.')

                    # Check for NaN's. '-9966699' is a NaN in SGeMS
                    num_nan = np.where(kriging_res == -9966699)[0].shape[0]
                    if num_nan:
                        num_of_points = self.dimens[0] * self.dimens[1] * self.dimens[2]
                        print(f"'Warning!\n'In run {number} property {self.property_name} has {np.round(num_nan*100/num_of_points, 2)}% of NaN's in the results.\n")
                        
            if self.alg == 'cokriging':
                    self._set_algorithm()
                    self._run_alg()
                    kriging_res = pd.read_csv(abs_join_path(self.path_to_save, 'results/results.grid'), skiprows=3, names=[self.prime_var]).to_numpy()
                    kriging_res = np.clip(kriging_res, a_min=0, a_max=None).ravel()
                    self.est_results[self.prime_var] = kriging_res
                    print(f'   For prime variable ({self.prime_var}) in run {number} estimation is completed.')

                    # Check for NaN's. '-9966699' is a NaN in SGeMS
                    num_nan = np.where(kriging_res == -9966699)[0].shape[0]
                    if num_nan:
                        num_of_points = self.dimens[0] * self.dimens[1] * self.dimens[2]
                        print(f"'Warning!\n'In run {number} prime variable {(self.prime_var)} has {np.round(num_nan*100/num_of_points, 2)}% of NaN's in the results.\n")
                    
                    self._utils_krig_run_for_poro(number) 
                    
            self._save_npz(number)
        return
    
    def _change_dirs(self, new_root=None, new_path_to_save=None):
        '''Change root directory and directory to save results.
        
           Arguments:
             new_root: str - a new root to change self.root.
             new_path_to_save - a new path to save to change self.path_to_save.
        '''
        if new_root:
            print(f'=> change root from {self.root} to {new_root}.')
            self.root = new_root
        if new_path_to_save:
            if os.path.exists(new_path_to_save):
                raise ValueError("Save path exists")
            else:
                os.mkdir(new_path_to_save)
                print(f'=> change save directory from {self.path_to_save} to {new_path_to_save}.')
                self.path_to_save = new_path_to_save        
        return

    def _set_generation_mode(self, size: int, alpha=None, n_wells=None):
        '''Set basic parameters of generation (e.g. size, alpha, n_wells).
        
           Arguments:
             size: int - a number of data sets to be created with this particular alpha/n_wells.
             alpha: float in (0, 1) - a part of points to be taken. If alpha=None considered that this run is for well data.
             n_wells: int - a number of unit wells to create data. If n_wells=None considered that this run is taking random points.
               Either 'alpha' or 'n_wells' has to be equal to 'None'.
        '''
        self.size = size
        self.alpha = alpha
        self.n_wells = n_wells
        
        if (not self.alpha) & (not self.n_wells):
            raise ValueError("Generation is stopped.\nEither 'alpha' or 'n_wells' has to be not 'None'.")

        elif self.alpha:
            self.size_dir = abs_join_path(self.path_to_save, self.run_name)
        elif self.n_wells:
            self.size_dir = abs_join_path(self.path_to_save, self.run_name)

        if os.path.exists(self.size_dir):
            raise ValueError("Folder for the specified 'alpha'\\'n_wells' and 'size' exists. Please check the save folder.\nDelete it if you want to proceed with these values.")
            
        os.mkdir(os.path.abspath(self.size_dir))
        return
        
    def _upload_data(self, ):
        '''Upload inital data of SPE10 model 2 from .dat file.'''
        if self.model == 'spe10m2':
            spe10_read_all = pd.read_csv(abs_join_path(self.root,'spe10_all_orig.dat'),
                                    sep=' ', skiprows=9, names=['x', 'y', 'z', 'poro', 'permx', 'permy', 'permz']).to_numpy()
            xyz = get_xyz()
            self.poro_full = np.hstack((xyz, spe10_read_all[:, 3].reshape(-1, 1)))
            self.permx_full = np.hstack((xyz, spe10_read_all[:, 4].reshape(-1, 1)))
            self.permy_full = np.hstack((xyz, spe10_read_all[:, 5].reshape(-1, 1)))
            self.permz_full = np.hstack((xyz, spe10_read_all[:, 6].reshape(-1, 1)))
        return
    
    def _match_coord(self, ):
        '''Match the cell_size and coordinate system for SGeMS.'''
        size_0, size_1, size_2 = self.cell_size
        self.poro_full[:, 0] *= size_0
        self.poro_full[:, 1] *= size_1
        self.poro_full[:, 2] *= size_2

        self.permx_full[:, 0] *= size_0
        self.permx_full[:, 1] *= size_1
        self.permx_full[:, 2] *= size_2

        self.permy_full[:, 0] *= size_0
        self.permy_full[:, 1] *= size_1
        self.permy_full[:, 2] *= size_2

        self.permz_full[:, 0] *= size_0
        self.permz_full[:, 1] *= size_1
        self.permz_full[:, 2] *= size_2
        return
    
    def _take_points(self, ):
        '''Take a specifeid percent of random points SPE10 model 2.'''
        if self.alpha == 1.:
            print('100% of points is chosen. Will return inital data.')
            self.poro = self.poro_full
            self.permx = self.permx_full
            self.permy = self.permy_full
            self.permz = self.permz_full
            return
        else:
            points_num = self.poro_full.shape[0]
            rng = np.random.default_rng()
            points_num_idx = rng.choice(points_num, size=int(self.alpha*points_num), replace=False)

            self.poro = self.poro_full[points_num_idx]
            self.permx = self.permx_full[points_num_idx]
            self.permy = self.permy_full[points_num_idx]
            self.permz = self.permz_full[points_num_idx]
            return
    
    def _take_wells(self, ):
        '''Take part of data from the properties of SPE10 model 2 as unit-well data.
           Well has unit width, and depth equal to a depth of the field.'''
        max_x_coord = np.unique(self.poro_full[:, 0]).shape[0]
        max_y_coord = np.unique(self.poro_full[:, 1]).shape[0]
                             
        # Need to muliply by cell_size because SGeMS bond to coordinate system when point set is created
        well_x_coord = np.random.randint(0, max_x_coord, size=self.n_wells) * self.cell_size[0]
        well_y_coord = np.random.randint(0, max_y_coord, size=self.n_wells) * self.cell_size[1]
      
        self.poro = np.vstack([self.poro_full[
                                          (self.poro_full[:, 0] == well_x_coord[i]) & (self.poro_full[:, 1] == well_y_coord[i])]
                                          for i in range(self.n_wells)])
        self.permx = np.vstack([self.permx_full[
                                          (self.permx_full[:, 0] == well_x_coord[i]) & (self.permx_full[:, 1] == well_y_coord[i])]
                                          for i in range(self.n_wells)])
        self.permy = np.vstack([self.permy_full[
                                          (self.permy_full[:, 0] == well_x_coord[i]) & (self.permy_full[:, 1] == well_y_coord[i])]
                                          for i in range(self.n_wells)])
        self.permz = np.vstack([self.permz_full[
                                          (self.permz_full[:, 0] == well_x_coord[i]) & (self.permz_full[:, 1] == well_y_coord[i])]
                                          for i in range(self.n_wells)])
        return
    
    def _save_gslib_all(self, number: int):
        '''Save data in GSLIB format for SGeMS. First row is for comments. Second row is for a number of columns.
           Then names of columns are one in a row; have to be in number specified on 2nd row. Then data
           columns in order of names from rows above.
           
           Arguments:
             numebr: int - a number of curent estimation run.
        '''
        
        all_prop_in_one = np.empty((self.poro.shape[0], 7))

        all_prop_in_one[:, 0:4] = self.poro
        all_prop_in_one[:, 4] = self.permx[:, -1]
        all_prop_in_one[:, 5] = self.permy[:, -1]
        all_prop_in_one[:, 6] = self.permz[:, -1]
        
        first_rows = np.array([[7, '', '', '', '', '', ''],
                               ['x', '', '', '', '', '', ''],
                               ['y', '', '', '', '', '', ''],
                               ['z', '', '', '', '', '', ''],
                               ['poro', '', '', '', '', '', ''],
                               ['permx', '', '', '', '', '', ''],
                               ['permy', '', '', '', '', '', ''],
                               ['permz', '', '', '', '', '', '']])

        all_gslib_extension = np.vstack((first_rows, all_prop_in_one))
        
        pd.DataFrame(all_gslib_extension).to_csv(abs_join_path(self.work_dir, 'all_prop.dat'), sep=' ', index=False,
                                              header=['Part of the all properties points from spe10 model 2 dataset.', '', '', '', '', '', ''])
        return
    
    def _set_algorithm(self, ):
        '''Create a script that will be run in the SGeMS.'''
        # Initiate sgems project
        cwd = abs_join_path(self.path_to_save, 'logs')         # Working directory
        rdir = abs_join_path(self.path_to_save, 'results')     # Results directory
        self.pjt = sg.Sgems(project_name="spe10_model2_OK",
                       project_wd=cwd,
                       res_dir=rdir,
                       exe_name=self.exe_name_sgems)    

        # Load dataset of points
        file_path = abs_join_path(self.work_dir, 'all_prop.dat')
        self.ps = PointSet(project=self.pjt, pointset_path=file_path)

        # Generate grid
        ds = Discretize(project=self.pjt, dx=self.cell_size[0], dy=self.cell_size[1],
                        dz=self.cell_size[2], xo=0, yo=0, zo=0,
                        x_lim=self.dimens[0]*self.cell_size[0], y_lim=self.dimens[1]*self.cell_size[1],
                        z_lim=self.dimens[2]*self.cell_size[2])

        # Upload an algorithm and update it for a current property
        algo_dir = abs_join_path(self.root, "algorithms")                
        self.al_pysgems = XML(project=self.pjt, algo_dir=algo_dir)
        if self.alg == 'ord_kriging':
            self.al_pysgems.xml_reader('permx_permy_ord_kriging')
                    
            if ('permx' == self.property_name) | ('permy' == self.property_name):
                pass
            elif 'permz' == self.property_name:
                self.al_pysgems.xml_update('Variogram', new_attribute_dict={'nugget': '0', 'structures_count': '1'})
                self.al_pysgems.xml_update('Variogram//structure_1', new_attribute_dict={'contribution': '135000', 'type': 'Exponential'})
            elif 'poro' == self.property_name:
                self.al_pysgems.xml_update('Variogram', new_attribute_dict={'nugget': '0', 'structures_count': '1'})
                self.al_pysgems.xml_update('Variogram//structure_1', new_attribute_dict={'contribution': '0.0087', 'type': 'Exponential'})
                self.al_pysgems.xml_update('Variogram//structure_1//ranges', new_attribute_dict={'max': '486', 'medium': '417', 'min': '125'})
                self.al_pysgems.xml_update('Variogram//structure_1//angles', new_attribute_dict={'x': '60', 'y': '0', 'z': '0'})
            
            grid_name = self.property_name + '_grid'

            # Update algoirthm
            self.al_pysgems.xml_update('Search_Ellipsoid', new_attribute_dict={'value': self.search_ellipsoid})
            self.al_pysgems.xml_update('Hard_Data', new_attribute_dict={'grid': grid_name, 'property': self.property_name})
            
        elif self.alg == 'cokriging':
            self.al_pysgems.xml_reader('corkiging_perm_from_poro')

            self.al_pysgems.xml_update('Primary_Harddata_Grid', new_attribute_dict={'value': self.prime_var+'_grid', 'region': ''})
            self.al_pysgems.xml_update('Primary_Variable', new_attribute_dict={'value': self.prime_var})
            self.al_pysgems.xml_update('Secondary_Harddata_Grid', new_attribute_dict={'value': self.second_var+'_grid', 'region': ''})
            self.al_pysgems.xml_update('Secondary_Variable', new_attribute_dict={'value': self.second_var})

        return
    
    def _print_alg_tree(self, ):
        '''Print the parameters of the chosen SGeMS algorithm.'''
        self.al_pysgems.show_tree()
        return
    
    def _change_alg_params(self, search_ellipsoid):
        '''Change the parameters of the SGeMS algorithm.'''
        self.al_pysgems.xml_update('Search_Ellipsoid', new_attribute_dict={'value': search_ellipsoid})
        pass

    def _run_alg(self, ):
        '''Run estimation in SGeMS.'''
        if self.alg == 'ord_kriging':
            # Export data in binary SGeMS format and rewrite the algorithm
            self.ps.export_01(self.property_name)
            self.pjt.write_command()
            
        elif self.alg == 'cokriging':
            self.ps.export_01(self.prime_var)
            self.ps.export_01(self.second_var)
            self.pjt.write_command()
            
        # Run estimation
        self.pjt.run()
        return

    def _utils_krig_run_for_poro(self, number: int):
        '''Utility run of ordinary kriging for porosity
           
           Arguments:
             numebr: int - a number of curent estimation run.
        '''
        cwd = abs_join_path(self.path_to_save, 'logs')
        rdir = abs_join_path(self.path_to_save, 'results')
        pjt = sg.Sgems(project_name="spe10_model2_OK",
                       project_wd=cwd,
                       res_dir=rdir,
                       exe_name=self.exe_name_sgems)    
        file_path = abs_join_path(self.work_dir, 'all_prop.dat')
        ps = PointSet(project=pjt, pointset_path=file_path)
        ds = Discretize(project=pjt, dx=self.cell_size[0], dy=self.cell_size[1],
                        dz=self.cell_size[2], xo=0, yo=0, zo=0,
                        x_lim=self.dimens[0]*self.cell_size[0], y_lim=self.dimens[1]*self.cell_size[1],
                        z_lim=self.dimens[2]*self.cell_size[2])
        algo_dir = abs_join_path(self.root, "algorithms/")                
        al_pysgems = XML(project=pjt, algo_dir=algo_dir)
        al_pysgems.xml_reader('permx_permy_ord_kriging')
        al_pysgems.xml_update('Search_Ellipsoid', new_attribute_dict={'value': '400 350 100  0 90 0'})
        al_pysgems.xml_update('Hard_Data', new_attribute_dict={'grid': self.second_var+'_grid', 'property': self.second_var})
        ps.export_01('poro')
        pjt.write_command()
        pjt.run()
        
        kriging_res = pd.read_csv(abs_join_path(self.path_to_save, 'results/results.grid'), skiprows=3, names=[self.second_var]).to_numpy()
        kriging_res = np.clip(kriging_res, a_min=0, a_max=None).ravel()
        self.est_results[self.second_var] = kriging_res
        print(f'   For secondary variable ({self.second_var}) in run {number} estimation is completed.')

        num_nan = np.where(kriging_res == -9966699)[0].shape[0]
        if num_nan:
            num_of_points = self.dimens[0] * self.dimens[1] * self.dimens[2]
            print(f"'Warning!\n'In run {number} secondary variable {(self.second_var)} has {np.round(num_nan*100/num_of_points, 2)}% of NaN's in the results.\n")
                        
        return
    
    def _save_npz(self, number: int):
        '''Save results of kriging for all properties of the field as .npz file; delete imtermediate dirs.
           
           Arguments:
             numebr: int - a number of curent estimation run.
        '''
        rmtree(self.work_dir)
        rmtree(abs_join_path(self.path_to_save, 'results'))
        npz_file_path = self.work_dir + '.npz'
        
        if self.alg == 'ord_kriging':
            np.savez(npz_file_path, permx=self.est_results['permx'], permy=self.est_results['permy'],
                     permz=self.est_results['permz'], poro=self.est_results['poro'])
            
        if self.alg == 'cokriging':
            np.savez(npz_file_path, permx=self.est_results['permx'], poro=self.est_results['poro'])
            
        print(f'   Run {number} is completed.\n   Full .npz file path of run {number}:', npz_file_path, end='\n\n')
        return
    
    def _upload_npz(self, npz_path: str):
        '''Load Results of estimation.
        
           Arguments:
             npz_dir - a path for .npz file to load.
        '''
        npz_file = np.load(npz_path)
        for prop in npz_file.keys():
            self.est_results[prop] = npz_file[prop]
        return
    
    def PlotResults(self, from_npz=False, npz_dir=None, z_coord=77, p_name='poro', fig_size=(11, 11)):
        '''Plot heatmap of result properties. Can be used for drawing results from both 
           multiple npz files and current estimation results.
           
           Arguments:
             from_npz: bool - if True, will draw every .npz file at the directory npz_dir. If False,
               will plot from current attributes self.poro, self.permx, self.permy, self.permz depending 
               on p_name.
             npz_dir: str - if not None, it has to be a directory with list of .npz files to draw.
             z_coord: int - level of cross-section to draw.
             p_name: str - name of a propetry to draw. Choices: poro, permx, permy, permz.
             fig_size: tuple(int, int) - size of picture to draw.
        '''
        dict_p = {
            'poro': self.poro_full,
            'permx': self.permx_full,
            'permy': self.permy_full,
            'permz': self.permz_full
        }
        p = dict_p[p_name]
        
        if not isinstance(p, np.ndarray):
            if 'all_prop.dat' not in os.listdir(self.root):
                raise ValueError('Preprocess data and put it to self.root to continue.')
            else:
                self._upload_data()
                dict_p = {
                    'poro': self.poro_full,
                    'permx': self.permx_full,
                    'permy': self.permy_full,
                    'permz': self.permz_full
                }
                p = dict_p[p_name]
        p = p[:, -1]
        p_init_visual = p.reshape((60, 220, 85), order='F')
        
        if not from_npz:
            if  not self.est_results:
                raise ValueError('Upload etimation results to continue.')
                
            p_part_visual = self.est_results[p_name].reshape((60, 220, 85), order='F')

            _, axes = plt.subplots(2, 1, figsize=fig_size)

            sns.heatmap(p_init_visual[:,:,z_coord], ax=axes[0], cmap='jet')
            sns.heatmap(p_part_visual[:,:,z_coord], ax=axes[1], cmap='jet')

            axes[0].set_title(f'Full data, z = {z_coord+1}')
            axes[1].set_title(f'Estimation results, z = {z_coord+1}')

            axes[1].axes.xaxis.set_visible(False)
            axes[1].axes.yaxis.set_visible(False)
            axes[0].axes.xaxis.set_visible(False)
            axes[0].axes.yaxis.set_visible(False)                  
            
        elif from_npz:
            if not isinstance(npz_dir, str):
                raise ValueError('Pass a directory with .npz files to plot.')
                
            npz_dir = abs_join_path(npz_dir, '*.npz')
            num_npz = len(glob.glob(npz_dir))

            _, axes = plt.subplots(num_npz+1, 1, figsize=fig_size)

            sns.heatmap(p_init_visual[:,:,z_coord], ax=axes[0], cmap='jet')
            axes[0].set_title(f'Full data, z = {z_coord+1}')
            axes[0].axes.xaxis.set_visible(False)
            axes[0].axes.yaxis.set_visible(False)

            for i, file_path in enumerate(glob.glob(npz_dir)):
                name_of_estimation = file_path[file_path.rfind('\\')+1:file_path.rfind('.')]
                npz_res = np.load(file_path)
                poro_part_visual = npz_res[p_name].reshape((60, 220, 85), order='F')

                sns.heatmap(poro_part_visual[:,:,z_coord], ax=axes[i+1], cmap='jet')

                axes[i+1].set_title(f'Kriging results for {name_of_estimation}, z = {z_coord+1}')
                axes[i+1].axes.xaxis.set_visible(False)
                axes[i+1].axes.yaxis.set_visible(False)           
        return
    
    def Plot3D(self, npz_path=None, fig_size=(11, 7)):
        '''Plot 3D heat map of a porosity.
        
           Arguments:
             npz_path: str - root to the npz file to plot. If npz_path=None 
               will plot initial model data.
        '''
        if npz_path:
            fig = plt.figure(figsize=fig_size)
            ax = Axes3D(fig)
            
            poro_npz = np.load(npz_path)['poro']
            xyz = get_xyz()

            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            c = poro_npz

            img = ax.scatter(x, y, z, c=c, cmap=plt.jet())
            
            fig.colorbar(img, shrink=.6)
            ax.set_xlim3d(0, 120)
            ax.set_ylim3d(0, 120)
            ax.set_zlim3d(0, 120)
            ax.axis("off")
            plt.show();
            return
        
        else:
            if 'spe10_all_orig.dat' not in os.listdir(self.root):
                raise ValueError('Preprocess data and put it to self.root to continue.')
            else:
                self._upload_data()

            fig = plt.figure(figsize=fig_size)
            ax = Axes3D(fig)

            img = ax.scatter(self.poro_full[:, 0], self.poro_full[:, 1], self.poro_full[:, 2],
                             c=self.poro_full[:, 3], cmap=plt.jet())

            fig.colorbar(img, shrink=.6)
            ax.set_xlim3d(0, 120)
            ax.set_ylim3d(0, 120)
            ax.set_zlim3d(0, 120)
            ax.axis("off")
            plt.show();
            return