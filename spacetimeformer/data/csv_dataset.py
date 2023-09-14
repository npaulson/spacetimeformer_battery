import wandb
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import h5py
import glob
import random
from typing import List
import os
import re

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt

import spacetimeformer as stf


class CSVTimeSeries:
    def __init__(
        self,
        data_path: str,
        target_cols: List[str],
        read_csv_kwargs={},
        val_split: float = 0.2,
        test_split: float = 0.15,
    ):
        self.data_path = data_path
        assert os.path.exists(self.data_path)

        # read the file and do some datetime conversions
        raw_df = pd.read_csv(
            self.data_path,
            **read_csv_kwargs,
        )

        time_df = pd.to_datetime(raw_df["Datetime"], format="%Y-%m-%d %H:%M")
        df = stf.data.timefeatures.time_features(time_df, raw_df)

        assert (df["Datetime"] > pd.Timestamp.min).all()
        assert (df["Datetime"] < pd.Timestamp.max).all()

        # Train/Val/Test Split using holdout approach #

        def mask_intervals(mask, intervals, cond):
            for (interval_low, interval_high) in intervals:
                if interval_low is None:
                    interval_low = df["Datetime"].iloc[0].year
                if interval_high is None:
                    interval_high = df["Datetime"].iloc[-1].year
                mask[
                    (df["Datetime"] >= interval_low) & (df["Datetime"] <= interval_high)
                ] = cond
            return mask

        test_cutoff = len(time_df) - round(test_split * len(time_df))
        val_cutoff = test_cutoff - round(val_split * len(time_df))

        val_interval_low = time_df.iloc[val_cutoff]
        val_interval_high = time_df.iloc[test_cutoff - 1]
        val_intervals = [(val_interval_low, val_interval_high)]

        test_interval_low = time_df.iloc[test_cutoff]
        test_interval_high = time_df.iloc[-1]
        test_intervals = [(test_interval_low, test_interval_high)]

        train_mask = df["Datetime"] > pd.Timestamp.min
        val_mask = df["Datetime"] > pd.Timestamp.max
        test_mask = df["Datetime"] > pd.Timestamp.max
        train_mask = mask_intervals(train_mask, test_intervals, False)
        train_mask = mask_intervals(train_mask, val_intervals, False)
        val_mask = mask_intervals(val_mask, val_intervals, True)
        test_mask = mask_intervals(test_mask, test_intervals, True)

        if (train_mask == False).all():
            print(f"No training data detected for file {data_path}")

        self._train_data = df[train_mask]
        self._scaler = StandardScaler()
        self.target_cols = target_cols

        self._scaler = self._scaler.fit(self._train_data[target_cols].values)

        self._train_data = self.apply_scaling(df[train_mask])
        self._val_data = self.apply_scaling(df[val_mask])
        self._test_data = self.apply_scaling(df[test_mask])

    def get_slice(self, split, start, stop, skip):
        assert split in ["train", "val", "test"]
        if split == "train":
            return self.train_data.iloc[start:stop:skip]
        elif split == "val":
            return self.val_data.iloc[start:stop:skip]
        else:
            return self.test_data.iloc[start:stop:skip]

    def apply_scaling(self, df):
        scaled = df.copy(deep=True)
        scaled[self.target_cols] = (df[self.target_cols].values - self._scaler.mean_) / self._scaler.scale_
        return scaled

    def reverse_scaling_df(self, df):
        scaled = df.copy(deep=True)
        scaled[self.target_cols] = (df[self.target_cols] * self._scaler.scale_) + self._scaler.mean_
        return scaled

    def reverse_scaling(self, array):
        return (array * self._scaler.scale_) + self._scaler.mean_

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def length(self, split):
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }[split]

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")


class CSVBatterySeries:
    def __init__(
        self,
        dset_name: str,
        data_path: str,
        context_points: int,
        stride: int,
        skip_context: int,
        skip_target: int,
        read_csv_kwargs={},
    ):

        loggercore = logging.getLogger('pytorch_lightning.core')
        self.loggercore = loggercore

        self.dset_name = dset_name
        self.data_path = data_path
        assert os.path.exists(self.data_path)

        self.context_points = context_points
        self.stride = stride
        self.skip_context = skip_context
        self.skip_target = skip_target

        np.random.seed(0)

        # process raw data
        dsets = ['trn', 'val', 'tst', 'ext']

        self._get_raw_data()

        loggercore.error(f"self.cellnamesD['tst']: {self.cellnamesD['tst']}")
        loggercore.error(f"len(self.dfD['tst']): {len(self.dfD['tst'])}")

        for dset in dsets:
            for ii, cell in enumerate(self.dfD[dset]):

                cyc_df = cell['cycle_number']
                cell = stf.data.timefeatures.cycle_features(cyc_df, cell)

                cell = self._fill_df_gaps(cell)

                self.dfD[dset][ii] = cell

        self.cols.remove('cycle_number')
        self.target_cols = self.cols

        self._scaler = StandardScaler()
        self._scaler = self._scaler.fit(pd.concat(self.dfD['trn'])[self.target_cols])

        for dset in dsets:
            for ii, cell in enumerate(self.dfD[dset]):
                self.dfD[dset][ii] = self.apply_scaling(self.dfD[dset][ii])

        self._train_data = self._create_dataset(
            'trn', self.dfD['trn'], self.cellnamesD['trn'], self.buildsD['trn'])
        self._val_data = self._create_dataset(
            'val', self.dfD['val'], self.cellnamesD['val'], self.buildsD['val'])
        self._test_data = self._create_dataset(
            'test', self.dfD['tst'], self.cellnamesD['tst'], self.buildsD['tst'])

        if self.dset_name in ['battery_CAMP']:
            self._extrapolate_data = self._create_dataset(
                'ext', self.dfD['ext'], self.cellnamesD['ext'], self.buildsD['ext'])

    def _create_dataset(self, dset_set, dfs, cellnames, builds):

        context_points = self.context_points
        stride = self.stride
        target_cols = self.target_cols
        skip_context = self.skip_context
        skip_target = self.skip_target

        inputs = []
        labels = []
        builds_ = []
        cellnames_ = []
        input_sts = []
        input_ends = []
        label_sts = []
        label_ends = []
    
        for df, build, cellname in zip(dfs, builds, cellnames):
    
            indices = df.index.values
            label_end = indices.max() + 1

            st_index_last = indices.max() - skip_target - context_points*skip_context
    
            st_indices = np.arange(
                indices.min(), st_index_last, stride)
    
            for st_index in st_indices:
    
                input_end = st_index + context_points*skip_context
                input_indices = np.arange(st_index, input_end, skip_context)

                label_st = input_end + skip_target
                label_indices = np.arange(label_st, label_end, skip_target)

                cols = list(df)

                input_datum = pd.DataFrame(
                    np.zeros((context_points, len(cols))), columns=cols)    
                input_datum_ = df.loc[input_indices]
                input_datum.iloc[:len(input_datum_), :] = input_datum_.values

                label = pd.DataFrame(
                    np.zeros((250, len(cols))), columns=cols)

                self.loggercore.debug(
                    f'label start: {label_st}, label end: {label_end}, ' +
                    f'len(label_indices): {len(label_indices)}')

                label_ = df.loc[label_indices]

                label.iloc[:len(label_), :] = label_.values
    
                inputs.append(input_datum)
                labels.append(label)
                builds_.append(build)
                cellnames_.append(cellname)
                input_sts.append(st_index)
                input_ends.append(input_end)
                label_sts.append(label_st)
                label_ends.append(label_end)

                # print(np.sum(np.isnan(label_.values)), np.sum(np.isnan(label.values)))
    
                # print(input_indices, label_indices, label)
                # print(label.shape)
                # plt.figure()
                # plt.imshow(label, aspect='auto', interpolation=None)
                # plt.show()

                # plt.figure()
                # plt.plot(input_indices, input_datum_.values[:, 1], 'b.')
                # plt.plot(label_indices, label_.values[:, 1], 'k.')
                # plt.savefig(f'./data/plots/inputs_labels_{self.dset_name}_{dset_set}_{label_st}_{label_end}.png')
                # plt.close()

        label_l = [len(label.values[label.values[:, 0] != 0, :]) for label in labels]
        plt.hist(label_l, bins=50)
        plt.savefig('label_len_hist.png')
        plt.close()

        # shuffle lists
        tmp = list(zip(inputs, labels, cellnames_, builds_,
                       input_sts, input_ends, label_sts, label_ends))

        random.shuffle(tmp)

        self.loggercore.error(f'len(inputs): {len(inputs)},' +\
                              f'len(labels): {len(labels)},' +\
                              f'len(cellnames): {len(cellnames_)},' +\
                              f'len(builds) : {len(builds_)},' +\
                              f'len(input_sts): {len(input_sts)},' +\
                              f'len(input_ends): {len(input_ends)},' +\
                              f'len(label_sts): {len(label_sts)},' +\
                              f'len(label_ends): {len(label_ends)}')

        inputs, labels, cellnames, builds, input_sts, input_ends, label_sts, label_ends = zip(*tmp)

        out_dict = {
            'cellname': cellnames_,
            'build': builds_,
            'input_st': input_sts,
            'input_end': input_ends,
            'label_st': label_sts,
            'label_end': label_ends}

        df_out = pd.DataFrame(out_dict)

        tab = wandb.Table(dataframe=df_out)
        name = f'inputs_labels_{self.dset_name}_{dset_set}'

        wandb.log({name: tab}) 

        self.loggercore.error(f'{name} table logged')
        self.loggercore.error(f'len(inputs): {len(inputs)}, len(labels): {len(labels)}')

        return inputs, labels

    def _fill_df_gaps(self, df):
    
        cycL = df.index.values
        cyc_min = cycL.min()
        cyc_max = cycL.max()
    
        for cyc in range(cyc_min, cyc_max+1):
            if cyc not in cycL:
                df.loc[cyc] = df.loc[cyc - 1]
    
        # fill in nans with preceeding values
        df.fillna(method='ffill', inplace=True)

        return df.sort_index()

    def _get_raw_data(self):
        
        name = self.dset_name
        fname = f'f_all_obj_{name}.pkl'
        if os.path.isfile(fname):
            f_all_obj = open(fname, 'rb')
            all_obj = pickle.load(f_all_obj)
            cols, cellnamesD, dfD, categoriesD, buildsD = all_obj
            self.cols = cols
            self.cellnamesD = cellnamesD
            self.dfD = dfD
            self.categoriesD = categoriesD
            self.buildsD = buildsD
            f_all_obj.close()
            self.loggercore.error('raw data objects loaded')
            return

        basedir = self.data_path

        dirs, cols, categoriesu = self._get_cols()

        self.loggercore.info(f'dirs: {dirs}, cols: {cols}, categoriesu: {categoriesu}')

        cellnames, all_df, categories, builds, cycles = [], [], [], [], []

        for dd, directory in enumerate(dirs):
    
            self.loggercore.info(f'directory number: {dd}, {directory}')

            if self.dset_name in ['battery_severson', 'battery_CAMP']:
                cellname = directory[len(basedir):-4]

            if self.dset_name == 'battery_severson':
                build = None
                category = cellname[:6]
            elif self.dset_name == 'battery_CAMP':
                locs = [m.start() for m in re.finditer('_', cellname)]
                build = cellname[locs[0]+1:locs[1]]
                category = cellname[:locs[0]]

            self.loggercore.info(f'cellname: {cellname}, build: {build}, category: {category}')
   
            df = pd.read_csv(directory)  # [cols]

            self.loggercore.debug('dataframe read')

            # for camp dataset, process step file to extract IR drop
            if self.dset_name == 'battery_CAMP':
                directory_step = directory[:-7] + 'step.csv'
                df_stp = pd.read_csv(directory_step)
                df_stp.loc[df_stp['ohmic'] <= 0, 'ohmic'] = np.nan

                ohmic = np.zeros((len(df),))
                for ii, step in df_stp.groupby(['cycle_number']):
                    ohmic[ii] = step['ohmic'].min()

                df['ohmic'] = ohmic

            if self.dset_name in ['battery_severson', 'battery_CAMP']:

                sel = (df['R_10'].values != 0).astype('bool')
                sel *= (df['R_10'].values != np.nan).astype('bool')
                if self.dset_name == 'battery_CAMP':
                    sel *= (df['cycle_type'].values == 'regular_cycle').astype('bool')
                df = df.iloc[sel]

            df = df[cols]

            self.loggercore.debug(f'dataframe shape: {df.shape}')

            if len(df) < 100:
                continue

            if self.dset_name in ['battery_severson', 'battery_CAMP']:
                nanfrac = np.sum(df.isna().values)/df.size
                self.loggercore.debug(f'dataframe nanfraction: {nanfrac}')
                if nanfrac > 0.05:
                    continue

            self.loggercore.debug('pre-sementation')

            df_filt, seg_bnds, jump_cycs = self._get_segment_bnds(df)

            self.loggercore.info(f'seg_bnds: {seg_bnds}')
            self.loggercore.info(f'jump_cycs: {jump_cycs}')

            # find segments with low noise for all properties
            segments = []
            segments_filt = []
            segments_err = []
            rng = df_filt.max(axis=0) # - df_filt.min(axis=0)
            self.loggercore.debug(f'range: \n{rng}')
            for ii, seg_bnd in enumerate(seg_bnds):

                sel_a = df['cycle_number'] >= seg_bnd[0]
                sel_b = df['cycle_number'] < seg_bnd[1]

                df_seg = df[sel_a & sel_b]
                df_filt_seg = df_filt[sel_a & sel_b]

                err = (df_seg.abs() - df_filt_seg.abs()).abs()

                self.loggercore.debug(f'segment errors:\n{err.describe().to_string()}')

                mape = 100*(err/rng).mean(axis=0)

                self.loggercore.debug(f'segment percent errors:\n{mape}')
                self.loggercore.debug(f'mape.shape: {mape.shape}')

                # err_lims: error limits for each feature
                if self.dset_name == 'battery_CAMP':
                    err_lims = np.array([1., 1., 1., 1., 2.])
                else:
                    err_lims = 5*np.ones(len(mape) - 1,)

                if np.all(mape.values[1:] < err_lims):
                    segments.append(df_seg)
                    segments_filt.append(df_filt_seg)
                    segments_err.append(mape.values[1:])

                    segname = f'{cellname}_seg{ii}'
                    cellnames.append(segname)
                    all_df.append(df_filt_seg)
                    categories.append(category)
                    cycles.append(df_seg['cycle_number'].max() -
                                  df_seg['cycle_number'].min())
                    builds.append(build)

                    self.loggercore.info(f'segment {segname} added')

            # plot original and filtered data overlaid with selected segments
            self._plot_raw_data(cellname, df, df_filt, segments_filt,
                       segments_err, jump_cycs, plot_kernel_size=False)

        # make train, val, test, extrapolate split
        unique, unique_counts = np.unique(categories, return_counts=True)
        self.loggercore.error(f'unique categories: {unique}, {unique_counts}')

        self.loggercore.info(f'cellnames: {cellnames}')

        if self.dset_name == 'battery_CAMP':
            df_cat = pd.DataFrame({'cellnames': cellnames, 'categories': categories, 'builds': builds, 'cycles': cycles})
            for ii, cells in df_cat.groupby(['categories', 'builds']):
                category = cells['categories'].values[0]
                build = cells['builds'].values[0]
                cycle_c = cells['cycles'].mean()
                n_cells = len(cells)

                self.loggercore.error(f'{category} {build}, cells:{n_cells}, cycles: {cycle_c}')

        n_cells, indxD = self._get_data_splits(cellnames, categories, builds)

        self.loggercore.debug(f'n_cells: {n_cells}')
        self.loggercore.debug(f'len(cellnames): {len(cellnames)}')
        self.loggercore.debug(f'indxD: {indxD}')

        cellnamesD, dfD, categoriesD, buildsD, = {}, {}, {}, {}
        for dset in ['trn', 'val', 'tst', 'ext']:
            cellnamesD[dset] = np.array(cellnames)[indxD[dset]]
            dfD[dset] = np.array(all_df)[indxD[dset]]
            categoriesD[dset] = np.array(categories)[indxD[dset]]
            buildsD[dset] = np.array(builds)[indxD[dset]]

        self.loggercore.error(cellnamesD)
    
        self.cellnamesD = cellnamesD
        self.dfD = dfD
        self.categoriesD = categoriesD
        self.buildsD = buildsD

        if not os.path.isfile(fname):
            all_obj = [cols, cellnamesD, dfD, categoriesD, buildsD]
            f_all_obj = open(fname, 'wb')
            pickle.dump(all_obj, f_all_obj)
            f_all_obj.close()
            self.loggercore.info('raw data objects pickled')

    def _get_standard_cycles(self, df):
        for col in list(df):
            df[col] = medfilt(df[col].values, kernel_size=5)
        return df

    def get_slice(self, split):
        assert split in ["train", "val", "test", "extrapolate"]
        if split == "train":
            return self.train_data
        elif split == "val":
            return self.val_data
        elif split == "test":
            return self.test_data
        elif split == "extrapolate":
            return self.extrapolate_data

    def apply_scaling(self, df):
        scaled = df.copy(deep=True)
        scaled[self.target_cols] = (df[self.target_cols].values - self._scaler.mean_) / self._scaler.scale_
        return scaled

    def _plot_raw_data(self, cellname, df, df_filt, segments_filt,
                       segments_err, jump_cycs, plot_kernel_size=False):

        cols = list(df)

        # plot original and filtered data overlaid with selected segments

        fs = 8
        lw = 1
        pad = 1
        tl = 1

        self.loggercore.debug(f'cols: {cols}, len(cols): {len(cols)}')

        n_plts = len(cols) - 1

        n_col = np.ceil(np.sqrt(n_plts)).astype('int8')
        n_row = np.ceil(n_plts/n_col).astype('int8')
        n_rem = n_plts - n_row*n_col

        fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col, 3*n_row), dpi=200)

        cs = sns.color_palette('hls', len(segments_filt))
        for ii, col in enumerate(cols[1:]):

            fmin = np.nanmin(df_filt[col])
            fmax = np.nanmax(df_filt[col])
            frng = fmax
            pmin = fmin - 0.1*frng
            pmax = fmax + 0.1*frng

            err = (df[col].abs() - df_filt[col].abs()).abs()
            mape = np.round(100*(err/frng).mean(), 1)

            aa, bb = np.unravel_index(ii, (n_row, n_col))
            axs[aa, bb].plot(df['cycle_number'], np.abs(df[col]), 'k-', lw=lw, alpha=0.5, label='orig')

            if plot_kernel_size:
                axs[aa, bb].plot(df['cycle_number'], medfilt(df[col].values, kernel_size=5),
                                 'r-', lw=lw, label='k=5')
                axs[aa, bb].plot(df['cycle_number'], medfilt(df[col].values, kernel_size=11),
                                 'g-', lw=lw, label='k=11')
                axs[aa, bb].plot(df['cycle_number'], medfilt(df[col].values, kernel_size=21),
                                 'b-', lw=lw, label='k=21')
                axs[aa, bb].plot(df['cycle_number'], medfilt(df[col].values, kernel_size=41),
                                 'y-', lw=lw, label='k=41')
                axs[aa, bb].plot(df['cycle_number'], medfilt(df[col].values, kernel_size=81),
                                 'm-', lw=lw, label='k=81')

            axs[aa, bb].plot(df['cycle_number'], df_filt[col],
                'k-', lw=0.5*lw, label=f'filt, MAPE {mape}%')

            for jj, (c, segf, sege) in enumerate(zip(cs, segments_filt, segments_err)):
                sege_ii = np.round(sege[ii], 2)
                axs[aa, bb].plot(segf['cycle_number'], segf[col], color=c,
                    label=f'seg {jj}, MAPE: {sege_ii}%')

            axs[aa, bb].vlines(jump_cycs, pmin, pmax, lw=0.5*lw, colors='r', alpha=0.5)
            axs[aa, bb].tick_params(axis='both', which='both', labelsize=fs, length=1)
            axs[aa, bb].set_xlabel('cycle number', fontsize=fs)
            axs[aa, bb].set_ylabel(col, fontsize=fs)
            axs[aa, bb].set_ylim(pmin, pmax)

            axs[aa, bb].legend(fontsize=fs)

        if self.dset_name == 'battery_CAMP': 
            fig.delaxes(axs[1, 2])

        fig.tight_layout(h_pad=pad, w_pad=pad)
        fig.savefig(f'./data/plots/{cellname}.png')
        plt.close('all')

        self.loggercore.info('plot made')

    def _get_cols(self):

        if self.dset_name in ['battery_severson']:
            dirs = glob.glob(f'{self.data_path}/*.csv')
        elif self.dset_name == 'battery_CAMP':
            dirs = glob.glob(f'{self.data_path}/*cyc.csv')

        if self.dset_name == 'battery_severson':
            categoriesu = ['Batch1', 'Batch2', 'Batch3', 'Batch4']
        elif self.dset_name == 'battery_CAMP':
            categoriesu = ['NMC111', 'NMC532', 'NMC622', 'NMC811',
                           'HE5050', '5Vspinel']
        else:
            categoriesu = []    

        self.loggercore.debug(f'categoriesu: {categoriesu}')

        if self.dset_name == 'battery_severson':
            cols_exclude = [
                'accumulated_capacity', 'accumulated_energy',
                'cI', 'dI', 'V_min', 'V_max', 'charge_capacity',
                'charge_energy', 'avg_discharge_current', 'avg_charge_current',
                'charge_avg_V', 'I_d2']
        elif self.dset_name == 'battery_CAMP':
            cols_exclude = [
                'file', 'nsteps', 't_i', 't_f', 't', 'ohm_avg', 'ohm_std', 'Q_acc_min',
                'Q_acc_max', 'E_acc_min', 'E_acc_max', 'V_min', 'V_max',
                'Q_c', 'E_c', 'I_c', 'I_d', 'cycle_type', 'I_d2',
                'R_10', 'OCV_10', 'R_20', 'OCV_20', 'R_30', 'OCV_30',
                'R_40', 'OCV_40', 'R_50', 'OCV_50', 'R_60', 'OCV_60',
                'R_70', 'OCV_70', 'R_80', 'OCV_80', 'R_90', 'OCV_90']
        else:
            cols_exclude = []

        cols = []
        for col in list(pd.read_csv(dirs[0])):
            if col in cols_exclude:
                continue
            else:
                cols.append(col)

        if self.dset_name == 'battery_CAMP':
            cols.append('ohmic')

        self.cols = cols

        return dirs, cols, categoriesu

    def _get_data_splits(self, cellnames, categories, builds):

        n_cells = len(cellnames)
        labels = np.zeros(n_cells)  # label for test/train/val/extrapolate splits

        if self.dset_name == 'battery_severson':

            n_ext = 0
            n_trn = np.int16(np.floor(0.7*(n_cells - n_ext)))
            n_val = np.int16(np.ceil(0.15*(n_cells - n_ext)))
            n_tst = n_cells - n_trn - n_val - n_ext

            labels[:n_val] = 1
            labels[n_val:n_val+n_tst] = 2
            if n_ext > 0:
                labels[-n_ext:] = 3

            np.random.shuffle(labels)

        elif self.dset_name == 'battery_CAMP':

            categories = np.array(categories)
            builds = np.array(builds)

            self.loggercore.error(f'categories: {categories}')
            self.loggercore.error(f'builds: {builds}')
            self.loggercore.error(f'cellnames: {cellnames}')

            # train
            builds_trn = [
                'B26E', 'B26L', 'B26Q', 'B26R', 'B26T', 'B26U',  # 5Vspinel 
                # 'B12B', 'B14A', 'B14C', 'B1A', 'B9A',  # HE5050
                'B21A', 'B27B',  # NMC111
                'B28B', 'B7B',  # NMC532
                'B38A', 'B28D',  # NMC622
            ]
            for build in builds_trn:
                labels[builds == build] = 0

            # val
            builds_val = [
                'B26F', 'B26P', 'B26V', 'B26X', 'B26Z', 'B3A', 'B5A', # 5Vspinel
                # 'B1', 'B14B', 'B14D', 'B8A', 'B9B', # HE5050
                'B27A', 'B6D',  # NMC111
                'B30A', 'B7A',  # NMC532
                'B38C',  # NMC622
            ]
            for build in builds_val:
                labels[builds == build] = 1

            # test
            builds_tst = [
                'B26K', 'B26N', 'B26O', 'B26S', 'B26W', 'B26Y', 'B3B', # 5Vspinel
                '# B12A', 'B13A', 'B4A', 'B4B', 'B9C',  # HE5050
                'B6A',  # NMC111
                'B28A', 'B30B',  # NMC532
                'B38B',  # NMC622
            ]
            for build in builds_tst:
                labels[builds == build] = 2

            # ext
            sel_ext = categories == 'HE5050'
            n_ext = np.sum(sel_ext)
            labels[sel_ext] = 3
  
            n_trn = np.sum(labels == 0)
            n_val = np.sum(labels == 1)
            n_tst = np.sum(labels == 2)
            n_ext = np.sum(labels == 3)

        self.loggercore.error(f'n_cells: {n_cells}, n_trn: {n_trn}, n_val: {n_val}, n_tst: {n_tst}, n_ext: {n_ext}')
 
        indxD = {}
        indxD['trn'] = labels == 0
        indxD['val'] = labels == 1
        indxD['tst'] = labels == 2
        indxD['ext'] = labels == 3

        return n_cells, indxD


    def _get_segment_bnds(self, df):

        all_jump_cycs = np.array([])
        df_filt = df.copy()
        cols = list(df)

        if self.dset_name == 'battery_CAMP':
            jump_lims = np.array([0.01, 0.007, 0.01, 0.007, 0.02])
        else:
            jump_lims = 0.1*np.ones((len(df)-1,))

        for ii, (col, jump_lim) in enumerate(zip(cols[1:], jump_lims)):

            med5 = medfilt(df[col].values, kernel_size=5)
            med11 = medfilt(df[col].values, kernel_size=11)
            med21 = medfilt(df[col].values, kernel_size=21)
            med41 = medfilt(df[col].values, kernel_size=41)
            med81 = medfilt(df[col].values, kernel_size=81)

            comb = med81.copy()
            comb[:41] = med41[:41]
            comb[-41:] = med41[-41:]
            comb[:21] = med11[:21]
            comb[-21:] = med11[-21:]
            comb[:11] = med5[:11]
            comb[-11:] = med5[-11:]
            comb = np.abs(comb)

            combrng = np.max(comb)
            jumps = np.abs(np.diff(comb, prepend=0)) > jump_lim*combrng
            jump_cycs = df['cycle_number'].values[jumps]
            all_jump_cycs = np.append(all_jump_cycs, jump_cycs)

            self.loggercore.debug(f'combrng: {combrng}, jump_lim: {jump_lim}, jump_lim*combrng: {jump_lim*combrng}')

            df_filt[col] = comb

        jump_cycs = np.unique(all_jump_cycs)  # large discontinuities

        jump_cycs_tmp = jump_cycs.copy()
        jump_cycs_tmp = np.append(jump_cycs_tmp, np.max(df['cycle_number']))
        jump_cycs_tmp = np.insert(jump_cycs_tmp, 0, np.min(df['cycle_number']))

        self.loggercore.debug(f'jump_cycs_tmp: {jump_cycs_tmp}')
        seg_bnds = []
        for cyc_ii, cyc_iip1 in zip(jump_cycs_tmp[:-1], jump_cycs_tmp[1:]):
            jdif = cyc_iip1 - cyc_ii
            if jdif >= 50:
                seg_bnds.append([cyc_ii, cyc_iip1])

        self.loggercore.debug(f'seg_bnds: {seg_bnds}')

        return df_filt, seg_bnds, jump_cycs

    def reverse_scaling_df(self, df):
        scaled = df.copy(deep=True)
        scaled[self.target_cols] = (df[self.target_cols] * self._scaler.scale_) + self._scaler.mean_
        return scaled

    def reverse_scaling(self, array):
        return (array * self._scaler.scale_) + self._scaler.mean_

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def extrapolate_data(self):
        return self._extrapolate_data

    def length(self, split):
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
            "extrapolate": len(self.extrapolate_data)
        }[split]

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
           "--data_path", type=str, default="auto")
        parser.add_argument(
           "--context_points", type=int, default=100,
           help="number of input timesteps")
        parser.add_argument(
           "--stride", type=int, default=5,
           help="number of steps to skip in between each saved reading in the input")
        parser.add_argument(
           "--skip_context", type=int, default=10,
           help="")
        parser.add_argument(
           "--skip_target", type=int, default=10,
           help="")


class CSVTorchDset_Battery(Dataset):
    def __init__(
        self,
        csv_battery_series: CSVBatterySeries,
        split: str = "train",
        context_points: int = 100,
    ):
        assert split in ["train", "val", "test", "extrapolate"]

        loggercore = logging.getLogger('pytorch_lightning.core')
        self.loggercore = loggercore

        self.split = split
        self.series = csv_battery_series

    def __len__(self):
        return len(self.series.get_slice(self.split)[0])

    def _torch(self, *np_arrays):
        t = []
        for x in np_arrays:
            t.append(torch.from_numpy(x).float())
        return tuple(t)

    def __getitem__(self, i):

        ctxt, trgt = self.series.get_slice(self.split)

        ctxt_slice = ctxt[i]
        trgt_slice = trgt[i] 

        ctxt_x = ctxt_slice[
            ctxt_slice.columns.difference(self.series.target_cols)
        ].values
        ctxt_y = ctxt_slice[self.series.target_cols].values

        trgt_x = trgt_slice[
            trgt_slice.columns.difference(self.series.target_cols)
        ].values
        trgt_y = trgt_slice[self.series.target_cols].values

        if np.any(np.isnan(ctxt_x)):
            self.loggercore.error('nan in ctxt_x')
        if np.any(np.isnan(ctxt_y)):
            self.loggercore.error('nan in ctxt_y')
        if np.any(np.isnan(trgt_x)):
            self.loggercore.error('nan in trgt_x')
        if np.any(np.isnan(trgt_y)):
            self.loggercore.error('nan in trgt_y')

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    @classmethod
    def add_cli(self, parser):
        None


class CSVTorchDset(Dataset):
    def __init__(
        self,
        csv_time_series: CSVTimeSeries,
        split: str = "train",
        context_points: int = 128,
        target_points: int = 32,
        time_resolution: int = 1,
    ):
        assert split in ["train", "val", "test", "extrapolate"]
        self.split = split
        self.series = csv_time_series
        self.context_points = context_points
        self.target_points = target_points
        self.time_resolution = time_resolution

        self._slice_start_points = [
            i
            for i in range(
                0,
                self.series.length(split)
                + time_resolution * (-target_points - context_points),
            )
        ]
        random.shuffle(self._slice_start_points)
        self._slice_start_points = self._slice_start_points

    def __len__(self):
        return len(self._slice_start_points)

    def _torch(self, *np_arrays):
        t = []
        for x in np_arrays:
            t.append(torch.from_numpy(x).float())
        return tuple(t)

    def __getitem__(self, i):
        start = self._slice_start_points[i]
        series_slice = self.series.get_slice(
            self.split,
            start=start,
            stop=start
            + self.time_resolution * (self.context_points + self.target_points),
            skip=self.time_resolution,
        ).drop(columns=["Datetime"])
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :],
        )
        ctxt_x = ctxt_slice[
            ctxt_slice.columns.difference(self.series.target_cols)
        ].values
        ctxt_y = ctxt_slice[self.series.target_cols].values

        trgt_x = trgt_slice[
            trgt_slice.columns.difference(self.series.target_cols)
        ].values
        trgt_y = trgt_slice[self.series.target_cols].values

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
            "--context_points",
            type=int,
            default=128,
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default=32,
            help="number of future timesteps to predict",
        )
        parser.add_argument(
            "--time_resolution",
            type=int,
            default=1,
        )


if __name__ == "__main__":
    test = CSVTimeSeries(
        "/p/qdatatext/jcg6dn/asos/temperature-v1.csv",
        ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"],
    )
    breakpoint()
    dset = CSVTorchDset(test)
    base = dset[0][0]
    for i in range(len(dset)):
        assert base.shape == dset[i][0].shape
    breakpoint()
