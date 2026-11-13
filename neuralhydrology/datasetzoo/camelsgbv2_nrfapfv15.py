from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CamelsGBv2_NRFAPFv15(BaseDataset):
    """Data set class for the CAMELS GB v2 dataset by [1] but with static attributes taken from the UK National River
    Flow Archive Peak Flows data set v15-0-2 [2]. Requires the NRFA Peak Flows attributes to have been combined into 
    a single, suitably structured csv file as a preprocessing step before to being loaded here.
    
    This file created from the Neural Hydrology Template class file for adding a new data set by Mike Vaughan.
    
    Each dataset class has to derive from `BaseDataset`, which implements most of the logic for preprocessing data and 
    preparing data for model training. Only two methods have to be implemented for each specific dataset class: 
    `_load_basin_data()`, which loads the time series data for a single basin, and `_load_attributes()`, which loads 
    the static attributes for the specific data set. 
    
    Usually, we outsource the functions to load the time series and attribute data into separate functions (in the
    same file), which we then call from the corresponding class methods. This way, we can also use specific basin data
    or dataset attributes without these classes.
    
    To make this dataset available for model training, don't forget to add it to the `get_dataset()` function in 
    'neuralhydrology.datasetzoo.__init__.py'

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    References
    ----------
    .. [1] Coxon, G., Zheng, Y., Barbedo, R., Cooper, H., Fileni, F., Fowler, H. J., Fry, M., Green, A., Gribbin, T., 
        Harfoot, H., Lewis, E., Neto, G. G. R., Qiu, X., Salwey, S., and Wendt, D. E.: CAMELS-GB v2: hydrometeorological
        time series and landscape attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data, 18, 4345–4371,
        https://doi.org/10.5194/essd-18-4345-2026, 2026
    .. [2] Turner, S., Suman, G., Crowhurst, D., Smith, R., Barr, A., Griffin, A., Vesuviano, G., and Meresa, H.: NRFA
        Peak Flow Dataset Version 15. https://nrfa.ceh.ac.uk/sites/default/files/2026-08/Note_on_version_15_FINAL.pdf
        (last accessed 9 November 2026)
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(CamelsGBv2_NRFAPFv15, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        df = load_camels_gb_v2_timeseries(data_dir=self.cfg.data_dir, basin=basin)
        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        # return load_camels_gb_attributes(self.cfg.data_dir, basins=self.basins)

        # New code
        # Temporary hard coding to get something working
        attrs_file = Path("/Volumes/prd_dash_lab/flood_forecasting_unrestricted/shared_external_volume/NRFAPeakFlow_v15-0-2/Processed/NRFAPeakFlow_catchment_descriptors.csv") 
        return load_nrfapfv15_attributes(attrs_file, basins=self.basins)


def load_camels_gb_v2_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the DAILY time series data for one basin of the CAMELS GB v2 data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS GB v2 directory. This folder must contain a folder called 'hydro-meteorological_daily'
        containing the forcing files for each basin as .csv file. The file names have to start with 'camels_gb_v2_hydromet'.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    """
    forcing_path = data_dir / 'hydro-meteorological_daily'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('**/camels_gb_v2_hydromet*.csv'))
    file_path = [f for f in files if f"_{basin}_" in f.name]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    df = pd.read_csv(file_path, sep=',', header=0, dtype={'date': str})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index("date")

    return df

def load_nrfapfv15_attributes(attrs_file: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load NRFA Peak Flows v15 attributes from a preprocessed csv file

    Parameters
    ----------
    attrs_file : Path
        Path and filename of the NRFA PF csv file    
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
        
    Raises
    ------
    FileNotFoundError
        If no subfolder called 'attributes' exists within the root directory of the CAMELS GB data set.

    """

    """ Code from CAMELS GB not required:
    attributes_path = data_dir / 'attributes'

    if not attributes_path.exists():
        raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('*_attributes.csv')

     Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=',', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    """
    # Add read statement here
    df = pd.read_csv(attrs_file)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df