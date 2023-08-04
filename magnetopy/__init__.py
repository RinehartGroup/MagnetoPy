from magnetopy.data_files import DatFile, GenericFile
from magnetopy.experiments import FC, ZFC, ZFCFC, MvsH, TrueFieldCorrection

# from magnetopy.plot import (
#     plot_all_in_single_voltage_scan,
#     plot_analyses,
#     plot_analyzed_voltage_scan,
#     plot_mvsh,
#     plot_mvsh_w_fits,
#     plot_voltage_scan,
#     plot_zfcfc,
#     plot_zfcfc_w_blocking,
# )
from magnetopy.plot_helpers import force_aspect, linear_color_gradient
from magnetopy.dataset import SampleInfo, Dataset
from magnetopy.analyses.simple_mvsh import (
    SimpleMvsHAnalysis,
    SimpleMvsHAnalysisResults,
    SimpleMvsHAnalysisParsingArgs,
)
