import inspect
from pathlib import Path
import pytest
import numpy as np
from lmfit import Parameters

from magnetopy import (
    cauchy_cdf,
    cauchy_pdf,
    Magnetometry,
    MvsH,
    fit_cauchy_cdf,
    fit_cauchy_pdf,
    CauchyCDFAnalysis,
    CauchyPDFAnalysis,
    CauchyParsingArgs,
    CauchyFittingArgs,
    CauchyParams,
)

TESTS_PATH = Path(inspect.getfile(inspect.currentframe())).parent
DATA_PATH = TESTS_PATH / "data"

dset5 = Magnetometry(DATA_PATH / "dataset5")
dset6 = Magnetometry(DATA_PATH / "dataset6")
mvsh12 = MvsH(DATA_PATH / "mvsh12.dat")
mvsh13 = MvsH(DATA_PATH / "mvsh13.dat", temperature=8)


params1 = Parameters()
params1.add("m_s_0", value=5.0)
params1.add("h_c_0", value=1.0)
params1.add("gamma_0", value=10)
params1.add("chi_pd", value=0.001)

params2 = Parameters()
params2.add("m_s_0", value=5.0)
params2.add("h_c_0", value=1.0)
params2.add("gamma_0", value=10)
params2.add("m_s_1", value=2.0)
params2.add("h_c_1", value=3.0)
params2.add("gamma_1", value=5)
params2.add("chi_pd", value=0.001)


@pytest.fixture
def fitting_args():
    return CauchyFittingArgs(
        [
            CauchyParams(
                m_s=(0.05, 0.0001, 0.3),
                h_c=(-20000, -60000, -5000),
                gamma=(25000, 5000, 200000),
            ),
            CauchyParams(m_s=(0.5, 0, 1.0), h_c=(0, -100, 100), gamma=(100, 5, 1000)),
            CauchyParams(
                m_s=(0.5, 0, 1), h_c=(15000, 10000, 30000), gamma=(10000, 2500, 40000)
            ),
        ]
    )


cdf_expected = [
    (10, params1, 2.3426),
    (np.array([100, 200]), params1, np.array([4.77956173, 5.04017972])),
    (10, params2, 3.5528),
    (np.array([100, 200]), params2, np.array([6.71398886, 7.00787093])),
]


@pytest.mark.parametrize("h, params, expected", cdf_expected)
def test_cauchy_cdf(h, params, expected):
    assert np.allclose(cauchy_cdf(h, params), expected, rtol=1e-4)


pdf_expected = [
    (10, params1, 0.092206),
    (np.array([100, 200]), params1, np.array([0.00181141, 0.00120092])),
    (10, params2, 0.123683),
    (np.array([100, 200]), params2, np.array([0.00198054, 0.00124192])),
]


@pytest.mark.parametrize("h, params, expected", pdf_expected)
def test_cauchy_pdf(h, params, expected):
    assert np.allclose(cauchy_pdf(h, params), expected, rtol=1e-4)


def test_fit_cauchy_cdf_num_terms():
    """An example of fitting data via the cdf function and using a single term."""
    x = mvsh12.simplified_data("forward")["field"]
    y = mvsh12.simplified_data("forward")["moment"]
    results = fit_cauchy_cdf(x, y, 1)
    assert len(results.terms) == 1
    assert np.allclose(results.terms[0].m_s, 0.09, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -34.49, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 431.06, rtol=0.02)
    assert np.allclose(results.chi_pd, -2.08e-8, rtol=0.02)
    assert np.allclose(results.chi_squared, 0.000209, rtol=0.02)


def test_fit_cauchy_pdf_num_terms():
    """An example of fitting data via the pdf function and using a single term."""
    x = mvsh12.simplified_data("forward")["field"]
    y = mvsh12.simplified_data("forward")["moment"]
    dydx = np.gradient(y, x)
    results = fit_cauchy_pdf(x, dydx, 1)
    assert len(results.terms) == 1
    assert np.allclose(results.terms[0].m_s, 0.0786, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -33.939, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 1443.011, rtol=0.02)
    assert np.allclose(results.chi_pd, 3.909e-6, rtol=0.02)
    assert np.allclose(results.chi_squared, 1.622e-8, rtol=0.02)


def test_fit_cauchy_cdf_params(
    fitting_args: CauchyFittingArgs,
):  # pylint: disable=redefined-outer-name
    """An example of fitting data via the cdf function and using multiple terms in the
    form of fitting args."""
    x = mvsh13.simplified_data("forward")["field"]
    y = mvsh13.simplified_data("forward")["moment"]
    results = fit_cauchy_cdf(x, y, fitting_args)
    assert len(results.terms) == 3
    assert np.allclose(results.terms[0].m_s, 0.073, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -17335.203, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 7708.642, rtol=0.02)
    assert np.allclose(results.terms[1].m_s, 0.081, rtol=0.02)
    assert np.allclose(results.terms[1].h_c, -44.452, rtol=0.02)
    assert np.allclose(results.terms[1].gamma, 73.309, rtol=0.02)
    assert np.allclose(results.terms[2].m_s, 0.167, rtol=0.02)
    assert np.allclose(results.terms[2].h_c, 14036.290, rtol=0.02)
    assert np.allclose(results.terms[2].gamma, 4785.800, rtol=0.02)
    assert np.allclose(results.chi_pd, 0.000000466, rtol=0.02)
    assert np.allclose(results.chi_squared, 0.018458, rtol=0.02)


def test_fit_cauchy_pdf_params(
    fitting_args: CauchyFittingArgs,
):  # pylint: disable=redefined-outer-name
    """An example of fitting data via the pdf function and using multiple terms in the
    form of fitting args."""
    x = mvsh13.simplified_data("forward")["field"]
    y = mvsh13.simplified_data("forward")["moment"]
    dydx = np.gradient(y, x)
    results = fit_cauchy_pdf(x, dydx, fitting_args)
    assert len(results.terms) == 3
    assert np.allclose(results.terms[0].m_s, 0.043087919, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -16732.827070410, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 19470.185801532, rtol=0.02)
    assert np.allclose(results.terms[1].m_s, 0.110738204, rtol=0.02)
    assert np.allclose(results.terms[1].h_c, -3.194245034, rtol=0.02)
    assert np.allclose(results.terms[1].gamma, 784.492158084, rtol=0.02)
    assert np.allclose(results.terms[2].m_s, 0.138365055, rtol=0.02)
    assert np.allclose(results.terms[2].h_c, 13030.298000446, rtol=0.02)
    assert np.allclose(results.terms[2].gamma, 12042.285605586, rtol=0.02)
    assert np.allclose(results.chi_pd, 0.000001383, rtol=0.02)
    assert np.allclose(results.chi_squared, 0.000000287, rtol=0.02)


def test_cauchy_cdf_analysis_num_terms():
    """An example of fitting data via the cdf class and using a single term."""
    analysis = CauchyCDFAnalysis(dset5, CauchyParsingArgs(300), 1)
    results = analysis.results
    assert len(results.terms) == 1
    assert np.allclose(results.terms[0].m_s, 75.037240206, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -34.311707446, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 430.538820228, rtol=0.02)
    assert np.allclose(results.chi_pd, -0.000017049, rtol=0.02)
    assert np.allclose(results.chi_squared, 294.090193155, rtol=0.02)


def test_cauchy_pdf_analysis_num_terms():
    """An example of fitting data via the pdf class and using a single term."""
    analysis = CauchyPDFAnalysis(dset5, CauchyParsingArgs(300), 1)
    results = analysis.results
    assert len(results.terms) == 1
    assert np.allclose(results.terms[0].m_s, 65.121491598, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -33.839534490, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 1432.274470987, rtol=0.02)
    assert np.allclose(results.chi_pd, 0.003522783, rtol=0.02)
    assert np.allclose(results.chi_squared, 0.022757120, rtol=0.02)


def test_cauchy_cdf_analysis_params(
    fitting_args: CauchyFittingArgs,
):  # pylint: disable=redefined-outer-name
    """An example of fitting data via the cdf class and using multiple terms in the
    form of fitting args."""
    m_sat = dset6.get_mvsh(2).simplified_data()["moment"].max()
    for arg in fitting_args.terms:
        arg.m_s = (m_sat * arg.m_s[0], m_sat * arg.m_s[1], m_sat * arg.m_s[2])
    analysis = CauchyCDFAnalysis(dset6, CauchyParsingArgs(8), fitting_args)
    results = analysis.results
    assert len(results.terms) == 3
    assert np.allclose(results.terms[0].m_s, 0.912120086, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -17335.185280059, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 7739.370073728, rtol=0.02)
    assert np.allclose(results.terms[1].m_s, 1.008925982, rtol=0.02)
    assert np.allclose(results.terms[1].h_c, -52.722320099, rtol=0.02)
    assert np.allclose(results.terms[1].gamma, 70.035927103, rtol=0.02)
    assert np.allclose(results.terms[2].m_s, 2.084670937, rtol=0.02)
    assert np.allclose(results.terms[2].h_c, 14043.014082755, rtol=0.02)
    assert np.allclose(results.terms[2].gamma, 4800.483295994, rtol=0.02)
    assert np.allclose(results.chi_pd, 0.000005944, rtol=0.02)
    assert np.allclose(results.chi_squared, 5.611427709, rtol=0.02)


def test_cauchy_pdf_analysis_params(
    fitting_args: CauchyFittingArgs,
):  # pylint: disable=redefined-outer-name
    """An example of fitting data via the pdf class and using multiple terms in the
    form of fitting args."""
    m_sat = dset6.get_mvsh(2).simplified_data()["moment"].max()
    for arg in fitting_args.terms:
        arg.m_s = (m_sat * arg.m_s[0], m_sat * arg.m_s[1], m_sat * arg.m_s[2])
    analysis = CauchyPDFAnalysis(dset6, CauchyParsingArgs(8), fitting_args)
    results = analysis.results
    assert len(results.terms) == 3
    assert np.allclose(results.terms[0].m_s, 0.777866808, rtol=0.02)
    assert np.allclose(results.terms[0].h_c, -17262.232384745, rtol=0.02)
    assert np.allclose(results.terms[0].gamma, 25016.867969794, rtol=0.02)
    assert np.allclose(results.terms[1].m_s, 1.369022700, rtol=0.02)
    assert np.allclose(results.terms[1].h_c, -15.675839493, rtol=0.02)
    assert np.allclose(results.terms[1].gamma, 773.151449611, rtol=0.02)
    assert np.allclose(results.terms[2].m_s, 1.940914543, rtol=0.02)
    assert np.allclose(results.terms[2].h_c, 13199.686677689, rtol=0.02)
    assert np.allclose(results.terms[2].gamma, 13740.289311196, rtol=0.02)
    assert np.allclose(results.chi_pd, 0.000005150, rtol=0.02)
    assert np.allclose(results.chi_squared, 0.000017446, rtol=0.02)
