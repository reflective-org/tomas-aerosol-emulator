"""Analytic burst feature: Dunne inorganic log-nucleation-rate from raw cond.
cond_raw[...,:5] = [pres(Pa), nh3(molec/cm3), T(K), RH, H2SO4_vmr]. Returns logJ (same lead shape).
org=0 in v4 data => Riccobono=0 => pure Dunne. (Validated burst DETECTOR; see validate_jdunne.py.)"""
import numpy as np
_PBN, _UBN, _VBN, _WBN = 3.95, 9.70, 12.6, -0.00707
_PTN, _UTN, _VTN, _WTN, _PAN, _AN = 2.89, 182.0, 1.20, -4.19, 8.00, 1.6e-6
_PBI, _UBI, _VBI, _WBI = 3.37, -11.5, 25.5, 0.181
_PTI, _UTI, _VTI, _WTI, _PAI, _AI = 3.14, -23.8, 37.0, 0.227, 3.07, 0.00485

def dunne_logJ(cond_raw):
    c = np.asarray(cond_raw, dtype=np.float64)
    pres = c[..., 0]; nh3 = c[..., 1]; T = c[..., 2]; hv = c[..., 4]
    nair = pres / (1.38e-23 * T) * 1e-6                 # molec/cm3
    gas = hv * nair
    cna = gas * 1e-6; ns_ = np.maximum(nh3 * 1e-6, 1e-300)
    alpha = 6e-8 * np.sqrt(300.0 / T) + 6e-26 * nair * (300.0 / T) ** 4
    ionc = np.sqrt(3.0 / alpha); t1 = T / 1000.0
    kbn = np.exp(_UBN - np.exp(_VBN * (t1 - _WBN))); ktn = np.exp(_UTN - np.exp(_VTN * (t1 - _WTN)))
    kbi = np.exp(_UBI - np.exp(_VBI * (t1 - _WBI))); kti = np.exp(_UTI - np.exp(_VTI * (t1 - _WTI)))
    ffn = np.where(nh3 * 1e-6 > 1e-10, ns_ * cna ** _PTN / (_AN + cna ** _PTN / ns_ ** _PAN), 0.0)
    ffi = np.where(nh3 * 1e-6 > 1e-10, ns_ * cna ** _PTI / (_AI + cna ** _PTI / ns_ ** _PAI), 0.0)
    J = kbn * cna ** _PBN + ktn * ffn + kbi * ionc * cna ** _PBI + kti * ionc * ffi
    return np.log(np.maximum(J, 1e-30)).astype(np.float32)
