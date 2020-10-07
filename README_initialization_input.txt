!!! See descriptions of other parameters in initialization input file of Silicene example

Compared with Shankar's initialization code, there are some new parameters:
ePhOnlyElec and ePhOnlyHole :
        if ePhOnlyElec (ePhOnlyHole) = 1, only e-ph of conduction (valence) bands will be written down;
nkBT :
        control energy range (Emargin = nkBT * Tmax)
        default is 7, which is enough for real-time but may not be for rate formula
nEphDelta :
        instead of a constant in Shankar code
        now you can set it to control the energy conservation
        This is important for spin lifetime computations inside the code, 
        since different smearings (e.g., max of ImSigme_ePh can be much larger than ePhDelta) will be used
band_skipped :
        starting band index of wannier relative to DFT, can be used to determine VBM and CBM correctly
        in case VBM is not zero (metal, Fermi smearing or finite electric field)