Description of input parameters:
(not all but almost all input parameters will be printed out in output)


1. Task control and algorithms
restart :
        0 (default) - a new task ; 1 - a restart task
code :
         jdftx - interface to jdftx ; mos2 - mos2 model
alg_scatt :
        lindblad (default) / conventional - two kinds of equations
alg_eph_sepr_eh, alg_eph_need_elec, alg_eph_need_hole :
        if you had used "ePhOnlyElec"("ePhOnlyHole") in initialization input, you must set
        alg_eph_sepr_eh = 1
        alg_eph_need_elec = 1(0)
        alg_eph_need_hole = 0(1)
alg_expt :
        1 (default) - considering exp(iwt)  factor everywhere; 0 - not consider at all
alg_sparseP :
        1 - make scattering matrices (P? = sum_{+-,mode} G G nq) sparse; 0 - not (default)
        in any case, you can search "ns_tot" in output to see
        how many elements of P? matrices are larger than several thresholds (default 1e-40)
        generally, larger energy range and smaller smearing make P? more sparse
alg_phenom_relax :
        1 - turn on phenomenon relaxation - \dot{rho} = - (rho - rho_eq) / tau_phenom; 0 default
ode_method :
        Odinary Differential Equation method
        rkf45 (default) / euler


1. Perturb system to generate spin inbalance
(i) Magenetic field perturbation
Bxpert, Bypert, Bzpert :
        in unit Tesla, default 0
(ii) Pump pulse (Gaussian)
pumpMode :
        coherent / lindblad / perturb
        coherent : real-time pump using coherent formula -i[P(t),rho]
        lindblad : real-time pump using lindblad formula
        perturb : generate an initial pump perturbation
pumpA0 : 
        pump amplitude, pump power = (pumpE * pumpA0)^2 / (8 pi alpha)
pumpPoltype :
        pump polarization type. possible values - LC / RC / Ex / Ey
pumpE :
        pump energy in eV
pumpTau :
        pump pulse width in unit fs
        introduce a weight function exp(-t^2/tau^2/2) / sqrt( sqrt(pi) * tau ) into pump amplitude
probePoltype1 :
        first probe polarization type. possible values - LC / RC / Ex / Ey
probePoltype? :
        ?th probe polarization type
probeEmin, probeEmax :
        probe energy range in eV
probeDE :
        probe energy step
probeTau :
        probe pulse width in unit fs


3. Time and measure
t0 :
        inital time, in fs, default 0
        in restart calculation, will be read from file restart/time_restart.dat
tend :
        ending time for reporting, in fs
tstep :
        time step in fs
tstep_pump :
        time step for reporting during pump (pump time center +- 6*pumpTau), in fs
freq_measure_ene :
        how often to print energy-resolved observables
de_measure :
        energy step in eV for energy-resolved observables
degauss_measure :
        gaussian smearing in eV for energy-resolved observables


4.
temperature :
        currently, it must be the same as temperature in initialization (since I use matrices P? including nq)
        this is not bad, since usually we need more k points at lower temperatures
mu :
        chemical potential in eV, usually relative to VBM, must be in range [dmuMin, dmuMax] in initialization


5. Phenomenon relaxation
tau_phenom :
        phenomenon relaxation time in "ps"
bStart_tau and bEnd_tau :
        only relavent when alg_phenom_relax = 1
        bands considering phenomenon relaxation, relative to starting band of density matrix !!!