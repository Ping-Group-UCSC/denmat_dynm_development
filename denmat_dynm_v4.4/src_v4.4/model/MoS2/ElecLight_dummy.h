class electronlight_dummy{
public:
	double dt;
	bool active;
	string pumpMode;
	complex **dm_pump;

	void compute_pumpprobeP(){ ; }
	void pump_pert(){ ; }
	bool during_pump(double t){ ; }
	bool enter_pump(double t, double tnext){ ; }
	bool leave_pump(double t, double tnext){ ; }
	void evolve_pump(double t, complex** dm, complex** dm1, complex** ddm_pump){ ; }
	void evolve_pump_coh(double t, complex** dm, complex** dm1, complex** ddm_pump){ ; }
	void evolve_pump_lindblad(double t, complex** dm, complex** dm1, complex** ddm_pump){ ; }
	void probe(double iprobe, double t, complex **dm, complex **dm1){ ; }
};