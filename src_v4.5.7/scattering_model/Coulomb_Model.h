#pragma once
#include "common_headers.h"
#include "Scatt_Param.h"
#include "lattice.h"
#include "parameters.h"
#include "electron.h"
#include "mymp.h"

struct homogeneous_electron_gas
{//homogeneous electron gas
	mymp *mp;
	lattice *latt;
	const double n, T, meff, epsb;
	double kF, vF, t, alpha, B, prefac1, prefac2, IFD;
	double alpha_, Q; //for transfering parameters of functions when using gsl
	double tol_qagiu, tol_qagp, thrQ;
	qIndexMap *qmap;
	vector<vector<double>> wq;
	vector<vector<complex>> epsqw;

	homogeneous_electron_gas(double& n, double& T, double& meff, double& epsb, double& kF, double& vF, double& EF,
		vector<vector3<double>>& qvec, int& iq_qmin, double& qmin, double& qmax, qIndexMap *qmap, lattice *latt,
		vector<double>& wqmax, double& wp)
		: n(n), T(T), meff(meff), epsb(epsb), kF(kF), vF(vF), t(T / EF), prefac1(0.25 / epsb / vF / M_PI), prefac2(0.125 * t / epsb / vF),
		tol_qagiu(1e-13), tol_qagp(1e-13), thrQ(1e-6), qmap(qmap), latt(latt),
		wq(qvec.size()), epsqw(qvec.size())
	{
		if (ionode) printf("t = %lg\n", t);

		alpha = solve_alpha();
		if (ionode) printf("mu = %lg eV alpha = %lg\n", alpha*EF / eV, alpha);

		B = alpha / t;
		IFD = calc_IFD();

		//mpi
		mp = new mymp();
		mp->mpi_init();
		mp->distribute_var("homogeneous_electron_gas", qvec.size());

		//static screening
		vector<double> eps0(qvec.size(),0);
		for (int iq = mp->varstart; iq < mp->varend; iq++){
			double q = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
			eps0[iq] = calc_eps(q).real();
		}
		mp->allreduce(eps0.data(), (int)qvec.size(), MPI_SUM);
		if (ionode) printf("static screening done\n");

		//write static screening
		if (ionode){
			string fname = "eps0.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#|q|^2 |qscr|^2 Eps\n");
			for (int iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq]));
				double qscr2 = q_length_square * (eps0[iq] - 1);
				fprintf(fp, "%14.7le %14.7le %14.7le\n", q_length_square, qscr2, eps0[iq]);
			}
			fclose(fp);
		}

		//construct wq and allocate epsqw
		double eps0_ref = calc_eps(qmin).real();
		double prefac_ratio = eps0_ref / (eps0_ref - 1) / wqmax[iq_qmin];

		for (size_t iq = 0; iq < qvec.size(); iq++){
			double q2 = latt->GGT.metric_length_squared(wrap(qvec[iq]));
			double ratio = q2 < 1e-20 ? 1 : prefac_ratio * wqmax[iq] * (eps0[iq] - 1) / eps0[iq];
			int nw = q2 < 1e-20 ? 2 : (int)round(ratio * clp.nomega) + 1; //will not deal with q=0 in this version
			if (nw < 6 && fabs(eps0[iq] - 1) > 0.1) nw = 6;
			if (nw < 2) nw = 2;
			double dw = wqmax[iq] / (nw - 1);
			wq[iq].resize(nw, 0);
			wq[iq][0] = 0; wq[iq][nw - 1] = wqmax[iq];
			for (int iw = 1; iw < nw - 1; iw++)
				wq[iq][iw] = iw * dw;

			//allocate epsqw
			epsqw[iq].resize(nw, c0);
		}
		if (ionode) printf("constructed frequency grids\n");

		//construct epsqw
		for (int iq = mp->varstart; iq < mp->varend; iq++){
			epsqw[iq][0] = eps0[iq];
			double q = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
			for (int iw = 1; iw < wq[iq].size(); iw++)
				epsqw[iq][iw] = calc_eps(q, wq[iq][iw]);
		}
		mp->allreduce(epsqw, MPI_SUM);
		if (ionode) printf("dynamic screening done\n");

		for (size_t iq = 1; iq < qvec.size(); iq++) //iq = 0 is Gamma
		for (int iw = 0; iw < wq[iq].size(); iw++)
		if (abs(epsqw[iq][iw]) < 1e-10){
			printf("rank = %d iq = %d iw = %d |eps| = %10.3le", mp->myrank, iq, iw, abs(epsqw[iq][iw]));
			exit(EXIT_FAILURE);
		}

		//write wq
		if (ionode){
			string fname = "wq.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#|q|^2  nw  dw (kBT)\n");
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				fprintf(fp, "%10.3le %d %10.3le\n", q_length_square, wq[iq].size(), wq[iq][1] / T);
			}
			fclose(fp);
		}

		/*
		//write static screening
		if (ionode){
			string fname = "eps0.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#|q|^2 |qscr|^2 Eps\n");
			for (int iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq]));
				double qscr2 = q_length_square * (epsqw[iq][0].real() - 1);
				fprintf(fp, "%14.7le %14.7le %14.7le\n", q_length_square, qscr2, epsqw[iq][0].real());
			}
			fclose(fp);
		}
		*/

		//test eps_intp
		vector<int> iq_test_arr{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (int)round(qvec.size() / 4) - 1, (int)round(qvec.size() / 2) - 1, (int)qvec.size() - 1 };
		for (int iqt = 0; iqt < iq_test_arr.size(); iqt++){
			int iq = iq_test_arr[iqt];
			double q = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
			if (q < 1e-10) continue;

			double wmin = 0;
			double wmax = wq[iq][wq[iq].size() - 1] + wq[iq][1];
			int nw = (int)round(sqrt(3)*wq[iq].size());
			double dw = (wmax - wmin) / (nw - 1);
			vector<double> w(nw);
			w[0] = wmin;
			for (int iw = 1; iw < nw; iw++)
				w[iw] = w[iw - 1] + dw;

			if (ionode){
				string fname = "eps_q" + int2str(iq) + ".out";
				FILE *fp = fopen(fname.c_str(), "w");
				fprintf(fp, "#|q| = %14.7le\n", q);
				fprintf(fp, "#w (kbt) ReEpsIntp ImEpsIntp |EpsIntp| ReEps ImEps |Eps|\n");
				for (int iw = 0; iw < nw; iw++){
					complex eps_intp_ = eps_intp(qvec[iq], w[iw]);
					complex eps = calc_eps(q, w[iw]);
					fprintf(fp, "%10.3le %10.3le %10.3le %10.3le %10.3le %10.3le %10.3le\n",
						w[iw] / T, eps_intp_.real(), eps_intp_.imag(), eps_intp_.abs(),
						eps.real(), eps.imag(), eps.abs());
				}
				fclose(fp);
			}
		}
	}
	~homogeneous_electron_gas(){
		wq = vector<vector<double>>();
		epsqw = vector<vector<complex>>();
	}
	complex eps_intp(vector3<> q, double w){
		double q_length_square = latt->GGT.metric_length_squared(wrap(q));
		if (q_length_square < 1e-20) return c0;
		size_t iq = qmap->q2iq(q);

		double dw = wq[iq][1];
		int iw = floor(fabs(w) / dw);
		int nw = wq[iq].size();
		complex result;
		if (iw >= nw - 1){//probably not happen at all
			double ratio = (fabs(w) - wq[iq][nw - 1]) / dw;
			result = epsqw[iq][nw - 1] + ratio * (epsqw[iq][nw - 1] - epsqw[iq][nw - 2]);
			double eps_nwm2_abs = abs(epsqw[iq][nw - 2]), eps_nwm1_abs = abs(epsqw[iq][nw - 1]), result_abs = abs(result);
			bool same_sign = (eps_nwm2_abs > 1 && eps_nwm1_abs > 1 && result_abs > 1) || (eps_nwm2_abs < 1 && eps_nwm1_abs < 1 && result_abs < 1);
			if (!same_sign) result = c1;
		}
		else{
			double ratio = (fabs(w) - wq[iq][iw]) / dw;
			result = epsqw[iq][iw] + ratio * (epsqw[iq][iw + 1] - epsqw[iq][iw]);
		}
		if (w < 0) return result.conj(); // eps(q,-w)=eps(q,w)^*
		else return result;
	}

	static double fd(double x, void *params){
		double alpha_ = ((homogeneous_electron_gas *)params)->alpha_;
		double t = ((homogeneous_electron_gas *)params)->t;
		return 1. / (1 + exp((x*x - alpha_) / t));
	}
	static double integrand_solve_alpha(double x, void *params){
		return x*x*((homogeneous_electron_gas *)params)->fd(x, params);
	}
	static double f_solve_alpha(double alpha_, void *params){
		((homogeneous_electron_gas *)params)->alpha_ = alpha_;
		double tol_qagiu = ((homogeneous_electron_gas *)params)->tol_qagiu;

		gsl_integration_workspace *w = gsl_integration_workspace_alloc(9999);

		double result, error;

		gsl_function F;
		F.function = &integrand_solve_alpha;
		F.params = params;

		gsl_integration_qagiu(&F, 0, tol_qagiu, tol_qagiu, 9999, w, &result, &error);

		gsl_integration_workspace_free(w);

		return result - 1. / 3;
	}
	double solve_alpha(){
		int status;
		int iter = 0, max_iter = 100;
		const gsl_root_fsolver_type *T;
		gsl_root_fsolver *s;
		double r = 0;
		double x_lo = -46 * t, x_hi = 1.0;
		gsl_function F;

		F.function = &f_solve_alpha;
		F.params = this;

		T = gsl_root_fsolver_brent;
		s = gsl_root_fsolver_alloc(T);
		gsl_root_fsolver_set(s, &F, x_lo, x_hi);

		if (ionode) printf("using %s method to find root\n", gsl_root_fsolver_name(s));

		do{
			iter++;
			status = gsl_root_fsolver_iterate(s);
			r = gsl_root_fsolver_root(s);
			x_lo = gsl_root_fsolver_x_lower(s);
			x_hi = gsl_root_fsolver_x_upper(s);
			status = gsl_root_test_interval(x_lo, x_hi, 1e-13, 1e-13);

			if (ionode && status == GSL_SUCCESS) printf("Converged in %5d iter\n", iter);
		} while (status == GSL_CONTINUE && iter < max_iter);

		gsl_root_fsolver_free(s);

		return r;
	}
	double calc_IFD(){
		gsl_integration_workspace *w = gsl_integration_workspace_alloc(9999);

		double result, error;

		gsl_function F;
		F.function = &fd;
		F.params = this;
		alpha_ = alpha;

		gsl_integration_qagiu(&F, 0, tol_qagiu, tol_qagiu, 9999, w, &result, &error);

		gsl_integration_workspace_free(w);

		return result;
	}

	double fp(double X, void *params){
		double Q = ((homogeneous_electron_gas *)params)->Q;
		double t = ((homogeneous_electron_gas *)params)->t;
		double B = ((homogeneous_electron_gas *)params)->B;
		double A = Q*Q / t;
		double rtmp = A*X*X - B;
		if (rtmp > 46)
			return -2 * A*X*exp(-rtmp);
		else{
			double rtmp2 = exp(rtmp);
			return -2 * A*X*rtmp2 / pow(1 + rtmp2, 2);
		}
	}
	static double integrand_g(double X, void *params){
		double Q = ((homogeneous_electron_gas *)params)->Q;
		double thrQ = ((homogeneous_electron_gas *)params)->thrQ;
		double rtmp = fabs(Q) > thrQ ? -X : X;
		if (fabs(X - 1) < 1e-20)
			return ((homogeneous_electron_gas *)params)->fp(X, params) * rtmp;
		else
			return ((homogeneous_electron_gas *)params)->fp(X, params) * (rtmp + 0.5*(1 - X*X)*log(fabs(X + 1) / fabs(X - 1)));
	}
	double g(double Q){
		double result = 0;
		if (Q == 0)
			return 0;
		else{
			if (t < 0.001){
				if (fabs(fabs(Q) - 1) < 1e-20)
					result = fabs(Q);
				else
					result = fabs(Q) + 0.5*(1 - fabs(Q)*fabs(Q)) * log(fabs(fabs(Q) + 1) / fabs(fabs(Q) - 1));
			}
			else{
				this->Q = fabs(Q);

				gsl_set_error_handler_off();
				gsl_integration_workspace *w = gsl_integration_workspace_alloc(9999);

				double integral, error;

				gsl_function F;
				F.function = &integrand_g;
				F.params = this;

				if (B > 0){
					double Xcenter = sqrt(alpha) / fabs(Q);
					if (t < 0.025 || Xcenter > 100){
						double i2, i3, e2, e3;
						double pts[4] = { 0, Xcenter, 2 * Xcenter, 5 * Xcenter };
						int status1 = gsl_integration_qagp(&F, pts, 3, tol_qagp, tol_qagp, 9999, w, &integral, &error);
						int status2 = gsl_integration_qag(&F, pts[2], pts[3], tol_qagiu, tol_qagiu, 9999, 1, w, &i2, &e2);
						int status3 = gsl_integration_qagiu(&F, pts[3], tol_qagiu, tol_qagiu, 9999, w, &i3, &e3);

						integral = integral + i2 + i3;

						if (status1 == GSL_EROUND || status2 == GSL_EROUND || status3 == GSL_EROUND){
							double error = error + e2 + e3;
							if (fabs(Q*Q*integral) > 1e-6 && (error / fabs(integral) > 1e-6)){
								printf("roundoff error; Q = %10.3le; ", Q);
								printf("i[0,2Xc] i[2Xc,5Xc] i[5Xc,inf] = %10.3le %10.3le %10.3le; error = %10.3le", integral - i2 - i3, i2, i3, error);
							}
						}
					}
					else{
						int status = gsl_integration_qagiu(&F, 0, tol_qagiu, tol_qagiu, 9999, w, &integral, &error);
						if (status == GSL_EROUND){
							if (fabs(Q*Q*integral) > 1e-6 && (error / fabs(integral) > 1e-6)){
								printf("roundoff error for integral in [0, inf] for B > 0\n");
								printf("integral = %10.3le error = %10.3le", integral, error);
							}
						}
					}
				}
				else{
					int status = gsl_integration_qagiu(&F, 0, tol_qagiu, tol_qagiu, 9999, w, &integral, &error);
					if (status == GSL_EROUND){
						if (fabs(Q*Q*integral) > 1e-6 && (error / fabs(integral) > 1e-6)){
							printf("roundoff error for integral in [0, inf] for B < 0\n");
							printf("integral = %10.3le error = %10.3le", integral, error);
						}
					}
				}

				gsl_integration_workspace_free(w);

				if (fabs(Q) > thrQ) result = Q*Q*integral;
				else result = 2*IFD*fabs(Q) + Q*Q*integral;
				if (Q < 0)
					result = -result;
			}
			return result;
		}
	}
	complex calc_eps(double q, double w = 0){
		if (q == 0) return c0;

		double qt = 0.5*q / kF;
		double qtinv3 = pow(qt, -3);
		double rtmp = w / q / vF;
		double q1t = qt - rtmp;
		double q2t = qt + rtmp;

		//real part
		double eps1; 
		if (w == 0) eps1 = 1 + 2 * prefac1 * qtinv3 * g(qt);
		else eps1 = 1 + prefac1 * qtinv3 * (g(q1t) + g(q2t));

		//imaginary part
		double a1 = (alpha - q1t*q1t) / t;
		double a2 = (alpha - q2t*q2t) / t;
		double l12;
		if (a2 > 46){
			if (a1 < -46)
				l12 = -a1 - log(1 + exp(-a2)) + a1 - a2;
			else
				l12 = log((1 + exp(-a1)) / (1 + exp(-a2))) + a1 - a2;
		}
		else{
			if (a1 > 46)
				l12 = a1 - log(1 + exp(a2));
			else
				l12 = log((1 + exp(a1)) / (1 + exp(a2)));
		}
		double eps2 = prefac2 * qtinv3 * l12;

		return complex(eps1, eps2);
	}
};

struct coulomb_model
{
	mymp *mp;
	bool ldebug;
	lattice *latt;
	electron *elec;
	int nk, bStart, bEnd, nb, nv; // bStart and bEnd relative to bStart_dm
	double T, nk_full, prefac_vq, prefac_vq_bare,
		kF, vF, EF, kF2, qTF2, fac0_Bechstedt, fac2_Bechstedt, fac4_Bechstedt, qscr2_debye, qscr2_TF;
	double **e, **f; double nfreetot_corr;
	vector<vector3<double>> qvec; double qmin, qmax; int iq_qmin;
	qIndexMap *qmap; kIndexMap *kmap;
	vector<complex> omega; double domega;
	vector<double> wqmax;
	vector<vector<complex>> omegaq; //frequencies for each q, currently we will only stores the max.
	vector<complex> qscr2_static_RPA;
	vector<vector<complex>> vq_RPA;
	vector<complex> Aq_ppa, Eq2_ppa; double wp2;
	complex *Uih, *ovlp;
	homogeneous_electron_gas *heg;

	coulomb_model(lattice *latt, parameters *param, electron *elec, int bStart, int bEnd, double dE)
		: latt(latt), elec(elec), T(param->temperature), nk(elec->nk), nk_full(elec->nk_full),
		bStart(bStart), bEnd(bEnd), nb(bEnd - bStart), nv(elec->nv_dm - bStart),
		qmap(nullptr), qmax(0), qmin(0), omega(clp.nomega), kmap(nullptr), ldebug(true)
	{
		if (ionode) printf("\nInitialize screening formula %s\n", clp.scrFormula.c_str());
		if (ionode) printf("bStart = %d bEnd = %d nv = %d\n", bStart, bEnd, nv);
		if (latt->dim < 3) error_message("debye screening model for lowD is not implemented");
		prefac_vq = 4 * M_PI / clp.eps / latt->cell_size;
		prefac_vq_bare = 4 * M_PI / latt->cell_size;
		//if (ionode) printf("prefac_vq = %10.3le\n", prefac_vq);
		e = trunc_alloccopy_array(elec->e_dm, nk, bStart, bEnd);
		f = trunc_alloccopy_array(elec->f_dm, nk, bStart, bEnd);
		if (clp.scrFormula == "RPA" || clp.scrFormula == "lindhard"){
			Uih = new complex[nb*elec->nb_wannier]{c0};
			ovlp = new complex[nb*nb]{c0};
		}

		//carrier density correction if two k-point lists are used
		nfreetot_corr = 0;
		if (elec->nk_morek){
			for (int ik = 0; ik < nk; ik++)
			for (int i = bStart; i < bEnd; i++)
			if (i >= nv) nfreetot_corr -= elec->f_dm[ik][i];
			else nfreetot_corr -= (elec->f_dm[ik][i] - 1.); // hole concentration is negative

			for (int ik = 0; ik < elec->nk_morek; ik++)
			for (int i = bStart; i < bEnd; i++)
			if (i >= nv) nfreetot_corr += elec->f_dm_morek[ik][i];
			else nfreetot_corr += (elec->f_dm_morek[ik][i] - 1.); // hole concentration is negative
		}
		nfreetot_corr /= (nk_full * latt->cell_size);

		//initialize qmap and qvec,
		if (qmap == nullptr){
			qmap = new qIndexMap(elec->kmesh); qmap->build(elec->kvec, qvec);

			if (ionode && DEBUG){
				string fnameq = dir_debug + "qIndexMap.out";
				FILE *fpq = fopen(fnameq.c_str(), "w");
				fprintf(fpq, "\nPrint qIndexMap:\n"); fflush(fpq);
				for (size_t iq = 0; iq < qvec.size(); iq++){
					std::map<vector3<int>, size_t>::iterator iter = qmap->the_map.find(qmap->iqvec3(qvec[iq]));
					fprintf(fpq, "iqvec3 = (%d,%d,%d) iq = %lu\n", iter->first[0], iter->first[1], iter->first[2], iter->second);  fflush(fpq);
				}
				fclose(fpq);
			}
		}

		kmap = new kIndexMap(elec->kmesh, elec->kvec);
		if (ionode && DEBUG){
			string fnamek = dir_debug + "kIndexMap.out";
			FILE *fpk = fopen(fnamek.c_str(), "w");
			fprintf(fpk, "\nPrint kIndexMap:\n");
			for (size_t ik = 0; ik < elec->kvec.size(); ik++){
				std::map<vector3<int>, size_t>::iterator iter = kmap->the_map.find(kmap->ikvec3(elec->kvec[ik]));
				fprintf(fpk, "ikvec3 = (%d,%d,%d) ik = %lu\n", iter->first[0], iter->first[1], iter->first[2], iter->second);
			}
			fclose(fpk);
		}

		//find out qmin and qmax
		if (qmax == 0 && qmin == 0){
			int iq_start = 0;
			for (int iq = 0; iq < qvec.size(); iq++){
				double q_length = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
				if (q_length < 1e-10) continue; // skip Gamma point in current version
				qmax = qmin = q_length;
				iq_qmin = iq;
				iq_start = iq + 1;
				break;
			}
			for (int iq = iq_start; iq < qvec.size(); iq++){
				double q_length = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
				if (q_length < 1e-10) continue; // skip Gamma point in current version
				if (q_length < qmin){ qmin = q_length; iq_qmin = iq; }
				if (q_length > qmax) qmax = q_length;
			}
			if (ionode) printf("qmin = %lg qmax = %lg\n", qmin, qmax);
		}

		//find out wmax for each q
		wqmax.resize(qvec.size(), 0);
		for (int ik1 = 0; ik1 < nk; ik1++)
		for (int ik2 = 0; ik2 < nk; ik2++){
			size_t iq = qmap->q2iq(elec->kvec[ik1] - elec->kvec[ik2]);

			for (int b1 = 0; b1 < nb; b1++)
			for (int b2 = 0; b2 < nb; b2++){
				double w = fabs(e[ik1][b1] - e[ik2][b2]);
				if (w > wqmax[iq]) wqmax[iq] = w;
			}
		}

		if (ionode){
			string fname = "wmax_q_kbt.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#|q|(au) wmax(kbT)\n");
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length = sqrt(latt->GGT.metric_length_squared(wrap(qvec[iq])));
				fprintf(fp, "%10.3le %10.3le\n", q_length, wqmax[iq] / T);
			}
			fclose(fp);
		}

		//for dynamics screening
		if (clp.omegamax == 0) clp.omegamax = dE;
		if (clp.dynamic == "ppa" && clp.ppamodel == "gn") { omega.resize(2); omega[0] = c0; }
		if (!(clp.dynamic == "ppa" && clp.ppamodel == "gn")) clp.smearing = (clp.smearing <= 0) ? 0.5 * T : clp.smearing;
		if (ionode) printf("smearing = %10.3le a.u. (%10.3le meV / %10.3le K)\n", clp.smearing, clp.smearing / eV * 1000, clp.smearing / Kelvin);

		//mpi
		mp = elec->mp;

		//initialization for models and RPA
		init_model(clp.nfreetot);
		if (clp.scrFormula == "RPA") init_RPA();
	}
	void init(double **ft){
		clp.nfreetot = 0;
		for (int ik = 0; ik < nk; ik++)
		for (int b = 0; b < nb; b++){
			if (b < nv) clp.nfreetot += (1 - ft[ik][b]);
			else clp.nfreetot += ft[ik][b];
		}
		clp.nfreetot /= (nk_full * latt->cell_size);
		clp.nfreetot += nfreetot_corr;
		if (ionode) printf("nfree = %lg cm-3 for screening\n", clp.nfreetot / std::pow(bohr2cm, 3));
		init_model(clp.nfreetot);
		if (clp.scrFormula == "RPA") init_RPA();
	}
	void init(complex **dm){
		double **ft = alloc_real_array(nk, nb);
		for (int ik = 0; ik < nk; ik++)
		for (int b = 0; b < nb; b++)
			ft[ik][b] = real(dm[ik][b*nb + b]);
		init(ft);
		dealloc_real_array(ft);
	}
	void init_model(double n, FILE *fp = stdout){
		if ((clp.scrFormula == "debye" || clp.scrFormula == "Bechstedt" || clp.scrFormula == "heg") && n <= 0)
			error_message("nfreetot must be postive");
		kF = std::pow(3 * M_PI*M_PI * n, 1. / 3.);
		vF = kF / clp.meff;
		kF2 = kF*kF;
		EF = kF2 / 2. / clp.meff;
		if (ionode) printf("kF = %lg EF = %lg\n", kF, EF);
		qTF2 = 6 * M_PI * n / clp.eps / EF;
		if (clp.scrFormula == "Bechstedt"){
			fac0_Bechstedt = 1. / (clp.eps - 1);
			fac2_Bechstedt = 1. / qTF2;
			fac4_Bechstedt = 3. / 4. / kF2 / qTF2;
		}
		qscr2_debye = 4 * M_PI * n / clp.eps / T; // Eq. 11 in PRB 94, 085204 (2016)
		qscr2_TF = 6 * M_PI * n / clp.eps / EF;
		if (ionode) printf("qscr2_debye = %lg qscr2_TF = %lg\n", qscr2_debye, qscr2_TF);

		double wp = sqrt(4 * M_PI * n / clp.meff / clp.eps);
		if (ionode) printf("wp = %10.3le a.u. (%10.3le meV / %10.3le K)\n", wp, wp / eV * 1000, wp / Kelvin);

		if (clp.scrFormula == "heg" || (ldebug && clp.dynamic == "real-axis")){
			if (heg != nullptr){
				bool update_heg = fabs(n - heg->n) / std::pow(bohr2cm, latt->dim) > 1; //if n is changed, heg needs to be updated as other parameters are all fixed
				if (update_heg){
					delete heg;
					heg = new homogeneous_electron_gas(n, T, clp.meff, clp.eps, kF, vF, EF, qvec, iq_qmin, qmin, qmax, qmap, latt, wqmax, wp);
				}
			}
			else
				heg = new homogeneous_electron_gas(n, T, clp.meff, clp.eps, kF, vF, EF, qvec, iq_qmin, qmin, qmax, qmap, latt, wqmax, wp);
		}
	}
	void init_RPA(double **ft = nullptr){
		if (clp.eppa == 0) clp.eppa = sqrt(4 * M_PI * clp.nfreetot / clp.meff / clp.eps);
		if (clp.dynamic == "ppa") omega[1] = ci * clp.eppa;
		if (ft != nullptr) trunc_copy_array(f, ft, nk, 0, nb);

		calc_qscr2_static_RPA();

		//determine frequency grids (for each q)		
		if (clp.dynamic == "real-axis"){
			domega = clp.omegamax / (clp.nomega - 1);
			omega[0] = c0;
			for (int io = 1; io < clp.nomega; io++)
				omega[io] = omega[io - 1] + domega;

			double prefac_ratio = (1 + qmin*qmin / qscr2_static_RPA[iq_qmin].abs()) / wqmax[iq_qmin];

			omegaq.resize(qvec.size());
			for (int iq = 0; iq < qvec.size(); iq++){
				//construct wq
				double q2 = latt->GGT.metric_length_squared(wrap(qvec[iq]));
				double qscr2 = qscr2_static_RPA[iq].abs();
				double ratio = q2 < 1e-20 ? 1 : prefac_ratio * wqmax[iq] / (1 + q2 / qscr2);
				int nw = q2 < 1e-20 ? 2 : nw = (int)round(ratio * clp.nomega) + 1; //will not deal with q=0 in this version
				if (nw < 6 && q2 > 1e-20 && fabs(qscr2 / q2) > 0.1) nw = 6;
				if (nw < 2) nw = 2;
				double dw = wqmax[iq] / (nw - 1);
				omegaq[iq].resize(nw);
				omegaq[iq][0] = c0; omegaq[iq][nw - 1] = complex(wqmax[iq], clp.smearing);
				for (int iw = 1; iw < nw - 1; iw++)
					omegaq[iq][iw] = complex(iw*dw, clp.smearing);
			}

			string fname = "wq.out";
			FILE *fp = fopen(fname.c_str(), "w");
			fprintf(fp, "#|q|^2  nw  dw (kBT)\n");
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				fprintf(fp, "%10.3le %d %10.3le %10.3le\n", q_length_square, omegaq[iq].size(), omegaq[iq][1].real() / T, omegaq[iq][1].imag() / T);
			}
			fclose(fp);
		}

		calc_vq_RPA();
	}
	
	complex vq(vector3<double> q, double w = 0){
		double q_length_square = latt->GGT.metric_length_squared(wrap(q));
		complex result;
		if (clp.scrFormula == "unscreened"){
			if (q_length_square < 1e-20) return c0; // skip Gamma point in current version
			return complex(prefac_vq / q_length_square, 0);
		}
		else if (clp.scrFormula == "heg"){
			if (q_length_square < 1e-20) return c0; // skip Gamma point in current version
			//if (ionode) printf("q = %lg  w = %lg in T\n", sqrt(q_length_square), w/T);
			complex eps = heg->eps_intp(q, w);
			//if (ionode) printf("|eps| = %lg\n", abs(eps));
			if (abs(eps) == 0)
				error_message("eps is zero!","vq");
			else{
				if (abs(eps) < 1e-20) printf("|eps| = %10.3le is too tiny!\n", abs(eps));
				return complex(prefac_vq,0) / eps / q_length_square;
			}
		}
		else if (clp.scrFormula == "debye"){
			return complex(prefac_vq / (qscr2_debye + q_length_square), 0);
		}
		else if (clp.scrFormula == "Bechstedt"){
			if (q_length_square < 1e-20) return c0; // skip Gamma point in current version
			double q2 = q_length_square;
			double epsq = 1 + 1. / (fac0_Bechstedt + fac2_Bechstedt * q2 + fac4_Bechstedt * q2 * q2);
			return complex(prefac_vq_bare / epsq / q2, 0);
		}
		else{
			size_t iq = qmap->q2iq(q);
			if (q_length_square < 1e-20) return c0; // skip Gamma point in current version
			if (clp.dynamic == "static" || w == 0)
				return vq_RPA[iq][0].real();
			else if (clp.dynamic == "ppa"){
				double q_length_square = latt->GGT.metric_length_squared(wrap(q));
				//if (clp.ppamodel == "hl"){
				complex cw = complex(fabs(w), clp.smearing);
				result = (prefac_vq + Aq_ppa[iq] / (cw*cw - Eq2_ppa[iq])) / q_length_square;
				//}
				//else if (clp.ppamodel == "gn"){ //for GN ppa, smearing is not needed
				//	result = (prefac_vq + Aq_ppa[iq] / (w*w - Eq2_ppa[iq])) / q_length_square; // Aq_ppa has been multiplied by prefac_vq
				//}
			}
			else if (clp.dynamic == "real-axis"){
				double dw = omegaq[iq][1].real();
				int iw = floor(fabs(w) / dw);
				int nw = omegaq[iq].size();
				if (iw >= nw - 1) result = vq_RPA[iq][nw - 1];
				else result = (vq_RPA[iq][iw] * (omegaq[iq][iw + 1].real() - w) + vq_RPA[iq][iw + 1] * (w - omegaq[iq][iw].real())) / dw; // linear interpolation
			}
			if (w < 0) return result.conj(); // eps(q,-w)=eps(q,w)^*
			else return result;
		}
	}
	void calc_ovlp(int ik, int jk){
		hermite(elec->U[ik], Uih, elec->nb_wannier, nb);
		zgemm_interface(ovlp, Uih, elec->U[jk], nb, nb, elec->nb_wannier);
	}
	void calc_qscr2_static_RPA(){
		qscr2_static_RPA.resize(qvec.size(), c0);
		// vq = vq0 / (1 - vq0 * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full)
		// vq0 = e^2 / V / (eps_r * eps_0) / q^2
		// Therefore, vq = e^2 / V / (eps_r * eps_0) / (q^2 + betas^2)
		// betas^2 = - e^2 / V / (eps_r * eps_0) * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full
		/*
		vector<complex> qscr2_ref(qvec.size());
		for (int ik = 0; ik < nk; ik++)
		for (int jk = 0; jk < nk; jk++){
			vector3<double> q = elec->kvec[ik] - elec->kvec[jk];
			int iq = qmap->q2iq(q);
			calc_ovlp(ik, jk);
			for (int b1 = 0; b1 < nb; b1++)
			for (int b2 = 0; b2 < nb; b2++){
				if (clp.scrFormula == "lindhard" && b1 != b2) continue;
				complex de = e[ik][b1] - e[jk][b2], dfde = c0;
				if (clp.fderavitive_technique){
					if (abs(de) < 1e-8){
						double favg = 0.5 * (f[ik][b1] + f[jk][b2]);
						dfde = complex((1 - favg) * favg / T, 0); // only true for Fermi-Dirac
					}
					else dfde = complex(f[jk][b2] - f[ik][b1], 0) / de;
				}
				else
					dfde = complex(f[jk][b2] - f[ik][b1], 0) / (de - complex(0, clp.smearing));
				if (clp.scrFormula == "RPA") qscr2_ref[iq] += dfde * ovlp[b1*nb + b2].norm();
				else if (clp.scrFormula == "lindhard") qscr2_ref[iq] += dfde;
			}
		}
		axbyc(qscr2_ref.data(), nullptr, qvec.size(), 0, complex(prefac_vq / nk_full, 0), c0); // y = ax + by + c
		*/

		for (int iq = 0; iq < qvec.size(); iq++){
			for (int ik = mp->varstart; ik < mp->varend; ik++){
				size_t jk = 0; vector3<> kj = elec->kvec[ik] - qvec[iq]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
				if (kmap->findk(kj, jk)){
					calc_ovlp(ik, jk);
					for (int b1 = 0; b1 < nb; b1++)
					for (int b2 = 0; b2 < nb; b2++){
						if (clp.scrFormula == "lindhard" && b1 != b2) continue;
						complex de = e[ik][b1] - e[jk][b2], dfde = c0;
						if (abs(de) < 1e-8){
							double favg = 0.5 * (f[ik][b1] + f[jk][b2]);
							dfde = complex((1 - favg) * favg / T, 0); // only true for Fermi-Dirac
						}
						else dfde = complex(f[jk][b2] - f[ik][b1], 0) / de;
						if (clp.scrFormula == "RPA") qscr2_static_RPA[iq] += dfde * ovlp[b1*nb + b2].norm();
						else if (clp.scrFormula == "lindhard") qscr2_static_RPA[iq] += dfde;
					}
				}
			}
			mp->allreduce(qscr2_static_RPA[iq], MPI_SUM);
			qscr2_static_RPA[iq] = complex(prefac_vq / nk_full, 0) * qscr2_static_RPA[iq];
		}

		//2nd implementation of static screening for comparison
		vector<complex> qscr2_2ndway(qvec.size());
		for (int iq = 0; iq < qvec.size(); iq++){
			for (int ik = mp->varstart; ik < mp->varend; ik++){
				size_t jk = 0; vector3<> kj = elec->kvec[ik] - qvec[iq]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
				if (kmap->findk(kj, jk)){
					calc_ovlp(ik, jk);
					for (int b1 = 0; b1 < nb; b1++)
					for (int b2 = 0; b2 < nb; b2++){
						if (clp.scrFormula == "lindhard" && b1 != b2) continue;
						complex de = e[ik][b1] - e[jk][b2] - complex(0, clp.smearing);
						complex dfde = complex(f[jk][b2] - f[ik][b1], 0) / de;
						if (clp.scrFormula == "RPA") qscr2_2ndway[iq] += dfde * ovlp[b1*nb + b2].norm();
						else if (clp.scrFormula == "lindhard") qscr2_2ndway[iq] += dfde;
					}
				}
			}
			mp->allreduce(qscr2_2ndway[iq], MPI_SUM);
			qscr2_2ndway[iq] = complex(prefac_vq / nk_full, 0) * qscr2_2ndway[iq];
		}
		if (ionode){
			string fnamevq = "qscr2_static_RPA.out";
			FILE *fpvq = fopen(fnamevq.c_str(), "w");
			init_model(clp.nfreetot, fpvq);
			fprintf(fpvq, "#|q|^2 |q_scr|^2 |q_scr_ref|^2 |2nd q_scr|^2\n");
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				fprintf(fpvq, "%14.7le %14.7le %14.7le\n", q_length_square, abs(qscr2_static_RPA[iq]), abs(qscr2_2ndway[iq]));
			}
			fclose(fpvq);
		}

		if (ionode) printf("\ncalc_qscr2_static_RPA done\n");
	}
	void calc_vq_RPA(){
		vector<vector<complex>> qscr2_RPA(qvec.size());
		for (int iq = 0; iq < qvec.size(); iq++){
			int nomega = 1;
			if (clp.dynamic == "ppa" && clp.ppamodel == "gn") nomega = 2;
			if (clp.dynamic == "real-axis") nomega = omegaq[iq].size();
			qscr2_RPA[iq].resize(nomega);
			qscr2_RPA[iq][0] = qscr2_static_RPA[iq];
		}
		qscr2_static_RPA = vector<complex>();
		// vq = vq0 / (1 - vq0 * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full)
		// vq0 = e^2 / V / (eps_r * eps_0) / q^2
		// Therefore, vq = e^2 / V / (eps_r * eps_0) / (q^2 + betas^2)
		// betas^2 = - e^2 / V / (eps_r * eps_0) * sum_k [(f_k - f_k-q) / (e_k - e_k-q - w - i0)] / nk_full
		for (int iq = 0; iq < qvec.size(); iq++){
			int nomega = 1;
			if (clp.dynamic == "ppa" && clp.ppamodel == "gn") nomega = 2;
			if (clp.dynamic == "real-axis") nomega = omegaq[iq].size();
			for (int iw = 1; iw < nomega; iw++){
				for (int ik = mp->varstart; ik < mp->varend; ik++){
					size_t jk = 0; vector3<> kj = elec->kvec[ik] - qvec[iq]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
					if (kmap->findk(kj, jk)){
						calc_ovlp(ik, jk);
						for (int b1 = 0; b1 < nb; b1++)
						for (int b2 = 0; b2 < nb; b2++){
							if (clp.scrFormula == "lindhard" && b1 != b2) continue;
							complex de;
							if (clp.dynamic == "ppa" && clp.ppamodel == "gn" && iw == 1)
								de = e[ik][b1] - e[jk][b2] - omega[iw];
							else
								de = e[ik][b1] - e[jk][b2] - omegaq[iq][iw];
							complex dfde = complex(f[jk][b2] - f[ik][b1], 0) / de;
							if (clp.scrFormula == "RPA") qscr2_RPA[iq][iw] += dfde * ovlp[b1*nb + b2].norm();
							else if (clp.scrFormula == "lindhard") qscr2_RPA[iq][iw] += dfde;
						}
					}
				}
				mp->allreduce(qscr2_RPA[iq][iw], MPI_SUM);
				qscr2_RPA[iq][iw] = complex(prefac_vq / nk_full, 0) * qscr2_RPA[iq][iw];
			}
		}
		if (clp.dynamic == "static" || clp.dynamic == "ppa"){
			vq_RPA.resize(qvec.size());
			for (size_t iq = 0; iq < qvec.size(); iq++){
				vq_RPA[iq].resize(1);
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				vq_RPA[iq][0] = complex(prefac_vq, 0) / (q_length_square + qscr2_RPA[iq][0]);
			}
		}
		if (clp.dynamic == "ppa"){
			double wp2 = clp.eppa * clp.eppa;

			/*
			//If we use qscr^2(w) = A / (w^2 - wq^2), we can include q=0 in Godby¨CNeeds PPA
			Aq_ppa.resize(qvec.size()); Eq2_ppa.resize(qvec.size());

			for (size_t iq = 0; iq < qvec.size(); iq++){
			if (iq == 0 && clp.ppamodel == "hl") continue;

			double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)

			if (clp.ppamodel == "hl"){
			//Hybertsen - Louie
			double dtmp = wp2 * q_length_square;
			Aq_ppa[iq] = -dtmp * qscr2_RPA[iq][0];
			Eq2_ppa[iq] = dtmp / qscr2_RPA[iq][0];
			}
			else if (clp.ppamodel == "gn"){
			//Godby¨CNeeds
			complex ctmp = wp2 * qscr2_RPA[iq][1];
			Aq_ppa[iq] = -ctmp * qscr2_RPA[iq][0];
			Eq2_ppa[iq] = ctmp / (qscr2_RPA[iq][0] - qscr2_RPA[iq][1]);
			}
			}
			*/

			//eps^-1(w) = 1 + A / (w^2 - wq^2)
			Aq_ppa.resize(qvec.size()); Eq2_ppa.resize(qvec.size());

			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				if (q_length_square < 1e-20) continue; //currently Gamma point is skipped

				if (clp.ppamodel == "hl"){
					//Hybertsen - Louie
					Aq_ppa[iq] = prefac_vq * wp2;
					Eq2_ppa[iq] = wp2 * (1 + complex(q_length_square, 0) / qscr2_RPA[iq][0]);
				}
				else if (clp.ppamodel == "gn"){
					//Godby¨CNeeds
					complex eps0inv = c1 / (1 + qscr2_RPA[iq][0] / q_length_square);
					complex epspinv = c1 / (1 + qscr2_RPA[iq][1] / q_length_square);
					Eq2_ppa[iq] = wp2 * (1 - epspinv) / (epspinv - eps0inv);
					Aq_ppa[iq] = prefac_vq * (1 - eps0inv) * Eq2_ppa[iq];
				}
			}
		}
		if (clp.dynamic == "real-axis"){
			vq_RPA.resize(qvec.size());
			for (size_t iq = 0; iq < qvec.size(); iq++){
				vq_RPA[iq].resize(omegaq[iq].size());
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)

				for (size_t iw = 0; iw < omegaq[iq].size(); iw++){
					vq_RPA[iq][iw] = complex(prefac_vq, 0) / (q_length_square + qscr2_RPA[iq][iw]);

					if (ldebug && ionode && q_length_square > 1e-20){
						complex eps = 1 + qscr2_RPA[iq][iw] / complex(q_length_square, 0);
						complex eps_heg = heg->eps_intp(qvec[iq], omegaq[iq][iw].real());
						if (abs(eps) < 0.2 || (abs(eps) / abs(eps_heg) < 0.5)){
							printf("iq = %lu  w = %10.3le T  |q|^2 = %10.3le  qscr2 = %10.3le %10.3le  eps_heg = %10.3le %10.3le\n", 
								iq, omegaq[iq][iw].real() / T, q_length_square, qscr2_RPA[iq][iw].real(), qscr2_RPA[iq][iw].imag(), eps_heg.real(), eps_heg.imag());
						}
					}
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (ionode){
			//static case is always written
			string fnamevq = "ldbd_vq.out";
			if (exists(fnamevq)) fnamevq = "ldbd_vq_updated.out";
			FILE *fpvq = fopen(fnamevq.c_str(), "w");
			init_model(clp.nfreetot, fpvq);
			fprintf(fpvq, "#|q|^2 |q_scr|^2 vq\n"); fflush(fpvq);
			for (size_t iq = 0; iq < qvec.size(); iq++){
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
				fprintf(fpvq, "%14.7le %14.7le %14.7le\n", q_length_square, abs(qscr2_RPA[iq][0]), abs(vq_RPA[iq][0])); fflush(fpvq);
			}
			fclose(fpvq);

			if (clp.dynamic == "ppa"){
				string fnamevq = "ldbd_vq_ppa.out";
				if (exists(fnamevq)) fnamevq = "ldbd_vq_ppa_updated.out";
				FILE *fpvq = fopen(fnamevq.c_str(), "w");

				if (clp.ppamodel == "hl")
					fprintf(fpvq, "#|q|^2 |q_scr|^2 wp^2 |Eq|^2\n");
				else if (clp.ppamodel == "gn")
					fprintf(fpvq, "#|q|^2 |q_scr|^2 vq(w=0) |q_scr(i*wp)|^2 Aq Eq^2\n");
				for (size_t iq = 0; iq < qvec.size(); iq++){
					double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)
					if (clp.ppamodel == "hl"){
						fprintf(fpvq, "%14.7le %14.7le  ", q_length_square, abs(qscr2_RPA[iq][0]));
						fprintf(fpvq, "%14.7le %14.7le\n", wp2, abs(Eq2_ppa[iq])); fflush(fpvq);
					}
					else if (clp.ppamodel == "gn"){
						fprintf(fpvq, "%14.7le %14.7le %14.7le   %14.7le %14.7le   ", q_length_square, abs(qscr2_RPA[iq][0]), abs(vq(qvec[iq], 0)), qscr2_RPA[iq][1].real(), qscr2_RPA[iq][1].imag());
						fprintf(fpvq, "%14.7le %14.7le   %14.7le %14.7le\n", Aq_ppa[iq].real() / prefac_vq, Aq_ppa[iq].imag() / prefac_vq, Eq2_ppa[iq].real(), Eq2_ppa[iq].imag()); fflush(fpvq);
					}
				}
				fclose(fpvq);

				double wmax = 14 * T;
				int nw = 141;
				double dw = wmax / (nw - 1);
				vector<double> w(nw);
				w[0] = 0;
				for (int iw = 1; iw < nw; iw++)
					w[iw] = w[iw - 1] + dw;
				int iq = 0;
				double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq]));
				if (q_length_square < 1e-20) iq = 1;
				q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq]));
				string fnamevqw = "ldbd_vqw_1stq.out";
				FILE *fpvqw = fopen(fnamevqw.c_str(), "wb");
				for (size_t iw = 0; iw < w.size(); iw++)
					fprintf(fpvqw, "%14.7le %14.7le %14.7le\n", w[iw] / T, abs(vq(qvec[iq], w[iw])), prefac_vq / q_length_square / abs(vq(qvec[iq], w[iw])));
				fclose(fpvqw);
			}
			else if (clp.dynamic == "real-axis"){
				vector<int> iq_test_arr{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (int)round(qvec.size() / 4) - 1, (int)round(qvec.size() / 2) - 1, (int)qvec.size() - 1 };
				for (int iqt = 0; iqt < iq_test_arr.size(); iqt++){
					int iq = iq_test_arr[iqt];
					double q_length_square = latt->GGT.metric_length_squared(wrap(qvec[iq])); // qvec is already wrapped to [-0.5,0.5)

					//directly calculated qscr2 and vqw
					string fnamevqw = "vqw_q" + int2str(iq) + ".out";
					FILE *fpvqw = fopen(fnamevqw.c_str(), "w");
					fprintf(fpvqw, "#w (kBT) qscr2 vqw (smearing = %7.3lf kBT |q|^2 = %14.7le)\n", clp.smearing / T, q_length_square);
					for (size_t iw = 0; iw < omegaq[iq].size(); iw++)
						fprintf(fpvqw, "%10.3le   %10.3le %10.3le   %10.3le %10.3le\n", omegaq[iq][iw].real() / T, qscr2_RPA[iq][iw].real(), qscr2_RPA[iq][iw].imag(), vq_RPA[iq][iw].real(), vq_RPA[iq][iw].imag());
					fclose(fpvqw);

					//interpolated quantities
					double wmax = wqmax[iq] + omegaq[iq][1].real();
					int nw = (int)round(sqrt(3)*omegaq[iq].size());
					double dw = wmax / (nw - 1);
					vector<double> w(nw);
					w[0] = 0; w[nw-1] = wmax;
					for (int iw = 1; iw < nw-1; iw++)
						w[iw] = iw*dw;

					fnamevqw = "vqw_intp_q" + int2str(iq) + ".out";
					fpvqw = fopen(fnamevqw.c_str(), "wb");
					fprintf(fpvqw, "#w (kBT) |vqw| ReEps ImEps |Eps| (smearing = %7.3lf kBT |q|^2 = %14.7le)\n", clp.smearing / T, q_length_square);
					for (size_t iw = 0; iw < w.size(); iw++){
						complex vqw = vq(qvec[iq], w[iw]);
						complex eps = complex(prefac_vq / q_length_square, 0) / vqw;
						fprintf(fpvqw, "%10.3le %10.3le %10.3le %10.3le %10.3le\n", w[iw] / T, abs(vq(qvec[iq], w[iw])), eps.real(), eps.imag(), eps.abs());
					}
					fclose(fpvqw);
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
};