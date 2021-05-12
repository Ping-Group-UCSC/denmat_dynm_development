#include "ElecElec_Model.h"

int elecelec_model::calc_P3P4(int ik1, int ika, complex* P3, complex* P4){
	// P3_{(k1,i1,i2),(ka,ia,ib)} = sum_{34} A_{(k1,i1),(k3,i3),(k4,i4),(ka,ia)} A_{(k1,i2),(k3,i3),(k4,i4),(ka,ib)}.conj() (1-f3) f4
	// P4_{(k1,i1,i2),(ka,ia,ib)} = sum_{34} A_{(k1,i1),(k3,i3),(k4,i4),(ka,ia)} A_{(k1,i2),(k3,i3),(k4,i4),(ka,ib)}.conj() f3 (1-f4)
	zeros(P3, nbpow4); zeros(P4, nbpow4);

	int nk3_count = 0;
	for (int ik3 = 0; ik3 < nk; ik3++){
		if (ika == ik3 && !eep.antisymmetry) continue;
		if (eep.antisymmetry && ika == ik3 && ik1 == ika) continue;
		size_t ik4 = 0; vector3<> k4 = elec->kvec[ik1] + elec->kvec[ik3] - elec->kvec[ika]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
		if (kmap->findk(k4, ik4)){
			// note that for A, I exchanged the 2nd and 3rd index compared with standard definition
			// here A_1234 = <1|vq|2> <3|4> delta(e1+e3-e2-e4)^0.5
			// while in paper, A_1234 = sqrt(2pi) <1|vq|3> <2|4> delta(e1+e2-e3-e4)^0.5
			if (!calc_A(ik1, (int)ik4, ik3, ika, true)) continue; // if energy conservation is not satisfied
			nk3_count++;

			for (int i1 = 0; i1 < nb; i1++)
			for (int ia = 0; ia < nb; ia++){

				for (int i4 = 0; i4 < nb; i4++){
					int n14 = (i1*nb + i4)*nb*nb;
					for (int i3 = 0; i3 < nb; i3++){
						int i43 = i4*nb + i3;
						int i143a = n14 + i3*nb + ia;
						A1rho[i43] = A1[i143a] * (1 - f[ik3][i3]) * f[ik4][i4];
						A1rhobar[i43] = A1[i143a] * f[ik3][i3] * (1 - f[ik4][i4]);
					}
				}

				for (int i2 = 0; i2 < nb; i2++){
					int n12 = (i1*nb + i2)*nb*nb;
					for (int ib = 0; ib < nb; ib++){
						int i12ab = n12 + ia*nb + ib;
						for (int i4 = 0; i4 < nb; i4++){
							int n24 = (i2*nb + i4)*nb*nb;
							for (int i3 = 0; i3 < nb; i3++){
								int i43 = i4*nb + i3;
								int i243b = n24 + i3*nb + ib;
								P3[i12ab] += A1rho[i43] * conj(A2[i243b]);
								P4[i12ab] += A1rhobar[i43] * conj(A2[i243b]);
							}
						}
					}
				}
			}
		}
	}
	complex prefac = complex(prefac_gauss / nk_full, 0);
	if (eep.antisymmetry) prefac *= 2;
	axbyc(P3, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
	axbyc(P4, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0

	return nk3_count;
}

int elecelec_model::calc_P5P6(int ik1, int ika, complex* P5, complex* P6){
	// P5_{(k1,i1,i2),(ka,ia,ib)} = sum_{34} A_{(k1,i1),(ka,ib),(k3,i3),(k4,i4)} A_{(k1,i2),(ka,ia),(k3,i3),(k4,i4)}.conj() f3 f4
	// P6_{(k1,i1,i2),(ka,ia,ib)} = sum_{34} A_{(k1,i1),(ka,ib),(k3,i3),(k4,i4)} A_{(k1,i2),(ka,ia),(k3,i3),(k4,i4)}.conj() (1-f3) (1-f4)
	zeros(P5, nbpow4); zeros(P6, nbpow4);

	int nk3_count = 0;
	for (int ik3 = 0; ik3 < nk; ik3++){
		if (ik1 == ik3 && !eep.antisymmetry) continue;
		if (eep.antisymmetry && ik1 == ik3 && ika == ik3) continue;
		size_t ik4 = 0; vector3<> k4 = elec->kvec[ik1] + elec->kvec[ika] - elec->kvec[ik3]; // not necessage to wrap k point around Gamma, kmap subroutines will wrap inside
		if (kmap->findk(k4, ik4)){
			// note that for A, I exchanged the 2nd and 3rd index compared with standard definition
			// here A_1234 = <1|vq|2> <3|4> delta(e1+e3-e2-e4)^0.5
			// while in paper, A_1234 = sqrt(2pi) <1|vq|3> <2|4> delta(e1+e2-e3-e4)^0.5
			if (!calc_A(ik1, ik3, ika, (int)ik4, true)) continue; // if energy conservation is not satisfied
			nk3_count++;

			for (int i1 = 0; i1 < nb; i1++)
			for (int ib = 0; ib < nb; ib++){

				for (int i3 = 0; i3 < nb; i3++){
					int n13 = (i1*nb + i3)*nb*nb;
					for (int i4 = 0; i4 < nb; i4++){
						int i34 = i3*nb + i4;
						int i13b4 = n13 + ib*nb + i4;
						A1rho[i34] = A1[i13b4] * f[ik3][i3] * f[ik4][i4];
						A1rhobar[i34] = A1[i13b4] * (1 - f[ik3][i3]) * (1 - f[ik4][i4]);
					}
				}

				for (int i2 = 0; i2 < nb; i2++){
					int n12 = (i1*nb + i2)*nb*nb;
					for (int ia = 0; ia < nb; ia++){
						int i12ab = n12 + ia*nb + ib;
						for (int i3 = 0; i3 < nb; i3++){
							int n23 = (i2*nb + i3)*nb*nb;
							for (int i4 = 0; i4 < nb; i4++){
								int i34 = i3*nb + i4;
								int i23a4 = n23 + ia*nb + i4;
								P5[i12ab] += A1rho[i34] * conj(A2[i23a4]);
								P6[i12ab] += A1rhobar[i34] * conj(A2[i23a4]);
							}
						}
					}
				}
			}
		}
	}
	complex prefac = complex(prefac_gauss / nk_full, 0);
	if (eep.antisymmetry) prefac *= 2;
	axbyc(P5, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0
	axbyc(P6, nullptr, nbpow4, c0, prefac); // y = ax + by + c with a = 0 and b = prefac and c = 0

	return nk3_count;
}