#pragma once
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>

struct GaussianSmapling
{
	double emin, de, emax;
	int ne;
	std::vector<double> out, out2, out3;
	std::vector<double> esampling;
	double degauss, prefacgauss, prefac_gaussexp;
	
	GaussianSmapling(double eminref, double de, double emaxref, double degauss)
		: emin(eminref - 7*degauss), de(de), emax(emaxref + 7*degauss), ne(int(ceil((emax - emin) / de))), 
		out(ne, 0.), out2(ne, 0), out3(ne, 0), esampling(ne), degauss(degauss),
		prefacgauss(1. / sqrt(2 * M_PI) / degauss), prefac_gaussexp(-0.5 / std::pow(degauss, 2))
	{
		esampling[0] = emin;
		for (int i = 1; i < ne; i++)
			esampling[i] = esampling[i - 1] + de;
	}

	void reset(){
		zeros(out.data(), ne);
		zeros(out2.data(), ne);
		zeros(out3.data(), ne);
	}
	
	void addEvent(double E, double weight){
		int ie = floor((E - emin) / de);
		int ne_ex = floor((7 * degauss) / de);
		ne_ex = std::min(1, ne_ex);
		int ie0 = std::max(0, ie - ne_ex);
		int ie1 = std::min(ie + ne_ex + 1, ne);
		for (int i = ie0; i < ie1; i++){
			out[i] += prefacgauss * weight * exp(prefac_gaussexp * std::pow(esampling[i] - E, 2));
		}
	}
	void addEvent2(double E, double weight){
		int ie = floor((E - emin) / de);
		int ne_ex = floor((7 * degauss) / de);
		ne_ex = std::min(1, ne_ex);
		int ie0 = std::max(0, ie - ne_ex);
		int ie1 = std::min(ie + ne_ex + 1, ne);
		for (int i = ie0; i < ie1; i++){
			out2[i] += prefacgauss * weight * exp(prefac_gaussexp * std::pow(esampling[i] - E, 2));
		}
	}
	void addEvent3(double E, double weight){
		int ie = floor((E - emin) / de);
		int ne_ex = floor((7 * degauss) / de);
		ne_ex = std::min(1, ne_ex);
		int ie0 = std::max(0, ie - ne_ex);
		int ie1 = std::min(ie + ne_ex + 1, ne);
		for (int i = ie0; i < ie1; i++){
			out3[i] += prefacgauss * weight * exp(prefac_gaussexp * std::pow(esampling[i] - E, 2));
		}
	}

	void print(FILE* fp, double Escale = 1.0, double histScale = 1.0) const{
		for (size_t i = 0; i < out.size(); i++)
			fprintf(fp, "%14.7le %14.7le\n", (emin + i*de)*Escale, out[i] * histScale);
	}
	void print(FILE* fp, double Escale, double histScale, double histScale_2) const{
		for (size_t i = 0; i < out.size(); i++)
			fprintf(fp, "%14.7le %14.7le\n", (emin + i*de)*Escale, out[i] * histScale, out[i] * histScale_2);
	}
	void print2(FILE* fp, double Escale = 1.0, double histScale = 1.0) const{
		for (size_t i = 0; i < out.size(); i++)
			fprintf(fp, "%14.7le %14.7le %14.7le\n", (emin + i*de)*Escale, out[i] * histScale, out2[i] * histScale);
	}
	void print3(FILE* fp, double Escale = 1.0, double histScale = 1.0) const{
		for (size_t i = 0; i < out.size(); i++)
			fprintf(fp, "%14.7le %14.7le %14.7le %14.7le\n", (emin + i*de)*Escale, out[i] * histScale, out2[i] * histScale, out3[i] * histScale);
	}
};
