/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#ifndef FEYNWANN_INTEGRATOR_H
#define FEYNWANN_INTEGRATOR_H

template<typename Vector> class Integrator
{
public:
	//! Implementation in derived classes should return the time derivative of v
	virtual Vector compute(double t, const Vector& v)=0;
	
	//! Optional reporting: override to extract any intermediate outputs
	virtual void report(double t, const Vector& v) const {}
	
	//! Integrates from tStart to tEnd with fixed time steps dt, input and output in v
	void integrateFixed(Vector& v, const double tStart, const double tEnd, double dt, double dtRep);

	//! Integrates from tStart to tEnd with adaptive step size controlled by tolerance tol, input and output in v
	void integrateAdaptive(Vector& v, const double tStart, const double tEnd, double tol, double dtRep);

private:
	//! Move the state in t space along dt (RK4 method)
	//! Initial vPrime is pre-calculated and passed-in to save re-computation duing step-size adjustments
	void stepRK45(const Vector& v, const Vector& vPrime, double t, double dt, Vector& vOut, Vector& vErr);
	
	//Adaptive step where t is updated with a safe step, and next step-size to try is returned in dt
	int stepRK45adaptive(Vector& v, const Vector& vPrime, double& t, double& dt, double tol);
};

//---------------------- Implementation ----------------------------
template<typename Vector> void Integrator<Vector>::integrateFixed(Vector& v, const double tStart, const double tEnd, double dt, double dtRep)
{
	double t = tStart;
	double tRep = std::min(tStart + dtRep, tEnd);
	int nCalls = 0, nCallsPrev = 0;
	double percentPrev = 0.; //previously reported progress percent
	report(t, v);
	logPrintf("Integrate: Evolving: "); logFlush();
	while(t < tEnd)
	{	if(t+dt > tRep) dt = tRep-t; //prevent reporting overshoot
		Vector vPrime = compute(t, v), vErr;
		stepRK45(v, vPrime, t, dt, v, vErr);
		t += dt;
		nCalls += 6;
		//Check and optionally report progress:
		double percent = 100. - (100./dtRep)*(tRep-t);
		if(percent - percentPrev >= 5.)
		{	logPrintf("%d%% ", int(round(percent))); logFlush();
			percentPrev = percent;
		}
		if(fabs(t - tRep) < 1e-6*dtRep)
		{	logPrintf("Done (%d function calls).\n", nCalls-nCallsPrev); logFlush();
			nCallsPrev = nCalls;
			percentPrev = 0.;
			report(t, v);
			if(tEnd-t < 1e-6*dtRep) //simulation effectively done
				break;
			else
			{	tRep = std::min(tRep + dtRep, tEnd);
				logPrintf("Integrate: Evolving: "); logFlush();
			}
		}
	}
	logPrintf("Integrate: Completed with %d function calls.\n", nCalls);
}


template<typename Vector> void Integrator<Vector>::integrateAdaptive(Vector& v, const double tStart, const double tEnd, double tol, double dtRep)
{
	double t=tStart;
	double tRep = std::min(tStart + dtRep, tEnd);
	double dt = tol*(tEnd-tStart); //initial time step estimate
	int nCalls = 0, nCallsPrev = 0;
	double percentPrev = 0.; //previously reported progress percent
	report(t, v);
	logPrintf("Integrate: Evolving: "); logFlush();
	while(t < tEnd)
	{	if(t+dt > tRep) dt = tRep-t; //prevent reporting overshoot
		Vector vPrime = compute(t, v);
		nCalls += 1 + 5*stepRK45adaptive(v, vPrime, t, dt, tol);
		//Check and optionally report progress:
		double percent = 100. - (100./dtRep)*(tRep-t);
		if(percent - percentPrev >= 5.)
		{	logPrintf("%d%% ", int(round(percent))); logFlush();
			percentPrev = percent;
		}
		if(fabs(t - tRep) < 1e-6*dtRep)
		{	logPrintf("Done (%d function calls).\n", nCalls-nCallsPrev); logFlush();
			nCallsPrev = nCalls;
			percentPrev = 0.;
			report(t, v);
			if(tEnd-t < 1e-6*dtRep) //simulation effectively done
				break;
			else
			{	tRep = std::min(tRep + dtRep, tEnd);
				logPrintf("Integrate: Evolving: "); logFlush();
			}
		}
	}
	logPrintf("Integrate: Completed with %d function calls.\n\n", nCalls);
}

template<typename Vector>
void Integrator<Vector>::stepRK45(const Vector& v, const Vector& vPrime, double t, double dt, Vector& vOut, Vector& vErr)
{	static const double a2=0.2, a3=0.3, a4=0.6, a5=1.0, a6=0.875,
		b21= 0.2,
		b31=-0.125000000000000, b32= 0.225,
		b41= 0.225000000000000, b42=-1.125000000000000, b43= 1.2,
		b51=-0.503703703703704, b52= 3.400000000000000, b53=-3.792592592592593, b54= 1.296296296296296,
		b61= 0.233199508101852, b62=-2.158203125000000, b63= 2.634186921296296, b64=-0.895950882523148, b65=0.061767578125,
		 c1= 0.09788359788359788,  c3= 0.4025764895330113,  c4= 0.21043771043771045,                            c6= 0.2891022021456804,
		dc1= 0.00429377480158731, dc3=-0.0186685860938579, dc4= 0.03415502683080807, dc5= 0.01932198660714286, dc6=-0.0391022021456804;
	//Test step 1 (at initial point, compute called externally with result in vPrime):
	Vector vTmp = clone(v);
	const Vector& k1 = vPrime;
	//Test step 2:
	axpy(dt*b21, k1, vTmp);
	Vector k2 = compute(t+dt*a2, vTmp);
	//Test step 3:
	axpy(dt*b31, k1, vTmp);
	axpy(dt*b32, k2, vTmp);
	Vector k3 = compute(t+dt*a3, vTmp);
	//Test step 4:
	axpy(dt*b41, k1, vTmp);
	axpy(dt*b42, k2, vTmp);
	axpy(dt*b43, k3, vTmp);
	Vector k4 = compute(t+dt*a4, vTmp);
	//Test step 5:
	axpy(dt*b51, k1, vTmp);
	axpy(dt*b52, k2, vTmp);
	axpy(dt*b53, k3, vTmp);
	axpy(dt*b54, k4, vTmp);
	Vector k5 = compute(t+dt*a5, vTmp);
	//Test step 6:
	axpy(dt*b61, k1, vTmp);
	axpy(dt*b62, k2, vTmp);
	axpy(dt*b63, k3, vTmp);
	axpy(dt*b64, k4, vTmp);
	axpy(dt*b65, k5, vTmp);
	Vector k6 = compute(t+dt*a6, vTmp);
	//Final result:
	vOut = clone(v);
	axpy(dt*c1, k1, vOut);
	axpy(dt*c3, k3, vOut);
	axpy(dt*c4 ,k4, vOut);
	axpy(dt*c6 ,k6, vOut);
	//Error estimate:
	vErr = clone(k1); vErr *= (dt*dc1);
	axpy(dt*dc3, k3, vErr);
	axpy(dt*dc4, k4, vErr);
	axpy(dt*dc5, k5, vErr);
	axpy(dt*dc6, k6, vErr);
}


template<typename Vector>
int Integrator<Vector>::stepRK45adaptive(Vector& v, const Vector& vPrime, double& t, double& dt, double tol)
{
	static const double safetyFactor=0.9, shrinkExponent=-0.25, growExponent=-0.25;
	double relErr = 0.; Vector vTmp;
	int nSteps = 0; //number of times stepRK45 is called
	while(true)
	{	Vector vErr;
		stepRK45(v, vPrime, t, dt, vTmp, vErr);
		relErr = sqrt(dot(vErr,vErr))/tol;
		nSteps++;
		if(relErr <= 1.) break;
		dt *= std::max(0.2, safetyFactor * std::pow(relErr, shrinkExponent));
		//logPrintf("at t= %lf, dt = %.10e.\n", t,dt);
		if(t + dt == t)
			die("Stepsize underflow: accuracy could not be achieved at t=%lf\n", t);
	}
	t += dt;
	v = vTmp;
	dt *= std::min(safetyFactor * std::pow(relErr, growExponent), 5.);
	return nSteps;
}

#endif //FEYNWANN_INTEGRATOR_H
