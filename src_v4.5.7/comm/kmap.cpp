#include "kmap.h"

bool kIndexMap::findk(vector3<> k, size_t& ik){
	const std::map<vector3<int>, size_t>::iterator iter = the_map.find(ikvec3(k));
	if (iter != the_map.end()){
		ik = iter->second;
		return true;
	}
	else return false;
}
size_t kIndexMap::k2ik(vector3<> k){ // if you are sure q already exists in qIndexMap
	return the_map[ikvec3(k)];
}

vector3<int> kIndexMap::ikvec3(vector3<> k){
	vector3<int> v3 = vector3<int>(0, 0, 0);
	for (int iDir = 0; iDir < 3; iDir++){
		double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
		v3[iDir] = (int)round(kmesh[iDir] * ki);
		if (v3[iDir] == kmesh[iDir]) v3[iDir] = 0;
	}
	return v3;
}

size_t qIndexMap::q2iq(vector3<> q){ // if you are sure q already exists in qIndexMap
	return the_map[iqvec3(q)];
}

vector3<int> qIndexMap::iqvec3(vector3<> q){
	vector3<int> v3 = vector3<int>(0, 0, 0);
	for (int iDir = 0; iDir < 3; iDir++){
		double qi = q[iDir] - floor(q[iDir]); //wrapped to [0,1)
		v3[iDir] = (int)round(kmesh[iDir] * qi);
		if (v3[iDir] == kmesh[iDir]) v3[iDir] = 0;
	}
	return v3;
}