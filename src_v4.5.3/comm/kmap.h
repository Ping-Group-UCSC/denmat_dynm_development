#pragma once
#include <map>
#include "vector3.h"
#include "myio.h"

static vector3<> wrap(const vector3<>& x, vector3<> center = vector3<>(0,0,0)){
	// wrap to [center - 0.5, center + 0.5)
	vector3<> result = x - center;
	for (int dir = 0; dir < 3; dir++){
		result[dir] -= floor(0.5 + result[dir]);
		if (fabs(result[dir] - 0.5) < 1e-6) result[dir] = -0.5;
	}
	result = result + center;
	return result;
}

struct kIndexMap{
	vector3<int> kmesh;
	std::map<vector3<int>, size_t> the_map;

	kIndexMap(vector3<int>& kmesh, std::vector<vector3<double>>& kvec)
		: kmesh(kmesh), the_map()
	{
		if (!the_map.empty()) error_message("the_map is not empty", "kIndexMap initialization");
		std::pair<std::map<vector3<int>, size_t>::iterator, bool> ret;
		for (size_t ik = 0; ik < kvec.size(); ik++)
			ret = the_map.insert(std::make_pair(ikvec3(kvec[ik]), ik));
	}
	bool findk(vector3<> k, size_t& ik);
	size_t k2ik(vector3<> k); // if you are sure q already exists in qIndexMap

	vector3<int> ikvec3(vector3<> k);
};

struct qIndexMap{
	vector3<int> kmesh;
	std::map<vector3<int>, size_t> the_map;

	qIndexMap(vector3<int>& kmesh) : kmesh(kmesh), the_map()
	{ if (!the_map.empty()) error_message("the_map is not empty", "qIndexMap initialization"); }

	void build(std::vector<vector3<double>>& kvec, std::vector<vector3<double>>& qvec){
		if (qvec.size() > 0) return;
		size_t iq = 0;
		std::pair<std::map<vector3<int>, size_t>::iterator, bool> ret;
		for (size_t ik = 0; ik < kvec.size(); ik++)
		for (size_t jk = 0; jk < kvec.size(); jk++){
			vector3<double> q = wrap(kvec[ik] - kvec[jk]);
			ret = the_map.insert(std::make_pair(iqvec3(q), iq));
			if (ret.second){
				qvec.push_back(q);
				iq++;
			}
		}
	}

	size_t q2iq(vector3<> q); // if you are sure q already exists in qIndexMap

	vector3<int> iqvec3(vector3<> q);
};