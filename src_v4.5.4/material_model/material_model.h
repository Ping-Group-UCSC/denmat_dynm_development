#pragma once
#include "parameters.h"
#include "MoS2_model.h"
#include "GaAs_model.h"

void init_model(parameters* param){
	if (material_model == "none") 
		return;
	else if (material_model == "mos2")
		mos2_model* model = new mos2_model(param);
	else if (material_model == "gaas")
		gaas_model* model = new gaas_model(param);
}