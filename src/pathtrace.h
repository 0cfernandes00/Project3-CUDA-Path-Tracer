#pragma once

#include "scene.h"
#include "utilities.h"

#define DENOISE 1

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(
	uchar4 *pbo, int frame, int iteration,
	bool materialSort, bool russianRoulette,
	bool enableBVH, bool antiAlias, bool dof, Texture envMap);


