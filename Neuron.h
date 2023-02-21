#include<iostream>
#include <cmath>

typedef struct IO
{
	float *X = new float[784];
	float *f[10];
	float *t_f[10];
};

typedef struct Layer_f
{
	float *X = new float[784];
	float *w = new float[784 * 200];	float* dw = new float[784 * 200];
	float *b = new float[200];			float* db = new float[200];
	float *Y = new float[200];

};

typedef struct Layer_h
{
	float *X = new float[200];
	float *w = new float[200 * 200];	float* dw = new float[200 * 200];
	float *b = new float[200];			float* db = new float[200];
	float *Y = new float[200];

};

typedef struct Layer_l
{
	float *X = new float[200];
	float *w = new float[200 * 10];		float* dw = new float[200 * 10];
	float *b = new float[10];			float* db = new float[10];
	float *Y = new float[10];

};

class E_Layer
{
public:
	IO io;
	Layer_f LF;
	Layer_h LH;
	Layer_l LL;
};

class S_Layer
{
	float sigmoid(float x);
	void Comp_y(Layer_f* L);
	void Comp_y(Layer_h* L);
	void Comp_y(Layer_l* L);
};