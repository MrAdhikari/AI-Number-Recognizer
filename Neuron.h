#include<iostream>
#include <cmath>

struct IO
{
	int x_digit[10];
	float *X = new float[784*10];
	float f[10];
	float t_f[10];
};

struct Layer_f
{
	float* X;
	float *w = new float[784 * 200];	float* dw = new float[784 * 200];
	float *b = new float[200];			float* db = new float[200];
	float *Y = new float[200];

};

struct Layer_h
{
	float* X;
	float *w = new float[200 * 200];	float* dw = new float[200 * 200];
	float *b = new float[200];			float* db = new float[200];
	float *Y = new float[200];

};

struct Layer_l
{
	float* X;
	float *w = new float[200 * 10];		float* dw = new float[200 * 10];
	float *b = new float[10];			float* db = new float[10];
	float *Y = new float[10];

};

class E_Layer
{
public:
	IO io;
	Layer_f LF;
	Layer_h LH[2];

	Layer_l LL;
};

class S_Layer
{
public:
	
	void Set_All_wby_zero(E_Layer& L);

	float sigmoid(float x);
	void Comp_y(Layer_f& L);
	void Comp_y(Layer_h& L);
	void Comp_y(Layer_l& L);
	

	//data should be loaded before execution
	void Comp_upto_last(E_Layer& EL);


	void cuda_comp_yl(E_Layer& EL);
		
};