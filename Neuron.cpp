#include "Neuron.h"



void S_Layer::Set_All_wby_zero(E_Layer& L)
{
	memset(L.LF.w, 0, 784 * 200 * sizeof(float));
	memset(L.LF.dw, 0, 784 * 200 * sizeof(float));
	memset(L.LF.b, 0, 200 * sizeof(float));
	memset(L.LF.db, 0, 200 * sizeof(float));
	memset(L.LF.Y, 0, 200 * sizeof(float));

	for (int i = 0; i < 2; i++)
	{
		memset(L.LH[i].w, 0, 200 * 200 * sizeof(float));
		memset(L.LH[i].dw, 0, 200 * 200 * sizeof(float));
		memset(L.LF.b, 0, 200 * sizeof(float));
		memset(L.LH[i].db, 0, 200 * sizeof(float));
		memset(L.LH[i].Y, 0, 200 * sizeof(float));

	}

	memset(L.LL.w, 0, 200 * 10 * sizeof(float));
	memset(L.LL.dw, 0, 200 * 10 * sizeof(float));
	memset(L.LL.b, 0, 10 * sizeof(float));
	memset(L.LL.db, 0, 10 * sizeof(float));
	memset(L.LL.Y, 0, 10 * sizeof(float));
}


inline float S_Layer::sigmoid(float x)
{

	return (1.0f / (1.0f + pow(2.71828f, (-x)))); 
}

// X value should be given to L
void S_Layer::Comp_y(Layer_f& L)
{
	//L.y[] = { 0 };
	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 784; i++)
		{
			L.Y[j] += (L.w[j * 784 + i] * L.X[i]);
			//cuda conversion

		}
		L.Y[j] += L.b[j];
		L.Y[j] = sigmoid(L.Y[j]);

	}
}

void S_Layer::Comp_y(Layer_h& L)
{
	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 200; i++)
		{
			L.Y[j] += (L.w[j * 200 + i] * L.X[i]);
			//cuda conversion

		}
		L.Y[j] += L.b[j];
		L.Y[j] = sigmoid(L.Y[j]);

	}
}

void S_Layer::Comp_y(Layer_l& L)
{
	for (int j = 0; j < 10; j++)
	{
		for (int i = 0; i < 200; i++)
		{
			L.Y[j] += (L.w[j * 200 + i] * L.X[i]);
			//cuda conversion

		}
		L.Y[j] += L.b[j];
		L.Y[j] = sigmoid(L.Y[j]);

	}
}


//data should be loaded before execution
void S_Layer::Comp_upto_last(E_Layer& EL)
{
	//for first element of minimap 1
	EL.LF.X = &EL.io.X[784 * 0];
	Comp_y(EL.LF);

	EL.LH[0].X = EL.LF.Y;
	Comp_y(EL.LH[0]);

	EL.LH[1].X = EL.LF.Y;
	Comp_y(EL.LH[1]);
	
	EL.LL.X = EL.LH[1].Y;
	Comp_y(EL.LL);

	for (int i = 0; i < 10; i++)
	{
		EL.io.f[i] = EL.LL.Y[i];
	}
}

void S_Layer::cuda_comp_yl(E_Layer& EL)
{

}
