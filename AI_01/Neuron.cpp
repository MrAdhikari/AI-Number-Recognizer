#include "Neuron.h"
#include "cmath"




void E_Layer::allocMemhost()
{
	io.X = new float[784 * 5];
	io.User_X = new float[784];
	io.Y = new float[10];
	io.Y_ref = new float[10];

	LF.w = new float[784 * 20];	 LF.dw = new float[784 * 20];
	LF.refw = new float[784 * 20];
	LF.b = new float[20];			 LF.db = new float[20];
	LF.Y = new float[20];




	LH[0].w = new float[20 * 20];		LH[0].dw = new float[20 * 20];
	LH[0].refw = new float[20 * 20];
	LH[0].b = new float[20];			LH[0].db = new float[20];
	LH[0].Y = new float[20];


	LH[1].w = new float[20 * 20];		LH[1].dw = new float[20 * 20];
	LH[1].refw = new float[20 * 20];
	LH[1].b = new float[20];			LH[1].db = new float[20];
	LH[1].Y = new float[20];




	LL.w = new float[20 * 10];		LL.dw = new float[20 * 10];
	LL.refw = new float[20 * 10];
	LL.b = new float[10];			LL.db = new float[10];
	LL.Y = new float[10];
}

float E_Layer::SqrMean(int numb)
{
	io.Y_ref[io.x_digit[numb]] -= 1;
	//std::cout << " d: " << io.x_digit[numb];
	io.cost = 0;

	for (int i = 0; i<10;i++)
	{
		io.cost += powf(io.Y_ref[i], 2);
	}

	return (io.cost/10.f);
}


void S_Layer::Set_All_wby_zero(E_Layer& L)
{
	memset(L.LF.w, 0, 784 * 20 * sizeof(float));
	memset(L.LF.refw, 0, 784 * 20 * sizeof(float));
	memset(L.LF.dw, 0, 784 * 20 * sizeof(float));
	memset(L.LF.b, 0, 20 * sizeof(float));
	memset(L.LF.db, 0, 20 * sizeof(float));
	memset(L.LF.Y, 0, 20 * sizeof(float));

	for (int i = 0; i < 2; i++)
	{
		memset(L.LH[i].w, 0, 20 * 20 * sizeof(float));
		memset(L.LH[i].refw, 0, 20 * 20 * sizeof(float));
		memset(L.LH[i].dw, 0, 20 * 20 * sizeof(float));
		memset(L.LH[i].b, 0, 20 * sizeof(float));
		memset(L.LH[i].db, 0, 20 * sizeof(float));
		memset(L.LH[i].Y, 0, 20 * sizeof(float));

	}

	memset(L.LL.w, 0, 20 * 10 * sizeof(float));
	memset(L.LL.refw, 0, 20 * 10 * sizeof(float));
	memset(L.LL.dw, 0, 20 * 10 * sizeof(float));
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
	for (int j = 0; j < 20; j++)
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
	for (int j = 0; j < 20; j++)
	{
		for (int i = 0; i < 20; i++)
		{
			L.Y[j] += (L.w[j * 20 + i] * L.X[i]);
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
		for (int i = 0; i < 20; i++)
		{
			L.Y[j] += (L.w[j * 20 + i] * L.X[i]);
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
	//EL.LF.X = &EL.io.X[784 * 0];
	//Comp_y(EL.LF);

	//EL.LH[0].X = EL.LF.Y;
	//Comp_y(EL.LH[0]);

	//EL.LH[1].X = EL.LF.Y;
	//Comp_y(EL.LH[1]);
	//
	//EL.LL.X = EL.LH[1].Y;
	//Comp_y(EL.LL);

	//for (int i = 0; i < 10; i++)
	//{
	//	EL.io.LF[i] = EL.LL.Y[i];
	//}
}


