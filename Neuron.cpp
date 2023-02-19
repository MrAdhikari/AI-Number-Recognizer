#include "Neuron.h"

inline float S_Layer::sigmoid(float x)
{
	return (1 / (1 + pow(2.71828, (-x))));
}

void S_Layer::Comp_y(Layer_f* L)
{
	float tempY = 0;
	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 784; i++)
		{
			 tempY += (L->w[j * 784 + i] * L->X[i]);
		}
		L->Y[j] = sigmoid(tempY + L->b[j]);
		tempY = 0;
	}
}

void S_Layer::Comp_y(Layer_h* L)
{
	float tempY = 0;
	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 200; i++)
		{
			tempY += (L->w[j * 200 + i] * L->X[i]);
		}
		L->Y[j] = sigmoid(tempY + L->b[j]);
		tempY = 0;
	}
}

void S_Layer::Comp_y(Layer_l* L)
{
	float tempY = 0;
	for (int j = 0; j < 10; j++)
	{
		for (int i = 0; i < 200; i++)
		{
			tempY += (L->w[j * 200 + i] * L->X[i]);
		}
		L->Y[j] = sigmoid(tempY + L->b[j]);
		tempY = 0;
	}
}
